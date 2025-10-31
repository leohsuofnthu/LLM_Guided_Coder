"""
HuggingFace Transformers-based LLM client for local model inference.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence

from .llm_client import LLMClient
from .utils import sample_texts


class HuggingFaceLLMClient(LLMClient):
    """Local HuggingFace model implementation of the LLM client."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize HuggingFace LLM client with local model.
        
        Parameters
        ----------
        model_name : str
            HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct", 
            "mistralai/Mistral-7B-Instruct-v0.2", "microsoft/Phi-3-mini-4k-instruct")
        device : str
            Device to run on ("auto", "cuda", "cpu")
        temperature : float
            Sampling temperature (0.0 = deterministic)
        max_new_tokens : int
            Maximum tokens to generate
        load_in_8bit : bool
            Use 8-bit quantization (requires bitsandbytes)
        load_in_4bit : bool
            Use 4-bit quantization (requires bitsandbytes)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HuggingFace models. "
                "Install it via `pip install transformers torch`."
            ) from exc

        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading model {model_name}... (this may take a few minutes)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optional quantization
        model_kwargs = {"device_map": device}
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
            trust_remote_code=False,
            attn_implementation="eager",  # Avoid flash attention issues
        )
        
        # Create text generation pipeline with proper defaults
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),  # Avoid exact 0
            do_sample=temperature > 0.01,
            repetition_penalty=1.2,  # Prevent repetition/gibberish
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
        )
        
        print(f"Model loaded successfully on {device}!")

    def _generate(self, prompt: str) -> str:
        """Generate text from prompt using the local model."""
        # Phi-3 uses a specific chat template format
        # Format: <|user|>\nprompt<|end|>\n<|assistant|>\n
        if "Phi-3" in self.model_name or "phi-3" in self.model_name:
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            # Use chat template for other instruct models
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                formatted_prompt = prompt
        else:
            # Direct prompt for base models
            formatted_prompt = prompt
        
        # Generate with stop tokens
        stop_strings = ["<|end|>", "<|user|>", "\n\n\n"]  # Stop at these tokens
        result = self.pipeline(
            formatted_prompt,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        output = result[0]["generated_text"].strip()
        
        # Clean up stop tokens if they appear in output
        for stop_str in stop_strings:
            if stop_str in output:
                output = output.split(stop_str)[0].strip()
        
        # Clean up: remove repeated tokens/nonsense if model goes haywire
        if output and len(set(output[:50].split())) < 3:  # Too repetitive
            print(f"[WARNING] Detected repetitive output, returning empty")
            return ""
        
        return output

    @staticmethod
    def _extract_json(payload: str):
        """Extract JSON from model output."""
        # Try to find JSON array or object
        match = re.search(r"(\[.*\]|\{.*\})", payload, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_line_proposals(payload: str, max_items: Optional[int] = None) -> List[dict]:
        """Fallback parser for simple line format: 'Label - seed1; seed2; seed3' per line."""
        items: List[dict] = []
        if not isinstance(payload, str) or not payload.strip():
            return items
        lines = [ln.strip() for ln in payload.strip().splitlines() if ln.strip()]
        for ln in lines:
            # Try " - " separator first
            if " - " in ln:
                label_part, seeds_part = ln.split(" - ", 1)
            # Try ": " as alternative
            elif ": " in ln and ln.count(":") == 1:
                label_part, seeds_part = ln.split(": ", 1)
            # Try numbered/bullet format "1. Label - seeds" or "- Label - seeds"
            elif re.match(r'^[0-9\-\.]\s+', ln):
                ln = re.sub(r'^[0-9\-\.]\s+', '', ln)
                if " - " in ln:
                    label_part, seeds_part = ln.split(" - ", 1)
                else:
                    continue
            else:
                continue
            
            label = label_part.strip()
            # Remove quotes if present
            label = label.strip('"\'')
            # Remove markdown bold formatting (**, __, etc.)
            label = re.sub(r'\*\*(.+?)\*\*', r'\1', label)  # **text**
            label = re.sub(r'__(.+?)__', r'\1', label)      # __text__
            label = re.sub(r'\*(.+?)\*', r'\1', label)      # *text*
            label = re.sub(r'_(.+?)_', r'\1', label)        # _text_
            label = label.strip()
            
            # Parse seeds - try semicolon, comma, or space
            raw_seeds = seeds_part.split(";") if ";" in seeds_part else seeds_part.split(",")
            if len(raw_seeds) < 3 and " " in seeds_part and "," not in seeds_part and ";" not in seeds_part:
                # Last resort: split by spaces, take first 5-7 words as seeds
                words = seeds_part.split()
                seeds = [" ".join(words[i:i+2]) for i in range(0, min(7, len(words)), 2)]
            else:
                seeds = [s.strip().strip('"\'') for s in raw_seeds if s.strip()]
            
            # Validation - be more lenient
            if not label or len(label.split()) > 10:  # Allow up to 10 words
                continue
            if len(seeds) < 2:  # Reduced from 3 to 2
                continue
            
            items.append({"label": label, "seeds": seeds})
            if max_items and len(items) >= max_items:
                break
        return items

    # ------------------------------------------------------------------
    # Interface implementations

    def propose_nets(self, question: str, num_nets: int, seeds_per_net: int, texts: Optional[Sequence[str]] = None) -> List[dict]:
        samples_section = ""
        if texts:
            from .utils import sample_texts
            sample_texts_list = sample_texts(list(texts), min(len(texts), 15))
            samples_section = "Here are sample survey responses:\n" + "\n".join(f"- {t}" for t in sample_texts_list) + "\n\n"
        
        prompt = (
            "You are a data analyst categorizing survey responses.\n\n"
            f"Question: {question}\n\n"
            f"{samples_section}"
            f"Create {num_nets} distinct categories (2-5 words each). For each category, list {seeds_per_net} short example phrases.\n\n"
            "Use this format (one category per line):\n"
            "Category Name - phrase1; phrase2; phrase3; phrase4\n\n"
            "Example:\n"
            "Whitening Effect - whiter teeth; bright smile; shade lighter; stain removal\n"
            "Fresh Breath - minty fresh; lasting freshness; clean breath\n\n"
            "Your output:"
        )
        raw = self._generate(prompt)
        # Log raw output for debugging (first 500 chars)
        print(f"[PHI3 RAW OUTPUT] {raw[:500]}...")
        result = self._extract_json(raw)
        if isinstance(result, list):
            print(f"[PHI3 PARSED] JSON format: {len(result)} items")
            return result
        parsed = self._parse_line_proposals(raw, max_items=num_nets)
        if parsed:
            print(f"[PHI3 PARSED] Line format: {len(parsed)} items")
        else:
            print(f"[PHI3 PARSED] FAILED - No valid proposals extracted from: {raw[:200]}")
        return parsed

    def summarize_net(
        self,
        texts: Sequence[str],
        max_words: int,
        seeds_per_net: int,
        fallback_label: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        samples = "\n".join(f"- {t}" for t in sample_texts(list(texts), min(len(texts), 40)))
        header = f"{system_prompt}\n\n" if system_prompt else ""
        prompt = (
            f"{header}"
            "Summarize these survey excerpts into a concise category label "
            f"(<= {max_words} words) and provide {seeds_per_net} representative short phrases.\n\n"
            "Survey excerpts:\n"
            f"{samples}\n\n"
            "Respond ONLY with valid JSON with 'label' and 'seeds' keys.\n"
            "Example: {\"label\": \"Customer Service\", \"seeds\": [\"helpful staff\", \"quick response\"]}\n\n"
            "JSON output:"
        )
        result = self._extract_json(self._generate(prompt))
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result:
            return {
                "label": result[0].get("label", fallback_label or "General Theme"),
                "seeds": result[0].get("seeds", []),
            }
        return {"label": fallback_label or "General Theme", "seeds": []}

    def confirm_merge(self, label_a: str, label_b: str) -> bool:
        prompt = (
            "Do these two survey categories describe the same concept?\n\n"
            f"Category A: {label_a}\n"
            f"Category B: {label_b}\n\n"
            "Answer ONLY with 'same' or 'different' (one word).\n"
            "Answer:"
        )
        response = self._generate(prompt).lower()
        return "same" in response

    def propose_split(self, texts: Sequence[str]) -> List[dict]:
        samples = "\n".join(f"- {t}" for t in sample_texts(list(texts), min(len(texts), 40)))
        prompt = (
            "These survey comments contain two distinct themes. "
            "Provide 2 short labels (<=4 words) and 5 representative phrases per label. "
            "Ensure the two labels are clearly different from each other.\n\n"
            "Survey comments:\n"
            f"{samples}\n\n"
            "Respond ONLY with valid JSON as a list of 2 objects with 'label' and 'seeds' keys.\n"
            "JSON output:"
        )
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        if isinstance(result, list):
            return result
        return self._parse_line_proposals(raw, max_items=2)

    def propose_subnets(
        self,
        parent_label: str,
        texts: Sequence[str],
        num_subnets: int,
        seeds_per_net: int,
        sibling_labels: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        samples = "\n".join(f"- {t}" for t in sample_texts(list(texts), min(len(texts), 40)))
        sibling_clause = ""
        if sibling_labels:
            formatted = "\n".join(f"- {label}" for label in sibling_labels)
            sibling_clause = (
                "\nExisting sibling subcategories under the same parent:\n"
                f"{formatted}\n"
                "Ensure your proposals are distinct from these siblings and do not overlap.\n"
            )
        prompt = (
            f"Parent category: {parent_label}\n\n"
            f"Propose {num_subnets} subcategories (<=4 words each) with {seeds_per_net} "
            "representative phrases per subcategory. Make each subcategory:\n"
            "1. Narrower and more specific than the parent\n"
            "2. Mutually exclusive with other subcategories\n"
            "3. Non-overlapping with the parent or any sibling subcategories\n"
            f"{sibling_clause}\n"
            "Survey comments:\n"
            f"{samples}\n\n"
            "Respond ONLY with valid JSON as a list of objects with 'label' and 'seeds' keys.\n"
            "JSON output:"
        )
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        if isinstance(result, list):
            return result
        return self._parse_line_proposals(raw, max_items=num_subnets)

    def propose_new_nets(
        self,
        texts: Sequence[str],
        max_nets: int,
        seeds_per_net: int,
        existing_labels: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        samples = "\n".join(f"- {t}" for t in sample_texts(list(texts), min(len(texts), 40)))
        existing_clause = ""
        if existing_labels:
            formatted = "\n".join(f"- {label}" for label in existing_labels)
            existing_clause = (
                "\nExisting categories already covered:\n"
                f"{formatted}\n"
                "Do not repeat or paraphrase these themes.\n"
            )
        prompt = (
            f"Identify up to {max_nets} new categories from these survey comments. "
            f"Provide <=5 word label and {seeds_per_net} short representative phrases per category. "
            "Each label must be meaningfully different from the existing categories already covered; "
            "skip or merge any redundant ideas."
            f"{existing_clause}\n"
            "Survey comments:\n"
            f"{samples}\n\n"
            "Respond ONLY with valid JSON as a list of objects with 'label' and 'seeds' keys.\n"
            "JSON output:"
        )
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        if isinstance(result, list):
            return result
        return self._parse_line_proposals(raw, max_items=max_nets)

