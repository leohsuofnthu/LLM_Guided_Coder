"""
Gemini-based LLM client utilities.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None

from .utils import sample_texts


class LLMClient(ABC):
    """Abstract interface describing LLM interactions needed by the pipeline."""

    @abstractmethod
    def propose_nets(self, question: str, num_nets: int, seeds_per_net: int, texts: Optional[Sequence[str]] = None) -> List[dict]:
        """Return a list of net proposals including seed phrases. Optionally accepts sample texts."""

    @abstractmethod
    def summarize_net(
        self,
        texts: Sequence[str],
        max_words: int,
        seeds_per_net: int,
        fallback_label: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Return an updated label + seed phrases for a net."""

    @abstractmethod
    def confirm_merge(self, label_a: str, label_b: str) -> bool:
        """Return True if two nets should be merged."""

    @abstractmethod
    def propose_split(self, texts: Sequence[str]) -> List[dict]:
        """Return candidate sublabels and seed phrases when splitting a net."""

    @abstractmethod
    def propose_subnets(
        self,
        parent_label: str,
        texts: Sequence[str],
        num_subnets: int,
        seeds_per_net: int,
        sibling_labels: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        """Return subnet proposals scoped under a parent label, narrower and non-overlapping with siblings."""

    @abstractmethod
    def propose_new_nets(
        self,
        texts: Sequence[str],
        max_nets: int,
        seeds_per_net: int,
        existing_labels: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        """Suggest new nets mined from the unknown bucket."""


class GeminiLLMClient(LLMClient):  # pragma: no cover - relies on external service
    """Google Gemini-backed implementation of the LLM client."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        if genai is None:
            raise ImportError("google-generativeai must be installed to use GeminiLLMClient.")
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided. Set gemini.api_key or GOOGLE_API_KEY.")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.model = genai.GenerativeModel(model_name)

    # ------------------------------------------------------------------
    # Helpers

    def _generate(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": max(self.temperature, 0.0)},
        )
        return getattr(response, "text", "") or ""

    @staticmethod
    def _extract_json(payload: str):
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
            if " - " not in ln:
                continue
            label_part, seeds_part = ln.split(" - ", 1)
            label = label_part.strip()
            raw_seeds = [s.strip() for s in seeds_part.split(";")]
            seeds = [s for s in raw_seeds if s]
            if not label or len(label.split()) > 8:  # be lenient but bounded
                continue
            if len(seeds) < 3:
                continue
            items.append({"label": label, "seeds": seeds})
            if max_items and len(items) >= max_items:
                break
        return items

    @staticmethod
    def _fallback_segments(text: str, max_segments: Optional[int]) -> List[str]:
        parts = re.split(r"[.;!?\\n]+", text)
        segments = [segment.strip() for segment in parts if segment.strip()]
        if max_segments:
            segments = segments[:max_segments]
        return segments or [text.strip()]

    # ------------------------------------------------------------------
    # Interface implementations

    def propose_nets(self, question: str, num_nets: int, seeds_per_net: int, texts: Optional[Sequence[str]] = None) -> List[dict]:
        samples_section = ""
        if texts:
            sample_texts_list = sample_texts(list(texts), min(len(texts), 20))
            samples_section = "Here are sample survey responses:\n" + "\n".join(f"- {t}" for t in sample_texts_list) + "\n\n"
        
        prompt = (
            "Given this survey question:\n"
            f"\"{question}\"\n"
            f"{samples_section}"
            f"Propose {num_nets} top-level categories (<=5 words each). Ensure categories are mutually distinct, non-overlapping, and avoid paraphrases. For each category include {seeds_per_net} representative short phrases.\n\n"
            "Preferred output 1 (JSON list of objects with 'label' and 'seeds'):\n"
            '[{"label": "Whitening Effect", "seeds": ["whiter teeth", "bright smile", "shade lighter"]}]\n\n'
            "Accepted output 2 (one per line):\n"
            "Label - seed1; seed2; seed3; seed4\n"
            "Output:"
        )
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        if isinstance(result, list):
            return result
        # fallback simple line parser
        return self._parse_line_proposals(raw, max_items=num_nets)

    def summarize_net(
        self,
        texts: Sequence[str],
        max_words: int,
        seeds_per_net: int,
        fallback_label: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict:
        samples = "\n".join(f"- {t}" for t in sample_texts(list(texts), min(len(texts), 40)))
        header = f"{system_prompt}\n" if system_prompt else ""
        prompt = (
            f"{header}"
            "Summarize these survey excerpts into a concise category label (<= "
            f"{max_words} words) and provide {seeds_per_net} representative short phrases. "
            "Respond as JSON with keys 'label' and 'seeds'.\n"
            f"{samples}"
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
            "Do these two survey categories describe the same concept?\n"
            f"A: {label_a}\nB: {label_b}\n"
            "Answer ONLY 'same' or 'different'."
        )
        response = self._generate(prompt).lower()
        return "same" in response

    def propose_split(self, texts: Sequence[str]) -> List[dict]:
        samples = "\n".join(f"- {t}" for t in sample_texts(list(texts), min(len(texts), 40)))
        prompt = (
            "These survey comments contain two distinct themes. Provide 2 short labels (<=4 words) "
            "and 5 representative phrases per label. Ensure the two labels are clearly different from each other and from the parent category name. Respond as JSON list of objects with 'label' "
            "and 'seeds'.\n"
            f"{samples}"
        )
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        if isinstance(result, list):
            return result
        # fallback simple line parser (accept top 2)
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
            f"Parent category: {parent_label}\n"
            f"Propose {num_subnets} subcategories (<=4 words each) with {seeds_per_net} "
            "representative phrases per subcategory. Make each subcategory:\n"
            "1. Narrower and more specific than the parent\n"
            "2. Mutually exclusive with other subcategories\n"
            "3. Non-overlapping with the parent or any sibling subcategories\n"
            f"{sibling_clause}"
            "Respond as JSON list of objects with 'label' and 'seeds'.\n"
            f"{samples}"
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
            f"Provide <=5 word label and {seeds_per_net} short representative phrases per category. Each label must be meaningfully different from the existing categories already covered; skip or merge any redundant ideas.{existing_clause}"
            "Respond as JSON list of objects with 'label' and 'seeds'.\n"
            f"{samples}"
        )
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        if isinstance(result, list):
            return result
        return self._parse_line_proposals(raw, max_items=max_nets)
