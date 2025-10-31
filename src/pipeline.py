"""
End-to-end orchestration for the survey topic discovery workflow.
"""

from __future__ import annotations

import json
import logging
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .coder import Coder
from .config import PipelineConfig, load_config
from .embedder import Embedder
from .llm_client import GeminiLLMClient, LLMClient
from .recursive_net import RecursiveNetDiscovery
from .segmenter import Segmenter

try:
    from .llm_client_hf import HuggingFaceLLMClient
except ImportError:
    HuggingFaceLLMClient = None
from .utils import (
    ensure_directory,
    get_logger,
    write_dataframe,
    write_json,
)


class SurveyTopicPipeline:
    """High-level pipeline that runs segmentation, embedding, discovery, and coding."""

    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or get_logger("survey_topic_pipeline")
        self.config.ensure_storage_paths()

        self.segmenter = Segmenter(self.config.segmenter, logger=self.logger.getChild("segmenter"))
        self.embedder = Embedder(self.config.embedder, logger=self.logger.getChild("embedder"))
        self.coder = Coder(self.config.coder, logger=self.logger.getChild("coder"))

        self.llm_client: LLMClient = self._build_llm_client()

    # ------------------------------------------------------------------ Public API

    def run(
        self,
        *,
        csv_path: Path | str,
        question: Optional[str] = None,
        text_column: Optional[str] = None,
        force_recompute: bool = False,  # noqa: ARG002 - retained for API compatibility
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline and return a dictionary of run statistics.

        Parameters
        ----------
        csv_path:
            Path to the CSV file containing raw survey responses.
        question:
            The survey question text to use for LLM prompts during discovery.
        text_column:
            Optional override for the column containing free-text responses.
        force_recompute:
            Retained for backwards compatibility. Currently always recomputes outputs.
        """

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        self.logger.info("Loading responses from %s", csv_path)

        responses_df = pd.read_csv(csv_path)
        effective_column = text_column or self.config.segmenter.text_column
        self.segmenter.config.text_column = effective_column

        segments_df = self.segmenter.segment_responses(responses_df)
        self.logger.info("Segmented %s responses into %s segments.", len(responses_df), len(segments_df))
        self._persist_segments(segments_df)

        embeddings = self.embedder.embed(segments_df["segment_text"].tolist())
        self.logger.info("Computed embeddings with shape %s.", embeddings.shape)
        self._persist_embeddings(embeddings)

        run_stats: Dict[str, Any]
        if self.config.discovery.enabled:
            run_stats = self._run_recursive_discovery(
                question or "Identify the key themes in these survey responses.",
                segments_df,
                embeddings,
            )
        else:
            run_stats = self._run_cluster_label_flow(segments_df, embeddings)

        run_stats.setdefault("artifacts", {})
        run_stats["artifacts"].update(
            {
                "segments": str(self.config.storage.segments_path),
                "embeddings": str(self.config.storage.embeddings_path),
                "assignments": str(self.config.storage.assignments_path),
                "topic_hierarchy": str(self.config.storage.topic_hierarchy_path),
            }
        )
        return run_stats

    # ------------------------------------------------------------------ Internal helpers

    def _build_llm_client(self) -> LLMClient:
        if self.config.llm_backend == "huggingface":
            if HuggingFaceLLMClient is None:
                raise ImportError(
                    "HuggingFace backend requires transformers and torch. "
                    "Install with: pip install transformers torch"
                )
            hf_cfg = self.config.huggingface
            return HuggingFaceLLMClient(
                model_name=hf_cfg.model_name,
                device=hf_cfg.device,
                temperature=hf_cfg.temperature,
                max_new_tokens=hf_cfg.max_new_tokens,
                load_in_8bit=hf_cfg.load_in_8bit,
                load_in_4bit=hf_cfg.load_in_4bit,
            )
        else:  # default to gemini
            gemini_cfg = self.config.gemini
            return GeminiLLMClient(
                model_name=gemini_cfg.model_name,
                api_key=gemini_cfg.api_key,
                temperature=gemini_cfg.temperature,
            )

    def _persist_segments(self, segments: pd.DataFrame) -> None:
        path = self.config.storage.segments_path
        write_dataframe(segments, path)
        self.logger.info("Persisted segments to %s", path)

    def _persist_embeddings(self, embeddings: np.ndarray) -> None:
        path = self.config.storage.embeddings_path
        ensure_directory(path)
        np.save(path, embeddings)
        self.logger.info("Persisted embeddings to %s", path)

    def _persist_assignments(self, assignments: pd.DataFrame) -> None:
        path = self.config.storage.assignments_path
        to_store = assignments.copy()
        if "alternative_nets" in to_store.columns:
            to_store["alternative_nets"] = to_store["alternative_nets"].apply(
                lambda value: json.dumps(value) if isinstance(value, list) else value
            )
        write_dataframe(to_store, path)
        self.logger.info("Persisted assignments to %s", path)

    def _persist_topic_hierarchy(self, hierarchy: Dict[str, Any]) -> None:
        path = self.config.storage.topic_hierarchy_path
        write_json(hierarchy, path)
        self.logger.info("Persisted topic hierarchy to %s", path)

    def _persist_nets(self, nets_payload: Dict[str, Any]) -> None:
        path = self.config.storage.nets_path
        write_json(nets_payload, path)
        self.logger.info("Persisted net registry to %s", path)

    def _run_recursive_discovery(
        self,
        question: str,
        segments_df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        discovery_logger = self.logger.getChild("discovery")
        discovery = RecursiveNetDiscovery(
            question=question,
            segments_df=segments_df,
            embeddings=embeddings,
            embedder=self.embedder,
            config=self.config.discovery,
            llm_client=self.llm_client,
            global_registry=[],
            labeler_prompt=self.config.labeler.system_prompt,
            logger=discovery_logger,
        )
        result = discovery.run()

        assignments_df = result.assignments
        self._persist_assignments(assignments_df)
        self._persist_topic_hierarchy(result.topic_hierarchy)

        nets_payload: Dict[str, Any] = {
            "llm_identifier": result.llm_identifier,
            "nets": [],
            "history": [asdict(record) for record in result.history],
        }
        for net in result.nets:
            payload = net.to_payload()
            payload["member_count"] = len(net.members)
            payload["iteration"] = net.iteration
            nets_payload["nets"].append(payload)
        self._persist_nets(nets_payload)

        coded = self.coder.assign(segments_df, assignments_df, result.label_lookup)
        self.coder.write_output(coded)

        run_stats: Dict[str, Any] = {
            "llm_calls": result.llm_calls,
            "llm_identifier": result.llm_identifier,
            "iterations_completed": len(result.history),
            "nets_total": len(result.nets),
            "iteration_metrics": result.iteration_metrics,
        }
        return run_stats

    def _run_cluster_label_flow(
        self,
        segments_df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        # Simple fallback path when recursive discovery is disabled.
        self.logger.info("Discovery disabled; all segments assigned to Unknown.")

        # In this simplified path, assignments default everyone into Unknown.
        assignments = pd.DataFrame(
            {
                "segment_id": segments_df["segment_id"],
                "net_id": ["Unknown"] * len(segments_df),
                "subnet_id": ["Unknown"] * len(segments_df),
            }
        )
        self._persist_assignments(assignments)

        hierarchy = {"nets": {}}
        self._persist_topic_hierarchy(hierarchy)

        label_lookup = {"nets": {}, "subnets": {}}
        coded = self.coder.assign(segments_df, assignments, label_lookup)
        self.coder.write_output(coded)

        return {
            "llm_calls": 0,
            "llm_identifier": getattr(self.llm_client, "model_name", "unknown"),
            "iterations_completed": 0,
            "nets_total": 0,
            "iteration_metrics": [],
        }


# ---------------------------------------------------------------------- Convenience API

def run_pipeline(
    csv_path: str | Path,
    *,
    config_path: Optional[str | Path] = None,
    question: Optional[str] = None,
    text_column: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function for running the pipeline without instantiating the class manually.
    """
    cfg = load_config(Path(config_path)) if config_path else PipelineConfig()
    pipeline = SurveyTopicPipeline(cfg)
    return pipeline.run(
        csv_path=csv_path,
        question=question,
        text_column=text_column,
        force_recompute=force,
    )


def _main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run the survey topic discovery pipeline.")
    parser.add_argument(
        "--input",
        "--csv-path",
        dest="csv_path",
        default="survey_topic_pipeline/data/responses.csv",
        help="Path to input CSV containing responses.",
    )
    parser.add_argument(
        "--config-path",
        dest="config_path",
        default=None,
        help="Optional path to a YAML/JSON pipeline configuration.",
    )
    parser.add_argument(
        "--text-column",
        dest="text_column",
        default=None,
        help="Name of the column that contains free-text responses.",
    )
    parser.add_argument(
        "--question",
        dest="question",
        default=None,
        help="Survey question prompt used for LLM guidance.",
    )
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Retained for compatibility; forces recomputation of artifacts.",
    )
    args = parser.parse_args(argv)

    stats = run_pipeline(
        csv_path=args.csv_path,
        config_path=args.config_path,
        question=args.question,
        text_column=args.text_column,
        force=args.force,
    )
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    _main()
