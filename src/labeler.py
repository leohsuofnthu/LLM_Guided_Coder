"""
LLM-based labeling of clustered survey segments.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import LabelerConfig
from .llm_client import LLMClient
from .utils import get_logger, sample_texts

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "have",
    "this",
    "from",
    "they",
    "their",
    "about",
    "would",
    "could",
    "should",
    "just",
    "into",
    "there",
    "were",
    "been",
    "when",
    "where",
    "while",
    "some",
    "only",
    "other",
    "more",
    "than",
    "because",
    "also",
    "these",
    "those",
}


class Labeler:
    """Generate short, human-readable labels for clusters."""

    def __init__(
        self,
        config: LabelerConfig,
        llm_client: LLMClient,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.llm_client = llm_client

    def label_clusters(
        self,
        df_segments: pd.DataFrame,
        df_clusters: pd.DataFrame,
    ) -> Tuple[Dict, Dict]:
        """Create topic hierarchy and label lookup dictionaries."""
        merged = df_segments.merge(df_clusters, on="segment_id", how="inner")
        topic_hierarchy = {"nets": {}}
        label_lookup = {"nets": {}, "subnets": {}}

        for net_id, net_df in merged.groupby("net_id"):
            net_examples = sample_texts(
                net_df["segment_text"].tolist(), self.config.max_examples_per_cluster
            )
            net_label = self._summarise_texts(net_examples)
            topic_hierarchy["nets"][net_id] = {
                "label": net_label,
                "examples": net_examples,
                "subnets": {},
            }
            label_lookup["nets"][net_id] = net_label

            for subnet_id, sub_df in net_df.groupby("subnet_id"):
                subnet_examples = sample_texts(
                    sub_df["segment_text"].tolist(), self.config.max_examples_per_cluster
                )
                subnet_label = self._summarise_texts(subnet_examples)
                topic_hierarchy["nets"][net_id]["subnets"][subnet_id] = {
                    "label": subnet_label,
                    "examples": subnet_examples,
                }
                label_lookup["subnets"][f"{net_id}|{subnet_id}"] = subnet_label

        return topic_hierarchy, label_lookup

    def _summarise_texts(self, texts: Sequence[str]) -> str:
        if not texts:
            return "Miscellaneous"

        try:
            payload = self.llm_client.summarize_net(
                texts,
                max_words=self.config.max_words,
                seeds_per_net=0,
                fallback_label=None,
                system_prompt=self.config.system_prompt,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.logger.warning("LLM labeling failed (%s). Using fallback summary.", exc)
            return self._fallback_keyword_summary(texts)

        label = payload.get("label") if isinstance(payload, dict) else None
        if not isinstance(label, str) or not label.strip():
            return self._fallback_keyword_summary(texts)
        return self._truncate_words(label.strip())

    def _fallback_keyword_summary(self, texts: Sequence[str]) -> str:
        tokens = re.findall(r"[A-Za-z]{3,}", " ".join(texts).lower())
        counts = Counter(t for t in tokens if t not in STOPWORDS)
        if not counts:
            return "General Feedback"
        top_words = [word for word, _ in counts.most_common(self.config.max_words)]
        label = " ".join(top_words[: self.config.max_words])
        return self._truncate_words(label or "General Feedback")

    def _truncate_words(self, label: str) -> str:
        words = label.split()
        return " ".join(words[: self.config.max_words])
