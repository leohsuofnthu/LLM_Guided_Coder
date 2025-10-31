"""
Map generated topic codes back to original survey responses.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from .config import CoderConfig
from .utils import get_logger, write_json


class Coder:
    """Align segment-level labels with original responses."""

    def __init__(self, config: CoderConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or get_logger(__name__)

    def assign(
        self,
        df_segments: pd.DataFrame,
        df_clusters: pd.DataFrame,
        label_lookup: Dict[str, Dict[str, str]],
    ) -> List[Dict]:
        merged = df_segments.merge(df_clusters, on="segment_id", how="inner")
        net_labels = label_lookup.get("nets", {})
        subnet_labels = label_lookup.get("subnets", {})

        coded_responses: List[Dict] = []
        for response_id, group in merged.groupby("response_id"):
            segments = []
            for _, row in group.iterrows():
                net_id = row["net_id"]
                subnet_id = row["subnet_id"]
                label_key = f"{net_id}|{subnet_id}"
                segments.append(
                    {
                        "segment_id": row["segment_id"],
                        "text": row["segment_text"],
                        "net_id": net_id,
                        "net": net_labels.get(net_id, net_id),
                        "subnet_id": subnet_id,
                        "subnet": subnet_labels.get(label_key, subnet_id),
                    }
                )
            coded_responses.append({"response_id": response_id, "segments": segments})
        return coded_responses

    def write_output(self, payload: List[Dict]) -> None:
        write_json(payload, self.config.output_path)
        self.logger.info("Wrote coded responses to %s", self.config.output_path)

