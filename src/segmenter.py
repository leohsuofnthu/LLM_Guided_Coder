"""
Segmentation logic for breaking survey responses into atomic idea units.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
import pysbd

from .config import SegmenterConfig
from .utils import get_logger


class Segmenter:
    """PySBD-based segmentation for consistent sentence splitting."""

    def __init__(
        self,
        config: SegmenterConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or get_logger(__name__)
        if getattr(self.config, "strategy", "pysbd") != "pysbd":
            self.logger.warning("Segmenter only supports PySBD; overriding strategy to 'pysbd'.")
            self.config.strategy = "pysbd"  # type: ignore[attr-defined]
        self._pysbd = pysbd.Segmenter(language=config.pysbd_language, clean=True)

    def segment_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.text_column not in df.columns:
            raise KeyError(f"Input DataFrame missing required column '{self.config.text_column}'")

        records: List[dict] = []
        for idx, row in df.iterrows():
            response_id = self._resolve_response_id(row, idx)
            raw_text = str(row[self.config.text_column])
            text = raw_text.strip()
            if not text:
                continue
            segments = self._segment_with_pysbd(text)
            for local_id, segment in enumerate(segments):
                segment = segment.strip()
                if not segment:
                    continue
                segment_id = f"{response_id}_{local_id}"
                records.append(
                    {
                        "response_id": response_id,
                        "segment_id": segment_id,
                        "segment_text": segment,
                        "response_text": text,
                    }
                )
        return pd.DataFrame.from_records(records)

    def _segment_with_pysbd(self, text: str) -> List[str]:
        raw_segments = [seg.strip() for seg in self._pysbd.segment(text)]
        if not raw_segments:
            return [text.strip()]
        if self.config.max_segments_per_response:
            raw_segments = raw_segments[: self.config.max_segments_per_response]
        return raw_segments

    @staticmethod
    def _resolve_response_id(row: pd.Series, idx: int) -> str:
        for candidate in ("response_id", "ResponseID", "id", "ID"):
            if candidate in row and pd.notna(row[candidate]):
                return str(row[candidate])
        return f"resp_{idx}"
