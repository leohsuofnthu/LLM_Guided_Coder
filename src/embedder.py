"""
Embedding generation for segmented survey texts.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np

from .config import EmbedderConfig
from .utils import batched, cosine_normalize, get_logger

MODEL = None


def _load_sentence_transformer(config: EmbedderConfig):
    """Load or cache the sentence transformer model."""
    global MODEL
    if MODEL is not None:
        return MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "sentence-transformers is required for embedding generation. "
            "Install it via `pip install sentence-transformers`."
        ) from exc

    device = config.device
    if device is None:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch optional
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL = SentenceTransformer(
        config.model_name,
        device=device,
        cache_folder=config.cache_dir,
    )
    return MODEL


class Embedder:
    """Generate embeddings for segment texts."""

    def __init__(self, config: EmbedderConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or get_logger(__name__)
        self._model = None

    def embed(self, segments: Sequence[str]) -> np.ndarray:
        if not segments:
            return np.empty((0, 0))

        model = self._load_model()
        embeddings = []
        for batch in batched(segments, self.config.batch_size):
            batch_embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False)
            embeddings.append(batch_embeddings)
        matrix = np.vstack(embeddings)
        if self.config.normalize:
            cosine_normalize(matrix)
        return matrix

    def _load_model(self):
        if self._model is None:
            self._model = _load_sentence_transformer(self.config)
        return self._model
