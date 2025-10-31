"""
Shared utilities for the survey topic discovery pipeline.
"""

from __future__ import annotations

import json
import logging
import random
from contextlib import suppress
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, TypeVar

import pandas as pd

try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    pyarrow = None

T = TypeVar("T")


def get_logger(name: str = "survey_topic_pipeline") -> logging.Logger:
    """Return a module-level logger with sensible defaults."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yield lists of length batch_size from iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def ensure_directory(path: Path) -> None:
    """Ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Persist a DataFrame to parquet when possible, otherwise fallback to CSV."""
    ensure_directory(path)
    if path.suffix == ".parquet":
        if pyarrow is None:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
        df.to_parquet(path, index=False)
        return path
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported DataFrame output format: {path}")


def write_json(payload: object, path: Path, indent: int = 2) -> None:
    """Persist JSON payload to disk."""
    ensure_directory(path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=indent, ensure_ascii=False)


def read_json(path: Path):
    """Load JSON payload from disk."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_dataframe(path: Path) -> pd.DataFrame:
    """Load a DataFrame from parquet or CSV."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet" and pyarrow is None:
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            return pd.read_csv(fallback)
        raise RuntimeError("pyarrow unavailable and no CSV fallback found.")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported DataFrame format: {path}")


def sample_texts(texts: Sequence[str], max_items: int) -> List[str]:
    """Randomly sample up to max_items texts while preserving reproducibility."""
    if len(texts) <= max_items:
        return list(texts)
    # Use deterministic sample by seeding from aggregate hash.
    seed = hash(tuple(texts)) & 0xFFFF
    rng = random.Random(seed)
    return rng.sample(list(texts), max_items)


def cosine_normalize(matrix) -> None:
    """In-place L2 normalization for a numpy matrix."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    with suppress(ZeroDivisionError):
        norms[norms == 0] = 1.0
    matrix /= norms
