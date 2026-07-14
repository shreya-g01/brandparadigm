"""Shared helpers for loading user-provided local dataset exports.

Amazon Reviews, TweetEval, and the historical Reddit archive are all
provisioned as local files (see docs/dataset_guide.md) at fixed, known
paths rather than downloaded, so their loaders share this small set of
"is it there yet, which column is it" helpers.
"""

from pathlib import Path

import pandas as pd


class NoLocalDataError(FileNotFoundError):
    """Raised when a required local dataset export hasn't been provided yet."""


def require_local_file(path: Path) -> Path:
    """Return `path` if it exists, else raise a NoLocalDataError with guidance."""
    if not path.exists():
        raise NoLocalDataError(
            f"Expected dataset file not found: '{path}'. "
            "Place it there first (see docs/dataset_guide.md)."
        )
    return path


def first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name in `candidates` present in `df`, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None
