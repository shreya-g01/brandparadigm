"""Shared helpers for loading user-provided local dataset exports.

Amazon Reviews and the historical Reddit dump are provisioned as local
files (see docs/dataset_guide.md) rather than downloaded, so both loaders
share the same "find files, detect columns, cap sample size" logic.
"""

from pathlib import Path

import pandas as pd

from brandparadigm.logging import get_logger

logger = get_logger(__name__)


class NoLocalDataError(FileNotFoundError):
    """Raised when a required local dataset export hasn't been provided yet."""


def find_source_files(raw_data_dir: str | Path, file_pattern: str) -> list[Path]:
    """Return files under `raw_data_dir` matching any of the comma-separated globs."""
    raw_data_dir = Path(raw_data_dir)
    matches: list[Path] = []
    for pattern in file_pattern.split(","):
        matches.extend(sorted(raw_data_dir.glob(pattern.strip())))
    return matches


def load_local_frame(raw_data_dir: str | Path, file_pattern: str) -> pd.DataFrame:
    """Load and concatenate every matching CSV/JSON/JSONL file into one frame."""
    files = find_source_files(raw_data_dir, file_pattern)
    if not files:
        raise NoLocalDataError(
            f"No dataset files found in '{raw_data_dir}' matching '{file_pattern}'. "
            "Place your export there first (see docs/dataset_guide.md)."
        )

    frames = []
    for f in files:
        logger.info("Loading local dataset file: %s", f)
        if f.suffix == ".csv":
            frames.append(pd.read_csv(f))
        elif f.suffix == ".jsonl":
            frames.append(pd.read_json(f, lines=True))
        elif f.suffix == ".json":
            frames.append(pd.read_json(f))
        else:
            logger.warning("Skipping unsupported file type: %s", f)
    if not frames:
        raise NoLocalDataError(f"No readable CSV/JSON/JSONL files found in '{raw_data_dir}'.")
    return pd.concat(frames, ignore_index=True)


def first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name in `candidates` present in `df`, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None
