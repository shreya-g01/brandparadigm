"""Dataset 1 — Amazon Review Polarity (training data for the sentiment model).

Headerless local CSV export (Zhang et al. "Amazon Review Polarity"):
`train.csv` / `test.csv` under `amazon_review_polarity_csv/`, fixed columns
`[polarity, title, text]`. `polarity` is 1 (Negative) or 2 (Positive) — this
dataset is binary, there is no Neutral class.
"""

from pathlib import Path

import pandas as pd

from brandparadigm.datasets.local_source import require_local_file
from brandparadigm.logging import get_logger

logger = get_logger(__name__)

_SPLIT_FILE_KEYS = {"train": "train_file", "test": "test_file"}


def _load_split_file(config: dict, split: str) -> pd.DataFrame:
    path = Path(config["raw_data_dir"]) / config[_SPLIT_FILE_KEYS[split]]
    require_local_file(path)
    return pd.read_csv(path, header=None, names=config["columns"])


def load_amazon_reviews(
    config: dict, split: str = "train", sample_size: int | None = None
) -> pd.DataFrame:
    """Load Amazon Review Polarity, normalized to [text, polarity, source].

    Args:
        config: the `amazon_reviews` section of configs/data_config.yaml.
        split: "train", "test", or "all" (concatenates both files).
        sample_size: if set, randomly sample at most this many rows.
    """
    if split == "all":
        raw = pd.concat([_load_split_file(config, s) for s in _SPLIT_FILE_KEYS], ignore_index=True)
    elif split in _SPLIT_FILE_KEYS:
        raw = _load_split_file(config, split)
    else:
        raise ValueError(f"Unknown split '{split}', expected 'train', 'test', or 'all'")

    title = raw["title"].fillna("").astype(str)
    text = raw["text"].fillna("").astype(str)
    df = pd.DataFrame(
        {
            "text": (title + " " + text).str.strip(),
            "polarity": pd.to_numeric(raw["polarity"], errors="coerce"),
        }
    )
    df = df.dropna(subset=["text", "polarity"])
    df = df[df["text"].str.strip() != ""]
    df["source"] = "amazon_reviews"

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info("Loaded %d Amazon Review Polarity rows (split=%s)", len(df), split)
    return df.reset_index(drop=True)
