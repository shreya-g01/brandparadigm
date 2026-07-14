"""Dataset 1 — Amazon Review Polarity (training data for the sentiment model).

Headerless local CSV export (Zhang et al. "Amazon Review Polarity"):
`train.csv` / `test.csv` under `amazon_review_polarity_csv/`, fixed columns
`[polarity, title, text]`. `polarity` is 1 (Negative) or 2 (Positive) — this
dataset is binary, there is no Neutral class.

Training profiles routinely request only a few hundred/thousand rows out
of a multi-million-row file — reading the whole CSV into memory first (as
a naive `pandas.read_csv` + `.sample()` would) defeats the purpose of
sampling in the first place. `read_csv_sampled` (see
`brandparadigm.datasets.local_source`) streams the file in
`chunk_size`-row chunks instead, so peak memory stays bounded by the
requested sample size rather than the file's total size.
"""

from pathlib import Path

import pandas as pd

from brandparadigm.datasets.local_source import (
    DEFAULT_CHUNK_SIZE,
    read_csv_sampled,
    require_local_file,
)
from brandparadigm.logging import get_logger

logger = get_logger(__name__)

_SPLIT_FILE_KEYS = {"train": "train_file", "test": "test_file"}


def _load_split_file(config: dict, split: str, sample_size: int | None) -> pd.DataFrame:
    path = Path(config["raw_data_dir"]) / config[_SPLIT_FILE_KEYS[split]]
    require_local_file(path)
    return read_csv_sampled(
        path,
        header=None,
        names=config["columns"],
        sample_size=sample_size,
        chunk_size=config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        strategy=config.get("sampling_strategy", "reservoir"),
    )


def load_amazon_reviews(
    config: dict, split: str = "train", sample_size: int | None = None
) -> pd.DataFrame:
    """Load Amazon Review Polarity, normalized to [text, polarity, source].

    Sampling happens during the CSV read itself (see module docstring), not
    after loading everything — the full file is never held in memory when
    `sample_size` is set.

    Args:
        config: the `amazon_reviews` section of configs/data_config.yaml
            (`chunk_size` and `sampling_strategy` are optional — see
            `brandparadigm.datasets.local_source.read_csv_sampled`).
        split: "train", "test", or "all" (concatenates both files). For
            "all" with `sample_size` set, the cap applies **per file**, so
            the combined result can hold up to `2 * sample_size` rows.
        sample_size: if set, cap each split file's read at this many rows.
    """
    if split == "all":
        raw = pd.concat(
            [_load_split_file(config, s, sample_size) for s in _SPLIT_FILE_KEYS],
            ignore_index=True,
        )
    elif split in _SPLIT_FILE_KEYS:
        raw = _load_split_file(config, split, sample_size)
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

    logger.info("Loaded %d Amazon Review Polarity rows (split=%s)", len(df), split)
    return df.reset_index(drop=True)
