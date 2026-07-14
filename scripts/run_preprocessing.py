#!/usr/bin/env python3
"""CLI: clean text and normalize labels for each dataset's `_raw.csv` cache.

Reads `raw_data/processed/{name}_raw.csv` (written by download_datasets.py)
and writes `raw_data/processed/{name}_clean.csv` with cleaned text and,
where applicable, a unified `sentiment_label` column.
"""

import argparse
import sys

import pandas as pd

from brandparadigm.config.paths import PROCESSED_DATA_DIR
from brandparadigm.datasets import DATASET_NAMES
from brandparadigm.logging import get_logger
from brandparadigm.preprocessing import clean_text, star_rating_to_sentiment
from brandparadigm.utils import ensure_dir

logger = get_logger(__name__)


def preprocess_amazon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df["sentiment_label"] = df["rating"].map(star_rating_to_sentiment)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def preprocess_tweeteval(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df["sentiment_label"] = df["label"]
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def preprocess_reddit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


_PREPROCESSORS = {
    "amazon": preprocess_amazon,
    "tweeteval": preprocess_tweeteval,
    "reddit": preprocess_reddit,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASET_NAMES, "all"], default="all")
    return parser.parse_args()


def run_one(name: str) -> bool:
    raw_path = PROCESSED_DATA_DIR / f"{name}_raw.csv"
    if not raw_path.exists():
        logger.error("%s not found — run scripts/download_datasets.py first.", raw_path)
        return False

    df = pd.read_csv(raw_path)
    df = _PREPROCESSORS[name](df)

    out_dir = ensure_dir(PROCESSED_DATA_DIR)
    out_path = out_dir / f"{name}_clean.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d cleaned rows to %s", len(df), out_path)
    return True


def main() -> int:
    args = parse_args()
    names = list(DATASET_NAMES) if args.dataset == "all" else [args.dataset]
    ok = all(run_one(name) for name in names)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
