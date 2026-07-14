"""Dataset 2 — TweetEval `sentiment` task only (evaluation, never trained on).

Local CSV export limited to the `sentiment` task
(`sentiment_train.csv` / `sentiment_validation.csv` / `sentiment_test.csv`)
— every other TweetEval task (emotion, hate, irony, offensive, ...) is
ignored entirely.

The production sentiment model is binary (Negative/Positive), but
TweetEval's `sentiment` task is natively 3-class in the source data. This
loader decodes labels faithfully — including "Neutral" — so no information
is silently dropped at load time; Neutral rows are filtered out downstream
during preprocessing (see `scripts/run_preprocessing.py::preprocess_tweeteval`
and docs/dataset_guide.md).
"""

from pathlib import Path

import pandas as pd

from brandparadigm.datasets.local_source import first_matching_column, require_local_file
from brandparadigm.logging import get_logger
from brandparadigm.preprocessing.label_mapping import (
    TWEETEVAL_RAW_CLASSES,
    tweeteval_label_to_sentiment,
)

logger = get_logger(__name__)

_CANONICAL_BY_LOWER = {label.lower(): label for label in TWEETEVAL_RAW_CLASSES}


def _normalize_label(value) -> str:
    """Accept either the original 0/1/2 int encoding or already-string labels."""
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return tweeteval_label_to_sentiment(int(text))
    canonical = _CANONICAL_BY_LOWER.get(text.lower())
    if canonical is None:
        raise ValueError(f"Unrecognized TweetEval sentiment label: {value!r}")
    return canonical


def load_tweeteval(
    config: dict, split: str = "test", sample_size: int | None = None
) -> pd.DataFrame:
    """Load a TweetEval `sentiment` split, normalized to [text, label, source].

    Args:
        config: the `tweet_eval` section of configs/data_config.yaml.
        split: one of config["files"] keys ("train"/"validation"/"test").
            Per the spec this dataset is for evaluation only — do not use
            "train" to fit the sentiment model.
        sample_size: if set, randomly sample at most this many rows.
    """
    files = config["files"]
    if split not in files:
        raise ValueError(f"Unknown split '{split}', expected one of {list(files)}")

    path = Path(config["raw_data_dir"]) / files[split]
    require_local_file(path)
    raw = pd.read_csv(path)

    text_col = first_matching_column(raw, config["text_columns"])
    label_col = first_matching_column(raw, config["label_columns"])
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find a text/label column in {list(raw.columns)}. "
            f"Expected one of {config['text_columns']} and {config['label_columns']}."
        )

    df = pd.DataFrame(
        {
            "text": raw[text_col].astype(str),
            "label": raw[label_col].map(_normalize_label),
        }
    )
    df = df[df["text"].str.strip() != ""]
    df["source"] = "tweet_eval"

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info("Loaded %d TweetEval sentiment (%s) rows", len(df), split)
    return df.reset_index(drop=True)
