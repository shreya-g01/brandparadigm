"""Metric computation for the binary sentiment model, and TweetEval evaluation-set prep."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from brandparadigm.datasets import load_dataset
from brandparadigm.logging import get_logger
from brandparadigm.preprocessing import (
    SENTIMENT_CLASSES,
    clean_text,
    prepare_binary_sentiment_labels,
)

logger = get_logger(__name__)


def compute_metrics(eval_pred) -> dict:
    """HF Trainer-compatible `compute_metrics`: accuracy/precision/recall/F1 (binary).

    `eval_pred` is a `transformers.EvalPrediction` (or any object/tuple
    exposing `.predictions`/`predictions[0]` logits and `.label_ids`).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary", zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", zero_division=0),
        "f1": f1_score(labels, predictions, average="binary", zero_division=0),
    }


def build_evaluation_report(y_true, y_pred) -> dict:
    """Confusion matrix + classification report, as JSON-serializable dicts."""
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "confusion_matrix_labels": SENTIMENT_CLASSES,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=SENTIMENT_CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def load_tweeteval_eval_set(
    data_config: dict, split: str = "test", sample_size: int | None = None
) -> pd.DataFrame:
    """Load TweetEval's `sentiment` split, cleaned, Neutral-dropped, binary-labeled.

    Self-contained: reads directly from `raw_data/tweeteval/` via
    `brandparadigm.datasets.load_dataset`, so evaluation can run without
    `scripts/run_preprocessing.py` having been run first.

    Args:
        data_config: parsed configs/data_config.yaml (the whole file — the
            `tweet_eval` section is looked up internally by the registry).
        split: "train", "validation", or "test".
        sample_size: if set, cap the number of rows loaded.
    """
    df = load_dataset("tweeteval", data_config, split=split, sample_size=sample_size)
    df["text"] = df["text"].map(clean_text)
    before = len(df)
    df = prepare_binary_sentiment_labels(df, label_column="label")
    logger.info("Dropped %d Neutral rows for binary evaluation", before - len(df))
    return df[df["text"].str.len() > 0].reset_index(drop=True)
