import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brandparadigm.sentiment.dataset import build_tokenized_dataset
from brandparadigm.sentiment.evaluate import (
    build_evaluation_report,
    compute_metrics,
    load_tweeteval_eval_set,
    run_batched_inference,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tweeteval_sentiment_sample.csv"


def test_compute_metrics_perfect_predictions():
    logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])  # argmax -> 0,1,0,1
    labels = np.array([0, 1, 0, 1])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_all_wrong_predictions():
    logits = np.array([[0.1, 0.9], [0.9, 0.1]])  # argmax -> 1,0
    labels = np.array([0, 1])  # true -> 0,1 (opposite)
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0


def test_build_evaluation_report_confusion_matrix_and_report_shape():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    report = build_evaluation_report(y_true, y_pred)

    assert report["confusion_matrix_labels"] == ["Negative", "Positive"]
    assert len(report["confusion_matrix"]) == 2
    assert "Negative" in report["classification_report"]
    assert "Positive" in report["classification_report"]
    assert "accuracy" in report["classification_report"]


@pytest.fixture
def data_config(tmp_path):
    data_dir = tmp_path / "tweeteval"
    data_dir.mkdir()
    shutil.copy(FIXTURE, data_dir / "sentiment_test.csv")
    return {
        "tweet_eval": {
            "raw_data_dir": str(data_dir),
            "files": {"test": "sentiment_test.csv"},
            "text_columns": ["text", "tweet"],
            "label_columns": ["label", "sentiment"],
        }
    }


def test_load_tweeteval_eval_set_drops_neutral_and_is_binary(data_config):
    df = load_tweeteval_eval_set(data_config, split="test")
    assert "Neutral" not in df["label"].values
    assert set(df["sentiment_label"].unique()) <= {0, 1}


def test_load_tweeteval_eval_set_sample_size_caps_before_neutral_filtering(data_config):
    # sample_size caps the raw load, which happens before Neutral rows are
    # dropped — so the final count can be <= sample_size, not necessarily ==.
    df = load_tweeteval_eval_set(data_config, split="test", sample_size=2)
    assert len(df) <= 2


def test_run_batched_inference_returns_predictions_and_labels(tiny_model, tiny_tokenizer):
    df = pd.DataFrame(
        {"text": ["good phone", "bad battery", "great camera"], "sentiment_label": [1, 0, 1]}
    )
    dataset = build_tokenized_dataset(df, tiny_tokenizer, max_length=8)

    y_pred, y_true = run_batched_inference(tiny_model, dataset, batch_size=2)

    assert len(y_pred) == 3
    assert list(y_true) == [1, 0, 1]
    assert set(y_pred).issubset({0, 1})


def test_run_batched_inference_batch_size_does_not_change_results(tiny_model, tiny_tokenizer):
    df = pd.DataFrame(
        {"text": ["good", "bad", "great", "terrible", "okay"], "sentiment_label": [1, 0, 1, 0, 1]}
    )
    dataset = build_tokenized_dataset(df, tiny_tokenizer, max_length=8)

    y_pred_batch_1, _ = run_batched_inference(tiny_model, dataset, batch_size=1)
    y_pred_batch_100, _ = run_batched_inference(tiny_model, dataset, batch_size=100)

    assert list(y_pred_batch_1) == list(y_pred_batch_100)
