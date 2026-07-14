import json
import shutil
from pathlib import Path

import pytest

from brandparadigm.sentiment.train import train

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "amazon_review_polarity_sample.csv"


@pytest.fixture
def data_config(tmp_path):
    amazon_dir = tmp_path / "amazon_review_polarity_csv"
    amazon_dir.mkdir()
    shutil.copy(FIXTURE, amazon_dir / "train.csv")
    shutil.copy(FIXTURE, amazon_dir / "test.csv")
    return {
        "amazon_reviews": {
            "raw_data_dir": str(amazon_dir),
            "train_file": "train.csv",
            "test_file": "test.csv",
            "columns": ["polarity", "title", "text"],
        }
    }


@pytest.fixture
def sentiment_config(tmp_path):
    return {
        "task_type": "binary",
        "num_labels": 2,
        "label_mapping": {0: "Negative", 1: "Positive"},
        "model": {"base_model": "fake/tiny-offline-model", "max_seq_length": 8},
        "training": {
            "train_dataset": "amazon_reviews",
            "validation_dataset": "amazon_reviews",
            "output_dir": str(tmp_path / "model_output"),
            "seed": 42,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "profiles": {
                "smoke_test": {
                    "train_sample_size": None,
                    "eval_sample_size": None,
                    "num_train_epochs": 2,
                    "per_device_train_batch_size": 2,
                    "per_device_eval_batch_size": 2,
                    "learning_rate": 5.0e-4,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.0,
                    "early_stopping_patience": 2,
                    "logging_steps": 1,
                }
            },
        },
        "evaluation": {"dataset": "tweet_eval", "split": "test", "exclude_labels": ["Neutral"]},
    }


def test_train_runs_end_to_end_fully_offline(
    sentiment_config, data_config, tiny_model, tiny_tokenizer
):
    """No network access anywhere: model/tokenizer are injected, both built
    from scratch in-memory (see conftest.py). This proves the Trainer
    orchestration (config-driven TrainingArguments, early stopping wiring,
    artifact saving) actually works, independent of Hugging Face Hub access.
    """

    def fake_model_loader(_model_name):
        return tiny_model, tiny_tokenizer

    result = train(
        sentiment_config, data_config, profile="smoke_test", model_loader=fake_model_loader
    )

    output_dir = Path(result["output_dir"])

    assert "eval_metrics" in result
    assert "eval_f1" in result["eval_metrics"] or "eval_loss" in result["eval_metrics"]

    # Best model + tokenizer saved.
    assert (output_dir / "config.json").exists()
    assert any(output_dir.glob("*.safetensors")) or (output_dir / "pytorch_model.bin").exists()
    assert (
        (output_dir / "tokenizer_config.json").exists()
        or (output_dir / "vocab.json").exists()
        or any(output_dir.glob("tokenizer*.json"))
    )

    # Required artifacts from the spec.
    for filename in [
        "metrics.json",
        "confusion_matrix.json",
        "classification_report.json",
        "training_history.json",
    ]:
        path = output_dir / filename
        assert path.exists(), f"missing {filename}"
        with open(path) as f:
            json.load(f)  # must be valid JSON


def test_train_evaluation_report_uses_binary_labels(
    sentiment_config, data_config, tiny_model, tiny_tokenizer
):
    def fake_model_loader(_model_name):
        return tiny_model, tiny_tokenizer

    result = train(
        sentiment_config, data_config, profile="smoke_test", model_loader=fake_model_loader
    )
    report = result["evaluation_report"]
    assert report["confusion_matrix_labels"] == ["Negative", "Positive"]
    assert len(report["confusion_matrix"]) == 2
    assert len(report["confusion_matrix"][0]) == 2


def test_build_training_arguments_reads_every_field_from_config(sentiment_config, tmp_path):
    from brandparadigm.sentiment.train import build_training_arguments

    params = sentiment_config["training"]["profiles"]["smoke_test"]
    args = build_training_arguments(sentiment_config, params, tmp_path)

    assert args.num_train_epochs == params["num_train_epochs"]
    assert args.per_device_train_batch_size == params["per_device_train_batch_size"]
    assert args.learning_rate == params["learning_rate"]
    assert args.eval_strategy.value == sentiment_config["training"]["eval_strategy"]
    assert args.metric_for_best_model == sentiment_config["training"]["metric_for_best_model"]
