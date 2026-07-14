"""Fine-tunes the binary sentiment model (Model 1) with the HF `Trainer`.

Training data: Amazon Review Polarity `train.csv` (`test.csv` is used for
in-training validation/testing — see docs/dataset_guide.md). All
hyperparameters come from `configs/sentiment_config.yaml`; nothing here is
hardcoded. Early stopping is wired via `EarlyStoppingCallback`, driven by
the profile's `early_stopping_patience`.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from brandparadigm.datasets import load_dataset
from brandparadigm.logging import get_logger
from brandparadigm.preprocessing import LABEL2ID, amazon_polarity_to_sentiment, clean_text
from brandparadigm.sentiment.dataset import build_tokenized_dataset
from brandparadigm.sentiment.evaluate import build_evaluation_report, compute_metrics
from brandparadigm.sentiment.model import load_model_and_tokenizer
from brandparadigm.utils import (
    ensure_dir,
    get_git_commit_hash,
    get_library_versions,
    set_seed,
    write_json,
)

logger = get_logger(__name__)

ModelLoader = Callable[[str], tuple[PreTrainedModel, PreTrainedTokenizerBase]]

# Packages whose versions matter for reproducing a training run.
_TRACKED_LIBRARIES = ["torch", "transformers", "datasets", "scikit-learn", "accelerate"]


def _prepare_amazon_split(data_config: dict, split: str, sample_size: int | None) -> pd.DataFrame:
    """Load an Amazon Review Polarity split, cleaned and binary-labeled."""
    df = load_dataset("amazon", data_config, split=split, sample_size=sample_size)
    df["text"] = df["text"].map(clean_text)
    df["sentiment_label"] = df["polarity"].map(amazon_polarity_to_sentiment).map(LABEL2ID)
    return df[df["text"].str.len() > 0].reset_index(drop=True)


def build_training_arguments(
    sentiment_config: dict, params: dict, output_dir: Path
) -> TrainingArguments:
    """Build `TrainingArguments` entirely from config — no hardcoded hyperparameters."""
    training_cfg = sentiment_config["training"]
    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=params["num_train_epochs"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        per_device_eval_batch_size=params["per_device_eval_batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        logging_steps=params["logging_steps"],
        eval_strategy=training_cfg["eval_strategy"],
        save_strategy=training_cfg["save_strategy"],
        save_total_limit=training_cfg["save_total_limit"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        greater_is_better=training_cfg["greater_is_better"],
        seed=training_cfg.get("seed", 42),
        report_to=[],
    )


def build_run_metadata(
    sentiment_config: dict,
    profile: str,
    params: dict,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble everything needed to reproduce/audit a training run.

    Written to `metadata.json` alongside the model — model info, the exact
    training configuration used, dataset details, the git commit that
    produced the run, and installed library versions, so a saved model
    directory is self-describing months later.
    """
    training_cfg = sentiment_config["training"]
    return {
        "model": {
            "base_model": sentiment_config["model"]["base_model"],
            "task_type": sentiment_config["task_type"],
            "num_labels": sentiment_config["num_labels"],
            "max_seq_length": sentiment_config["model"]["max_seq_length"],
            "label_mapping": sentiment_config["label_mapping"],
        },
        "training_config": {
            "profile": profile,
            **params,
            "eval_strategy": training_cfg["eval_strategy"],
            "save_strategy": training_cfg["save_strategy"],
            "save_total_limit": training_cfg["save_total_limit"],
            "metric_for_best_model": training_cfg["metric_for_best_model"],
            "greater_is_better": training_cfg["greater_is_better"],
            "seed": training_cfg.get("seed", 42),
        },
        "dataset": {
            "train_dataset": training_cfg["train_dataset"],
            "validation_dataset": training_cfg["validation_dataset"],
            "train_rows": len(train_df),
            "eval_rows": len(eval_df),
        },
        "git_commit_hash": get_git_commit_hash(),
        "library_versions": get_library_versions(_TRACKED_LIBRARIES),
    }


def train(
    sentiment_config: dict,
    data_config: dict,
    profile: str = "smoke_test",
    model_loader: ModelLoader = load_model_and_tokenizer,
) -> dict[str, Any]:
    """Fine-tune the sentiment model per `sentiment_config`'s `profile`.

    Args:
        sentiment_config: parsed configs/sentiment_config.yaml.
        data_config: parsed configs/data_config.yaml (for the Amazon loader).
        profile: key into `sentiment_config["training"]["profiles"]`
            (`"smoke_test"` or `"full"`) — selects every hyperparameter.
        model_loader: loads `(model, tokenizer)` given the base model name.
            Defaults to the real HF loader; tests inject a fake one so the
            training loop can be exercised without network access.

    Returns:
        dict with `eval_metrics`, `output_dir`, `evaluation_report`
        (confusion matrix + classification report on the held-out Amazon
        Review Polarity test split), and `metadata` (the same dict written
        to `metadata.json` — see `build_run_metadata`).
    """
    training_cfg = sentiment_config["training"]
    set_seed(training_cfg.get("seed", 42))

    params = training_cfg["profiles"][profile]
    output_dir = ensure_dir(Path(training_cfg["output_dir"]))

    logger.info("Loading Amazon Review Polarity train/test splits (profile=%s)", profile)
    train_df = _prepare_amazon_split(data_config, "train", params.get("train_sample_size"))
    eval_df = _prepare_amazon_split(data_config, "test", params.get("eval_sample_size"))

    model, tokenizer = model_loader(sentiment_config["model"]["base_model"])
    max_length = sentiment_config["model"]["max_seq_length"]

    train_dataset = build_tokenized_dataset(train_df, tokenizer, max_length=max_length)
    eval_dataset = build_tokenized_dataset(eval_df, tokenizer, max_length=max_length)

    training_args = build_training_arguments(sentiment_config, params, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=params["early_stopping_patience"])
        ],
    )

    logger.info(
        "Starting training: %d train / %d eval examples", len(train_dataset), len(eval_dataset)
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    predictions = trainer.predict(eval_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)
    y_true = predictions.label_ids
    report = build_evaluation_report(y_true, y_pred)

    write_json(eval_metrics, output_dir / "metrics.json")
    write_json(
        {"matrix": report["confusion_matrix"], "labels": report["confusion_matrix_labels"]},
        output_dir / "confusion_matrix.json",
    )
    write_json(report["classification_report"], output_dir / "classification_report.json")
    write_json(trainer.state.log_history, output_dir / "training_history.json")

    metadata = build_run_metadata(sentiment_config, profile, params, train_df, eval_df)
    write_json(metadata, output_dir / "metadata.json")

    logger.info("Saved best model, tokenizer, and evaluation artifacts to %s", output_dir)
    return {
        "eval_metrics": eval_metrics,
        "output_dir": str(output_dir),
        "evaluation_report": report,
        "metadata": metadata,
    }
