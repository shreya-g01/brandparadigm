"""Turns cleaned sentiment DataFrames into tokenized `datasets.Dataset` objects for the Trainer."""

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from brandparadigm.logging import get_logger

logger = get_logger(__name__)


def build_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    text_column: str = "text",
    label_column: str = "sentiment_label",
) -> Dataset:
    """Tokenize `df[text_column]` and attach `df[label_column]` as `labels`.

    Returns a `datasets.Dataset` formatted for PyTorch (`input_ids`,
    `attention_mask`, `labels`), ready to hand straight to `Trainer`.
    """
    dataset = Dataset.from_pandas(
        df[[text_column, label_column]].rename(columns={label_column: "labels"}),
        preserve_index=False,
    )

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    dataset = dataset.map(_tokenize, batched=True, remove_columns=[text_column])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info(
        "Built tokenized dataset with %d examples (max_length=%d)", len(dataset), max_length
    )
    return dataset
