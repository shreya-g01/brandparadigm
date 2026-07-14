import pandas as pd

from brandparadigm.sentiment.dataset import build_tokenized_dataset


def test_build_tokenized_dataset_has_expected_columns(tiny_tokenizer):
    df = pd.DataFrame(
        {"text": ["good phone battery", "terrible bad product"], "sentiment_label": [1, 0]}
    )
    dataset = build_tokenized_dataset(df, tiny_tokenizer, max_length=8)

    assert dataset.column_names == ["labels", "input_ids", "attention_mask"]
    assert len(dataset) == 2


def test_build_tokenized_dataset_pads_to_max_length(tiny_tokenizer):
    df = pd.DataFrame({"text": ["good"], "sentiment_label": [1]})
    dataset = build_tokenized_dataset(df, tiny_tokenizer, max_length=8)

    assert len(dataset[0]["input_ids"]) == 8
    assert len(dataset[0]["attention_mask"]) == 8


def test_build_tokenized_dataset_preserves_labels(tiny_tokenizer):
    df = pd.DataFrame({"text": ["good", "bad", "okay"], "sentiment_label": [1, 0, 1]})
    dataset = build_tokenized_dataset(df, tiny_tokenizer, max_length=8)

    assert [int(x) for x in dataset["labels"]] == [1, 0, 1]


def test_build_tokenized_dataset_is_torch_formatted(tiny_tokenizer):
    import torch

    df = pd.DataFrame({"text": ["good phone"], "sentiment_label": [1]})
    dataset = build_tokenized_dataset(df, tiny_tokenizer, max_length=8)

    assert isinstance(dataset[0]["input_ids"], torch.Tensor)
