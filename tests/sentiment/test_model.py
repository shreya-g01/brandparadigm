from unittest.mock import MagicMock, patch

from brandparadigm.preprocessing.label_mapping import ID2LABEL, LABEL2ID
from brandparadigm.sentiment.model import load_model_and_tokenizer, load_trained_model_and_tokenizer


def test_load_model_and_tokenizer_requests_binary_head():
    fake_tokenizer = MagicMock()
    fake_model = MagicMock()

    with (
        patch(
            "brandparadigm.sentiment.model.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as tok_call,
        patch(
            "brandparadigm.sentiment.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake_model,
        ) as model_call,
    ):
        model, tokenizer = load_model_and_tokenizer("cardiffnlp/twitter-roberta-base-sentiment")

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    tok_call.assert_called_once_with("cardiffnlp/twitter-roberta-base-sentiment")
    model_call.assert_called_once_with(
        "cardiffnlp/twitter-roberta-base-sentiment",
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )


def test_load_trained_model_and_tokenizer_loads_from_local_dir():
    fake_tokenizer = MagicMock()
    fake_model = MagicMock()

    with (
        patch(
            "brandparadigm.sentiment.model.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as tok_call,
        patch(
            "brandparadigm.sentiment.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=fake_model,
        ) as model_call,
    ):
        model, tokenizer = load_trained_model_and_tokenizer("models/sentiment")

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    tok_call.assert_called_once_with("models/sentiment")
    model_call.assert_called_once_with("models/sentiment")
