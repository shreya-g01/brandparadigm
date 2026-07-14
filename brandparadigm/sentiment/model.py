"""Model 1 — binary sentiment classifier: cardiffnlp/twitter-roberta-base-sentiment.

The base checkpoint's pretrained classification head is 3-class
(negative/neutral/positive, trained on Twitter data); fine-tuning here
replaces it with a fresh binary head (num_labels=2) since the production
sentiment scheme is binary — see docs/model_cards/roberta_sentiment.md.
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from brandparadigm.logging import get_logger
from brandparadigm.preprocessing.label_mapping import ID2LABEL, LABEL2ID

logger = get_logger(__name__)


def load_model_and_tokenizer(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the base checkpoint with a fresh binary classification head.

    `ignore_mismatched_sizes=True` is required because the base checkpoint
    ships a 3-class head: loading it with `num_labels=2` replaces that head
    with a freshly initialized binary one instead of raising a
    shape-mismatch error.
    """
    logger.info("Loading tokenizer and model from '%s'", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def load_trained_model_and_tokenizer(
    model_dir: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a previously fine-tuned model/tokenizer from a local directory.

    Used by inference (`brandparadigm.sentiment.predict`) and evaluation
    once a model has been trained and saved — as opposed to
    `load_model_and_tokenizer`, which starts from the pretrained base
    checkpoint for fine-tuning.
    """
    logger.info("Loading fine-tuned model and tokenizer from '%s'", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer
