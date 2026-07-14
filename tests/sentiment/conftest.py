"""Fully offline tokenizer + tiny model fixtures for sentiment pipeline tests.

Both are constructed from scratch in-memory (a hand-built WordLevel vocab,
a randomly-initialized `RobertaForSequenceClassification`) — no network
access, no Hugging Face Hub download. This lets tests exercise the real
`transformers.Trainer`/tokenization machinery end-to-end without depending
on `cardiffnlp/twitter-roberta-base-sentiment` being downloadable in this
environment, per the "don't attempt to download models" instruction.
"""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

_VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "good": 4,
    "bad": 5,
    "great": 6,
    "terrible": 7,
    "phone": 8,
    "battery": 9,
    "camera": 10,
    "love": 11,
    "hate": 12,
    "okay": 13,
    "product": 14,
    "service": 15,
}


@pytest.fixture
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel(_VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", _VOCAB["[CLS]"]), ("[SEP]", _VOCAB["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


@pytest.fixture
def tiny_model() -> RobertaForSequenceClassification:
    config = RobertaConfig(
        vocab_size=len(_VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=2,
    )
    return RobertaForSequenceClassification(config)


@pytest.fixture
def tiny_model_and_tokenizer(tiny_model, tiny_tokenizer):
    return tiny_model, tiny_tokenizer
