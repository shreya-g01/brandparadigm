"""Lightweight, model-agnostic tokenization helpers used for EDA/QC.

Model-specific tokenizers (RoBERTa, DistilBERT) live next to the model that
uses them (`brandparadigm.sentiment.model`, `brandparadigm.topic_classifier.distilbert`)
since padding/truncation/vocab choices are model-specific. This module only
provides quick word-count style stats used during dataset validation.
"""

import re

_WORD_RE = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def truncate_to_word_limit(text: str, max_words: int) -> str:
    words = _WORD_RE.findall(text)
    return " ".join(words[:max_words])
