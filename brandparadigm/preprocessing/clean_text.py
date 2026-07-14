"""Text cleaning shared by every dataset before it reaches a model."""

import re
import unicodedata

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def strip_html(text: str) -> str:
    """Remove HTML tags/entities (reviews sometimes embed raw markup)."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_unicode(text: str) -> str:
    """NFKC-normalize unicode (smart quotes, accented chars, etc.)."""
    return unicodedata.normalize("NFKC", text)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def clean_text(text: str, *, remove_links: bool = True) -> str:
    """Apply the full cleaning pipeline used before tokenization/training."""
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    text = normalize_unicode(text)
    if remove_links:
        text = remove_urls(text)
    text = collapse_whitespace(text)
    return text
