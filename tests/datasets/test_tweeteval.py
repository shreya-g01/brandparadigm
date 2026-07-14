import httpx
import pytest

from brandparadigm.datasets.tweeteval import load_tweeteval

CONFIG = {
    "base_url": "https://example.invalid/sentiment",
    "splits": ["train", "val", "test"],
}


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture
def mock_fetch(monkeypatch):
    texts = "great phone\nterrible battery\nit is fine\n"
    labels = "2\n0\n1\n"

    def fake_get(url, timeout=None, follow_redirects=None):
        if url.endswith("_text.txt"):
            return _FakeResponse(texts)
        if url.endswith("_labels.txt"):
            return _FakeResponse(labels)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(httpx, "get", fake_get)


def test_load_tweeteval_maps_labels(mock_fetch):
    df = load_tweeteval(CONFIG, split="test")
    assert list(df["label"]) == ["Positive", "Negative", "Neutral"]
    assert (df["source"] == "tweet_eval").all()


def test_load_tweeteval_rejects_unknown_split(mock_fetch):
    with pytest.raises(ValueError):
        load_tweeteval(CONFIG, split="bogus")


def test_load_tweeteval_sample_size(mock_fetch):
    df = load_tweeteval(CONFIG, split="test", sample_size=2)
    assert len(df) == 2
