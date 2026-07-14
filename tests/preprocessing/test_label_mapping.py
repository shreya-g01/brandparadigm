import pytest

from brandparadigm.preprocessing.label_mapping import (
    ID2LABEL,
    LABEL2ID,
    SENTIMENT_CLASSES,
    amazon_polarity_to_sentiment,
    tweeteval_label_to_sentiment,
)


@pytest.mark.parametrize("polarity,expected", [(1, "Negative"), (2, "Positive")])
def test_amazon_polarity_to_sentiment(polarity, expected):
    assert amazon_polarity_to_sentiment(polarity) == expected


@pytest.mark.parametrize(
    "label,expected",
    [(0, "Negative"), (1, "Neutral"), (2, "Positive")],
)
def test_tweeteval_label_to_sentiment(label, expected):
    assert tweeteval_label_to_sentiment(label) == expected


def test_label_id_roundtrip():
    for label in SENTIMENT_CLASSES:
        assert ID2LABEL[LABEL2ID[label]] == label
