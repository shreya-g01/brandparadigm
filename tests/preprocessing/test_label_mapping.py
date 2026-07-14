import pytest

from brandparadigm.preprocessing.label_mapping import (
    ID2LABEL,
    LABEL2ID,
    SENTIMENT_CLASSES,
    TWEETEVAL_RAW_CLASSES,
    amazon_polarity_to_sentiment,
    tweeteval_label_to_sentiment,
)


def test_production_sentiment_scheme_is_binary():
    assert SENTIMENT_CLASSES == ["Negative", "Positive"]
    assert LABEL2ID == {"Negative": 0, "Positive": 1}
    assert ID2LABEL == {0: "Negative", 1: "Positive"}
    assert "Neutral" not in LABEL2ID


@pytest.mark.parametrize("polarity,expected", [(1, "Negative"), (2, "Positive")])
def test_amazon_polarity_to_sentiment(polarity, expected):
    assert amazon_polarity_to_sentiment(polarity) == expected
    assert amazon_polarity_to_sentiment(polarity) in SENTIMENT_CLASSES


@pytest.mark.parametrize(
    "label,expected",
    [(0, "Negative"), (1, "Neutral"), (2, "Positive")],
)
def test_tweeteval_label_to_sentiment_decodes_raw_3_class_encoding(label, expected):
    # TweetEval's source data is natively 3-class; decoding it faithfully
    # (including "Neutral") is separate from the binary production scheme —
    # Neutral rows are dropped later, during preprocessing.
    assert tweeteval_label_to_sentiment(label) == expected
    assert expected in TWEETEVAL_RAW_CLASSES


def test_label_id_roundtrip():
    for label in SENTIMENT_CLASSES:
        assert ID2LABEL[LABEL2ID[label]] == label
