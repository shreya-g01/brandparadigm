import pandas as pd

from brandparadigm.preprocessing import LABEL2ID
from scripts.run_preprocessing import preprocess_amazon, preprocess_reddit, preprocess_tweeteval


def test_preprocess_amazon_maps_polarity_to_binary_int_labels():
    df = pd.DataFrame({"text": ["great phone", "bad battery"], "polarity": [2, 1]})
    out = preprocess_amazon(df)
    assert list(out["sentiment_label"]) == [1, 0]
    assert set(out["sentiment_label"].unique()) <= {0, 1}


def test_preprocess_tweeteval_drops_neutral_rows_before_mapping():
    df = pd.DataFrame(
        {
            "text": ["great stuff", "bad stuff", "meh stuff"],
            "label": ["Positive", "Negative", "Neutral"],
        }
    )
    out = preprocess_tweeteval(df)
    assert len(out) == 2
    assert "Neutral" not in out["label"].values
    assert list(out["sentiment_label"]) == [1, 0]


def test_preprocess_tweeteval_sentiment_label_matches_label2id():
    df = pd.DataFrame({"text": ["a", "b"], "label": ["Negative", "Positive"]})
    out = preprocess_tweeteval(df)
    assert out.loc[out["label"] == "Negative", "sentiment_label"].iloc[0] == LABEL2ID["Negative"]
    assert out.loc[out["label"] == "Positive", "sentiment_label"].iloc[0] == LABEL2ID["Positive"]


def test_preprocess_tweeteval_all_neutral_yields_empty_frame():
    df = pd.DataFrame({"text": ["meh", "okay"], "label": ["Neutral", "Neutral"]})
    out = preprocess_tweeteval(df)
    assert len(out) == 0


def test_preprocess_reddit_has_no_sentiment_label_column():
    df = pd.DataFrame({"text": ["some post"], "subreddit": ["apple"]})
    out = preprocess_reddit(df)
    assert "sentiment_label" not in out.columns
