"""Maps raw dataset labels onto the project's binary sentiment scheme.

Production sentiment classification is **binary**: 0 = Negative,
1 = Positive. This matches the only training dataset, Amazon Review
Polarity, which is itself binary — there is no third "Neutral" class to
train on, and introducing an artificial one would mean fabricating labels
rather than modeling the data. See docs/model_cards/roberta_sentiment.md
for the full rationale.

TweetEval's `sentiment` task is natively 3-class in the source data
(Negative/Neutral/Positive). Its raw labels are decoded faithfully by
`tweeteval_label_to_sentiment` so the loader doesn't silently drop
information — Neutral rows are filtered out downstream, during
preprocessing (see `scripts/run_preprocessing.py::preprocess_tweeteval`),
before they ever reach evaluation metrics.
"""

# Production label scheme: what the trained model actually predicts.
SENTIMENT_CLASSES = ["Negative", "Positive"]
LABEL2ID = {label: idx for idx, label in enumerate(SENTIMENT_CLASSES)}
ID2LABEL = dict(enumerate(SENTIMENT_CLASSES))

# TweetEval's raw `sentiment` task encoding — 3-class, as published. Used
# only to decode the source data; not the production label scheme above.
TWEETEVAL_RAW_CLASSES = ["Negative", "Neutral", "Positive"]
TWEETEVAL_ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def amazon_polarity_to_sentiment(polarity: int) -> str:
    """Amazon Review Polarity: 1 -> Negative, 2 -> Positive (binary, matches production scheme)."""
    return "Negative" if int(polarity) == 1 else "Positive"


def tweeteval_label_to_sentiment(label: int) -> str:
    """Decode TweetEval's raw 0/1/2 encoding — may return "Neutral"."""
    return TWEETEVAL_ID2LABEL[int(label)]
