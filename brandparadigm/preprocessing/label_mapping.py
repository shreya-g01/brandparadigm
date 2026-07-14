"""Maps raw dataset labels onto the project's 3-class sentiment scheme."""

SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: idx for idx, label in enumerate(SENTIMENT_CLASSES)}
ID2LABEL = dict(enumerate(SENTIMENT_CLASSES))

# tweet_eval's `sentiment` config already uses this exact 0/1/2 ordering.
TWEETEVAL_ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def amazon_polarity_to_sentiment(polarity: int) -> str:
    """Amazon Review Polarity: 1 -> Negative, 2 -> Positive (binary, no Neutral class)."""
    return "Negative" if int(polarity) == 1 else "Positive"


def tweeteval_label_to_sentiment(label: int) -> str:
    return TWEETEVAL_ID2LABEL[int(label)]
