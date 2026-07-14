"""Maps raw dataset labels onto the project's 3-class sentiment scheme."""

SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: idx for idx, label in enumerate(SENTIMENT_CLASSES)}
ID2LABEL = dict(enumerate(SENTIMENT_CLASSES))

# tweet_eval's `sentiment` config already uses this exact 0/1/2 ordering.
TWEETEVAL_ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def star_rating_to_sentiment(rating: float) -> str:
    """1-2 stars -> Negative, 3 -> Neutral, 4-5 -> Positive (per spec)."""
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def tweeteval_label_to_sentiment(label: int) -> str:
    return TWEETEVAL_ID2LABEL[int(label)]
