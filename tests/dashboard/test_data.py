import pandas as pd

from dashboard.components.data import _normalise


def test_normalise_supports_baseline_output_schema():
    frame = pd.DataFrame(
        [{"brand": "Apple", "text": "Great", "predicted_sentiment": "positive", "sentiment_confidence": 0.9}]
    )
    result = _normalise(frame)
    assert result.loc[0, "sentiment"] == "Positive"
    assert result.loc[0, "confidence"] == 0.9
    assert result.loc[0, "source"] == "Analysis output"
