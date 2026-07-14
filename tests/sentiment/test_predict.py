from unittest.mock import patch

from brandparadigm.sentiment.predict import SentimentPredictor


def test_predict_returns_label_and_scores(tiny_model, tiny_tokenizer):
    predictor = SentimentPredictor(tiny_model, tiny_tokenizer, max_length=8)
    result = predictor.predict("good phone battery")

    assert result["label"] in {"Negative", "Positive"}
    assert set(result["scores"].keys()) == {"Negative", "Positive"}
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-4


def test_predict_batch_returns_one_result_per_input(tiny_model, tiny_tokenizer):
    predictor = SentimentPredictor(tiny_model, tiny_tokenizer, max_length=8)
    results = predictor.predict_batch(["great product", "terrible service", "okay"])

    assert len(results) == 3
    for result in results:
        assert result["label"] in {"Negative", "Positive"}


def test_predict_cleans_text_before_tokenizing(tiny_model, tiny_tokenizer):
    predictor = SentimentPredictor(tiny_model, tiny_tokenizer, max_length=8)
    # Should not raise even with HTML/whitespace noise — clean_text handles it.
    result = predictor.predict("<b>good</b>   phone  ")
    assert result["label"] in {"Negative", "Positive"}


def test_from_pretrained_dir_loads_via_load_trained_model_and_tokenizer(tiny_model, tiny_tokenizer):
    with patch(
        "brandparadigm.sentiment.predict.load_trained_model_and_tokenizer",
        return_value=(tiny_model, tiny_tokenizer),
    ) as loader:
        predictor = SentimentPredictor.from_pretrained_dir("models/sentiment", max_length=8)

    loader.assert_called_once_with("models/sentiment")
    assert predictor.model is tiny_model
    assert predictor.tokenizer is tiny_tokenizer
