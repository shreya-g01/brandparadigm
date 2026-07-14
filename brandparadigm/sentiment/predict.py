"""Inference wrapper for the fine-tuned binary sentiment model.

Kept decoupled from FastAPI/Streamlit — `api/` (Phase 5) will import
`SentimentPredictor` rather than reimplementing model loading/inference.
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from brandparadigm.logging import get_logger
from brandparadigm.preprocessing import ID2LABEL, clean_text
from brandparadigm.sentiment.model import load_trained_model_and_tokenizer

logger = get_logger(__name__)


class SentimentPredictor:
    """Loads a fine-tuned sentiment model once and serves predictions from it."""

    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, max_length: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model.eval()

    @classmethod
    def from_pretrained_dir(cls, model_dir: str, max_length: int = 128) -> "SentimentPredictor":
        model, tokenizer = load_trained_model_and_tokenizer(model_dir)
        return cls(model, tokenizer, max_length=max_length)

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single piece of text.

        Returns `{"label": "Negative"|"Positive", "scores": {"Negative": p0, "Positive": p1}}`.
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        cleaned = [clean_text(t) for t in texts]
        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

        results = []
        for row in probs:
            scores = {ID2LABEL[i]: float(row[i]) for i in range(row.shape[0])}
            label = ID2LABEL[int(row.argmax())]
            results.append({"label": label, "scores": scores})
        return results
