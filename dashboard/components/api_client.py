from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd


API_BASE_URL = os.getenv(
    "DASHBOARD_API_BASE_URL",
    "https://bparadigm-api-554595681018.europe-west1.run.app",
).rstrip("/")


class DashboardAPIError(RuntimeError):
    pass


def request_api(
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> Any:
    try:
        response = httpx.request(
            method,
            f"{API_BASE_URL}{endpoint}",
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPError as exc:
        raise DashboardAPIError(
            f"API request failed: {exc}"
        ) from exc


def get_model_health() -> dict[str, Any]:
    result = request_api(
        "GET",
        "/",
        timeout=30,
    )

    return {
        "status": "healthy",
        "service": result.get(
            "service",
            "sentiment-baseline",
        ),
        "models": [
            {
                "name": "sentiment",
                "loaded": True,
                "version": result.get(
                    "service",
                    "sentiment-baseline",
                ),
                "error": None,
            }
        ],
    }


def analyze_review(
    text: str,
    brand: str = "Unknown",
    product: str = "Unknown",
    source: str = "Manual",
) -> dict[str, Any]:
    result = request_api(
        "GET",
        "/predict",
        params={"text": text},
        timeout=120,
    )

    return {
        "text": text,
        "brand": brand,
        "product": product,
        "source": source,
        "sentiment": result["label"],
        "sentiment_confidence": result["confidence"],
        "sentiment_scores": {
            result["label"]: result["confidence"],
        },
        "model_version": "sentiment-baseline",
    }


def analyze_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if "text" not in dataframe.columns:
        raise ValueError(
            "The CSV must contain a 'text' column."
        )

    texts = dataframe["text"].fillna("").astype(str).tolist()

    predictions = request_api(
        "POST",
        "/predict_batch",
        json={"texts": texts},
        timeout=600,
    )

    results = dataframe.copy().reset_index(drop=True)

    results["sentiment"] = [
        prediction["label"]
        for prediction in predictions
    ]

    results["sentiment_confidence"] = [
        prediction["confidence"]
        for prediction in predictions
    ]

    results["model_version"] = "sentiment-baseline"

    return results
