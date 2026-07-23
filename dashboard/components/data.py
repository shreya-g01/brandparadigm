"""Dashboard data adapter.

Real analyzed CSV output is preferred. Demo rows are isolated here so they can
be removed as soon as the analytics/API phases expose persisted review data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DashboardData:
    reviews: pd.DataFrame
    mode: str
    source_path: str | None


DEMO_REVIEWS = [
    ("Apple", "iPhone 15", "Amazon", "The camera is exceptional, even at night.", "Positive", 0.98, "Product Quality"),
    ("Samsung", "Galaxy S24", "Reddit", "Battery life easily gets me through the day.", "Positive", 0.94, "Battery"),
    ("Apple", "iPhone 15", "Reddit", "The latest update made the phone run hot.", "Negative", 0.91, "Features"),
    ("Google", "Pixel 8", "Amazon", "Clean software and genuinely useful AI features.", "Positive", 0.96, "Features"),
    ("Samsung", "Galaxy S24", "Amazon", "Customer support kept transferring my case.", "Negative", 0.93, "Customer Service"),
    ("Google", "Pixel 8", "Reddit", "Great photos but charging is still too slow.", "Negative", 0.87, "Battery"),
    ("Apple", "AirPods Pro", "Amazon", "Noise cancellation is worth every penny.", "Positive", 0.97, "Pricing"),
    ("Samsung", "Galaxy Buds", "Reddit", "Comfortable design, but the app disconnects.", "Negative", 0.88, "Design"),
    ("Google", "Pixel Buds", "Amazon", "Delivery was fast and setup took seconds.", "Positive", 0.92, "Delivery"),
    ("Apple", "iPhone 15", "Reddit", "Premium build quality and a very bright display.", "Positive", 0.95, "Design"),
    ("Samsung", "Galaxy S24", "Amazon", "The screen is excellent but the price is steep.", "Negative", 0.84, "Pricing"),
    ("Google", "Pixel 8", "Reddit", "Support replaced my faulty device quickly.", "Positive", 0.90, "Customer Service"),
]


def _demo_frame() -> pd.DataFrame:
    today = date.today()
    rows = []
    for index in range(72):
        brand, product, source, text, sentiment, confidence, topic = DEMO_REVIEWS[index % len(DEMO_REVIEWS)]
        rows.append(
            {
                "brand": brand,
                "product": product,
                "source": source,
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "topic": topic,
                "date": today - timedelta(days=71 - index),
            }
        )
    return pd.DataFrame(rows)


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    aliases = {
        "review": "text",
        "content": "text",
        "predicted_sentiment": "sentiment",
        "sentiment_confidence": "confidence",
        "category": "topic",
        "label": "sentiment",
        "created_at": "date",
    }
    df = df.rename(columns={key: value for key, value in aliases.items() if key in df.columns})
    defaults = {
        "brand": "Unspecified",
        "product": "All products",
        "source": "Analysis output",
        "text": "Review text unavailable",
        "sentiment": "Unknown",
        "confidence": 0.0,
        "topic": "Others",
    }
    for column, value in defaults.items():
        if column not in df:
            df[column] = value

    df["topic"] = (
        df["topic"]
        .fillna("Others")
        .astype(str)
        .str.strip()
        .replace({
            "": "Others",
            "Unclassified": "Others",
        })
    )

    if "date" not in df:
        df["date"] = pd.Timestamp.today().normalize()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).fillna(pd.Timestamp.today())

    df["sentiment"] = df["sentiment"].astype(str).str.title()

    df["confidence"] = (
        pd.to_numeric(df["confidence"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
    )

    return df[list(defaults) + ["date"]]



def load_dashboard_data() -> DashboardData:
    candidates = [
        PROJECT_ROOT / "sentiment_baseline_results.csv",
        PROJECT_ROOT / "raw_data" / "processed" / "reviews_analyzed.csv",
        PROJECT_ROOT / "data" / "processed" / "reviews_analyzed.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = _normalise(pd.read_csv(path))
            if not frame.empty:
                return DashboardData(frame, "analysis output", str(path.relative_to(PROJECT_ROOT)))
    return DashboardData(_normalise(_demo_frame()), "demo", None)
