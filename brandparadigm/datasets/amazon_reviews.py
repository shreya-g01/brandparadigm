"""Dataset 1 — Amazon Reviews (training data for the sentiment model).

Provisioned as a user-supplied local export (see docs/dataset_guide.md);
this loader normalizes whatever schema was exported into
`{text, rating, category, source}`.
"""

import pandas as pd

from brandparadigm.datasets.local_source import first_matching_column, load_local_frame
from brandparadigm.logging import get_logger

logger = get_logger(__name__)


def load_amazon_reviews(config: dict, sample_size: int | None = None) -> pd.DataFrame:
    """Load and normalize the local Amazon Reviews export.

    Args:
        config: the `amazon_reviews` section of configs/data_config.yaml.
        sample_size: if set, randomly sample at most this many rows.

    Returns:
        DataFrame with columns [text, rating, category, source].
    """
    raw = load_local_frame(config["raw_data_dir"], config["file_pattern"])

    text_col = first_matching_column(raw, config["text_columns"])
    rating_col = first_matching_column(raw, config["rating_columns"])
    if text_col is None or rating_col is None:
        raise ValueError(
            f"Could not find a text/rating column in {list(raw.columns)}. "
            f"Expected one of {config['text_columns']} and {config['rating_columns']}."
        )
    category_col = first_matching_column(raw, config.get("category_columns", []))

    df = pd.DataFrame(
        {
            "text": raw[text_col].astype(str),
            "rating": pd.to_numeric(raw[rating_col], errors="coerce"),
            "category": raw[category_col] if category_col else "Unknown",
        }
    )
    df = df.dropna(subset=["text", "rating"])
    df = df[df["text"].str.strip() != ""]
    df["source"] = "amazon_reviews"

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info("Loaded %d Amazon review rows", len(df))
    return df.reset_index(drop=True)
