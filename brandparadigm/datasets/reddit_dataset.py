"""Dataset 3 — historical Reddit dump (inference only, no live Reddit API).

Provisioned as a user-supplied local export (see docs/dataset_guide.md);
this loader normalizes whatever schema was exported into
`{text, subreddit, source}` and optionally filters to a subreddit allowlist.
"""

import pandas as pd

from brandparadigm.datasets.local_source import first_matching_column, load_local_frame
from brandparadigm.logging import get_logger

logger = get_logger(__name__)


def load_reddit_posts(config: dict, sample_size: int | None = None) -> pd.DataFrame:
    """Load and normalize the local historical Reddit export.

    Args:
        config: the `reddit` section of configs/data_config.yaml.
        sample_size: if set, randomly sample at most this many rows.

    Returns:
        DataFrame with columns [text, subreddit, source].
    """
    raw = load_local_frame(config["raw_data_dir"], config["file_pattern"])

    text_col = first_matching_column(raw, config["text_columns"])
    if text_col is None:
        raise ValueError(
            f"Could not find a text column in {list(raw.columns)}. "
            f"Expected one of {config['text_columns']}."
        )
    title_col = first_matching_column(raw, config.get("title_columns", []))
    subreddit_col = first_matching_column(raw, config.get("subreddit_columns", []))

    text = raw[text_col].astype(str)
    if title_col:
        text = raw[title_col].astype(str).str.cat(text, sep=" ", na_rep="")

    df = pd.DataFrame(
        {
            "text": text,
            "subreddit": raw[subreddit_col] if subreddit_col else "unknown",
        }
    )
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    allowlist = config.get("subreddit_allowlist")
    if allowlist and subreddit_col:
        before = len(df)
        df = df[df["subreddit"].isin(allowlist)]
        logger.info("Filtered to allowlisted subreddits: %d -> %d rows", before, len(df))

    df["source"] = "reddit"

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info("Loaded %d Reddit post rows", len(df))
    return df.reset_index(drop=True)
