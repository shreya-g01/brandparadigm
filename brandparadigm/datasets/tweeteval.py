"""Dataset 2 — TweetEval (evaluation only, never trained on).

Downloaded directly from the official cardiffnlp/tweeteval GitHub repo —
the `sentiment` task's `train`/`val`/`test` splits are plain text files
hosted at `config["base_url"]`, no Hugging Face Hub access required.
"""

import httpx
import pandas as pd

from brandparadigm.logging import get_logger
from brandparadigm.preprocessing.label_mapping import tweeteval_label_to_sentiment

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(30.0)


def _fetch_text(url: str) -> list[str]:
    response = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    return response.text.splitlines()


def load_tweeteval(
    config: dict, split: str = "test", sample_size: int | None = None
) -> pd.DataFrame:
    """Load a TweetEval sentiment split, normalized to [text, label, source].

    Args:
        config: the `tweet_eval` section of configs/data_config.yaml.
        split: one of config["splits"] ("train"/"val"/"test"). Per the spec
            this dataset is for evaluation only — do not use "train" to fit
            the sentiment model.
        sample_size: if set, take at most this many rows (deterministic head,
            since the upstream file order is not itself random).
    """
    if split not in config["splits"]:
        raise ValueError(f"Unknown split '{split}', expected one of {config['splits']}")

    base_url = config["base_url"]
    texts = _fetch_text(f"{base_url}/{split}_text.txt")
    labels = _fetch_text(f"{base_url}/{split}_labels.txt")
    if len(texts) != len(labels):
        raise ValueError(
            f"Mismatched text/label counts for split '{split}': {len(texts)} vs {len(labels)}"
        )

    df = pd.DataFrame(
        {"text": texts, "label": [tweeteval_label_to_sentiment(int(raw)) for raw in labels]}
    )
    df["source"] = "tweet_eval"

    if sample_size is not None and len(df) > sample_size:
        df = df.head(sample_size).reset_index(drop=True)

    logger.info("Loaded %d TweetEval (%s) rows", len(df), split)
    return df
