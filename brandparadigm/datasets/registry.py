"""Single entrypoint for loading any of the three project datasets by name."""

import pandas as pd

from brandparadigm.datasets.amazon_reviews import load_amazon_reviews
from brandparadigm.datasets.reddit_dataset import load_reddit_posts
from brandparadigm.datasets.tweeteval import load_tweeteval

_LOADERS = {
    "amazon": lambda cfg, **kw: load_amazon_reviews(cfg["amazon_reviews"], **kw),
    "tweeteval": lambda cfg, **kw: load_tweeteval(cfg["tweet_eval"], **kw),
    "reddit": lambda cfg, **kw: load_reddit_posts(cfg["reddit"], **kw),
}

DATASET_NAMES = tuple(_LOADERS)


def load_dataset(name: str, config: dict, **kwargs) -> pd.DataFrame:
    """Load one of "amazon", "tweeteval", or "reddit" using data_config.yaml."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}', expected one of {DATASET_NAMES}")
    return _LOADERS[name](config, **kwargs)
