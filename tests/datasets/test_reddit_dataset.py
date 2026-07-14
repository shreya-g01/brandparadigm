import shutil
from pathlib import Path

import pytest

from brandparadigm.datasets.reddit_dataset import load_reddit_posts

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reddit_sample.csv"


@pytest.fixture
def config(tmp_path):
    data_dir = tmp_path / "reddit"
    data_dir.mkdir()
    shutil.copy(FIXTURE, data_dir / "sample.csv")
    return {
        "raw_data_dir": str(data_dir),
        "file_pattern": "*.csv",
        "text_columns": ["selftext", "text"],
        "title_columns": ["title"],
        "subreddit_columns": ["subreddit"],
        "subreddit_allowlist": ["apple", "android"],
    }


def test_loads_and_filters_subreddit_allowlist(config):
    df = load_reddit_posts(config)
    assert list(df.columns) == ["text", "subreddit", "source"]
    assert len(df) == 2
    assert set(df["subreddit"]) == {"apple", "android"}


def test_title_is_concatenated_with_body(config):
    df = load_reddit_posts(config)
    battery_row = df[df["subreddit"] == "apple"].iloc[0]
    assert "Battery drains fast" in battery_row["text"]
    assert "battery drains way too fast" in battery_row["text"]


def test_no_allowlist_keeps_all_rows(config):
    config = {**config, "subreddit_allowlist": None}
    df = load_reddit_posts(config)
    assert len(df) == 3
