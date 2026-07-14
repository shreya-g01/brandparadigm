import shutil
from pathlib import Path

import pytest

from brandparadigm.datasets.local_source import NoLocalDataError
from brandparadigm.datasets.reddit_dataset import load_reddit_posts

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reddit_submissions_sample.zst"


@pytest.fixture
def config(tmp_path):
    data_dir = tmp_path / "reddit"
    data_dir.mkdir()
    shutil.copy(FIXTURE, data_dir / "RS_sample.zst")
    return {
        "raw_data_dir": str(data_dir),
        "archive_file": "RS_sample.zst",
        "subreddit_allowlist": [
            "apple",
            "samsung",
            "GooglePixel",
            "Android",
            "OnePlus",
            "gadgets",
            "technology",
        ],
        "sample_size": 20000,
    }


def test_streams_and_filters_subreddit_allowlist(config):
    df = load_reddit_posts(config)
    assert list(df.columns) == [
        "title",
        "selftext",
        "subreddit",
        "score",
        "created_utc",
        "text",
        "source",
    ]
    # fixture has 5 posts; "funny" is not allowlisted, the other 4 are.
    assert len(df) == 4
    assert "funny" not in set(df["subreddit"])


def test_allowlist_is_case_insensitive(config):
    df = load_reddit_posts(config)
    assert "Android" in set(df["subreddit"])


def test_title_and_selftext_combined_into_text(config):
    df = load_reddit_posts(config)
    row = df[df["subreddit"] == "apple"].iloc[0]
    assert "Battery drains fast" in row["text"]
    assert "battery drains way too fast" in row["text"]


def test_sample_size_caps_matched_rows(config):
    df = load_reddit_posts(config, sample_size=2)
    assert len(df) == 2


def test_missing_archive_raises(tmp_path):
    config = {
        "raw_data_dir": str(tmp_path / "does_not_exist"),
        "archive_file": "RS_missing.zst",
        "subreddit_allowlist": ["apple"],
        "sample_size": 100,
    }
    with pytest.raises(NoLocalDataError):
        load_reddit_posts(config)
