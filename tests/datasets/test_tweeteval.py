import shutil
from pathlib import Path

import pytest

from brandparadigm.datasets.local_source import NoLocalDataError
from brandparadigm.datasets.tweeteval import load_tweeteval

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tweeteval_sentiment_sample.csv"


@pytest.fixture
def config(tmp_path):
    data_dir = tmp_path / "tweeteval"
    data_dir.mkdir()
    shutil.copy(FIXTURE, data_dir / "sentiment_train.csv")
    shutil.copy(FIXTURE, data_dir / "sentiment_validation.csv")
    shutil.copy(FIXTURE, data_dir / "sentiment_test.csv")
    return {
        "raw_data_dir": str(data_dir),
        "files": {
            "train": "sentiment_train.csv",
            "validation": "sentiment_validation.csv",
            "test": "sentiment_test.csv",
        },
        "text_columns": ["text", "tweet"],
        "label_columns": ["label", "sentiment"],
    }


def test_load_tweeteval_maps_numeric_labels(config):
    df = load_tweeteval(config, split="test")
    assert list(df.columns) == ["text", "label", "source"]
    assert list(df["label"]) == ["Positive", "Negative", "Neutral", "Positive", "Negative"]
    assert (df["source"] == "tweet_eval").all()


def test_load_tweeteval_does_not_filter_neutral_rows(config):
    # The production sentiment model is binary, but this raw loader decodes
    # TweetEval's source data faithfully (including "Neutral") — filtering
    # happens downstream in scripts/run_preprocessing.py, not here.
    df = load_tweeteval(config, split="test")
    assert "Neutral" in set(df["label"])


def test_load_tweeteval_accepts_neutral_string_label(tmp_path, config):
    data_dir = Path(config["raw_data_dir"])
    (data_dir / "sentiment_test.csv").write_text("text,label\nmeh,neutral\ngreat,Positive\n")
    df = load_tweeteval(config, split="test")
    assert list(df["label"]) == ["Neutral", "Positive"]


def test_load_tweeteval_accepts_string_labels(tmp_path, config):
    data_dir = Path(config["raw_data_dir"])
    (data_dir / "sentiment_test.csv").write_text(
        "text,label\ngreat stuff,positive\nbad stuff,Negative\n"
    )
    df = load_tweeteval(config, split="test")
    assert list(df["label"]) == ["Positive", "Negative"]


def test_load_tweeteval_rejects_unknown_split(config):
    with pytest.raises(ValueError):
        load_tweeteval(config, split="bogus")


def test_load_tweeteval_sample_size(config):
    df = load_tweeteval(config, split="test", sample_size=2)
    assert len(df) == 2


def test_missing_file_raises(tmp_path):
    config = {
        "raw_data_dir": str(tmp_path / "does_not_exist"),
        "files": {"test": "sentiment_test.csv"},
        "text_columns": ["text"],
        "label_columns": ["label"],
    }
    with pytest.raises(NoLocalDataError):
        load_tweeteval(config, split="test")
