import shutil
from pathlib import Path

import pytest

from brandparadigm.datasets.amazon_reviews import load_amazon_reviews
from brandparadigm.datasets.local_source import NoLocalDataError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "amazon_review_polarity_sample.csv"


@pytest.fixture
def config(tmp_path):
    data_dir = tmp_path / "amazon_review_polarity_csv"
    data_dir.mkdir()
    shutil.copy(FIXTURE, data_dir / "train.csv")
    shutil.copy(FIXTURE, data_dir / "test.csv")
    return {
        "raw_data_dir": str(data_dir),
        "train_file": "train.csv",
        "test_file": "test.csv",
        "columns": ["polarity", "title", "text"],
    }


def test_loads_and_normalizes_columns(config):
    df = load_amazon_reviews(config, split="train")
    assert list(df.columns) == ["text", "polarity", "source"]
    assert len(df) == 5
    assert (df["source"] == "amazon_reviews").all()
    assert set(df["polarity"]) == {1, 2}


def test_title_is_combined_into_text(config):
    df = load_amazon_reviews(config, split="train")
    row = df[df["text"].str.contains("Battery died")].iloc[0]
    assert "Terrible battery" in row["text"]


def test_split_all_concatenates_train_and_test(config):
    df = load_amazon_reviews(config, split="all")
    assert len(df) == 10


def test_unknown_split_raises(config):
    with pytest.raises(ValueError):
        load_amazon_reviews(config, split="bogus")


def test_sample_size_caps_rows(config):
    df = load_amazon_reviews(config, split="train", sample_size=2)
    assert len(df) == 2


def test_missing_file_raises(tmp_path):
    config = {
        "raw_data_dir": str(tmp_path / "does_not_exist"),
        "train_file": "train.csv",
        "test_file": "test.csv",
        "columns": ["polarity", "title", "text"],
    }
    with pytest.raises(NoLocalDataError):
        load_amazon_reviews(config, split="train")
