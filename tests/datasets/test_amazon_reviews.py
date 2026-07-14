import shutil
from pathlib import Path

import pytest

from brandparadigm.datasets.amazon_reviews import load_amazon_reviews
from brandparadigm.datasets.local_source import NoLocalDataError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "amazon_reviews_sample.csv"


@pytest.fixture
def config(tmp_path):
    data_dir = tmp_path / "amazon_reviews"
    data_dir.mkdir()
    shutil.copy(FIXTURE, data_dir / "sample.csv")
    return {
        "raw_data_dir": str(data_dir),
        "file_pattern": "*.csv",
        "text_columns": ["reviewText", "text"],
        "rating_columns": ["overall", "rating"],
        "category_columns": ["category"],
    }


def test_loads_and_normalizes_columns(config):
    df = load_amazon_reviews(config)
    assert list(df.columns) == ["text", "rating", "category", "source"]
    assert len(df) == 5
    assert (df["source"] == "amazon_reviews").all()


def test_sample_size_caps_rows(config):
    df = load_amazon_reviews(config, sample_size=2)
    assert len(df) == 2


def test_missing_directory_raises(tmp_path):
    config = {
        "raw_data_dir": str(tmp_path / "does_not_exist"),
        "file_pattern": "*.csv",
        "text_columns": ["reviewText"],
        "rating_columns": ["overall"],
    }
    with pytest.raises(NoLocalDataError):
        load_amazon_reviews(config)
