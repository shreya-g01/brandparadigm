import pandas as pd
import pytest

from brandparadigm.datasets.local_source import read_csv_sampled


@pytest.fixture
def large_csv(tmp_path):
    """A 1000-row headerless CSV, large enough to span many small chunks."""
    path = tmp_path / "data.csv"
    rows = [f"{i},value_{i}" for i in range(1000)]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_read_csv_sampled_no_sample_size_reads_every_row(large_csv):
    df = read_csv_sampled(large_csv, names=["id", "value"], header=None, sample_size=None)
    assert len(df) == 1000


def test_read_csv_sampled_respects_chunk_size_boundaries(large_csv):
    df = read_csv_sampled(
        large_csv, names=["id", "value"], header=None, sample_size=None, chunk_size=37
    )
    assert len(df) == 1000
    assert list(df["id"]) == list(range(1000))


def test_read_csv_sampled_reservoir_caps_at_sample_size(large_csv):
    df = read_csv_sampled(
        large_csv, names=["id", "value"], header=None, sample_size=50, chunk_size=64
    )
    assert len(df) == 50
    # A true sample of the whole file, not just the head — with 50 drawn
    # from 1000 rows, the odds every sampled id is < 500 are astronomically
    # small for a correct reservoir sample (unlike "head", which would cap
    # at id 49).
    assert df["id"].max() > 500


def test_read_csv_sampled_reservoir_is_deterministic_given_seed(large_csv):
    df1 = read_csv_sampled(
        large_csv, names=["id", "value"], header=None, sample_size=20, chunk_size=64, seed=7
    )
    df2 = read_csv_sampled(
        large_csv, names=["id", "value"], header=None, sample_size=20, chunk_size=64, seed=7
    )
    assert list(df1["id"]) == list(df2["id"])


def test_read_csv_sampled_reservoir_differs_across_seeds(large_csv):
    df1 = read_csv_sampled(
        large_csv, names=["id", "value"], header=None, sample_size=20, chunk_size=64, seed=1
    )
    df2 = read_csv_sampled(
        large_csv, names=["id", "value"], header=None, sample_size=20, chunk_size=64, seed=2
    )
    assert list(df1["id"]) != list(df2["id"])


def test_read_csv_sampled_head_strategy_takes_first_n_rows_only(large_csv):
    df = read_csv_sampled(
        large_csv,
        names=["id", "value"],
        header=None,
        sample_size=10,
        strategy="head",
    )
    assert list(df["id"]) == list(range(10))


def test_read_csv_sampled_sample_size_larger_than_file(large_csv):
    df = read_csv_sampled(large_csv, names=["id", "value"], header=None, sample_size=5000)
    assert len(df) == 1000


def test_read_csv_sampled_unknown_strategy_raises(large_csv):
    with pytest.raises(ValueError):
        read_csv_sampled(
            large_csv, names=["id", "value"], header=None, sample_size=10, strategy="bogus"
        )


def test_read_csv_sampled_with_header_row(tmp_path):
    path = tmp_path / "with_header.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(path, index=False)
    df = read_csv_sampled(path, sample_size=None)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 3
