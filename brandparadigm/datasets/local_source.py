"""Shared helpers for loading user-provided local dataset exports.

Amazon Reviews, TweetEval, and the historical Reddit archive are all
provisioned as local files (see docs/dataset_guide.md) at fixed, known
paths rather than downloaded, so their loaders share this small set of
"is it there yet, which column is it" helpers.
"""

import random
from pathlib import Path

import pandas as pd

from brandparadigm.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 50_000
SAMPLING_STRATEGIES = ("reservoir", "head")


class NoLocalDataError(FileNotFoundError):
    """Raised when a required local dataset export hasn't been provided yet."""


def require_local_file(path: Path) -> Path:
    """Return `path` if it exists, else raise a NoLocalDataError with guidance."""
    if not path.exists():
        raise NoLocalDataError(
            f"Expected dataset file not found: '{path}'. "
            "Place it there first (see docs/dataset_guide.md)."
        )
    return path


def first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name in `candidates` present in `df`, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def read_csv_sampled(
    path: Path,
    *,
    names: list[str] | None = None,
    header: int | str | None = "infer",
    sample_size: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    strategy: str = "reservoir",
    seed: int = 42,
) -> pd.DataFrame:
    """Read a (possibly very large) CSV without ever holding the full file in memory.

    Always reads `path` in `chunk_size`-row chunks rather than one monolithic
    parse. If `sample_size` is None, every row is needed by definition, so
    chunks are simply concatenated as they arrive (the full result is still
    held once fully read — unavoidable when nothing is being sampled out).

    If `sample_size` is set, one of two sampling strategies applies,
    trading statistical correctness for speed:

    - `"reservoir"` (default): Algorithm R reservoir sampling — every row in
      the file has an equal probability of ending up in the final sample.
      Correct, but requires one full sequential pass over the file (every
      row is visited once), so runtime scales with file size regardless of
      how small `sample_size` is. Peak memory stays bounded by roughly
      `sample_size + chunk_size` rows either way — the file is never
      materialized as a whole.
    - `"head"`: reads only the first `sample_size` rows (`pandas.read_csv`'s
      `nrows`) and stops — fast and memory-bounded, but not a random sample
      across the whole file (biased toward the file's existing order). Best
      for quick, iterative smoke tests where speed matters more than
      statistical representativeness; not recommended for real evaluation.

    Args:
        path: CSV file to read.
        names: explicit column names (for headerless files); mirrors
            `pandas.read_csv`'s `names` argument.
        header: row to use as the header, or `None` for headerless files;
            mirrors `pandas.read_csv`'s `header` argument.
        sample_size: if set, return at most this many rows (fewer only if
            the file itself has fewer rows).
        chunk_size: rows read per chunk. Larger values trade memory for
            fewer, bigger I/O reads. Ignored by the `"head"` strategy.
        strategy: `"reservoir"` or `"head"` — see above.
        seed: random seed for reservoir sampling — deterministic given the
            same file, `sample_size`, and seed.
    """
    if sample_size is None:
        reader = pd.read_csv(path, names=names, header=header, chunksize=chunk_size)
        return pd.concat(reader, ignore_index=True)

    if strategy == "head":
        return pd.read_csv(path, names=names, header=header, nrows=sample_size)

    if strategy != "reservoir":
        raise ValueError(
            f"Unknown sampling strategy '{strategy}', expected one of {SAMPLING_STRATEGIES}"
        )

    rng = random.Random(seed)
    reservoir: list[tuple] = []
    rows_seen = 0

    reader = pd.read_csv(path, names=names, header=header, chunksize=chunk_size)
    for chunk in reader:
        for row in chunk.itertuples(index=False, name=None):
            rows_seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = rng.randint(0, rows_seen - 1)
                if j < sample_size:
                    reservoir[j] = row

    columns = names if names is not None else list(pd.read_csv(path, nrows=0).columns)
    logger.info(
        "Reservoir-sampled %d/%d rows from %s (chunk_size=%d)",
        len(reservoir),
        rows_seen,
        path,
        chunk_size,
    )
    return pd.DataFrame(reservoir, columns=columns)
