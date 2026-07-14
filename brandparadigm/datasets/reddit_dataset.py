"""Dataset 3 — historical Reddit dump (inference only, no live Reddit API).

Reads a Pushshift Reddit Submissions archive (`RS_YYYY-MM.zst`) directly —
no manual decompression/conversion required. Streams and filters to the
configured subreddit allowlist, keeping only
`[title, selftext, subreddit, score, created_utc]` and combining
`title` + `selftext` into a single `text` column.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import zstandard

from brandparadigm.datasets.local_source import require_local_file
from brandparadigm.logging import get_logger

logger = get_logger(__name__)

_KEEP_FIELDS = ["title", "selftext", "subreddit", "score", "created_utc"]
_READ_CHUNK_SIZE = 2**20  # 1 MiB


def _iter_submissions(archive_path: Path) -> Iterator[dict[str, Any]]:
    """Yield one decoded JSON object per line of a Pushshift .zst NDJSON dump."""
    with open(archive_path, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        buffer = b""
        while chunk := reader.read(_READ_CHUNK_SIZE):
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        if buffer.strip():
            try:
                yield json.loads(buffer)
            except json.JSONDecodeError:
                pass


def load_reddit_posts(config: dict, sample_size: int | None = None) -> pd.DataFrame:
    """Stream-filter the Reddit submissions archive into a normalized DataFrame.

    Args:
        config: the `reddit` section of configs/data_config.yaml.
        sample_size: cap on the number of matching rows collected (falls
            back to `config["sample_size"]`, e.g. 20000, when not given).
            Rows are the first N matches encountered while scanning the
            archive forward — not a reservoir sample — since the archive is
            read as a single streaming pass.

    Returns:
        DataFrame with columns
        [title, selftext, subreddit, score, created_utc, text, source].
    """
    archive_path = Path(config["raw_data_dir"]) / config["archive_file"]
    require_local_file(archive_path)

    allowlist = {s.lower() for s in (config.get("subreddit_allowlist") or [])}
    limit = sample_size if sample_size is not None else config.get("sample_size")

    rows: list[dict[str, Any]] = []
    scanned = 0
    for post in _iter_submissions(archive_path):
        scanned += 1
        subreddit = post.get("subreddit")
        if not subreddit or (allowlist and subreddit.lower() not in allowlist):
            continue
        rows.append({field: post.get(field) for field in _KEEP_FIELDS})
        if limit is not None and len(rows) >= limit:
            break

    logger.info("Scanned %d submissions in %s, matched %d rows", scanned, archive_path, len(rows))

    df = pd.DataFrame(rows, columns=_KEEP_FIELDS)
    title = df["title"].fillna("").astype(str)
    selftext = df["selftext"].fillna("").astype(str)
    df["text"] = (title + " " + selftext).str.strip()
    df["source"] = "reddit"
    return df.reset_index(drop=True)
