#!/usr/bin/env python3
"""CLI: fetch/validate the three project datasets.

TweetEval is downloaded live from GitHub. Amazon Reviews and the historical
Reddit dump are user-provided local exports (see docs/dataset_guide.md) —
for those this script validates the files are present and readable, then
writes a normalized cache; it does not attempt any network fetch for them.
"""

import argparse
import sys

from brandparadigm.config.paths import CONFIGS_DIR, PROCESSED_DATA_DIR
from brandparadigm.datasets import DATASET_NAMES, NoLocalDataError, load_dataset
from brandparadigm.logging import get_logger
from brandparadigm.utils import ensure_dir, read_yaml

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASET_NAMES, "all"],
        default="all",
        help="Which dataset to fetch/validate.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Cap the number of rows loaded (e.g. for a fast smoke test).",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIGS_DIR / "data_config.yaml"),
        help="Path to data_config.yaml",
    )
    return parser.parse_args()


def run_one(name: str, config: dict, sample_size: int | None) -> bool:
    try:
        df = load_dataset(name, config, sample_size=sample_size)
    except NoLocalDataError as exc:
        logger.error("Dataset '%s' unavailable: %s", name, exc)
        return False

    out_dir = ensure_dir(PROCESSED_DATA_DIR)
    out_path = out_dir / f"{name}_raw.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return True


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config)
    names = list(DATASET_NAMES) if args.dataset == "all" else [args.dataset]

    ok = True
    for name in names:
        ok = run_one(name, config, args.sample_size) and ok

    if not ok:
        logger.error("One or more datasets could not be loaded — see docs/dataset_guide.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
