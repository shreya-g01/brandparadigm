#!/usr/bin/env python3
"""CLI: fine-tune the binary sentiment model (Model 1).

Requires network access to Hugging Face Hub to download the base
checkpoint (`configs/sentiment_config.yaml::model.base_model`) — this
script itself makes no attempt to work around a blocked Hub; it assumes
that access exists in the environment it's actually run in.
"""

import argparse
import sys

from brandparadigm.config.paths import CONFIGS_DIR
from brandparadigm.logging import get_logger
from brandparadigm.sentiment.train import train
from brandparadigm.utils import read_yaml

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke_test", "full"],
        default="smoke_test",
        help="Hyperparameter profile from configs/sentiment_config.yaml.",
    )
    parser.add_argument(
        "--sentiment-config",
        default=str(CONFIGS_DIR / "sentiment_config.yaml"),
    )
    parser.add_argument(
        "--data-config",
        default=str(CONFIGS_DIR / "data_config.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sentiment_config = read_yaml(args.sentiment_config)
    data_config = read_yaml(args.data_config)

    result = train(sentiment_config, data_config, profile=args.profile)

    logger.info("Training complete. Eval metrics: %s", result["eval_metrics"])
    logger.info("Artifacts saved to %s", result["output_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
