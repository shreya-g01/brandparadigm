#!/usr/bin/env python3
"""CLI: evaluate a fine-tuned sentiment model against TweetEval (Neutral rows dropped).

Per the spec, TweetEval is evaluation-only — this script never trains on
it. Requires network access to Hugging Face Hub to load the fine-tuned
model's tokenizer/config the same way training did.
"""

import argparse
import sys

import numpy as np
import torch

from brandparadigm.config.paths import CONFIGS_DIR
from brandparadigm.logging import get_logger
from brandparadigm.sentiment.dataset import build_tokenized_dataset
from brandparadigm.sentiment.evaluate import build_evaluation_report, load_tweeteval_eval_set
from brandparadigm.sentiment.model import load_trained_model_and_tokenizer
from brandparadigm.utils import read_yaml, write_json

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory of the fine-tuned model (defaults to sentiment_config.yaml::training.output_dir).",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sentiment-config", default=str(CONFIGS_DIR / "sentiment_config.yaml"))
    parser.add_argument("--data-config", default=str(CONFIGS_DIR / "data_config.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sentiment_config = read_yaml(args.sentiment_config)
    data_config = read_yaml(args.data_config)

    model_dir = args.model_dir or sentiment_config["training"]["output_dir"]
    model, tokenizer = load_trained_model_and_tokenizer(model_dir)
    model.eval()

    eval_df = load_tweeteval_eval_set(
        data_config, split=sentiment_config["evaluation"]["split"], sample_size=args.sample_size
    )
    max_length = sentiment_config["model"]["max_seq_length"]
    eval_dataset = build_tokenized_dataset(eval_df, tokenizer, max_length=max_length)

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(eval_dataset), 32):
            batch = eval_dataset[i : i + 32]
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            all_logits.append(outputs.logits)

    logits = torch.cat(all_logits, dim=0)
    y_pred = logits.argmax(dim=-1).numpy()
    y_true = np.array(eval_dataset["labels"])

    report = build_evaluation_report(y_true, y_pred)
    out_path = f"{model_dir}/tweeteval_evaluation_report.json"
    write_json(report, out_path)

    logger.info("Evaluated %d TweetEval examples (Neutral rows already dropped)", len(eval_dataset))
    logger.info("Report written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
