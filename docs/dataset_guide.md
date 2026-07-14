# Dataset Guide

## Environment constraint: dataset provisioning

This session's egress policy blocks `huggingface.co`, `kaggle.com`,
`zenodo.org`, `archive.org`, and the UCSD/McAuley Lab hosts (confirmed via
the proxy's `/__agentproxy/status` endpoint — policy denials, not
transient failures). Only `raw.githubusercontent.com`,
`storage.googleapis.com`, and `pypi.org` are reachable. As a result:

- **TweetEval** is downloaded live — its official data lives in the
  `cardiffnlp/tweeteval` GitHub repo, unaffected by the block.
- **Amazon Reviews** and the **historical Reddit dump** are **user-provided
  local files** rather than fetched by this codebase. If your deployment
  environment has open internet access, point `brandparadigm.datasets` at
  the original sources instead (McAuley-Lab/Amazon-Reviews-2023 on HF Hub;
  any static historical Reddit export) — the loaders only care about the
  normalized schema below, not where the file came from.

## Dataset 1 — Amazon Reviews (training)

**Role**: fine-tune the RoBERTa sentiment model (Model 1).

**How to provide it**: drop one or more `.csv` / `.json` / `.jsonl` files
into `raw_data/amazon_reviews/`. Recommended subset per the spec:
Electronics, Cell Phones and Accessories.

**Expected columns** (configurable in `configs/data_config.yaml` —
`amazon_reviews.text_columns` / `rating_columns` / `category_columns`,
first match wins):

| Purpose | Accepted column names |
|---|---|
| Review text | `reviewText`, `text`, `review_text`, `review_body` |
| Star rating (1-5) | `overall`, `rating`, `star_rating`, `score` |
| Category (optional) | `category`, `main_category`, `product_category` |

**Label mapping** (`brandparadigm.preprocessing.star_rating_to_sentiment`):
1-2★ → Negative, 3★ → Neutral, 4-5★ → Positive.

**Loader**: `brandparadigm.datasets.load_amazon_reviews` →
`scripts/download_datasets.py --dataset amazon` → normalized
`raw_data/processed/amazon_raw.csv` with columns
`[text, rating, category, source]`, then
`scripts/run_preprocessing.py --dataset amazon` → `amazon_clean.csv` with
an added `sentiment_label` column.

Verified against a 5-row fixture (`tests/fixtures/amazon_reviews_sample.csv`)
mirroring the real schema — see `tests/datasets/test_amazon_reviews.py`.

## Dataset 2 — TweetEval (evaluation only)

**Role**: evaluate Model 1 — **never train on this dataset**, per the spec.

**Source**: `tweet_eval`'s `sentiment` task, official files at
`https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/{split}_text.txt`
and `..._labels.txt`, splits `train` / `val` / `test`. Labels are already
`0=Negative, 1=Neutral, 2=Positive` — identical ordering to our scheme.

**Confirmed live** (2026-07-14): `test` split has 12,284 rows.

**Loader**: `brandparadigm.datasets.load_tweeteval` →
`scripts/download_datasets.py --dataset tweeteval` → `tweeteval_raw.csv`
`[text, label, source]` → `run_preprocessing.py` → `tweeteval_clean.csv`
adds `sentiment_label` (a direct copy of `label`, kept as a separate column
so every dataset's clean CSV shares the same final schema).

## Dataset 3 — Historical Reddit dump (inference)

**Role**: input to sentiment prediction, BERTopic discovery, the topic
classifier, and the dashboard. No live Reddit API is used anywhere.

**How to provide it**: drop one or more `.csv` / `.json` / `.jsonl` files
into `raw_data/reddit/`. A static historical export (not a live pull).

**Expected columns** (configurable in `configs/data_config.yaml` —
`reddit.text_columns` / `title_columns` / `subreddit_columns`):

| Purpose | Accepted column names |
|---|---|
| Post body | `selftext`, `body`, `text`, `review_text` |
| Title (optional, concatenated with body) | `title` |
| Subreddit (optional, for allowlist filtering) | `subreddit` |

`reddit.subreddit_allowlist` in the config filters to
tech/product-relevant subreddits (`apple`, `samsung`, `GooglePixel`,
`oneplus`, `android`, `gadgets`, `technology`) for brand-comparison
relevance — set it to `null` to keep every row.

**Loader**: `brandparadigm.datasets.load_reddit_posts` →
`scripts/download_datasets.py --dataset reddit` → `reddit_raw.csv`
`[text, subreddit, source]` → `run_preprocessing.py` → `reddit_clean.csv`
(text cleaned, no sentiment label — this dataset is inference-only).

Verified against a 3-row fixture (`tests/fixtures/reddit_sample.csv`) — see
`tests/datasets/test_reddit_dataset.py`.

## Running the pipeline

```bash
make download_datasets   # or: python scripts/download_datasets.py --dataset all --sample-size 500
make preprocess           # or: python scripts/run_preprocessing.py --dataset all
```

`download_datasets.py` exits non-zero with a clear message if
`raw_data/amazon_reviews/` or `raw_data/reddit/` is empty — it never
crashes silently or fabricates data.
