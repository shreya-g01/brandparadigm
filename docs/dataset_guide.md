# Dataset Guide

## Environment constraint: dataset provisioning

This session's egress policy blocks `huggingface.co`, `kaggle.com`,
`zenodo.org`, `archive.org`, and the UCSD/McAuley Lab hosts (confirmed via
the proxy's `/__agentproxy/status` endpoint — policy denials, not
transient failures). Only `raw.githubusercontent.com`,
`storage.googleapis.com`, and `pypi.org` are reachable. As a result, **all
three datasets are user-provided local files** — none are downloaded by
this codebase. If your deployment environment has open internet access,
swap in the original hosted sources instead; the loaders only care about
the file layout and schema below, not where the file came from.

## Dataset 1 — Amazon Review Polarity (training)

**Role**: fine-tune the RoBERTa sentiment model (Model 1).

**Source**: Amazon Review Polarity (Zhang et al.), a binary-sentiment
version of the Amazon Reviews corpus.

**How to provide it**: place the two files at
`raw_data/amazon_reviews/amazon_review_polarity_csv/train.csv` and
`.../test.csv`. Headerless CSVs, fixed columns `[polarity, title, text]`
(configurable in `configs/data_config.yaml` — `amazon_reviews.columns`).

**Label mapping** (`brandparadigm.preprocessing.amazon_polarity_to_sentiment`):
`polarity == 1` → Negative, `polarity == 2` → Positive. This dataset is
binary — there is no Neutral class from this source (Neutral coverage for
Model 1 comes from TweetEval during evaluation and, at inference time, from
Reddit posts the model itself classifies as Neutral).

**Loader**: `brandparadigm.datasets.load_amazon_reviews(config, split=...)`
— `split` is `"train"`, `"test"`, or `"all"` (concatenates both files).
`title` and `text` are combined into a single `text` field. →
`scripts/download_datasets.py --dataset amazon` → normalized
`raw_data/processed/amazon_raw.csv` `[text, polarity, source]` →
`scripts/run_preprocessing.py --dataset amazon` → `amazon_clean.csv` adds
`sentiment_label`.

Verified against a 5-row fixture
(`tests/fixtures/amazon_review_polarity_sample.csv`) — see
`tests/datasets/test_amazon_reviews.py`.

## Dataset 2 — TweetEval `sentiment` (evaluation only)

**Role**: evaluate Model 1 — **never train on this dataset**, per the spec.

**Source**: TweetEval, **`sentiment` task only** — every other TweetEval
task (emotion, hate, irony, offensive, ...) is ignored entirely.

**How to provide it**: place three CSVs under `raw_data/tweeteval/`:
`sentiment_train.csv`, `sentiment_validation.csv`, `sentiment_test.csv`.

**Expected columns** (configurable — `tweet_eval.text_columns` /
`label_columns`, first match wins):

| Purpose | Accepted column names |
|---|---|
| Tweet text | `text`, `tweet` |
| Sentiment label | `label`, `sentiment` |

Labels may be either the original `0=Negative, 1=Neutral, 2=Positive` int
encoding or already-string labels (`"positive"`, `"Negative"`, ... —
matched case-insensitively); both are normalized to the project's 3-class
scheme by `brandparadigm.datasets.tweeteval._normalize_label`.

**Loader**: `brandparadigm.datasets.load_tweeteval(config, split=...)` —
`split` is `"train"`, `"validation"`, or `"test"` →
`scripts/download_datasets.py --dataset tweeteval` → `tweeteval_raw.csv`
`[text, label, source]` → `run_preprocessing.py` → `tweeteval_clean.csv`.

Verified against a 5-row fixture
(`tests/fixtures/tweeteval_sentiment_sample.csv`) — see
`tests/datasets/test_tweeteval.py`.

## Dataset 3 — Historical Reddit archive (inference)

**Role**: input to sentiment prediction, BERTopic discovery, the topic
classifier, and the dashboard. No live Reddit API is used anywhere.

**Source**: a Pushshift Reddit **Submissions** archive, `RS_YYYY-MM.zst`.

**How to provide it**: place the archive at `raw_data/reddit/RS_2019-04.zst`
(filename configurable — `reddit.archive_file`). **No manual decompression
or conversion is needed** — the loader reads the `.zst` archive directly,
streaming and decoding its NDJSON contents with the `zstandard` package.

**Filtering**: only posts from the allowlisted subreddits are kept
(`reddit.subreddit_allowlist`, matched case-insensitively): `apple`,
`samsung`, `GooglePixel`, `Android`, `OnePlus`, `gadgets`, `technology`.

**Fields kept** from each matching submission: `title`, `selftext`,
`subreddit`, `score`, `created_utc` — every other field in the raw dump
(id, author, url, ...) is dropped. `title` and `selftext` are combined into
a single `text` column.

**Sample size**: configurable, **default 20,000 rows**
(`reddit.sample_size` in `configs/data_config.yaml`, overridable via
`--sample-size`). Rows are the first N matches encountered while streaming
the archive forward, not a reservoir sample, since the archive is read as a
single pass — the practical trade-off for scanning a multi-GB archive
without loading it into memory.

**Loader**: `brandparadigm.datasets.load_reddit_posts(config, sample_size=...)`
→ `scripts/download_datasets.py --dataset reddit` → `reddit_raw.csv`
`[title, selftext, subreddit, score, created_utc, text, source]` →
`run_preprocessing.py` → `reddit_clean.csv` (text cleaned; no sentiment
label — this dataset is inference-only).

Verified against a 5-post synthetic `.zst` fixture
(`tests/fixtures/reddit_submissions_sample.zst`) — see
`tests/datasets/test_reddit_dataset.py`.

## Running the pipeline

```bash
make download_datasets   # or: python scripts/download_datasets.py --dataset all --sample-size 500
make preprocess           # or: python scripts/run_preprocessing.py --dataset all
```

`download_datasets.py` exits non-zero with a clear message if any expected
file is missing — it never crashes silently or fabricates data. The
`--sample-size` flag overrides every dataset's default sampling (including
Reddit's 20,000-row default) for a fast smoke test.
