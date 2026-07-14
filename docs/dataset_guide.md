# Dataset Guide

## Binary sentiment: why, and what it means for each dataset

Model 1's production label scheme is **binary**: 0 = Negative,
1 = Positive (`brandparadigm.preprocessing.label_mapping.LABEL2ID`,
`configs/sentiment_config.yaml`). The only training dataset, Amazon Review
Polarity, is itself binary — it has no Neutral class — so rather than
fabricate one or switch training datasets, the whole sentiment task is
defined as binary end-to-end. Concretely:

- **Amazon Review Polarity** (training + validation): already binary,
  maps directly to 0/1.
- **TweetEval `sentiment`** (evaluation only): natively 3-class in the
  source data. The loader decodes it faithfully (including "Neutral"), but
  `brandparadigm.preprocessing.label_mapping.prepare_binary_sentiment_labels`
  **drops every Neutral row before computing the binary
  `sentiment_label`** — used by both `scripts/run_preprocessing.py` and
  `brandparadigm.sentiment.evaluate.load_tweeteval_eval_set` — so
  evaluation only ever compares Negative vs Positive, the same label space
  the model actually predicts.
- **Reddit**: unlabeled, inference-only; sentiment on Reddit posts comes
  from Model 1's predictions, not ground-truth labels, so binary vs 3-class
  is not applicable here.

See `docs/architecture.md` ("Why binary sentiment, not 3-class") and
`docs/model_cards/roberta_sentiment.md` for the full rationale.

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

**Label mapping**: `polarity == 1` → Negative, `polarity == 2` → Positive
(`brandparadigm.preprocessing.amazon_polarity_to_sentiment`), then to the
binary int scheme via `LABEL2ID` (`Negative` → 0, `Positive` → 1). This
dataset is binary — there is no Neutral class from this source, which is
why the project's production sentiment scheme is binary end-to-end (see
"Binary sentiment" above).

**Loader**: `brandparadigm.datasets.load_amazon_reviews(config, split=...)`
— `split` is `"train"`, `"test"`, or `"all"` (concatenates both files).
Per this refactor, **`train.csv` is used for training and `test.csv` for
validation/testing** (`configs/sentiment_config.yaml`). `title` and `text`
are combined into a single `text` field. →
`scripts/download_datasets.py --dataset amazon` → normalized
`raw_data/processed/amazon_raw.csv` `[text, polarity, source]` →
`scripts/run_preprocessing.py --dataset amazon` → `amazon_clean.csv` adds
`sentiment_label` (int, 0 or 1).

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
encoding or already-string labels (`"positive"`, `"Negative"`, `"neutral"`,
... — matched case-insensitively); both are decoded faithfully — including
Neutral — by `brandparadigm.datasets.tweeteval._normalize_label`. **The raw
loader does not filter anything out**; it exists to represent TweetEval's
actual 3-class source data without information loss.

**Loader**: `brandparadigm.datasets.load_tweeteval(config, split=...)` —
`split` is `"train"`, `"validation"`, or `"test"` →
`scripts/download_datasets.py --dataset tweeteval` → `tweeteval_raw.csv`
`[text, label, source]` (may still contain "Neutral" rows) →
`run_preprocessing.py` → **`preprocess_tweeteval` drops every row where
`label == "Neutral"`, then maps the remaining Negative/Positive rows to
`sentiment_label` (int, 0 or 1) via `LABEL2ID`** → `tweeteval_clean.csv`
(binary-only, ready to evaluate directly against Model 1's predictions).
`brandparadigm.sentiment.evaluate.load_tweeteval_eval_set` does the same
filtering self-contained (reads straight from `raw_data/tweeteval/`,
doesn't require the CSV cache to exist) for use during model evaluation.

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
