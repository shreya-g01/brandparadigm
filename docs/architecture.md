# Architecture

## Business problem

Businesses collect thousands of customer reviews but lack a reliable way to
turn that unstructured text into structured intelligence: overall sentiment,
recurring pain points, and how they compare to competitors. Generic
sentiment tools and keyword search don't capture domain-specific nuance.
BrandParadigm turns raw reviews/posts into three structured signals
(sentiment, discovered topics, business-category topic) and aggregates them
into brand-level insights.

## ML pipeline

```
Amazon Review Polarity (train.csv)
        │
Preprocessing (clean text, polarity → binary label: 0=Negative, 1=Positive)
        │
Fine-tune RoBERTa  ───────────────► Model 1: Binary Sentiment Classifier
        │
Validate using Amazon Review Polarity (test.csv)
        │
Evaluate using TweetEval sentiment (Neutral rows dropped, held-out, never trained on)
        │
Save Best Model
        │
Historical Reddit Dataset (inference-only, no live API)
        │
Sentiment Prediction (Model 1 applied to Reddit posts)
        │
BERTopic  ─────────────────────────► Model 2: Topic Discovery
        │
Manual Topic Validation (discovered topics → 7 business categories)
        │
Topic Classifier (LogisticRegression vs DistilBERT, best selected)
                                    ► Model 3: Business-Category Classifier
        │
Business Insights (brandparadigm.analytics)
        │
Dashboard (Streamlit) / API (FastAPI)
```

## Three ML components

| # | Model | Purpose | Input | Output |
|---|-------|---------|-------|--------|
| 1 | Fine-tuned RoBERTa (`roberta-base`) | **Binary** sentiment classification | Review/post text | Negative (0) / Positive (1) |
| 2 | BERTopic | Unsupervised topic discovery | Reddit post text | Discovered discussion themes |
| 3 | Topic Classifier (best of LogisticRegression / DistilBERT) | Business-category prediction | Review/post text | Product Quality, Battery, Customer Service, Delivery, Pricing, Features, Design |

### Why binary sentiment, not 3-class

The master spec originally called for Positive/Neutral/Negative. The only
dataset Model 1 trains on — Amazon Review Polarity — is itself binary (no
Neutral class exists in the source data). Rather than fabricate a Neutral
label (e.g. by treating some score threshold as "Neutral" on a dataset that
doesn't encode one) or switch to a different, star-rated training dataset,
the sentiment task was redefined as binary end-to-end: 0=Negative,
1=Positive. This keeps every label the model ever sees grounded in real
annotations. See `docs/model_cards/roberta_sentiment.md` for the full
rationale and `configs/sentiment_config.yaml` for the task definition
(`task_type: binary`, `num_labels: 2`).

TweetEval's `sentiment` task is natively 3-class, so its Neutral rows are
filtered out during preprocessing before evaluation metrics are computed —
evaluation compares apples to apples against the binary production label
space. See `docs/dataset_guide.md`.

## Datasets

| # | Dataset | Role | Notes |
|---|---------|------|-------|
| 1 | Amazon Review Polarity (Zhang et al.), local `train.csv`/`test.csv` | Train (`train.csv`) + validate (`test.csv`) Model 1 | Binary `polarity` (1=Negative, 2=Positive) — the source of the project's binary label scheme |
| 2 | TweetEval, `sentiment` task only, local `sentiment_{train,validation,test}.csv` | Evaluate Model 1 only | Never trained on; Neutral rows dropped during preprocessing so evaluation is binary-only; every other TweetEval task ignored |
| 3 | Historical Reddit Submissions archive (`RS_2019-04.zst`), read directly, filtered to tech/product subreddits | Inference for Models 1–3, dashboard | Static historical dump, no live Reddit API, no labels; default 20,000-row sample |

See `docs/dataset_guide.md` for the exact configs confirmed during Phase 2
and their schemas.

## Software architecture

Monorepo, Le Wagon base-project layout:

- **`brandparadigm/`** — the installable package. All business logic lives
  here (preprocessing, dataset loaders, model training/inference, analytics,
  visualization, config, logging, utils). No subpackage imports from `api/`
  or `dashboard/` — dependency direction is one-way: `api`/`dashboard` →
  `brandparadigm`.
- **`api/`** — FastAPI service. Thin HTTP layer: request/response schemas,
  dependency-injected model loading, routers that call into
  `brandparadigm.sentiment` / `topic_classifier` / `analytics`.
- **`dashboard/`** — Streamlit app. Pages contain only layout/UI calls;
  data loading and chart-building logic live in `dashboard/components/` and
  `brandparadigm/visualization/`.
- **`scripts/`** — CLI entrypoints (`download_datasets.py`,
  `run_training_sentiment.py`, ...) that wrap package functions for the
  command line / `Makefile` targets. No business logic here either.
- **`notebooks/`** — experimentation only. Once validated, code is moved
  into `brandparadigm/`; notebooks call the package, they don't reimplement it.
- **`tests/`** — mirrors `brandparadigm/`, plus `api/` and `dashboard/`
  smoke tests.
- **`configs/`** — YAML configuration per component (dataset sampling,
  training hyperparameters, topic label mapping).

## Compute constraints and reproducibility

Training was developed and smoke-tested on CPU-only hardware. Every
training script exposes a `smoke_test` profile (tiny sample, 1 epoch) used
for correctness verification in CI/dev, and a `full` profile intended to run
on a GPU machine before Demo Day. This keeps the repository reproducible
without requiring GPU access to validate the code.
