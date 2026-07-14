# Model Card — Sentiment Classifier (Model 1)

**Status**: the full training/evaluation pipeline is implemented and
integration-tested (`tests/sentiment/`, including an end-to-end `Trainer`
run against a fully offline, from-scratch tiny model/tokenizer — no
network access required to verify the pipeline's correctness). **An actual
fine-tuning run has not been executed** in this development session
because `huggingface.co` — required to download the base checkpoint below
— is blocked by this session's egress policy (see
`docs/project_management.md`, "Current status"). Per the maintainer's
instruction, the code is written as production-ready and assumes the
checkpoint downloads normally wherever it's actually run
(`make train_sentiment` / `scripts/run_training_sentiment.py`); this card's
Metrics section will be filled in with real numbers from that run.

## Task

**Binary sentiment classification**: given a piece of review/post text,
predict whether it expresses **Negative (0)** or **Positive (1)**
sentiment. There is no Neutral class in the production model.

## Intended use

**In scope**:
- Classifying the sentiment of product reviews and review-like text
  (Amazon reviews, the training domain) as clearly Negative or Positive.
- Classifying the sentiment of short social-media posts (Reddit
  titles/selftext) that express clear positive or negative opinion about a
  product or brand — the primary use case for BrandParadigm's brand
  intelligence pipeline (Phase 4+).
- A component in an aggregate pipeline (sentiment + topic + business
  category) — not intended to be the sole signal for a business decision.

**Out of scope / not intended**:
- Detecting genuinely neutral, mixed, or ambivalent opinions — the model
  cannot express "neutral," it will force every input toward Negative or
  Positive (see Limitations).
- Any use where a Neutral/no-opinion output is required (e.g. filtering
  out off-topic content) — that's not this model's job; topic relevance is
  Model 3's job (Phase 4).
- High-stakes or adversarial settings (content moderation, automated
  customer response, anything affecting a real person's outcome) without
  human review — this is a business-intelligence aid, not a moderation or
  decision system.
- Languages other than English — both training (Amazon Review Polarity)
  and evaluation (TweetEval) data are English-only; behavior on other
  languages is undefined and untested.

## Why binary, not 3-class

The master project spec originally called for a 3-class
Positive/Neutral/Negative scheme. That was changed to binary for one
concrete reason: **the only dataset this model trains on — Amazon Review
Polarity — is itself binary**. It has exactly two classes (`polarity` 1 or
2); there is no Neutral label anywhere in the source data.

Two alternatives were considered and rejected:

1. **Fabricate a Neutral class** (e.g. by treating some score/confidence
   band as "Neutral") — rejected because it would mean training the model
   on invented labels rather than real annotations, undermining the
   model's validity.
2. **Switch to a star-rated dataset** (1–5★, mappable to
   Negative/Neutral/Positive) — rejected because the maintainer explicitly
   chose to keep Amazon Review Polarity as the dataset (see
   `docs/dataset_guide.md`) rather than change datasets to accommodate a
   label scheme the data doesn't naturally support.

Binary sentiment is the architecture that's actually grounded in the data
available. See `configs/sentiment_config.yaml` (`task_type: binary`,
`num_labels: 2`) and `brandparadigm/preprocessing/label_mapping.py`
(`SENTIMENT_CLASSES = ["Negative", "Positive"]`) for where this is
enforced in code.

## Data

| Split | Source | Role |
|---|---|---|
| Train | Amazon Review Polarity `train.csv` | Fine-tuning |
| Validation/Test | Amazon Review Polarity `test.csv` | In-training validation and held-out testing on the same distribution |
| Evaluation | TweetEval `sentiment` task (`sentiment_test.csv`), **Neutral rows dropped** | Out-of-domain generalization check — tweets, not reviews; never trained on |
| Inference | Historical Reddit Submissions archive (`RS_2019-04.zst`) | Production inference target; unlabeled |

TweetEval's `sentiment` task is natively 3-class in its source data. Its
Neutral rows are dropped automatically by
`brandparadigm.preprocessing.label_mapping.prepare_binary_sentiment_labels`
*before* evaluation metrics are computed (used by both
`brandparadigm.sentiment.evaluate.load_tweeteval_eval_set` and
`scripts/run_preprocessing.py`), so evaluation always compares Negative vs
Positive — the same label space the model predicts. This filtering is
intentional and is the only place TweetEval's third class is discarded;
the raw dataset loader (`brandparadigm.datasets.tweeteval`) decodes it
faithfully and does not filter anything.

## Label mapping

```
0 = Negative
1 = Positive
```

Defined once in `brandparadigm.preprocessing.label_mapping.LABEL2ID` /
`ID2LABEL`, and mirrored in `configs/sentiment_config.yaml` for
config-driven tooling. All three dataset preprocessing paths
(`scripts/run_preprocessing.py`) produce a `sentiment_label` column in this
exact 0/1 encoding.

## Architecture

- **Base model**: `cardiffnlp/twitter-roberta-base-sentiment` — a RoBERTa
  already fine-tuned for 3-class sentiment on Twitter data, a strong
  starting point for social-text sentiment. Its pretrained 3-class head is
  discarded (`ignore_mismatched_sizes=True`) and replaced with a freshly
  initialized binary head (`num_labels=2`), since this project's
  production label space is binary. See
  `brandparadigm/sentiment/model.py::load_model_and_tokenizer`.
- **Fine-tuning**: HF `Trainer`, entirely config-driven via
  `configs/sentiment_config.yaml` — no hardcoded hyperparameters. Two
  named profiles: `smoke_test` (tiny sample, 1 epoch, fast correctness
  check) and `full` (the real training run, intended for a GPU-enabled
  environment). See `brandparadigm/sentiment/train.py`.
- **Early stopping**: `transformers.EarlyStoppingCallback`, patience from
  the active profile's `early_stopping_patience`, driven by
  `metric_for_best_model: f1` (`load_best_model_at_end: true`).
- **Checkpointing**: `save_total_limit` (config, default 2) caps how many
  epoch checkpoints accumulate under `models/sentiment/checkpoints/` — each
  is a full model+optimizer+scheduler snapshot, so this bounds disk usage
  on multi-epoch runs.
- **Persistence**: best model + tokenizer saved to `models/sentiment/`
  (gitignored — regenerate via `scripts/run_training_sentiment.py`, not
  committed as a binary), alongside `metrics.json`,
  `confusion_matrix.json`, `classification_report.json`,
  `training_history.json` (the `Trainer`'s full `log_history`), and
  `metadata.json` — see "Reproducibility" below.
- **Inference**: `brandparadigm.sentiment.predict.SentimentPredictor` —
  decoupled from FastAPI/Streamlit, loads a saved model directory and
  exposes `predict(text) -> {label, scores}`.
- **Large-file loading**: `configs/data_config.yaml::amazon_reviews`
  (`chunk_size`, `sampling_strategy`) controls how the Amazon Review
  Polarity CSVs are read — see "Data loading" below.

## Data loading

Amazon Review Polarity's `train.csv` is several million rows; naively
reading the whole file into memory before sampling down to a training
profile's `train_sample_size` (e.g. 200 rows for `smoke_test`) would waste
most of that read. `brandparadigm.datasets.local_source.read_csv_sampled`
instead streams the file in `chunk_size`-row chunks (default 50,000) and
applies one of two sampling strategies, chosen via
`amazon_reviews.sampling_strategy`:

- **`reservoir`** (default): Algorithm R reservoir sampling — a true
  uniform random sample across the entire file. Correct, but requires one
  full sequential pass (every row is visited once), so runtime scales with
  file size regardless of `sample_size`. Peak memory stays bounded by
  roughly `sample_size + chunk_size` rows.
- **`head`**: reads only the first `sample_size` rows and stops — fast,
  but not statistically representative of the whole file (biased toward
  file order). Useful for very fast dev-loop iteration; not recommended
  for the `full` profile or real evaluation.

When no `sample_size` is requested (the `full` profile trains on
everything), all rows are needed regardless, so this only reduces parsing
overhead, not final memory usage — that case is unavoidable when nothing
is being sampled out.

## Reproducibility

Every training run writes `models/sentiment/metadata.json`, making a saved
model directory self-describing without cross-referencing the codebase at
a particular point in time:

- **Model info**: base checkpoint, task type, `num_labels`,
  `max_seq_length`, label mapping.
- **Training config**: the exact profile name and every hyperparameter
  used (epochs, batch sizes, learning rate, early stopping patience,
  `save_total_limit`, seed, ...).
- **Dataset details**: which datasets/splits were used and how many rows
  each actually contributed after sampling and cleaning.
- **Git commit hash**: `brandparadigm.utils.metadata.get_git_commit_hash()`
  — `git rev-parse HEAD` at training time (`None` if unavailable, e.g. not
  a git checkout — this never blocks a training run).
- **Library versions**: installed versions of `torch`, `transformers`,
  `datasets`, `scikit-learn`, `accelerate` at training time.

## Testing

`tests/sentiment/` covers every module. Because this session cannot reach
`huggingface.co`, the `Trainer` orchestration itself is verified with a
**fully offline** substitute: a from-scratch `RobertaConfig`-based tiny
model and a from-scratch `WordLevel` tokenizer (`tests/sentiment/conftest.py`,
`tiny_model`/`tiny_tokenizer` fixtures), injected into `train()` via its
`model_loader` parameter — zero network calls, but real `Trainer`,
`TrainingArguments`, and `EarlyStoppingCallback` wiring, so the test
actually exercises the training loop rather than only mocking it.
`brandparadigm.sentiment.model.load_model_and_tokenizer` itself (the real
HF Hub call) is covered separately by mocking `from_pretrained` and
asserting it's called with the right checkpoint name and binary-head
arguments.

## Evaluation metrics

**Not yet available** — an actual fine-tuning run has not been executed
(see Status above). Once run, this section reports the following, sourced
from `models/sentiment/metrics.json` (Amazon test split, computed by
`brandparadigm.sentiment.evaluate.compute_metrics` during training) and
`models/sentiment/tweeteval_evaluation_report.json` (TweetEval, via
`scripts/run_evaluation.py`):

| Metric | Definition | Where computed |
|---|---|---|
| Accuracy | Fraction of correct predictions | Both sets |
| Precision | Of predicted-Positive, fraction actually Positive | Both sets |
| Recall | Of actually-Positive, fraction predicted Positive | Both sets |
| F1 | Harmonic mean of precision/recall (the model-selection metric — `metric_for_best_model: f1`) | Both sets |
| Confusion matrix | 2×2 Negative/Positive matrix | Both sets, via `build_evaluation_report` |
| Per-class report | sklearn `classification_report` (precision/recall/F1 per class + macro/weighted averages) | Both sets |

**How to read the two evaluation sets**: the Amazon test-split metrics
measure in-domain performance (held-out reviews, same distribution as
training). The TweetEval metrics measure out-of-domain generalization
(tweets, a different register and length than reviews) — expect these to
be meaningfully lower; that gap is itself informative, not a bug.

## Hardware requirements

Not benchmarked yet (no run has executed), but sizeable from the
architecture and profiles:

- **Training (`full` profile)**: a GPU is strongly recommended — full
  fine-tuning of a 125M-parameter RoBERTa on the complete Amazon Review
  Polarity training set (millions of rows, 3 epochs) is impractical on CPU
  in reasonable time. Memory: model + optimizer state + activations for
  `per_device_train_batch_size: 16` at `max_seq_length: 128` fits
  comfortably on a single consumer/cloud GPU with ≥8GB VRAM; more VRAM
  allows a larger batch size (also config-driven).
- **Training (`smoke_test` profile)**: runs on CPU in seconds to low
  minutes (200 train rows, 1 epoch) — this is its purpose, a fast
  correctness check, not a representative training run.
- **Inference** (`SentimentPredictor`): CPU-viable for interactive
  single-request or small-batch use (the API's expected load pattern);
  a GPU helps for high-throughput batch inference (e.g. scoring the full
  Reddit inference set) but isn't required.
- **Disk**: `save_total_limit: 2` bounds checkpoint accumulation to 2 full
  model snapshots at a time under `models/sentiment/checkpoints/`, plus
  the final best-model copy in `models/sentiment/` itself.

## Bias considerations

No fairness/bias audit has been run (no training run has executed yet).
Known structural biases inherited from the data and base checkpoint,
worth accounting for when interpreting predictions:

- **Product-category skew**: Amazon Review Polarity aggregates reviews
  across all Amazon product categories at the time of its construction —
  whatever categories dominated that corpus (electronics, books, etc.)
  dominate the model's notion of "typical" review language. Its
  performance on a category poorly represented in that corpus is
  untested.
- **Selection bias in who leaves reviews**: reviewers are not a random
  sample of purchasers/users — people with strong (often negative)
  opinions are disproportionately likely to write a review at all. The
  model inherits whatever sentiment-expression patterns are typical of
  reviewers, not of the general population of product users.
- **Register mismatch between training and inference domains**: trained
  on Amazon reviews (longer, product-focused), but applied at inference
  time to Reddit posts (shorter, more casual, more likely to include
  sarcasm/slang/memes) — sentiment models are known to underperform on
  sarcasm and irony, and this gap is not measured by either evaluation
  set (Amazon test = same domain as training; TweetEval = tweets, not
  Reddit).
- **English-only, US-centric**: both training and evaluation data are
  English-language, predominantly US-market text. No claims are made
  about performance on other languages, dialects, or regional product
  vocabularies.
- **Inherited base-model bias**: `cardiffnlp/twitter-roberta-base-sentiment`
  was itself pretrained/fine-tuned on Twitter data with its own
  demographic and topical skew (documented by its authors); fine-tuning
  here adapts the classification head and encoder weights but does not
  audit or correct for whatever biases that base model already encodes.
- **No demographic attribute is modeled or available** — there's no
  mechanism here to measure performance disparities across reviewer/poster
  demographics, since the datasets don't carry that information at all.

## Known limitations

- Binary sentiment cannot express ambivalent/mixed reviews — a genuine
  product limitation of training on Amazon Review Polarity, not a modeling
  choice that can be fixed without a differently-labeled dataset. A review
  reading "great screen, terrible battery" will be forced to one label.
- TweetEval evaluation only measures the subset of tweets with clear
  Negative/Positive sentiment; the ~third of TweetEval examples labeled
  Neutral are excluded from every reported evaluation metric by design.
- No calibration analysis: the softmax scores `SentimentPredictor` returns
  are not verified to be well-calibrated probabilities (a 0.9 score isn't
  necessarily "90% likely correct") — treat them as relative confidence
  signals, not calibrated probabilities, until an actual run is analyzed.
- Domain gap between training (Amazon reviews) and the primary production
  input (Reddit posts) is real and unmeasured until an evaluation against
  labeled Reddit data exists — TweetEval is the closest available proxy
  (social text) but is not Reddit.
- Short or ambiguous inputs (a lone emoji, a single word, sarcasm without
  other context) are a known weak point for sentence-level sentiment
  classifiers generally; this model has no special handling for them.
