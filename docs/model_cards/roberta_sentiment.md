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
- **Persistence**: best model + tokenizer saved to `models/sentiment/`
  (gitignored — regenerate via `scripts/run_training_sentiment.py`, not
  committed as a binary), alongside `metrics.json`,
  `confusion_matrix.json`, `classification_report.json`, and
  `training_history.json` (the `Trainer`'s full `log_history`).
- **Inference**: `brandparadigm.sentiment.predict.SentimentPredictor` —
  decoupled from FastAPI/Streamlit, loads a saved model directory and
  exposes `predict(text) -> {label, scores}`.

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

## Metrics

Not yet available — an actual fine-tuning run has not been executed (see
Status above). Once run, this section reports accuracy/F1/precision/recall
on both the Amazon Review Polarity test split and the Neutral-filtered
TweetEval evaluation set, sourced from `models/sentiment/metrics.json` and
`.../tweeteval_evaluation_report.json`.

## Known limitations

- Binary sentiment cannot express ambivalent/mixed reviews — a genuine
  product limitation of training on Amazon Review Polarity, not a modeling
  choice that can be fixed without a differently-labeled dataset.
- TweetEval evaluation only measures the subset of tweets with clear
  Negative/Positive sentiment; the ~third of TweetEval examples labeled
  Neutral are excluded from every reported evaluation metric by design.
