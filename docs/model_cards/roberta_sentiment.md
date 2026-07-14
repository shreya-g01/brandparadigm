# Model Card — Sentiment Classifier (Model 1)

**Status**: architecture and data pipeline finalized; training not yet run.
Phases 3–7 are paused in the current development session because
`huggingface.co` — required to download the `roberta-base` pretrained
checkpoint — is blocked by this session's egress policy (see
`docs/project_management.md`, "Current status"). This card documents the
design decisions made so far; it will be updated with real metrics once
training runs (either after egress is allowlisted, or on a separate
GPU-enabled environment).

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
Neutral rows are dropped by `scripts/run_preprocessing.py::preprocess_tweeteval`
*before* evaluation metrics are computed, so evaluation always compares
Negative vs Positive — the same label space the model predicts. This
filtering is intentional and is the only place TweetEval's third class is
discarded; the raw dataset loader (`brandparadigm.datasets.tweeteval`)
decodes it faithfully and does not filter anything.

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

## Architecture (planned)

- **Base model**: `roberta-base` (`transformers.AutoModelForSequenceClassification`, `num_labels=2`).
- **Fine-tuning**: HF `Trainer`, config-driven via `configs/sentiment_config.yaml`.
- **Persistence**: `models/sentiment_roberta/` (gitignored — regenerate via the training script, not committed as a binary).

## Metrics

Not yet available — training has not run. Once Phase 3 resumes, this
section will report accuracy/F1/precision/recall on both the Amazon
Review Polarity test split and the Neutral-filtered TweetEval evaluation
set.

## Known limitations

- Binary sentiment cannot express ambivalent/mixed reviews — a genuine
  product limitation of training on Amazon Review Polarity, not a modeling
  choice that can be fixed without a differently-labeled dataset.
- TweetEval evaluation only measures the subset of tweets with clear
  Negative/Positive sentiment; the ~third of TweetEval examples labeled
  Neutral are excluded from every reported evaluation metric by design.
