# brandparadigm.sentiment

Model 1: binary sentiment classification (0=Negative, 1=Positive), fine-tuned
from `cardiffnlp/twitter-roberta-base-sentiment`. See
`docs/model_cards/roberta_sentiment.md` for the full rationale and
`configs/sentiment_config.yaml` for every hyperparameter — nothing in this
package hardcodes a hyperparameter that belongs in config.

- `model.py` — `load_model_and_tokenizer(base_model_name)` loads the
  pretrained checkpoint with a fresh binary head
  (`ignore_mismatched_sizes=True`, since the checkpoint's native head is
  3-class). `load_trained_model_and_tokenizer(model_dir)` loads an
  already-fine-tuned model for evaluation/inference.
- `dataset.py` — `build_tokenized_dataset(df, tokenizer, ...)` turns a
  cleaned, binary-labeled DataFrame into a tokenized `datasets.Dataset`
  ready for `Trainer`.
- `train.py` — `train(sentiment_config, data_config, profile=...)`
  orchestrates the full fine-tuning run: loads Amazon Review Polarity
  (`train.csv` for training, `test.csv` for validation/testing), builds
  `TrainingArguments` entirely from config (including `save_total_limit`),
  wires `EarlyStoppingCallback`, trains, saves the best model/tokenizer
  plus `metrics.json`, `confusion_matrix.json`, `classification_report.json`,
  `training_history.json`, and `metadata.json` (model info, training
  config, dataset details, git commit hash, library versions, label
  mapping — see `build_run_metadata`) under `models/sentiment/`. Accepts
  an injectable `model_loader` so tests can exercise the full
  orchestration without network access.
- `evaluate.py` — `compute_metrics` (HF Trainer-compatible: accuracy/
  precision/recall/F1), `run_batched_inference(model, dataset, batch_size)`
  (used by `scripts/run_evaluation.py` for standalone evaluation outside
  the `Trainer`), and `load_tweeteval_eval_set(...)`, which loads
  TweetEval's `sentiment` split and **automatically drops Neutral rows**
  before returning — evaluation only ever sees the binary label space.
- `predict.py` — `SentimentPredictor`, the inference wrapper `api/`
  (Phase 5) will use. Decoupled from FastAPI/Streamlit on purpose.

## Why `cardiffnlp/twitter-roberta-base-sentiment`, fine-tuned binary

That checkpoint is itself a RoBERTa fine-tuned for 3-class sentiment on
Twitter data — a strong starting point for social-text sentiment. Its
original 3-class head is discarded and replaced with a fresh binary head
because this project's production label space is binary (see
`docs/model_cards/roberta_sentiment.md` for why).
