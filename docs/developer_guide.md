# Developer Guide

## Package structure

All business logic lives in the installable `brandparadigm/` package;
`api/`, `dashboard/`, `scripts/`, and `notebooks/` are thin callers of it
(see `docs/architecture.md` for the full dependency-direction rationale).
Each subpackage has its own `README.md` — start there for module-level
detail. Currently implemented (Phases 1–3):

- `brandparadigm/config/` — `get_settings()` (env-driven), canonical paths.
- `brandparadigm/logging/` — `get_logger(name)`, one consistent format everywhere.
- `brandparadigm/utils/` — YAML/JSON IO, `set_seed()` for reproducibility.
- `brandparadigm/datasets/` — the three dataset loaders (Amazon Review Polarity, TweetEval, Reddit).
- `brandparadigm/preprocessing/` — text cleaning and label mapping.
- `brandparadigm/sentiment/` — Model 1: binary sentiment fine-tuning, evaluation, and inference (`train.py`, `evaluate.py`, `model.py`, `dataset.py`, `predict.py`).

`brandparadigm/topic_model/`, `topic_classifier/`, `analytics/`,
`visualization/` exist as empty package stubs, populated as their phases
land (see `docs/project_management.md` for phase status).

## Coding conventions

- **PEP8 + type hints + docstrings** on every public function; `black`
  (line length 100) and `ruff` enforce formatting/lint — run `make format`
  before committing, `make lint` to check without modifying.
- **No business logic in notebooks or scripts.** Notebooks
  (`notebooks/`) are experimentation only and call into the package.
  Scripts (`scripts/`) are CLI entrypoints that parse args, call package
  functions, and write output — the logic itself lives in `brandparadigm/`.
- **Config over hardcoding.** Anything that varies by environment or
  dataset (paths, column names, sample sizes, label schemes) belongs in
  `configs/*.yaml` or `brandparadigm.config.settings`, not hardcoded in
  function bodies.
- **Logging, not print.** Use `get_logger(__name__)`; never `print()` in
  package code (scripts' final CLI output is the one exception).
- **Fail loudly and specifically.** Missing local dataset files raise
  `NoLocalDataError` with the exact path and a pointer to
  `docs/dataset_guide.md`, rather than silently returning empty results.

## The sentiment label architecture

Model 1 is **binary** sentiment (0=Negative, 1=Positive) — see
`docs/model_cards/roberta_sentiment.md` for why. This has one important
consequence for anyone touching the sentiment pipeline: **raw dataset
loaders never filter or reinterpret labels — they decode the source data
faithfully.** Label-scheme decisions (dropping TweetEval's Neutral rows,
mapping to 0/1) live in exactly one shared function so the rule is
enforced once, not duplicated per caller. Concretely:

- `brandparadigm.preprocessing.label_mapping.LABEL2ID` /
  `ID2LABEL` is the single source of truth for the production 0/1 scheme.
  `configs/sentiment_config.yaml` mirrors it for config-driven tooling —
  if you ever change one, change both.
- `brandparadigm.datasets.tweeteval` decodes TweetEval's native 3-class
  encoding (`TWEETEVAL_RAW_CLASSES`, including "Neutral") without
  filtering — that loader's job is fidelity to the source, not the
  production label space.
- `brandparadigm.preprocessing.label_mapping.prepare_binary_sentiment_labels`
  is where Neutral rows actually get dropped and the rest mapped to
  `LABEL2ID`. Both `scripts/run_preprocessing.py::preprocess_tweeteval` and
  `brandparadigm.sentiment.evaluate.load_tweeteval_eval_set` call this same
  function — reuse it rather than reimplementing the filter.

If you add a fourth dataset or change the label scheme, keep this
separation: loaders decode, this one shared function decides what the
model sees.

## Testing HF-dependent code without network access

This project's sentiment (and later, topic) models depend on Hugging Face
Hub for pretrained checkpoints. Per the "never stop because Hugging
Face/GPU/internet is unavailable" development rule, the pattern used in
`tests/sentiment/` (see `conftest.py`) is: build a **fully offline**
substitute — a tiny `RobertaConfig`-based model (random init, no
`from_pretrained` call) and a from-scratch `tokenizers.Tokenizer` with a
handful of vocab entries — and inject it into the production function via
a `model_loader`/similar callable parameter. This lets integration tests
exercise the real `Trainer`/tokenization machinery end-to-end, while unit
tests for the actual `from_pretrained` call site mock that one boundary
and assert it's invoked with the right arguments. Reuse this pattern for
BERTopic/DistilBERT in later phases rather than inventing a new one.

## Running things locally

```bash
make install_dev          # deps + editable install
make test                 # pytest, all suites
make lint / make format   # ruff + black
make download_datasets    # fetch/validate all 3 datasets (needs raw_data/ files — see docs/dataset_guide.md)
make preprocess           # clean text, apply label mapping
make train_sentiment      # fine-tune Model 1 (smoke_test profile) — needs Hugging Face Hub access
make evaluate_sentiment   # evaluate a trained model against TweetEval (Neutral rows auto-dropped)
```

## Testing conventions

`tests/` mirrors the package: `tests/datasets/`, `tests/preprocessing/`,
`tests/sentiment/`, etc. Dataset loader tests use small fixtures under
`tests/fixtures/` (including a synthetic `.zst` archive for the Reddit
loader) rather than live network calls or real user data, so the suite is
fast and deterministic. `tests/scripts/` tests the CLI scripts' underlying
functions directly (e.g. `preprocess_amazon`, `preprocess_tweeteval`)
rather than shelling out to the script, so failures point at the exact
function. `tests/sentiment/conftest.py` provides fully offline
model/tokenizer fixtures (see "Testing HF-dependent code without network
access" above) reused across the sentiment test modules.
