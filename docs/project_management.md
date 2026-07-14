# Project Management: Epic → Feature → Task Breakdown

## Current status (2026-07-14)

**Epics 1–2 (Phases 1–2) are complete, tested, and pushed** to
`claude/brandparadigm-spec-wubwi9`: full repo scaffold, tooling, and the
dataset loading/preprocessing pipeline. All three loaders now target the
maintainer's actual local data sources: Amazon Review Polarity
(`train.csv`/`test.csv`), TweetEval's `sentiment` task only
(`sentiment_{train,validation,test}.csv`), and a Pushshift Reddit
Submissions archive (`RS_2019-04.zst`) read directly via `zstandard` — no
manual conversion needed. All verified against fixtures (including a
synthetic `.zst`); pending the maintainer's real data drop into
`raw_data/` — see `docs/dataset_guide.md`.

**Standing rule as of Phase 3** (applies to every remaining phase): this
session never stops or pauses because Hugging Face Hub, a GPU, general
internet access, or a dataset is unavailable. Code is written
production-ready, assuming those resources are available wherever it
actually runs; where this session itself can't reach them (e.g.
`huggingface.co` is blocked by egress policy — confirmed via
`/__agentproxy/status`, a policy denial, not transient), verification uses
fully offline substitutes (see `docs/developer_guide.md`, "Testing
HF-dependent code without network access") rather than stopping. The
earlier "Epics 3-7 paused" state below no longer applies as of Phase 3.

**Sentiment architecture finalized before implementing Phase 3**: Model 1
is **binary** (0=Negative, 1=Positive), not 3-class, because the only
training dataset (Amazon Review Polarity) has no Neutral class.
`brandparadigm.preprocessing.label_mapping.prepare_binary_sentiment_labels`
is the single shared function that drops TweetEval's Neutral rows and maps
the rest to 0/1 — used by both `scripts/run_preprocessing.py` and
`brandparadigm.sentiment.evaluate`. See
`docs/model_cards/roberta_sentiment.md` for the full rationale.

**Epic 3 (Phase 3) is implemented, tested, and pushed**: the full binary
sentiment pipeline (`brandparadigm/sentiment/` — `model.py`, `dataset.py`,
`train.py`, `evaluate.py`, `predict.py`), config-driven hyperparameters
(`configs/sentiment_config.yaml`), early stopping, and artifact saving
(`models/sentiment/metrics.json`, `confusion_matrix.json`,
`classification_report.json`, `training_history.json`). Base checkpoint:
`cardiffnlp/twitter-roberta-base-sentiment`, fine-tuned with a fresh binary
head. **No actual fine-tuning run has executed** in this session (still
blocked on `huggingface.co`), but the `Trainer` orchestration itself is
integration-tested end-to-end with a fully offline tiny model/tokenizer —
see `tests/sentiment/` and `docs/model_cards/roberta_sentiment.md`,
"Testing".

**Epics 4–7 (Phases 4–7) are next**, not paused — to be implemented under
the same standing rule above.

This document is the project-management source of truth in lieu of a live
GitHub Projects board (the tools available to this session can create
issues but not a Projects v2 board with custom columns/fields — see note at
the bottom). Copy the structure below into a Project board manually;
columns are `Backlog / Ready / In Progress / Review / Testing / Done`.

Each Epic below maps 1:1 to a project phase. Every Task lists acceptance
criteria, dependencies, priority (P0 highest), and an hour estimate.

## Epic 1 — Repository Scaffold (Phase 1)

| Task | Acceptance Criteria | Depends on | Priority | Est. (h) |
|---|---|---|---|---|
| Create monorepo folder structure | All directories in `docs/architecture.md` exist with `__init__.py`/`.gitkeep` | — | P0 | 1 |
| Packaging (`setup.py`, `pyproject.toml`, requirements) | `pip install -e .` succeeds; black/ruff/pytest configured | Folder structure | P0 | 2 |
| Config/logging/utils core modules | `get_settings()`, `get_logger()`, `set_seed()` importable and unit-tested | Packaging | P0 | 2 |
| Makefile + Docker base stage | `make install`, `make lint`, `make test` work; `docker build` succeeds for base stage | Packaging | P1 | 2 |
| Root docs (README, architecture, installation, this file) | Docs describe the real repo state, no TODOs | All above | P1 | 3 |

## Epic 2 — Datasets (Phase 2)

| Task | Acceptance Criteria | Depends on | Priority | Est. (h) |
|---|---|---|---|---|
| Amazon Reviews loader | Reads local Amazon Review Polarity `train.csv`/`test.csv`, returns normalized `{text, polarity, source}` | Epic 1 | P0 | 3 |
| TweetEval loader | Reads local `sentiment_{train,validation,test}.csv`, normalized schema | Epic 1 | P0 | 1 |
| Reddit historical loader | Streams local `RS_YYYY-MM.zst` directly via `zstandard`, filters to product/tech subreddits | Epic 1 | P0 | 3 |
| Preprocessing (clean text, label mapping) | Unit-tested against fixtures; binary label mapping (0=Negative, 1=Positive) matches spec; TweetEval Neutral rows dropped before evaluation | Loaders | P0 | 3 |
| Dataset guide + EDA | `docs/dataset_guide.md` documents confirmed configs, schema, label distribution | Loaders, preprocessing | P1 | 2 |

## Epic 3 — Sentiment Model (Phase 3)

| Task | Acceptance Criteria | Depends on | Priority | Est. (h) |
|---|---|---|---|---|
| RoBERTa model/tokenizer wrapper | Binary head (num_labels=2), label2id fixed and tested | Epic 2 | P0 | 2 |
| Training pipeline (`Trainer`, config-driven) | `smoke_test` profile runs end-to-end on CPU, checkpoint persists | Model wrapper | P0 | 4 |
| Evaluation on TweetEval | Never trains on TweetEval; reports accuracy/F1/confusion matrix | Training pipeline | P0 | 2 |
| Inference wrapper for API | `predict(text) -> {label, scores}` | Training pipeline | P0 | 1 |
| Model card | Data, procedure, smoke-test metrics, GPU full-run instructions | All above | P1 | 1 |

## Epic 4 — Topic Discovery & Classifier (Phase 4)

| Task | Acceptance Criteria | Depends on | Priority | Est. (h) |
|---|---|---|---|---|
| BERTopic pipeline | Fits on Reddit sample with CPU-sized UMAP/HDBSCAN params | Epic 2 | P0 | 4 |
| Manual topic validation mapping | `configs/topic_label_mapping.yaml` maps discovered topics → 7 categories | BERTopic pipeline | P0 | 2 |
| Logistic Regression classifier | TF-IDF + LogReg trained/evaluated on mapped labels | Topic mapping | P0 | 2 |
| DistilBERT classifier | Fine-tuned smoke-test run on same labels | Topic mapping | P0 | 3 |
| Model comparison + selection | `compare.py` picks best by F1, writes model card | Both classifiers | P0 | 1 |

## Epic 5 — API (Phase 5)

| Task | Acceptance Criteria | Depends on | Priority | Est. (h) |
|---|---|---|---|---|
| FastAPI app skeleton + DI | `/health` returns 200; models loaded via dependency injection | Epics 3–4 | P0 | 2 |
| `/predict_sentiment`, `/predict_topic` | Pydantic-validated request/response, unit-tested | App skeleton | P0 | 2 |
| `/analyze_brand`, `/model_metrics` | Aggregates via `brandparadigm.analytics`; metrics read from model cards | App skeleton | P0 | 3 |
| API tests + OpenAPI docs | `pytest tests/api` green; `/docs` renders all 5 endpoints | All above | P1 | 2 |

## Epic 6 — Dashboard (Phase 6)

| Task | Acceptance Criteria | Depends on | Priority | Est. (h) |
|---|---|---|---|---|
| Home + navigation | Streamlit multipage app boots | Epic 5 | P0 | 1 |
| Sentiment Analysis, Topic Explorer pages | Charts render from API/package data | Home | P0 | 3 |
| Brand Comparison, Model Performance, About pages | Charts render, no business logic in page files | Home | P0 | 3 |
| Dashboard smoke tests (`AppTest`) | Every page loads without exception in `pytest` | All pages | P1 | 1 |

## Epic 7 — Docker, Testing, Deployment (Phase 7)

| Task | Acceptance Criteria | Depends on | Priority | Est. (h) |
|---|---|---|---|---|
| Finalize Dockerfile (API + dashboard CMDs) | `docker build` + `docker run` serve `/health` | Epics 5–6 | P0 | 2 |
| Full test suite pass | `pytest` green across all mirrored test dirs | All epics | P0 | 2 |
| Deployment/demo/developer/API docs | All docs in `docs/` reflect final state, no TODOs | All epics | P1 | 3 |
| Final README polish | Roadmap marked complete, quickstart verified | All above | P1 | 1 |

---

**Note on GitHub Projects tooling**: the GitHub MCP tools available to this
session can create issues, labels, milestones, and sub-issue links, but
cannot create or configure a Projects v2 board (columns, custom fields like
Priority/Estimated Hours). Per the maintainer's decision, no GitHub Issues
were created for this breakdown — copy the tables above into issues/a board
manually if you want them tracked in GitHub's UI.
