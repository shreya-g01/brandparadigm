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

**Epics 3–7 (Phases 3–7) are paused**, at the maintainer's request, on a
hard environment blocker discovered while starting Phase 3: this session's
egress policy blocks `huggingface.co` entirely, so no pretrained model
checkpoint can be downloaded — `AutoTokenizer.from_pretrained("roberta-base")`
fails outright, and the same applies to DistilBERT and any
sentence-transformers embedding model BERTopic would use. This isn't a
compute/scale limitation (that was already scoped via smoke-test profiles)
— it's a total access issue: none of the three ML models can be
instantiated at all in this session, pretrained or not.

**To resume**: allowlist `huggingface.co` (and its CDN,
`cdn-lfs.huggingface.co` / `cdn-lfs-us-1.huggingface.co`) in this session's
egress policy, then continue from Epic 3 in this same breakdown.

**Sentiment architecture finalized before resuming Phase 3**: Model 1 is
now **binary** (0=Negative, 1=Positive), not 3-class, because the only
training dataset (Amazon Review Polarity) has no Neutral class.
`brandparadigm.preprocessing.label_mapping`, `configs/sentiment_config.yaml`
(new — `task_type: binary`, `num_labels: 2`), `scripts/run_preprocessing.py`
(TweetEval's Neutral rows now dropped before binary label mapping), and all
affected tests/docs were updated accordingly — see
`docs/model_cards/roberta_sentiment.md` for the full rationale. No dataset
loader logic changed as a result — `brandparadigm/datasets` still decodes
every raw label faithfully (including TweetEval's Neutral); only
preprocessing's label-mapping step is binary-aware.

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
