# BrandParadigm

**AI-powered competitor intelligence through transformer-based Natural
Language Processing.**

BrandParadigm turns unstructured customer reviews and social discussion
into structured business intelligence: sentiment, discovered discussion
topics, and business-category breakdowns, aggregated into brand-level
insights and served through an API and dashboard.

## Business problem

Businesses receive thousands of customer reviews but struggle to answer
questions like: What do customers think about our products? What are our
biggest pain points? How do we compare with competitors? Which topics
dominate customer discussion? Keyword search and generic sentiment models
don't capture domain-specific nuance — BrandParadigm fine-tunes and
combines three purpose-built models to answer these questions from
historical review/social data (no live APIs — see `docs/dataset_guide.md`).

## Architecture

Three ML components, chained into one pipeline: a fine-tuned RoBERTa
**binary** sentiment classifier (Negative/Positive — the only training
dataset, Amazon Review Polarity, has no Neutral class; see
[docs/model_cards/roberta_sentiment.md](docs/model_cards/roberta_sentiment.md)),
BERTopic for unsupervised topic discovery, and a topic classifier (best of
Logistic Regression / DistilBERT) that maps text to one of seven business
categories. Full pipeline diagram, dataset choices, and software layout:
**[docs/architecture.md](docs/architecture.md)**.

```
brandparadigm/
├── api/            FastAPI inference service
├── dashboard/      Streamlit dashboard
├── brandparadigm/  installable package — all business logic
├── notebooks/      experimentation only (validated code moves to the package)
├── scripts/        CLI entrypoints (download, preprocess, train, evaluate)
├── tests/          mirrors the package + api/ + dashboard/
├── configs/        YAML config per component
├── docs/           architecture, guides, model cards
└── models/         trained model artifacts (gitignored)
```

## Quickstart

```bash
git clone <repo-url> && cd brandparadigm
python3 -m venv .venv && source .venv/bin/activate
make install_dev
cp .env.sample .env
make test
```

Full setup details: **[docs/installation.md](docs/installation.md)**.

Once trained models exist (`docs/model_cards/`):

```bash
make run_api         # FastAPI on :8000, OpenAPI docs at /docs
make run_dashboard    # Streamlit on :8501
```

## Documentation

- [Architecture](docs/architecture.md)
- [Installation Guide](docs/installation.md)
- [Dataset Guide](docs/dataset_guide.md)
- [Developer Guide](docs/developer_guide.md)
- [API Documentation](docs/api_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Demo Guide](docs/demo_guide.md)
- [Model Cards](docs/model_cards/)
- [Project Management (Epic/Feature/Task breakdown)](docs/project_management.md)

## Project phases

Built sequentially; each phase is completed and verified before the next
begins.

- [x] **Phase 1** — Repository scaffold, tooling, documentation
- [x] **Phase 2** — Dataset loaders, preprocessing, dataset guide
- [ ] **Phase 3** — RoBERTa sentiment fine-tuning, evaluation on TweetEval
- [ ] **Phase 4** — BERTopic discovery, topic classifier (LogReg vs DistilBERT)
- [ ] **Phase 5** — FastAPI inference service
- [ ] **Phase 6** — Streamlit dashboard
- [ ] **Phase 7** — Docker packaging, full test suite, deployment docs

## Testing

```bash
make lint   # ruff + black --check
make test   # pytest across brandparadigm/, api/, dashboard/
```

## License

Built for Le Wagon Demo Day.
