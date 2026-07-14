# brandparadigm.datasets

Loaders for the three project datasets, each normalized to a common,
predictable schema — see `docs/dataset_guide.md` for exact columns,
provisioning instructions, and the local-file requirement forced by this
environment's egress policy (huggingface.co/kaggle.com/etc. are blocked;
all three datasets are read from local files under `raw_data/`).

- `amazon_reviews.py` — Dataset 1 (training). Amazon Review Polarity, local headerless `train.csv`/`test.csv`, `[text, polarity, source]`. Reads via `local_source.read_csv_sampled` — sampling happens during the streaming read itself, so a `train_sample_size: 200` smoke-test profile never loads the full multi-million-row file into memory.
- `tweeteval.py` — Dataset 2 (evaluation only). TweetEval `sentiment` task only, local `sentiment_{train,validation,test}.csv`, `[text, label, source]`.
- `reddit_dataset.py` — Dataset 3 (inference). Pushshift Reddit Submissions archive, read directly from its local `.zst` file (no manual conversion), `[title, selftext, subreddit, score, created_utc, text, source]`.
- `local_source.py` — shared "is the file there yet / which column is it" helpers used by all three loaders, plus `read_csv_sampled` — a chunked CSV reader with two configurable sampling strategies (`"reservoir"`: true uniform random sample, one full streaming pass, memory bounded by `sample_size + chunk_size`; `"head"`: fast first-N-rows read, not statistically representative). See `docs/model_cards/roberta_sentiment.md` ("Data loading") for the full trade-off.
- `registry.py` — `load_dataset(name, config)` dispatches to the three loaders above by name (`"amazon"`, `"tweeteval"`, `"reddit"`); used by `scripts/download_datasets.py`.
