# brandparadigm.datasets

Loaders for the three project datasets, each normalized to a common,
predictable schema — see `docs/dataset_guide.md` for exact columns,
provisioning instructions, and the local-file requirement forced by this
environment's egress policy (huggingface.co/kaggle.com/etc. are blocked;
all three datasets are read from local files under `raw_data/`).

- `amazon_reviews.py` — Dataset 1 (training). Amazon Review Polarity, local headerless `train.csv`/`test.csv`, `[text, polarity, source]`.
- `tweeteval.py` — Dataset 2 (evaluation only). TweetEval `sentiment` task only, local `sentiment_{train,validation,test}.csv`, `[text, label, source]`.
- `reddit_dataset.py` — Dataset 3 (inference). Pushshift Reddit Submissions archive, read directly from its local `.zst` file (no manual conversion), `[title, selftext, subreddit, score, created_utc, text, source]`.
- `local_source.py` — shared "is the file there yet / which column is it" helpers used by all three loaders.
- `registry.py` — `load_dataset(name, config)` dispatches to the three loaders above by name (`"amazon"`, `"tweeteval"`, `"reddit"`); used by `scripts/download_datasets.py`.
