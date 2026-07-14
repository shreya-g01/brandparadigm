# brandparadigm.datasets

Loaders for the three project datasets, each normalized to a common,
predictable schema — see `docs/dataset_guide.md` for exact columns,
provisioning instructions, and the live-vs-local-file split forced by this
environment's egress policy.

- `amazon_reviews.py` — Dataset 1 (training). Local file, `[text, rating, category, source]`.
- `tweeteval.py` — Dataset 2 (evaluation only). Live HTTP download from GitHub, `[text, label, source]`.
- `reddit_dataset.py` — Dataset 3 (inference). Local file, `[text, subreddit, source]`.
- `local_source.py` — shared file-discovery/column-detection helpers used by the two local-file loaders.
- `registry.py` — `load_dataset(name, config)` dispatches to the three loaders above by name (`"amazon"`, `"tweeteval"`, `"reddit"`); used by `scripts/download_datasets.py`.
