# brandparadigm.preprocessing

Text cleaning and label normalization shared by every dataset before it
reaches a model.

- `clean_text.py` — `clean_text(text)`: strips HTML, NFKC-normalizes
  unicode, removes URLs, collapses whitespace. Used identically for
  training data and inference input so the model sees consistent text.
- `label_mapping.py` — the project's 3-class scheme (`SENTIMENT_CLASSES`,
  `LABEL2ID`/`ID2LABEL`) plus `star_rating_to_sentiment` (Amazon 1-5★ → 3
  classes) and `tweeteval_label_to_sentiment` (TweetEval's 0/1/2 → the same
  3 classes).
- `tokenization.py` — model-agnostic word-count helpers used for EDA/QC
  only. Model-specific tokenizers (RoBERTa, DistilBERT) live next to the
  model that owns them, not here, since vocab/padding choices differ per
  model.
