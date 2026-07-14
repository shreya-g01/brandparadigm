# brandparadigm.preprocessing

Text cleaning and label normalization shared by every dataset before it
reaches a model.

- `clean_text.py` — `clean_text(text)`: strips HTML, NFKC-normalizes
  unicode, removes URLs, collapses whitespace. Used identically for
  training data and inference input so the model sees consistent text.
- `label_mapping.py` — the project's **binary** production sentiment scheme
  (`SENTIMENT_CLASSES = ["Negative", "Positive"]`, `LABEL2ID`/`ID2LABEL`,
  0/1), plus `amazon_polarity_to_sentiment` (Amazon Review Polarity's 1/2
  → the binary scheme directly) and `tweeteval_label_to_sentiment` (decodes
  TweetEval's native 3-class 0/1/2 encoding, including "Neutral" —
  `TWEETEVAL_RAW_CLASSES` — since that's what the source data actually
  contains; Neutral rows are dropped downstream in
  `scripts/run_preprocessing.py`, not here). See
  `docs/dataset_guide.md` and `docs/model_cards/roberta_sentiment.md` for
  why the production scheme is binary rather than 3-class.
- `tokenization.py` — model-agnostic word-count helpers used for EDA/QC
  only. Model-specific tokenizers (RoBERTa, DistilBERT) live next to the
  model that owns them, not here, since vocab/padding choices differ per
  model.
