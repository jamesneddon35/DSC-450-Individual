# IMDB 50K Sentiment Analysis (NLP)

Two-stage NLP pipeline classifying sentiment on the IMDB 50K Movie Reviews
dataset, built for DSC360 (Text Analytics).

## Notebooks

1. **[01 — Feature engineering](01_feature_engineering_imdb.ipynb)** — text
   cleaning (HTML stripping, contraction expansion, lemmatization, stop-word
   removal) and feature construction: TF-IDF vectorization and Word2Vec
   embeddings, with per-class TF-IDF sanity checks.
2. **[02 — Sentiment models](02_sentiment_models_imdb.ipynb)** — Logistic
   Regression and Linear SVC classifiers on the engineered features, with
   accuracy/precision/recall comparison and a discussion of where
   transformer models (BERT) would improve on bag-of-words approaches.

## Side project: CRF named-entity tagger

**[crf_ner_tagger.ipynb](crf_ner_tagger.ipynb)** — a conditional random field
(sklearn-crfsuite) NER tagger using token, casing, and context features, built
on the GMB corpus, evaluated with per-entity classification reports.

## Data

The [IMDB 50K Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
is not committed; download it from the source (also mirrored on Kaggle) and
update the read path at the top of notebook 01.
