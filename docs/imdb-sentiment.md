# IMDB 50K Sentiment Analysis

**A two-stage NLP pipeline: engineer the features properly, then let simple
models shine.**

[← Back to portfolio](index.md) ·
[Code on GitHub]({{ site.github.repository_url }}/tree/main/imdb-sentiment-nlp)

## Stage 1 — Feature engineering

Raw movie reviews are messy: HTML fragments, contractions, inflected forms,
stop-word noise. The first notebook
([01]({{ site.github.repository_url }}/blob/main/imdb-sentiment-nlp/01_feature_engineering_imdb.ipynb))
builds a normalization pipeline — HTML stripping, contraction expansion,
tokenization, lemmatization, stop-word removal — then constructs two feature
representations:

- **TF-IDF vectors**, with vocabulary limits chosen to cut noise from very
  rare tokens, validated by inspecting mean TF-IDF scores per sentiment class;
- **Word2Vec embeddings** for a dense, semantic alternative.

## Stage 2 — Classification

The second notebook
([02]({{ site.github.repository_url }}/blob/main/imdb-sentiment-nlp/02_sentiment_models_imdb.ipynb))
trains and compares **Logistic Regression** and **Linear SVC** on the
engineered features, evaluating accuracy, precision, and recall — and closes
with a discussion of where contextual transformer models like BERT would beat
bag-of-words features, and at what cost.

## Side project — CRF named-entity recognition

A conditional random field tagger
([notebook]({{ site.github.repository_url }}/blob/main/imdb-sentiment-nlp/crf_ner_tagger.ipynb))
built with `sklearn-crfsuite` on the GMB corpus: hand-crafted token, casing,
and context-window features, with per-entity-type evaluation. A nice
illustration that sequence models with good features remain competitive for
structured NLP tasks.

## Stack

`NLTK` · `scikit-learn` · `sklearn-crfsuite` · `spaCy` · `gensim` · `BeautifulSoup`
