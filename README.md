# James Sneddon — Data Science Portfolio

Selected projects from my data science program at Bellevue University, spanning
sports analytics, natural language processing, and agricultural modeling.

📄 **Portfolio site:** enable GitHub Pages (Settings → Pages → deploy from `main`, `/docs` folder) and this repo doubles as a website.

## Projects

| Project | Domain | Highlights |
|---|---|---|
| [USL Match Points Predictor](usl-match-points-predictor/) | Sports analytics | End-to-end pipeline: EDA → feature engineering → Ridge & gradient-boosted models → interactive Dash app. Integrates American Soccer Analysis goals-added (g+) goalkeeper data. |
| [IMDB 50K Sentiment Analysis](imdb-sentiment-nlp/) | NLP | Text normalization and TF-IDF/Word2Vec feature engineering, then Logistic Regression and Linear SVC sentiment classifiers. Includes a CRF named-entity tagger side project. |
| [Almond Chill Accumulation vs. Yield](almond-chill-accumulation/) | Agriculture / regression | USDA NASS + NOAA weather wrangling, chill-hours feature engineering, OLS modeling of Fresno County almond yield, with an ethics writeup. |

## Data availability

Raw datasets are **not** included in this repo:

- **USL Championship 2025 match data** — compiled from public match statistics; feature-engineered CSV available on request.
- **Goalkeeper goals-added (g+)** — from [American Soccer Analysis](https://www.americansocceranalysis.com/); download their goalkeeper g+ tables directly.
- **IMDB 50K Movie Reviews** — the standard [Kaggle/Stanford dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
- **USDA NASS Quick Stats** and **NOAA LCD** exports — public data; pull parameters are documented inside the almond notebooks.

Notebooks are committed with executed outputs so results are viewable without re-running.

## About

James Sneddon — B.S. Data Science (in progress), Bellevue University.
Goalkeeper & data analyst with USL League One experience; I build models where
football and data meet.
