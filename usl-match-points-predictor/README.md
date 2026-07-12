# USL Match Points Predictor

Predicting match points (0 / 1 / 3) for USL Championship 2025 teams from
match-level team statistics and goalkeeper **goals-added (g+)** metrics, ending
in an interactive Dash app.

## Pipeline

1. **[01 — EDA](notebooks/01_eda_usl_championship_2025.ipynb)** — exploratory
   analysis of match-level team statistics across all 24 USL Championship
   clubs (ydata-profiling, distributions, correlations).
2. **[02 — Feature engineering](notebooks/02_feature_engineering_usl_championship_2025.ipynb)** —
   derived features (xG differential, shot conversion rate, possession tiers,
   cleaned formations), scaling and encoding.
3. **[03 — Modeling](notebooks/03_model_usl_championship_2025.ipynb)** — Ridge
   regression baseline vs. gradient-boosted trees
   (`HistGradientBoostingRegressor`), tuned with GridSearchCV
   (best CV RMSE ≈ 0.94 points).
4. **[App](app/)** — a Dash web app that loads the serialized pipeline and
   predicts expected match points from user-entered team and goalkeeper stats.

## The goalkeeper angle

Goalkeeper performance enters the model through American Soccer Analysis
**goals-added (g+)** components — shot-stopping, claiming, sweeping, handling,
fielding, and passing — aggregated to team level as minutes-weighted season
averages and joined onto each match row (`app/model_training_usl.py`).

## Running the app

```bash
cd app
pip install -r requirements.txt
python predict_usl.py        # uses the committed Model/*.joblib artifacts
```

To retrain, place the two source CSVs in `app/data/` (see
`model_training_usl.py` for expected filenames) and run
`python model_training_usl.py`.

## Data

Not included in the repo: USL Championship 2025 match statistics
(feature-engineered CSV) and the ASA goalkeeper g+ export — see
[americansocceranalysis.com](https://www.americansocceranalysis.com/).
