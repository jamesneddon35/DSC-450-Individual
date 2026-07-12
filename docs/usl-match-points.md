# USL Match Points Predictor

**Predicting match outcomes in the USL Championship — and measuring how much
the goalkeeper matters.**

[← Back to portfolio](index.md) ·
[Code on GitHub]({{ site.github.repository_url }}/tree/main/usl-match-points-predictor)

## The question

How many points (0 = loss, 1 = draw, 3 = win) should a team expect from a
match, given its performance profile — and how much predictive signal comes
from the goalkeeper?

## The pipeline

**1. EDA.** Match-level team statistics for all 24 USL Championship 2025
clubs profiled for distributions, missingness, and correlations
([notebook]({{ site.github.repository_url }}/blob/main/usl-match-points-predictor/notebooks/01_eda_usl_championship_2025.ipynb)).

**2. Feature engineering.** Derived features including xG differential, shot
conversion rate, possession tiers, and cleaned formation labels; standard
scaling and one-hot encoding
([notebook]({{ site.github.repository_url }}/blob/main/usl-match-points-predictor/notebooks/02_feature_engineering_usl_championship_2025.ipynb)).

**3. Goalkeeper goals-added (g+).** American Soccer Analysis publishes
goalkeeper value decomposed into shot-stopping, claiming, sweeping, handling,
fielding, and passing. I aggregated these to team level as minutes-weighted
season averages and joined them to every match row — so the model sees both
how the team played and who was in goal.

**4. Modeling.** Ridge regression baseline against gradient-boosted trees
(`HistGradientBoostingRegressor`), tuned via GridSearchCV to a cross-validated
RMSE of ≈ 0.94 points
([notebook]({{ site.github.repository_url }}/blob/main/usl-match-points-predictor/notebooks/03_model_usl_championship_2025.ipynb)).

**5. Deployment.** The fitted preprocessing + model pipeline is serialized
with joblib and served by a Dash web app: adjust any team or goalkeeper input
and watch the expected-points prediction update.

## Why it's interesting

As a professional goalkeeper, I wanted a model that doesn't treat the keeper
as an afterthought. Splitting goalkeeper value into g+ components lets the
model (and the app user) ask questions like: *what's a good sweeper-keeper
worth in expected points per match?*

## Stack

`pandas` · `scikit-learn` · `Dash` · `joblib` · ydata-profiling
