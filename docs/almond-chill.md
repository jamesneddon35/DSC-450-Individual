# Almond Chill Accumulation vs. Yield

**Does a cold winter make a good almond harvest? Public agricultural and
weather data, engineered features, and honest regression.**

[← Back to portfolio](index.md) ·
[Code on GitHub]({{ site.github.repository_url }}/tree/main/almond-chill-accumulation)

## The question

Almond trees need winter chill to break dormancy and set fruit. Using Fresno
County — the heart of California's almond belt — can we quantify the
relationship between winter chill accumulation and yield?

## The data work

- **Yield** from USDA NASS Quick Stats: derived as production ÷ bearing acres,
  because raw acreage climbs over time for economic reasons that have nothing
  to do with weather (a confound the analysis addresses head-on).
- **Weather** from NOAA Local Climatological Data (Fresno station): three
  export files merged, de-duplicated on timestamp, temperature fields
  hardened against suspect-flag artifacts, and Nov/Dec weather mapped to the
  *following* harvest year.

## Feature engineering

Per harvest year: average/min/max/std winter temperature, **freeze hours**
(≤ 32°F), **chill hours** (32–45°F — the horticulturally meaningful band),
and fog hours from present-weather codes.

## Modeling & findings

Single-feature OLS models for every predictor, a multiple regression, and an
explicit head-to-head between chill hours and plain average temperature —
testing whether the horticultural definition of chill beats a naive
temperature average. A reusable `ModelPerformance` class reports R², adjusted
R², and RMSE with the sample size printed prominently, because with yearly
agricultural data, *n* is always the elephant in the room.

![Chill vs yield]({{ site.github.repository_url }}/raw/main/almond-chill-accumulation/figures/fig_chill_vs_yield.png)

## Ethics & limits

The writeup addresses the derived-yield caveat, the acreage confound,
small-sample degrees of freedom, and why "cold causes yield" oversimplifies a
system where growers can actively mitigate (misting, cooling).

*Group project for DSC450; teammate names withheld. The modeling pass
published here is my own work.*

## Stack

`pandas` · `statsmodels` · `scikit-learn` · `matplotlib`
