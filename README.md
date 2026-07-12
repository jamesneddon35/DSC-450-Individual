# James Sneddon — Data Science Portfolio

I'm a data science student at Bellevue University and a professional goalkeeper
in USL League One. Most of my work sits somewhere between soccer and stats,
but not all of it. These are the three projects from my program I'm most proud
of.

## Projects

**[USL Match Points Predictor](usl-match-points-predictor/)**
I wanted to know how many points a team should expect from a match based on
how it played, and how much of that comes down to the goalkeeper. Three
notebooks take USL Championship 2025 match data from EDA through feature
engineering to a tuned gradient-boosted model, and a small Dash app lets you
play with the inputs and see the prediction change. Goalkeeper value comes in
through American Soccer Analysis goals-added (g+) numbers, which I care about
for obvious reasons.

**[IMDB 50K Sentiment Analysis](imdb-sentiment-nlp/)**
Text analytics coursework on the classic IMDB reviews dataset. One notebook
does the cleanup and feature engineering (TF-IDF and Word2Vec), the other
trains and compares Logistic Regression and Linear SVC classifiers. There's
also a CRF named-entity tagger in there from the same course.

**[Almond Chill Accumulation vs. Yield](almond-chill-accumulation/)**
A group project asking whether cold winters predict almond harvests in Fresno
County. I handled the modeling pass: wrangling USDA and NOAA data, building
chill-hour features, and running the regressions. It's small data with real
confounds, and the notebook is honest about both.

## About the data

None of the raw data lives in this repo. The IMDB dataset is the standard
[Stanford/Kaggle release](https://ai.stanford.edu/~amaas/data/sentiment/), the
goalkeeper g+ tables come from
[American Soccer Analysis](https://www.americansocceranalysis.com/), and the
almond project uses public USDA NASS and NOAA exports (pull details are in the
notebooks). The USL match dataset I compiled myself; happy to share it if you
ask. All notebooks are committed with their outputs, so you can read the
results without running anything.

## Site

This repo doubles as a website via GitHub Pages:
[jamesneddon35.github.io/DSC-450-Individual](https://jamesneddon35.github.io/DSC-450-Individual/)
