import pandas as pd
import numpy as np
import re
import os
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MATCH_CSV = os.path.join(DATA_DIR, 'USL_Championship_2025_Feature_Engineered.csv')
GK_CSV    = os.path.join(DATA_DIR, 'american_soccer_analysis_uslc_goals-added_goalkeepers_2026-05-09.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

match_df = pd.read_csv(MATCH_CSV)
gk_df    = pd.read_csv(GK_CSV, index_col=0)

print(f'Match rows : {len(match_df):,}')
print(f'GK rows    : {len(gk_df)}')

# ── GK abbreviation mapping and minutes-weighted season aggregates ─────────────

abbrev_map = {
    'BHM': 'Birmingham Legion',     'CHS': 'Charleston Battery',
    'DET': 'Detroit City',           'HFD': 'Hartford Athletic',
    'IND': 'Indy Eleven',            'LEX': 'Lexington SC',
    'LDN': 'Loudoun United',         'LOU': 'Louisville City',
    'MIA': 'Miami',                  'NC':  'North Carolina FC',
    'PIT': 'Pittsburgh Riverhounds', 'RI':  'Rhode Island',
    'TBR': 'Tampa Bay Rowdies',      'COS': 'Colorado Springs',
    'ELP': 'El Paso Locomotive',     'TUL': 'FC Tulsa',
    'LV':  'Las Vegas Lights',       'MB':  'Monterey Bay FC',
    'NM':  'New Mexico United',      'OAK': 'Oakland Roots',
    'OC':  'Orange County SC',       'PHX': 'Phoenix Rising',
    'SAC': 'Sacramento Republic',    'SA':  'San Antonio',
}

gk_cols = ['Claiming', 'Fielding', 'Handling', 'Passing', 'Shotstopping', 'Sweeping', 'Goals Added']

gk_single = gk_df[~gk_df['Team'].str.contains(',')].copy()
gk_single['Team_Full'] = gk_single['Team'].map(abbrev_map)
gk_single['Minutes'] = pd.to_numeric(gk_single['Minutes'], errors='coerce')
for c in gk_cols:
    gk_single[c] = pd.to_numeric(gk_single[c], errors='coerce')

def weighted_avg(group):
    w = group['Minutes']
    out = {f'GK_{c.replace(" ", "_")}': np.average(group[c], weights=w) for c in gk_cols}
    out['GK_Minutes'] = w.sum()
    return pd.Series(out)

gk_team = (
    gk_single
    .groupby('Team_Full', group_keys=False)
    .apply(weighted_avg)
    .reset_index()
)

df = match_df.merge(gk_team, left_on='Team', right_on='Team_Full', how='inner')
print(f'Teams with GK data : {len(gk_team)}')
print(f'Rows after join    : {len(df):,}  |  Teams : {df["Team"].nunique()}')

# ── get_features() ────────────────────────────────────────────────────────────

def get_features(row):
    m = re.search(r'(\d+):(\d+)', row['Match'])
    hs, as_ = int(m.group(1)), int(m.group(2))
    home = re.split(r'\s*\(?\w\)?\s*\d+:\d+', row['Match'])[0].split(' - ')[0].strip()
    is_home = row['Team'] == home
    ms, os_ = (hs, as_) if is_home else (as_, hs)
    r = 'W' if ms > os_ else ('L' if ms < os_ else 'D')
    return pd.Series(
        [int(is_home), r, 3 if r == 'W' else 1 if r == 'D' else 0],
        index=['Is_Home', 'Result', 'Points']
    )

if 'Points' not in df.columns:
    df = pd.concat([df, df.apply(get_features, axis=1)], axis=1)

# ── Exclude playoff matches ───────────────────────────────────────────────────

df = df[df['Season_Phase'] != 'Playoffs'].copy()
print(f'Rows after playoff exclusion : {len(df):,}')

# ── Feature lists (verbatim from notebook) ────────────────────────────────────

numeric_features = [
    'GK_Shotstopping', 'GK_Claiming', 'GK_Sweeping',
    'GK_Handling', 'GK_Fielding', 'GK_Passing', 'GK_Goals_Added',
    'xG', 'Shots on target', 'Possession %', 'PPDA',
    'xG_Diff', 'Shot_Conversion_Rate',
    'Shots against on target', 'Deep completed passes',
    'Touches in penalty area', 'Is_Home',
]
categorical_features = ['Formation_Clean', 'Season_Phase', 'Possession_Tier']

for col in categorical_features:
    df[col] = df[col].astype(str)

X = df[numeric_features + categorical_features]
y = df['Points']

print(f'Features : {len(numeric_features)} numeric + {len(categorical_features)} categorical')
print(f'Samples  : {len(X)}')

# ── GBT pipeline — best params from GridSearchCV in notebook ──────────────────
# Best: learning_rate=0.05, max_depth=2, max_iter=200  CV RMSE=0.9388

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
])

best_pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=2,
        max_iter=200,
        random_state=42,
    )),
])

best_pipeline.fit(X, y)
print('Pipeline fitted on full dataset.')

# ── Serialize ─────────────────────────────────────────────────────────────────

joblib.dump(best_pipeline,         os.path.join(MODEL_DIR, 'pipeline.joblib'))
joblib.dump(numeric_features,      os.path.join(MODEL_DIR, 'numeric_features.joblib'))
joblib.dump(categorical_features,  os.path.join(MODEL_DIR, 'categorical_features.joblib'))

print(f'\nSaved to {MODEL_DIR}')
print('Numeric features   :', numeric_features)
print('Categorical features:', categorical_features)
