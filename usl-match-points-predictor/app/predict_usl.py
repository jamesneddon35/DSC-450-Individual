import os
import joblib
import pandas as pd
import dash
from dash import dcc, html, Input, Output

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')

pipeline             = joblib.load(os.path.join(MODEL_DIR, 'pipeline.joblib'))
numeric_features     = joblib.load(os.path.join(MODEL_DIR, 'numeric_features.joblib'))
categorical_features = joblib.load(os.path.join(MODEL_DIR, 'categorical_features.joblib'))

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# ── Helpers ───────────────────────────────────────────────────────────────────

def num_input(id_, default):
    return dcc.Input(id=id_, type='number', value=default, style={'width': '100%'})

def dropdown(id_, options, default):
    return dcc.Dropdown(id=id_, options=[{'label': l, 'value': v} for l, v in options],
                        value=default, clearable=False)

def labeled(label, component):
    return html.Div([html.Label(label), component], className='four columns')

# ── Prediction ────────────────────────────────────────────────────────────────

def get_prediction(gk_ss, gk_claim, gk_sweep, gk_hand, gk_field, gk_pass, gk_ga,
                   xg, sot, poss, ppda, xg_diff, conv, sag, deep, tpa,
                   is_home, formation, season_phase, poss_tier):
    row = pd.DataFrame([{
        'GK_Shotstopping':         float(gk_ss),
        'GK_Claiming':             float(gk_claim),
        'GK_Sweeping':             float(gk_sweep),
        'GK_Handling':             float(gk_hand),
        'GK_Fielding':             float(gk_field),
        'GK_Passing':              float(gk_pass),
        'GK_Goals_Added':          float(gk_ga),
        'xG':                      float(xg),
        'Shots on target':         float(sot),
        'Possession %':            float(poss),
        'PPDA':                    float(ppda),
        'xG_Diff':                 float(xg_diff),
        'Shot_Conversion_Rate':    float(conv),
        'Shots against on target': float(sag),
        'Deep completed passes':   float(deep),
        'Touches in penalty area': float(tpa),
        'Is_Home':                 int(is_home),
        'Formation_Clean':         str(formation),
        'Season_Phase':            str(season_phase),
        'Possession_Tier':         str(poss_tier),
    }])
    pred = pipeline.predict(row)[0]
    return f'Predicted Points: {round(pred, 1)}'

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div([
    html.H1('USL Championship 2025 — Match Points Predictor'),
    html.H2('Gradient Boosted Trees — predicted points (0=loss, 1=draw, 3=win)'),

    # Row 1
    html.Div([
        labeled('GK Shotstopping',  num_input('gk_ss',    0)),
        labeled('GK Claiming',      num_input('gk_claim', 0)),
        labeled('GK Sweeping',      num_input('gk_sweep', 0)),
    ], className='row'),

    # Row 2
    html.Div([
        labeled('GK Handling', num_input('gk_hand',  0)),
        labeled('GK Fielding', num_input('gk_field', 0)),
        labeled('GK Passing',  num_input('gk_pass',  0)),
    ], className='row'),

    # Row 3
    html.Div([
        labeled('GK Goals Added',   num_input('gk_ga',   0)),
        labeled('xG',               num_input('xg',      0)),
        labeled('Shots on Target',  num_input('sot',     0)),
    ], className='row'),

    # Row 4
    html.Div([
        labeled('Possession %', num_input('poss',    50)),
        labeled('PPDA',         num_input('ppda',    10)),
        labeled('xG Diff',      num_input('xg_diff', 0)),
    ], className='row'),

    # Row 5
    html.Div([
        labeled('Shot Conversion Rate',      num_input('conv', 0)),
        labeled('Shots Against on Target',   num_input('sag',  0)),
        labeled('Deep Completed Passes',     num_input('deep', 0)),
    ], className='row'),

    # Row 6
    html.Div([
        labeled('Touches in Penalty Area', num_input('tpa', 0)),
        labeled('Is Home', dropdown('is_home', [('Home', 1), ('Away', 0)], 1)),
        labeled('Formation', dropdown('formation', [
            ('4-4-2', '4-4-2'), ('4-3-3', '4-3-3'), ('4-2-3-1', '4-2-3-1'),
            ('3-5-2', '3-5-2'), ('5-3-2', '5-3-2'), ('4-1-4-1', '4-1-4-1'),
        ], '4-4-2')),
    ], className='row'),

    # Row 7
    html.Div([
        labeled('Season Phase', dropdown('season_phase', [
            ('Regular Season', 'Regular Season'), ('Playoffs', 'Playoffs'),
        ], 'Regular Season')),
        labeled('Possession Tier', dropdown('poss_tier', [
            ('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High'),
        ], 'Medium')),
    ], className='row'),

    html.Br(),
    html.H1(id='output'),
], style={'padding': '20px'})

# ── Callback ──────────────────────────────────────────────────────────────────

@app.callback(
    Output('output', 'children'),
    Input('gk_ss',       'value'), Input('gk_claim', 'value'),
    Input('gk_sweep',    'value'), Input('gk_hand',  'value'),
    Input('gk_field',    'value'), Input('gk_pass',  'value'),
    Input('gk_ga',       'value'), Input('xg',       'value'),
    Input('sot',         'value'), Input('poss',      'value'),
    Input('ppda',        'value'), Input('xg_diff',   'value'),
    Input('conv',        'value'), Input('sag',       'value'),
    Input('deep',        'value'), Input('tpa',       'value'),
    Input('is_home',     'value'), Input('formation',  'value'),
    Input('season_phase','value'), Input('poss_tier',  'value'),
)
def update_output(gk_ss, gk_claim, gk_sweep, gk_hand, gk_field, gk_pass, gk_ga,
                  xg, sot, poss, ppda, xg_diff, conv, sag, deep, tpa,
                  is_home, formation, season_phase, poss_tier):
    if any(v is None for v in [gk_ss, gk_claim, gk_sweep, gk_hand, gk_field,
                                gk_pass, gk_ga, xg, sot, poss, ppda, xg_diff,
                                conv, sag, deep, tpa, is_home, formation,
                                season_phase, poss_tier]):
        return 'Prediction: —'
    return get_prediction(gk_ss, gk_claim, gk_sweep, gk_hand, gk_field, gk_pass, gk_ga,
                          xg, sot, poss, ppda, xg_diff, conv, sag, deep, tpa,
                          is_home, formation, season_phase, poss_tier)


if __name__ == '__main__':
    app.run(debug=True)
