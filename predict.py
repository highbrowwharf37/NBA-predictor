import pandas as pd
import pickle

# Load model and historical averages
df = pd.read_csv("features.csv")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load team season averages to look up stats
games = pd.read_csv("playoff_games.csv")
games['WIN'] = (games['WL'] == 'W').astype(int)

team_avgs = games.groupby('TEAM_ABBREVIATION').agg(
    avg_pts=('PTS', 'mean'),
    avg_fg=('FG_PCT', 'mean'),
    avg_fg3=('FG3_PCT', 'mean'),
    avg_reb=('REB', 'mean'),
    avg_ast=('AST', 'mean'),
    avg_tov=('TOV', 'mean'),
    avg_stl=('STL', 'mean'),
    avg_pm=('PLUS_MINUS', 'mean'),
    win_pct=('WIN', 'mean'),
    avg_pts_allowed=('PTS', 'mean')
).reset_index()

print("Available teams:")
print(sorted(team_avgs['TEAM_ABBREVIATION'].tolist()))
print()

home_team = input("Enter HOME team abbreviation (e.g. BOS): ").strip().upper()
away_team = input("Enter AWAY team abbreviation (e.g. NYK): ").strip().upper()
home_rest = float(input("Home team rest days: "))
away_rest = float(input("Away team rest days: "))

h = team_avgs[team_avgs['TEAM_ABBREVIATION'] == home_team].iloc[0]
a = team_avgs[team_avgs['TEAM_ABBREVIATION'] == away_team].iloc[0]

game = {
    'pts_diff':     h['avg_pts']         - a['avg_pts'],
    'fg_diff':      h['avg_fg']          - a['avg_fg'],
    'fg3_diff':     h['avg_fg3']         - a['avg_fg3'],
    'reb_diff':     h['avg_reb']         - a['avg_reb'],
    'ast_diff':     h['avg_ast']         - a['avg_ast'],
    'tov_diff':     h['avg_tov']         - a['avg_tov'],
    'stl_diff':     h['avg_stl']         - a['avg_stl'],
    'pm_diff':      h['avg_pm']          - a['avg_pm'],
    'rest_diff':    home_rest            - away_rest,
    'win_pct_diff': h['win_pct']         - a['win_pct'],
    'def_diff':     a['avg_pts_allowed'] - h['avg_pts_allowed'],
    'home_court':   1
}

features = ['pts_diff','fg_diff','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','pm_diff','rest_diff',
            'win_pct_diff','def_diff','home_court']

X = pd.DataFrame([game])[features]
prob = model.predict_proba(X)[0][1]
prediction = home_team if prob >= 0.5 else away_team

print()
print("=" * 40)
print(home_team + " (home) vs " + away_team + " (away)")
print("Home win probability: " + str(round(prob * 100, 1)) + "%")
print("Away win probability: " + str(round((1 - prob) * 100, 1)) + "%")
print("Prediction: " + prediction + " wins")
print("=" * 40)