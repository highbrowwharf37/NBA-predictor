import pandas as pd
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

games = pd.read_csv("playoff_games.csv")
games['WIN'] = (games['WL'] == 'W').astype(int)
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

team_avgs = games.groupby(['SEASON_ID', 'TEAM_ABBREVIATION']).agg(
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

features = ['pts_diff','fg_diff','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','pm_diff','rest_diff',
            'win_pct_diff','def_diff','home_court']

games['REST_DAYS'] = games.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(3)

home = games[games['MATCHUP'].str.contains('vs.')].copy()
away = games[games['MATCHUP'].str.contains('@')].copy()
merged = home.merge(away, on='GAME_ID', suffixes=('_home', '_away'))

home_avgs = team_avgs.rename(columns={
    'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION_home',
    'SEASON_ID': 'SEASON_ID_home',
    'avg_pts': 'avg_pts_home', 'avg_fg': 'avg_fg_home',
    'avg_fg3': 'avg_fg3_home', 'avg_reb': 'avg_reb_home',
    'avg_ast': 'avg_ast_home', 'avg_tov': 'avg_tov_home',
    'avg_stl': 'avg_stl_home', 'avg_pm': 'avg_pm_home',
    'win_pct': 'win_pct_home', 'avg_pts_allowed': 'avg_pts_allowed_home'
})
merged = merged.merge(home_avgs, on=['SEASON_ID_home', 'TEAM_ABBREVIATION_home'])

away_avgs = team_avgs.rename(columns={
    'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION_away',
    'SEASON_ID': 'SEASON_ID_away',
    'avg_pts': 'avg_pts_away', 'avg_fg': 'avg_fg_away',
    'avg_fg3': 'avg_fg3_away', 'avg_reb': 'avg_reb_away',
    'avg_ast': 'avg_ast_away', 'avg_tov': 'avg_tov_away',
    'avg_stl': 'avg_stl_away', 'avg_pm': 'avg_pm_away',
    'win_pct': 'win_pct_away', 'avg_pts_allowed': 'avg_pts_allowed_away'
})
merged = merged.merge(away_avgs, on=['SEASON_ID_away', 'TEAM_ABBREVIATION_away'])

merged['pts_diff']     = merged['avg_pts_home']         - merged['avg_pts_away']
merged['fg_diff']      = merged['avg_fg_home']          - merged['avg_fg_away']
merged['fg3_diff']     = merged['avg_fg3_home']         - merged['avg_fg3_away']
merged['reb_diff']     = merged['avg_reb_home']         - merged['avg_reb_away']
merged['ast_diff']     = merged['avg_ast_home']         - merged['avg_ast_away']
merged['tov_diff']     = merged['avg_tov_home']         - merged['avg_tov_away']
merged['stl_diff']     = merged['avg_stl_home']         - merged['avg_stl_away']
merged['pm_diff']      = merged['avg_pm_home']          - merged['avg_pm_away']
merged['rest_diff']    = merged['REST_DAYS_home']       - merged['REST_DAYS_away']
merged['win_pct_diff'] = merged['win_pct_home']         - merged['win_pct_away']
merged['def_diff']     = merged['avg_pts_allowed_away'] - merged['avg_pts_allowed_home']
merged['home_court']   = 1
merged['home_win']     = (merged['WL_home'] == 'W').astype(int)

df = merged[features + ['home_win', 'SEASON_ID_home',
            'TEAM_ABBREVIATION_home', 'TEAM_ABBREVIATION_away']].dropna()

X = df[features]
y = df['home_win']
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:,1]

df['predicted'] = y_pred
df['correct'] = (df['predicted'] == df['home_win']).astype(int)
df['prob'] = y_prob

# Accuracy by season
print("=" * 45)
print("ACCURACY BY SEASON")
print("=" * 45)
season_acc = df.groupby('SEASON_ID_home')['correct'].mean()
for season, acc in season_acc.items():
    print(str(season) + ": " + str(round(acc * 100, 1)) + "%")

print()
print("=" * 45)
print("OVERALL ACCURACY: " + str(round(df['correct'].mean() * 100, 1)) + "%")
print("=" * 45)

# Most confident correct predictions
print()
print("TOP 5 MOST CONFIDENT CORRECT PREDICTIONS")
print("=" * 45)
correct = df[df['correct'] == 1].nlargest(5, 'prob')
for _, row in correct.iterrows():
    print(row['TEAM_ABBREVIATION_home'] + " vs " + row['TEAM_ABBREVIATION_away'] +
          " | Confidence: " + str(round(row['prob'] * 100, 1)) + "%")

# Most confident wrong predictions
print()
print("TOP 5 MOST CONFIDENT WRONG PREDICTIONS")
print("=" * 45)
wrong = df[df['correct'] == 0].nlargest(5, 'prob')
for _, row in wrong.iterrows():
    print(row['TEAM_ABBREVIATION_home'] + " vs " + row['TEAM_ABBREVIATION_away'] +
          " | Confidence: " + str(round(row['prob'] * 100, 1)) + "%")

df.to_csv("historical_accuracy.csv", index=False)
print()
print("Full results saved to historical_accuracy.csv")