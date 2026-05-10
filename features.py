import pandas as pd

games = pd.read_csv("playoff_games.csv")
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games = games.sort_values(['TEAM_ID', 'GAME_DATE'])

# Rest days
games['REST_DAYS'] = games.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(3)

# Win/loss as 1/0
games['WIN'] = (games['WL'] == 'W').astype(int)

# Season averages per team
season_avgs = games.groupby(['SEASON_ID', 'TEAM_ID']).agg(
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

# Playoff seeding — extract from SEASON_ID and game order
# Use win_pct as proxy for seed (higher win pct = better seed)

# Separate home and away
home = games[games['MATCHUP'].str.contains('vs.')].copy()
away = games[games['MATCHUP'].str.contains('@')].copy()

merged = home.merge(away, on='GAME_ID', suffixes=('_home', '_away'))

# Attach home team averages
home_avgs = season_avgs.rename(columns={
    'TEAM_ID': 'TEAM_ID_home',
    'SEASON_ID': 'SEASON_ID_home',
    'avg_pts': 'avg_pts_home',
    'avg_fg': 'avg_fg_home',
    'avg_fg3': 'avg_fg3_home',
    'avg_reb': 'avg_reb_home',
    'avg_ast': 'avg_ast_home',
    'avg_tov': 'avg_tov_home',
    'avg_stl': 'avg_stl_home',
    'avg_pm': 'avg_pm_home',
    'win_pct': 'win_pct_home',
    'avg_pts_allowed': 'avg_pts_allowed_home'
})
merged = merged.merge(home_avgs, on=['SEASON_ID_home', 'TEAM_ID_home'])

# Attach away team averages
away_avgs = season_avgs.rename(columns={
    'TEAM_ID': 'TEAM_ID_away',
    'SEASON_ID': 'SEASON_ID_away',
    'avg_pts': 'avg_pts_away',
    'avg_fg': 'avg_fg_away',
    'avg_fg3': 'avg_fg3_away',
    'avg_reb': 'avg_reb_away',
    'avg_ast': 'avg_ast_away',
    'avg_tov': 'avg_tov_away',
    'avg_stl': 'avg_stl_away',
    'avg_pm': 'avg_pm_away',
    'win_pct': 'win_pct_away',
    'avg_pts_allowed': 'avg_pts_allowed_away'
})
merged = merged.merge(away_avgs, on=['SEASON_ID_away', 'TEAM_ID_away'])

# Feature differentials
merged['pts_diff']         = merged['avg_pts_home']         - merged['avg_pts_away']
merged['fg_diff']          = merged['avg_fg_home']          - merged['avg_fg_away']
merged['fg3_diff']         = merged['avg_fg3_home']         - merged['avg_fg3_away']
merged['reb_diff']         = merged['avg_reb_home']         - merged['avg_reb_away']
merged['ast_diff']         = merged['avg_ast_home']         - merged['avg_ast_away']
merged['tov_diff']         = merged['avg_tov_home']         - merged['avg_tov_away']
merged['stl_diff']         = merged['avg_stl_home']         - merged['avg_stl_away']
merged['pm_diff']          = merged['avg_pm_home']          - merged['avg_pm_away']
merged['rest_diff']        = merged['REST_DAYS_home']       - merged['REST_DAYS_away']
merged['win_pct_diff']     = merged['win_pct_home']         - merged['win_pct_away']
merged['def_diff']         = merged['avg_pts_allowed_away'] - merged['avg_pts_allowed_home']

# Home court is always 1 for home team (explicit feature)
merged['home_court'] = 1

merged['home_win'] = (merged['WL_home'] == 'W').astype(int)

features = ['pts_diff','fg_diff','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','pm_diff','rest_diff',
            'win_pct_diff','def_diff','home_court']

df = merged[features + ['home_win']].dropna()
df.to_csv("features.csv", index=False)
print("Done! " + str(len(df)) + " games ready for modeling")
print(df.head())