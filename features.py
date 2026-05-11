import pandas as pd

games = pd.read_csv("playoff_games.csv")
advanced = pd.read_csv("advanced_stats.csv")
regular = pd.read_csv("regular_season_stats.csv")

games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games = games.sort_values(['TEAM_ID', 'GAME_DATE'])
games['WIN'] = (games['WL'] == 'W').astype(int)

def convert_season_id(sid):
    sid = str(sid)
    if '-' in sid:
        return sid
    year = int(sid[-4:])
    return str(year) + "-" + str(year + 1)[2:]

# Convert all season IDs to same format
games['SEASON_ID'] = games['SEASON_ID'].apply(convert_season_id)
advanced['SEASON_ID'] = advanced['SEASON_ID'].apply(convert_season_id)
regular['SEASON_ID'] = regular['SEASON_ID'].apply(convert_season_id)

# Rest days
games['REST_DAYS'] = games.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(3)

# Rolling last 15 games
games['last15_win'] = games.groupby('TEAM_ID')['WIN'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=1).mean()
)
games['last15_pm'] = games.groupby('TEAM_ID')['PLUS_MINUS'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=1).mean()
)

# Separate home and away
home = games[games['MATCHUP'].str.contains('vs.')].copy()
away = games[games['MATCHUP'].str.contains('@')].copy()
merged = home.merge(away, on='GAME_ID', suffixes=('_home', '_away'))

# Attach regular season stats for home team
adv_home = regular.rename(columns={
    'TEAM_ID': 'TEAM_ID_home',
    'SEASON_ID': 'SEASON_ID_home',
    'OFF_RATING': 'off_rtg_home',
    'DEF_RATING': 'def_rtg_home',
    'NET_RATING': 'net_rtg_home',
    'PACE': 'pace_home',
    'TS_PCT': 'ts_pct_home',
    'W_PCT': 'w_pct_home',
    'AST_PCT': 'ast_pct_home',
    'OREB_PCT': 'oreb_pct_home',
    'DREB_PCT': 'dreb_pct_home',
    'TM_TOV_PCT': 'tov_pct_home',
    'EFG_PCT': 'efg_pct_home',
})[['TEAM_ID_home','SEASON_ID_home','off_rtg_home','def_rtg_home',
    'net_rtg_home','pace_home','ts_pct_home','w_pct_home',
    'ast_pct_home','oreb_pct_home','dreb_pct_home','tov_pct_home','efg_pct_home']]

# Attach regular season stats for away team
adv_away = regular.rename(columns={
    'TEAM_ID': 'TEAM_ID_away',
    'SEASON_ID': 'SEASON_ID_away',
    'OFF_RATING': 'off_rtg_away',
    'DEF_RATING': 'def_rtg_away',
    'NET_RATING': 'net_rtg_away',
    'PACE': 'pace_away',
    'TS_PCT': 'ts_pct_away',
    'W_PCT': 'w_pct_away',
    'AST_PCT': 'ast_pct_away',
    'OREB_PCT': 'oreb_pct_away',
    'DREB_PCT': 'dreb_pct_away',
    'TM_TOV_PCT': 'tov_pct_away',
    'EFG_PCT': 'efg_pct_away',
})[['TEAM_ID_away','SEASON_ID_away','off_rtg_away','def_rtg_away',
    'net_rtg_away','pace_away','ts_pct_away','w_pct_away',
    'ast_pct_away','oreb_pct_away','dreb_pct_away','tov_pct_away','efg_pct_away']]

merged = merged.merge(adv_home, on=['SEASON_ID_home','TEAM_ID_home'], how='left')
merged = merged.merge(adv_away, on=['SEASON_ID_away','TEAM_ID_away'], how='left')

# Build features — all from season-specific regular season stats
merged['off_rtg_diff']    = merged['off_rtg_home']   - merged['off_rtg_away']
merged['def_rtg_diff']    = merged['def_rtg_home']   - merged['def_rtg_away']
merged['net_rtg_diff']    = merged['net_rtg_home']   - merged['net_rtg_away']
merged['pace_diff']       = merged['pace_home']      - merged['pace_away']
merged['ts_pct_diff']     = merged['ts_pct_home']    - merged['ts_pct_away']
merged['w_pct_diff']      = merged['w_pct_home']     - merged['w_pct_away']
merged['ast_pct_diff']    = merged['ast_pct_home']   - merged['ast_pct_away']
merged['oreb_pct_diff']   = merged['oreb_pct_home']  - merged['oreb_pct_away']
merged['dreb_pct_diff']   = merged['dreb_pct_home']  - merged['dreb_pct_away']
merged['tov_pct_diff']    = merged['tov_pct_home']   - merged['tov_pct_away']
merged['efg_pct_diff']    = merged['efg_pct_home']   - merged['efg_pct_away']
merged['rest_diff']       = merged['REST_DAYS_home'] - merged['REST_DAYS_away']
merged['last15_win_diff'] = merged['last15_win_home']- merged['last15_win_away']
merged['last15_pm_diff']  = merged['last15_pm_home'] - merged['last15_pm_away']
merged['home_court']      = 1
merged['home_win']        = (merged['WL_home'] == 'W').astype(int)

features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','w_pct_diff','ast_pct_diff','oreb_pct_diff',
            'dreb_pct_diff','tov_pct_diff','efg_pct_diff',
            'rest_diff','last15_win_diff','last15_pm_diff','home_court']

df = merged[features + ['home_win']].dropna()
df.to_csv("features.csv", index=False)
print("Done! " + str(len(df)) + " games ready for modeling")
print(df.head())