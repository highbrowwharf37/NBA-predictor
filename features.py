import pandas as pd

games = pd.read_csv("playoff_games.csv")
advanced = pd.read_csv("advanced_stats.csv")

games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games = games.sort_values(['TEAM_ID', 'GAME_DATE'])
games['WIN'] = (games['WL'] == 'W').astype(int)

# Rest days
games['REST_DAYS'] = games.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(3)

# Rolling last 15 games form per team
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

# Attach advanced stats for home team
adv_home = advanced.rename(columns={
    'TEAM_ID': 'TEAM_ID_home',
    'SEASON_ID': 'SEASON_ID_home',
    'OFF_RATING': 'off_rtg_home',
    'DEF_RATING': 'def_rtg_home',
    'NET_RATING': 'net_rtg_home',
    'PACE': 'pace_home',
    'TS_PCT': 'ts_pct_home'
})[['TEAM_ID_home','SEASON_ID_home','off_rtg_home',
    'def_rtg_home','net_rtg_home','pace_home','ts_pct_home']]

# Attach advanced stats for away team
adv_away = advanced.rename(columns={
    'TEAM_ID': 'TEAM_ID_away',
    'SEASON_ID': 'SEASON_ID_away',
    'OFF_RATING': 'off_rtg_away',
    'DEF_RATING': 'def_rtg_away',
    'NET_RATING': 'net_rtg_away',
    'PACE': 'pace_away',
    'TS_PCT': 'ts_pct_away'
})[['TEAM_ID_away','SEASON_ID_away','off_rtg_away',
    'def_rtg_away','net_rtg_away','pace_away','ts_pct_away']]

# Need to match SEASON_ID format
# advanced uses "2014-15", games uses "42014" format — fix this
def convert_season_id(sid):
    sid = str(sid)
    if '-' in sid:
        return sid
    year = int(sid[-4:])
    return str(year) + "-" + str(year + 1)[2:]

advanced['SEASON_ID'] = advanced['SEASON_ID'].apply(convert_season_id)

adv_home = advanced.rename(columns={
    'TEAM_ID': 'TEAM_ID_home',
    'SEASON_ID': 'SEASON_ID_home',
    'OFF_RATING': 'off_rtg_home',
    'DEF_RATING': 'def_rtg_home',
    'NET_RATING': 'net_rtg_home',
    'PACE': 'pace_home',
    'TS_PCT': 'ts_pct_home'
})[['TEAM_ID_home','SEASON_ID_home','off_rtg_home',
    'def_rtg_home','net_rtg_home','pace_home','ts_pct_home']]

adv_away = advanced.rename(columns={
    'TEAM_ID': 'TEAM_ID_away',
    'SEASON_ID': 'SEASON_ID_away',
    'OFF_RATING': 'off_rtg_away',
    'DEF_RATING': 'def_rtg_away',
    'NET_RATING': 'net_rtg_away',
    'PACE': 'pace_away',
    'TS_PCT': 'ts_pct_away'
})[['TEAM_ID_away','SEASON_ID_away','off_rtg_away',
    'def_rtg_away','net_rtg_away','pace_away','ts_pct_away']]

# Convert games SEASON_ID too
merged['SEASON_ID_home'] = merged['SEASON_ID_home'].apply(convert_season_id)
merged['SEASON_ID_away'] = merged['SEASON_ID_away'].apply(convert_season_id)

merged = merged.merge(adv_home, on=['SEASON_ID_home','TEAM_ID_home'], how='left')
merged = merged.merge(adv_away, on=['SEASON_ID_away','TEAM_ID_away'], how='left')

# Build features
merged['off_rtg_diff']      = merged['off_rtg_home']    - merged['off_rtg_away']
merged['def_rtg_diff']      = merged['def_rtg_home']    - merged['def_rtg_away']
merged['net_rtg_diff']      = merged['net_rtg_home']    - merged['net_rtg_away']
merged['pace_diff']         = merged['pace_home']       - merged['pace_away']
merged['ts_pct_diff']       = merged['ts_pct_home']     - merged['ts_pct_away']
merged['rest_diff']         = merged['REST_DAYS_home']  - merged['REST_DAYS_away']
merged['last15_win_diff']   = merged['last15_win_home'] - merged['last15_win_away']
merged['last15_pm_diff']    = merged['last15_pm_home']  - merged['last15_pm_away']
merged['home_court']        = 1
merged['fg3_diff']      = merged['FG3_PCT_home']   - merged['FG3_PCT_away']
merged['reb_diff']      = merged['REB_home']        - merged['REB_away']
merged['ast_diff']      = merged['AST_home']        - merged['AST_away']
merged['tov_diff']      = merged['TOV_home']        - merged['TOV_away']
merged['stl_diff']      = merged['STL_home']        - merged['STL_away']
merged['win_pct_diff']  = merged['last15_win_home'] - merged['last15_win_away']

merged['home_win'] = (merged['WL_home'] == 'W').astype(int)

features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
            'home_court','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','win_pct_diff']

df = merged[features + ['home_win']].dropna()
df.to_csv("features.csv", index=False)
print("Done! " + str(len(df)) + " games ready for modeling")
print(df.head())