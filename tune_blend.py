import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

games = pd.read_csv("playoff_games.csv")
advanced = pd.read_csv("advanced_stats.csv")
regular = pd.read_csv("regular_season_stats.csv")
features_df = pd.read_csv("features.csv")

games['WIN'] = (games['WL'] == 'W').astype(int)
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games = games.sort_values(['TEAM_ID', 'GAME_DATE'])
games['REST_DAYS'] = games.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(3)
games['last15_win'] = games.groupby('TEAM_ID')['WIN'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=1).mean()
)
games['last15_pm'] = games.groupby('TEAM_ID')['PLUS_MINUS'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=1).mean()
)

team_avgs = games.groupby('TEAM_ABBREVIATION').agg(
    avg_fg3=('FG3_PCT', 'mean'),
    avg_reb=('REB', 'mean'),
    avg_ast=('AST', 'mean'),
    avg_tov=('TOV', 'mean'),
    avg_stl=('STL', 'mean'),
    win_pct=('WIN', 'mean'),
).reset_index()

team_ids = games[['TEAM_ID','TEAM_ABBREVIATION']].drop_duplicates()

def convert_season_id(sid):
    sid = str(sid)
    if '-' in sid:
        return sid
    year = int(sid[-4:])
    return str(year) + "-" + str(year + 1)[2:]

advanced['SEASON_ID'] = advanced['SEASON_ID'].apply(convert_season_id)
regular['SEASON_ID'] = regular['SEASON_ID'].apply(convert_season_id)

home = games[games['MATCHUP'].str.contains('vs.')].copy()
away = games[games['MATCHUP'].str.contains('@')].copy()
merged = home.merge(away, on='GAME_ID', suffixes=('_home', '_away'))

def convert_col(df, col):
    df[col] = df[col].apply(convert_season_id)
    return df

merged = convert_col(merged, 'SEASON_ID_home')
merged = convert_col(merged, 'SEASON_ID_away')

print("Testing blend ratios (playoff% / regular season%)...")
print("-" * 50)

best_auc = 0
best_blend = 0

for playoff_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    regular_weight = round(1.0 - playoff_weight, 1)

    # Build blended stats per season per team
    rows = []
    for _, season_group in advanced.groupby('SEASON_ID'):
        season = season_group['SEASON_ID'].iloc[0]
        reg_season = regular[regular['SEASON_ID'] == season]
        
        for _, p_row in season_group.iterrows():
            tid = p_row['TEAM_ID']
            r_row = reg_season[reg_season['TEAM_ID'] == tid]
            if r_row.empty:
                continue
            r_row = r_row.iloc[0]
            
            blended = {
                'TEAM_ID': tid,
                'SEASON_ID': season,
                'OFF_RATING': p_row['OFF_RATING'] * playoff_weight + r_row['OFF_RATING'] * regular_weight,
                'DEF_RATING': p_row['DEF_RATING'] * playoff_weight + r_row['DEF_RATING'] * regular_weight,
                'NET_RATING': p_row['NET_RATING'] * playoff_weight + r_row['NET_RATING'] * regular_weight,
                'PACE':       p_row['PACE']       * playoff_weight + r_row['PACE']       * regular_weight,
                'TS_PCT':     p_row['TS_PCT']      * playoff_weight + r_row['TS_PCT']      * regular_weight,
            }
            rows.append(blended)

    blended_df = pd.DataFrame(rows)
    blended_df = blended_df.merge(team_ids, on='TEAM_ID')

    # Attach to games
    adv_home = blended_df.rename(columns={
        'TEAM_ID': 'TEAM_ID_home', 'SEASON_ID': 'SEASON_ID_home',
        'OFF_RATING': 'off_rtg_home', 'DEF_RATING': 'def_rtg_home',
        'NET_RATING': 'net_rtg_home', 'PACE': 'pace_home', 'TS_PCT': 'ts_pct_home'
    })
    adv_away = blended_df.rename(columns={
        'TEAM_ID': 'TEAM_ID_away', 'SEASON_ID': 'SEASON_ID_away',
        'OFF_RATING': 'off_rtg_away', 'DEF_RATING': 'def_rtg_away',
        'NET_RATING': 'net_rtg_away', 'PACE': 'pace_away', 'TS_PCT': 'ts_pct_away'
    })

    m = merged.merge(adv_home[['TEAM_ID_home','SEASON_ID_home','off_rtg_home',
                                'def_rtg_home','net_rtg_home','pace_home','ts_pct_home']],
                     on=['SEASON_ID_home','TEAM_ID_home'], how='left')
    m = m.merge(adv_away[['TEAM_ID_away','SEASON_ID_away','off_rtg_away',
                           'def_rtg_away','net_rtg_away','pace_away','ts_pct_away']],
                on=['SEASON_ID_away','TEAM_ID_away'], how='left')

    m['off_rtg_diff']    = m['off_rtg_home']   - m['off_rtg_away']
    m['def_rtg_diff']    = m['def_rtg_home']   - m['def_rtg_away']
    m['net_rtg_diff']    = m['net_rtg_home']   - m['net_rtg_away']
    m['pace_diff']       = m['pace_home']      - m['pace_away']
    m['ts_pct_diff']     = m['ts_pct_home']    - m['ts_pct_away']
    m['rest_diff']       = m['REST_DAYS_home'] - m['REST_DAYS_away']
    m['last15_win_diff'] = m['last15_win_home']- m['last15_win_away']
    m['last15_pm_diff']  = m['last15_pm_home'] - m['last15_pm_away']
    m['home_court']      = 1
    m['fg3_diff']        = m['FG3_PCT_home']   - m['FG3_PCT_away']
    m['reb_diff']        = m['REB_home']       - m['REB_away']
    m['ast_diff']        = m['AST_home']       - m['AST_away']
    m['tov_diff']        = m['TOV_home']       - m['TOV_away']
    m['stl_diff']        = m['STL_home']       - m['STL_away']
    m['win_pct_diff']    = m['last15_win_home']- m['last15_win_away']
    m['home_win']        = (m['WL_home'] == 'W').astype(int)

    feat = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
            'home_court','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','win_pct_diff']

    df = m[feat + ['home_win']].dropna()
    split = int(len(df) * 0.8)
    X_test = df[feat].iloc[split:]
    y_test = df['home_win'].iloc[split:]

    if len(X_test) == 0:
        continue

    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Playoff {int(playoff_weight*100)}% / Regular {int(regular_weight*100)}%  →  Accuracy: {acc:.1%}  AUC: {auc:.3f}")

    if auc > best_auc:
        best_auc = auc
        best_blend = playoff_weight

print()
print(f"Best blend: {int(best_blend*100)}% playoff / {int((1-best_blend)*100)}% regular season (AUC: {best_auc:.3f})")