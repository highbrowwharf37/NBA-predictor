import pandas as pd
import pickle
from player_data import get_team_summary
from injury_data import get_injury_report, is_star_injured

# Load base model and team averages
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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

print("NBA Playoff Predictor v2 — with live player data & injuries")
print("=" * 60)

home_team = input("Enter HOME team abbreviation (e.g. BOS): ").strip().upper()
away_team = input("Enter AWAY team abbreviation (e.g. NYK): ").strip().upper()
home_rest = float(input("Home team rest days: "))
away_rest = float(input("Away team rest days: "))

# Fetch live player stats
print("\nFetching live player data...")
home_players = get_team_summary(home_team)
away_players = get_team_summary(away_team)

# Fetch injury report
print("\nFetching injury report...")
injury_df = get_injury_report()

# Base team stats
h = team_avgs[team_avgs['TEAM_ABBREVIATION'] == home_team].iloc[0]
a = team_avgs[team_avgs['TEAM_ABBREVIATION'] == away_team].iloc[0]

# Player level features
home_top5_pts = home_players['top5_pts'] if home_players else h['avg_pts'] * 5
away_top5_pts = away_players['top5_pts'] if away_players else a['avg_pts'] * 5
home_star_pts = home_players['star_pts'] if home_players else 0
away_star_pts = away_players['star_pts'] if away_players else 0
home_star_name = home_players['star_name'] if home_players else "Unknown"
away_star_name = away_players['star_name'] if away_players else "Unknown"

# Injury flags
home_star_out = is_star_injured(home_team, home_star_name, injury_df)
away_star_out = is_star_injured(away_team, away_star_name, injury_df)

# Adjust star pts if injured
if home_star_out:
    print("⚠️  " + home_star_name + " is OUT for " + home_team)
    home_star_pts *= 0.1
    home_top5_pts -= home_star_pts * 0.9

if away_star_out:
    print("⚠️  " + away_star_name + " is OUT for " + away_team)
    away_star_pts *= 0.1
    away_top5_pts -= away_star_pts * 0.9

# Build features
game = {
    'pts_diff':       h['avg_pts']         - a['avg_pts'],
    'fg_diff':        h['avg_fg']          - a['avg_fg'],
    'fg3_diff':       h['avg_fg3']         - a['avg_fg3'],
    'reb_diff':       h['avg_reb']         - a['avg_reb'],
    'ast_diff':       h['avg_ast']         - a['avg_ast'],
    'tov_diff':       h['avg_tov']         - a['avg_tov'],
    'stl_diff':       h['avg_stl']         - a['avg_stl'],
    'pm_diff':        h['avg_pm']          - a['avg_pm'],
    'rest_diff':      home_rest            - away_rest,
    'win_pct_diff':   h['win_pct']         - a['win_pct'],
    'def_diff':       a['avg_pts_allowed'] - h['avg_pts_allowed'],
    'home_court':     1
}

features = ['pts_diff','fg_diff','fg3_diff','reb_diff','ast_diff',
            'tov_diff','stl_diff','pm_diff','rest_diff',
            'win_pct_diff','def_diff','home_court']

X = pd.DataFrame([game])[features]
prob = model.predict_proba(X)[0][1]

# Adjust probability for player-level data
top5_diff = home_top5_pts - away_top5_pts
star_diff  = home_star_pts - away_star_pts

# Small nudge based on player data (cap at 8%)
player_adjustment = max(-0.08, min(0.08, top5_diff * 0.002 + star_diff * 0.001))
prob = max(0.05, min(0.95, prob + player_adjustment))
away_prob = 1 - prob

# Print results
print("\n" + "=" * 60)
print(home_team + " (home) vs " + away_team + " (away)")
print("=" * 60)

if home_players:
    print("\n" + home_team + " top players (last 5 playoff games):")
    for p in home_players['players']:
        print("  " + p['name'] + " — " + str(round(p['avg_pts'],1)) + " pts, " + str(round(p['avg_reb'],1)) + " reb, " + str(round(p['avg_ast'],1)) + " ast")

if away_players:
    print("\n" + away_team + " top players (last 5 playoff games):")
    for p in away_players['players']:
        print("  " + p['name'] + " — " + str(round(p['avg_pts'],1)) + " pts, " + str(round(p['avg_reb'],1)) + " reb, " + str(round(p['avg_ast'],1)) + " ast")

print("\n" + home_team + " win probability: " + str(round(prob * 100, 1)) + "%")
print(away_team + " win probability:  " + str(round(away_prob * 100, 1)) + "%")

if prob >= 0.5:
    print("\n🏆 Predicted winner: " + home_team)
else:
    print("\n🏆 Predicted winner: " + away_team)
print("=" * 60)