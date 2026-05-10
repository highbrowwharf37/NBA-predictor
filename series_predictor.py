import pandas as pd
import pickle
import numpy as np

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

def get_win_prob(home_abbr, away_abbr, home_rest, away_rest):
    h = team_avgs[team_avgs['TEAM_ABBREVIATION'] == home_abbr].iloc[0]
    a = team_avgs[team_avgs['TEAM_ABBREVIATION'] == away_abbr].iloc[0]
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
    return model.predict_proba(X)[0][1]

def simulate_series(team1, team2, n_simulations=10000):
    # team1 has home court (games 1,2,5,7)
    # team2 has home court (games 3,4,6)
    home_court_schedule = [team1, team1, team2, team2, team1, team2, team1]

    series_wins = {team1: 0, team2: 0}
    game_count_wins = {4: 0, 5: 0, 6: 0, 7: 0}

    for _ in range(n_simulations):
        t1_wins = 0
        t2_wins = 0
        for game_num in range(7):
            home = home_court_schedule[game_num]
            away = team2 if home == team1 else team1
            p = get_win_prob(home, away, 2, 2)
            if np.random.random() < p:
                home_wins = True
            else:
                home_wins = False
            if home_wins and home == team1:
                t1_wins += 1
            elif not home_wins and home == team2:
                t1_wins += 1
            elif home_wins and home == team2:
                t2_wins += 1
            elif not home_wins and home == team1:
                t2_wins += 1

            if t1_wins == 4:
                series_wins[team1] += 1
                game_count_wins[game_num + 1] += 1
                break
            if t2_wins == 4:
                series_wins[team2] += 1
                game_count_wins[game_num + 1] += 1
                break

    print("\n" + "=" * 45)
    print("SERIES PREDICTION: " + team1 + " vs " + team2)
    print("=" * 45)
    print(team1 + " wins series: " + str(round(series_wins[team1] / n_simulations * 100, 1)) + "%")
    print(team2 + " wins series: " + str(round(series_wins[team2] / n_simulations * 100, 1)) + "%")
    print()
    print("Series length probabilities:")
    for games, count in game_count_wins.items():
        print("  " + str(games) + " games: " + str(round(count / n_simulations * 100, 1)) + "%")
    print("=" * 45)

print("Available teams:")
print(sorted(team_avgs['TEAM_ABBREVIATION'].tolist()))
print()

team1 = input("Enter team with HOME COURT (e.g. BOS): ").strip().upper()
team2 = input("Enter away team (e.g. NYK): ").strip().upper()

simulate_series(team1, team2)