import streamlit as st
import pandas as pd
import pickle
import random
regular = pd.read_csv("regular_season_stats.csv")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

games = pd.read_csv("playoff_games.csv")
advanced = pd.read_csv("advanced_stats.csv")
games['WIN'] = (games['WL'] == 'W').astype(int)

team_avgs = games.groupby('TEAM_ABBREVIATION').agg(
    avg_fg3=('FG3_PCT', 'mean'),
    avg_reb=('REB', 'mean'),
    avg_ast=('AST', 'mean'),
    avg_tov=('TOV', 'mean'),
    avg_stl=('STL', 'mean'),
    win_pct=('WIN', 'mean'),
).reset_index()

regular = pd.read_csv("regular_season_stats.csv")

team_ids = games[['TEAM_ID','TEAM_ABBREVIATION']].drop_duplicates()

# Get latest season for each team from both sources
latest_advanced = pd.read_csv("regular_season_stats.csv")
latest_advanced = latest_advanced.sort_values('SEASON_ID').groupby('TEAM_ID').last().reset_index()
latest_advanced = latest_advanced.merge(team_ids, on='TEAM_ID')

# Blend: 40% playoff history, 60% regular season (more data = mor

def predict_series(home, away, n_simulations=1000):
    ha_row = latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == home]
    aa_row = latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == away]
    h_avg = team_avgs[team_avgs['TEAM_ABBREVIATION'] == home]
    a_avg = team_avgs[team_avgs['TEAM_ABBREVIATION'] == away]

    if ha_row.empty or aa_row.empty or h_avg.empty or a_avg.empty:
        return home, 4, 0, 0, 0.5, 0.5

    ha = ha_row.iloc[0]
    aa = aa_row.iloc[0]
    h = h_avg.iloc[0]
    a = a_avg.iloc[0]

    game = {
        'off_rtg_diff':    ha['OFF_RATING']  - aa['OFF_RATING'],
        'def_rtg_diff':    ha['DEF_RATING']  - aa['DEF_RATING'],
        'net_rtg_diff':    ha['NET_RATING']  - aa['NET_RATING'],
        'pace_diff':       ha['PACE']        - aa['PACE'],
        'ts_pct_diff':     ha['TS_PCT']      - aa['TS_PCT'],
        'rest_diff':       0,
        'last15_win_diff': h['win_pct']      - a['win_pct'],
        'last15_pm_diff':  0,
        'home_court':      1,
        'fg3_diff':        h['avg_fg3']      - a['avg_fg3'],
        'reb_diff':        h['avg_reb']      - a['avg_reb'],
        'ast_diff':        h['avg_ast']      - a['avg_ast'],
        'tov_diff':        h['avg_tov']      - a['avg_tov'],
        'stl_diff':        h['avg_stl']      - a['avg_stl'],
        'win_pct_diff':    h['win_pct']      - a['win_pct'],
    }

    features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
                'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
                'home_court','fg3_diff','reb_diff','ast_diff',
                'tov_diff','stl_diff','win_pct_diff']

    X = pd.DataFrame([game])[features]
    home_win_prob = model.predict_proba(X)[0][1]

    # Monte Carlo — simulate series 1000 times
    home_series_wins = 0
    total_games_list = []

    for sim in range(n_simulations):
        hw = 0
        aw = 0
        game_num = 0
        while hw < 4 and aw < 4:
            game_num += 1
            if game_num in [1, 2, 5, 7]:
                prob = home_win_prob
            else:
                prob = 1 - home_win_prob
            if random.random() < prob:
                hw += 1
            else:
                aw += 1
        if hw == 4:
            home_series_wins += 1
        total_games_list.append(hw + aw)

    home_series_prob = home_series_wins / n_simulations
    avg_games = sum(total_games_list) / len(total_games_list)

    # Pick winner based on series probability
    if home_series_prob >= 0.5:
        winner = home
        win_prob = home_series_prob
    else:
        winner = away
        win_prob = 1 - home_series_prob

    return winner, round(avg_games, 1), 0, 0, home_win_prob, win_prob
st.set_page_config(page_title="2026 NBA Playoff Bracket", page_icon="🏀", layout="wide")
st.title("🏀 2026 NBA Playoff Bracket Predictor")
st.markdown("ML-powered bracket prediction using advanced stats, net rating, and rolling form.")
st.divider()

# First round matchups
west_matchups = [
    ("OKC", "PHX"),
    ("SAS", "POR"),
    ("DEN", "MIN"),
    ("LAL", "HOU"),
]
east_matchups = [
    ("DET", "ORL"),
    ("BOS", "PHI"),
    ("NYK", "TOR"),
    ("CLE", "ATL"),
]

def run_round(matchups):
    results = []
    for home, away in matchups:
        result = predict_series(home, away)
        winner = result[0]
        loser = away if winner == home else home
        results.append({
            'home': home,
            'away': away,
            'winner': winner,
            'loser': loser,
            'result': result
        })
    return results

def display_round(results, round_name):
    st.subheader(round_name)
    cols = st.columns(len(results))
    winners = []
    for i, r in enumerate(results):
        with cols[i]:
            winner, avg_games, hw, aw, game_prob, series_prob = r['result']
            st.markdown(f"**{r['home']} vs {r['away']}**")
            st.success(f"🏆 {winner}")
            st.metric("Series win probability", str(round(series_prob * 100, 1)) + "%")
            st.caption(f"Avg series length: {avg_games} games | Per-game prob: {round(game_prob*100,1)}%")
            if series_prob >= 0.9:
                st.progress(1.0)
                st.caption("🔒 Very likely")
            elif series_prob >= 0.7:
                st.progress(series_prob)
                st.caption("📈 Favored")
            else:
                st.progress(series_prob)
                st.caption("⚠️ Toss-up")
        winners.append(winner)
    return winners
# WEST
st.header("🟠 Western Conference")
west_r1 = run_round(west_matchups)
west_r1_winners = display_round(west_r1, "First Round")

# Reseed: pair 1st remaining seed vs 4th, 2nd vs 3rd
west_seeds = {
    "OKC": 1, "SAS": 2, "DEN": 3, "LAL": 4,
    "HOU": 5, "MIN": 6, "POR": 7, "PHX": 8
}
east_seeds = {
    "DET": 1, "BOS": 2, "NYK": 3, "CLE": 4,
    "ATL": 5, "TOR": 6, "PHI": 7, "ORL": 8
}

def reseed(winners, seed_dict):
    sorted_winners = sorted(winners, key=lambda x: seed_dict.get(x, 9))
    if len(sorted_winners) == 2:
        return [(sorted_winners[0], sorted_winners[1])]
    return [(sorted_winners[0], sorted_winners[3]),
            (sorted_winners[1], sorted_winners[2])]

west_semi_matchups = reseed(west_r1_winners, west_seeds)
west_r2 = run_round(west_semi_matchups)
west_r2_winners = display_round(west_r2, "Conference Semifinals")

west_final_matchup = reseed(west_r2_winners, west_seeds)
west_r3 = run_round(west_final_matchup)
west_r3_winners = display_round(west_r3, "Conference Finals")
west_champion = west_r3_winners[0]

st.divider()

# EAST
st.header("🔵 Eastern Conference")
east_r1 = run_round(east_matchups)
east_r1_winners = display_round(east_r1, "First Round")

east_semi_matchups = reseed(east_r1_winners, east_seeds)

east_r2 = run_round(east_semi_matchups)
east_r2_winners = display_round(east_r2, "Conference Semifinals")

east_final_matchup = reseed(east_r2_winners, east_seeds)
east_r3 = run_round(east_final_matchup)
east_r3_winners = display_round(east_r3, "Conference Finals")
east_champion = east_r3_winners[0]

st.divider()

# NBA Finals
st.header("🏆 NBA Finals")
finals = run_round([(west_champion, east_champion)])
finals_winners = display_round(finals, "NBA Finals")

st.divider()
st.balloons()
st.markdown(f"## 🎉 Predicted NBA Champion: **{finals_winners[0]}**")