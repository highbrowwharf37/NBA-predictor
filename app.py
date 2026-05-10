import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and data
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

teams = sorted(team_avgs['TEAM_ABBREVIATION'].tolist())

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
            if (home_wins and home == team1) or (not home_wins and home == team2):
                t1_wins += 1
            else:
                t2_wins += 1
            if t1_wins == 4:
                series_wins[team1] += 1
                game_count_wins[game_num + 1] += 1
                break
            if t2_wins == 4:
                series_wins[team2] += 1
                game_count_wins[game_num + 1] += 1
                break

    return series_wins, game_count_wins, n_simulations

# Page config
st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="centered")
st.title("🏀 NBA Playoff Predictor")
st.markdown("Predict playoff outcomes using a machine learning model trained on 11 years of NBA data.")
st.divider()

tab1, tab2 = st.tabs(["Single Game", "Full Series"])

# ── Tab 1: Single Game ──
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select home team", teams, index=teams.index("BOS") if "BOS" in teams else 0, key="g_home")
        home_rest = st.slider("Rest days", 0, 7, 2, key="g_home_rest")
    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Select away team", teams, index=teams.index("NYK") if "NYK" in teams else 1, key="g_away")
        away_rest = st.slider("Rest days", 0, 7, 1, key="g_away_rest")

    if st.button("🔮 Predict Game", use_container_width=True):
        if home_team == away_team:
            st.error("Please select two different teams!")
        else:
            prob = get_win_prob(home_team, away_team, home_rest, away_rest)
            away_prob = 1 - prob
            col3, col4 = st.columns(2)
            with col3:
                st.metric(label=home_team + " (Home)", value=str(round(prob * 100, 1)) + "%")
            with col4:
                st.metric(label=away_team + " (Away)", value=str(round(away_prob * 100, 1)) + "%")
            if prob >= 0.5:
                st.success("🏆 Predicted winner: " + home_team)
            else:
                st.success("🏆 Predicted winner: " + away_team)
            st.progress(prob)
            st.caption("Bar shows home team win probability")

# ── Tab 2: Full Series ──
with tab2:
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("🏠 Home Court Team")
        s_team1 = st.selectbox("Select team with home court", teams, index=teams.index("BOS") if "BOS" in teams else 0, key="s_home")
    with col6:
        st.subheader("✈️ Away Team")
        s_team2 = st.selectbox("Select away team", teams, index=teams.index("NYK") if "NYK" in teams else 1, key="s_away")

    if st.button("🔮 Simulate Series", use_container_width=True):
        if s_team1 == s_team2:
            st.error("Please select two different teams!")
        else:
            with st.spinner("Simulating 10,000 series..."):
                series_wins, game_count_wins, n_sims = simulate_series(s_team1, s_team2)

            col7, col8 = st.columns(2)
            with col7:
                st.metric(s_team1 + " wins series", str(round(series_wins[s_team1] / n_sims * 100, 1)) + "%")
            with col8:
                st.metric(s_team2 + " wins series", str(round(series_wins[s_team2] / n_sims * 100, 1)) + "%")

            winner = s_team1 if series_wins[s_team1] > series_wins[s_team2] else s_team2
            st.success("🏆 Predicted series winner: " + winner)

            st.subheader("Series length probabilities")
            length_data = pd.DataFrame({
                'Games': [str(g) + ' games' for g in game_count_wins.keys()],
                'Probability': [round(v / n_sims * 100, 1) for v in game_count_wins.values()]
            })
            st.bar_chart(length_data.set_index('Games'))