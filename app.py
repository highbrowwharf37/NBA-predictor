import streamlit as st
import pandas as pd
import pickle

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

# Page config
st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="centered")

st.title("🏀 NBA Playoff Predictor")
st.markdown("Predict the winner of any playoff matchup using a machine learning model trained on 11 years of NBA data.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select home team", teams, index=teams.index("BOS") if "BOS" in teams else 0)
    home_rest = st.slider("Rest days", 0, 7, 2, key="home_rest")

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox("Select away team", teams, index=teams.index("NYK") if "NYK" in teams else 1)
    away_rest = st.slider("Rest days", 0, 7, 1, key="away_rest")

st.divider()

if st.button("🔮 Predict Winner", use_container_width=True):
    if home_team == away_team:
        st.error("Please select two different teams!")
    else:
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
        away_prob = 1 - prob

        st.subheader("Prediction")

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