import streamlit as st
import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

games = pd.read_csv("playoff_games.csv")
regular = pd.read_csv("regular_season_stats.csv")
games['WIN'] = (games['WL'] == 'W').astype(int)

def convert_season_id(sid):
    sid = str(sid)
    if '-' in sid:
        return sid
    year = int(sid[-4:])
    return str(year) + "-" + str(year + 1)[2:]

regular['SEASON_ID'] = regular['SEASON_ID'].apply(convert_season_id)

team_avgs = games.groupby('TEAM_ABBREVIATION').agg(
    avg_fg3=('FG3_PCT', 'mean'),
    avg_reb=('REB', 'mean'),
    avg_ast=('AST', 'mean'),
    avg_tov=('TOV', 'mean'),
    avg_stl=('STL', 'mean'),
    win_pct=('WIN', 'mean'),
).reset_index()

team_ids = games[['TEAM_ID','TEAM_ABBREVIATION']].drop_duplicates()
latest_advanced = regular.sort_values('SEASON_ID').groupby('TEAM_ID').last().reset_index()
latest_advanced = latest_advanced.merge(team_ids, on='TEAM_ID')

games['last15_win'] = games.groupby('TEAM_ID')['WIN'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=1).mean()
)
games['last15_pm'] = games.groupby('TEAM_ID')['PLUS_MINUS'].transform(
    lambda x: x.shift(1).rolling(15, min_periods=1).mean()
)
last15 = games.groupby('TEAM_ABBREVIATION').agg(
    last15_win=('last15_win', 'last'),
    last15_pm=('last15_pm', 'last')
).reset_index()

teams = sorted(team_avgs['TEAM_ABBREVIATION'].tolist())

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="centered")
st.title("🏀 NBA Playoff Predictor")
st.markdown("Predict any playoff matchup using ML trained on 12 years of NBA data — advanced stats, rolling form, and live injuries.")
st.divider()

page = st.sidebar.radio("Navigate", ["Game Predictor", "2026 Bracket"])

if page == "Game Predictor":
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
            ha = latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == home_team]
            aa = latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == away_team]
            hl = last15[last15['TEAM_ABBREVIATION'] == home_team]
            al = last15[last15['TEAM_ABBREVIATION'] == away_team]

            if ha.empty or aa.empty:
                st.error("Advanced stats not found for one of these teams.")
            else:
                ha = ha.iloc[0]
                aa = aa.iloc[0]
                hl = hl.iloc[0] if not hl.empty else None
                al = al.iloc[0] if not al.empty else None

                game = {
                    'off_rtg_diff':    ha['OFF_RATING']  - aa['OFF_RATING'],
                    'def_rtg_diff':    ha['DEF_RATING']  - aa['DEF_RATING'],
                    'net_rtg_diff':    ha['NET_RATING']  - aa['NET_RATING'],
                    'pace_diff':       ha['PACE']        - aa['PACE'],
                    'ts_pct_diff':     ha['TS_PCT']      - aa['TS_PCT'],
                    'rest_diff':       home_rest         - away_rest,
                    'last15_win_diff': (hl['last15_win'] - al['last15_win']) if hl is not None and al is not None else 0,
                    'last15_pm_diff':  (hl['last15_pm']  - al['last15_pm'])  if hl is not None and al is not None else 0,
                    'fg3_diff':        h['avg_fg3']      - a['avg_fg3'],
                    'reb_diff':        h['avg_reb']      - a['avg_reb'],
                    'ast_diff':        h['avg_ast']      - a['avg_ast'],
                    'tov_diff':        h['avg_tov']      - a['avg_tov'],
                    'stl_diff':        h['avg_stl']      - a['avg_stl'],
                    'win_pct_diff':    h['win_pct']      - a['win_pct'],
                    'home_court':      1
                }

                features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
                            'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
                            'fg3_diff','reb_diff','ast_diff','tov_diff','stl_diff',
                            'win_pct_diff','home_court']

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

                st.divider()
                st.subheader("📊 Key stats comparison")
                stats_df = pd.DataFrame({
                    'Stat': ['Off Rating','Def Rating','Net Rating','Pace','TS%','Win %'],
                    home_team: [
                        round(ha['OFF_RATING'],1), round(ha['DEF_RATING'],1),
                        round(ha['NET_RATING'],1), round(ha['PACE'],1),
                        str(round(ha['TS_PCT']*100,1))+"%",
                        str(round(h['win_pct']*100,1))+"%"
                    ],
                    away_team: [
                        round(aa['OFF_RATING'],1), round(aa['DEF_RATING'],1),
                        round(aa['NET_RATING'],1), round(aa['PACE'],1),
                        str(round(aa['TS_PCT']*100,1))+"%",
                        str(round(a['win_pct']*100,1))+"%"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

elif page == "2026 Bracket":
    st.switch_page("pages/bracket.py") if hasattr(st, 'switch_page') else st.info("Run bracket.py separately with: streamlit run bracket.py")