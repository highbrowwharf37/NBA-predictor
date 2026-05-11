import streamlit as st
import pandas as pd
import pickle
import random

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")

# Load data
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

# Use only current season stats for predictions
current_season = regular[regular['SEASON_ID'] == '2025-26'].copy()
current_season = current_season.merge(team_ids, on='TEAM_ID')

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

features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','rest_diff','last15_win_diff','last15_pm_diff',
            'fg3_diff','reb_diff','ast_diff','tov_diff','stl_diff',
            'win_pct_diff','home_court']

def get_game_features(home_team, away_team, home_rest, away_rest):
    h = team_avgs[team_avgs['TEAM_ABBREVIATION'] == home_team].iloc[0]
    a = team_avgs[team_avgs['TEAM_ABBREVIATION'] == away_team].iloc[0]
    ha = latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == home_team].iloc[0]
    aa = latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == away_team].iloc[0]
    hl = last15[last15['TEAM_ABBREVIATION'] == home_team]
    al = last15[last15['TEAM_ABBREVIATION'] == away_team]
    hl = hl.iloc[0] if not hl.empty else None
    al = al.iloc[0] if not al.empty else None

    return {
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
        'win_pct_diff':    ha['W_PCT']       - aa['W_PCT'],
        'home_court':      1
    }, ha, aa, h, a

def predict_series(home, away, n_simulations=1000):
    if latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == home].empty or \
       latest_advanced[latest_advanced['TEAM_ABBREVIATION'] == away].empty:
        return home, 4.0, 0.5, 0.5

    game, ha, aa, h, a = get_game_features(home, away, 2, 2)
    X = pd.DataFrame([game])[features]
    home_win_prob = model.predict_proba(X)[0][1]

    home_series_wins = 0
    total_games_list = []

    for _ in range(n_simulations):
        hw, aw, gn = 0, 0, 0
        while hw < 4 and aw < 4:
            gn += 1
            prob = home_win_prob if gn in [1,2,5,7] else 1 - home_win_prob
            if random.random() < prob:
                hw += 1
            else:
                aw += 1
        if hw == 4:
            home_series_wins += 1
        total_games_list.append(hw + aw)

    series_prob = home_series_wins / n_simulations
    avg_games = sum(total_games_list) / len(total_games_list)
    winner = home if series_prob >= 0.5 else away
    win_prob = series_prob if winner == home else 1 - series_prob
    return winner, round(avg_games, 1), win_prob, home_win_prob

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .big-title { font-size: 3rem; font-weight: 800; color: #FF6B35; text-align: center; margin-bottom: 0; }
    .subtitle { font-size: 1.1rem; color: #888; text-align: center; margin-bottom: 2rem; }
    .team-card { background: #1a1f2e; border-radius: 12px; padding: 1.5rem; border: 1px solid #2d3748; }
    .winner-box { background: linear-gradient(135deg, #1a1f2e, #0d1b2a); border: 2px solid #FF6B35; border-radius: 12px; padding: 2rem; text-align: center; }
    .prob-text { font-size: 3rem; font-weight: 800; color: #FF6B35; }
    .series-card { background: #1a1f2e; border-radius: 10px; padding: 1rem; border: 1px solid #2d3748; margin: 0.5rem 0; text-align: center; }
    .nav-btn { width: 100%; }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #FF6B35; }
</style>
""", unsafe_allow_html=True)

# Navigation
st.markdown('<p class="big-title">🏀 NBA Playoff Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML-powered predictions trained on 12 years of NBA data</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 Game Predictor", "🏆 2026 Bracket"])

# ── TAB 1: GAME PREDICTOR ──
with tab1:
    st.markdown("### Pick your matchup")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏠 Home Team")
        home_team = st.selectbox("", teams, index=teams.index("BOS") if "BOS" in teams else 0, key="home_sel")
        home_rest = st.slider("Rest days", 0, 7, 2, key="home_rest")

    with col2:
        st.markdown("#### ✈️ Away Team")
        away_team = st.selectbox("", teams, index=teams.index("NYK") if "NYK" in teams else 1, key="away_sel")
        away_rest = st.slider("Rest days", 0, 7, 1, key="away_rest")

    st.markdown("")
    predict_btn = st.button("🔮 Predict Winner", use_container_width=True, type="primary")

    if predict_btn:
        if home_team == away_team:
            st.error("Please select two different teams!")
        else:
            try:
                game, ha, aa, h, a = get_game_features(home_team, away_team, home_rest, away_rest)
                X = pd.DataFrame([game])[features]
                prob = model.predict_proba(X)[0][1]
                away_prob = 1 - prob

                st.markdown("---")
                col3, col4, col5 = st.columns([2,1,2])

                with col3:
                    st.markdown(f"""
                    <div class="winner-box">
                        <h2>{home_team}</h2>
                        <p style="color:#888">Home</p>
                        <p class="prob-text">{round(prob*100,1)}%</p>
                    </div>""", unsafe_allow_html=True)

                with col4:
                    st.markdown("<br><br><h2 style='text-align:center;color:#888'>VS</h2>", unsafe_allow_html=True)

                with col5:
                    st.markdown(f"""
                    <div class="winner-box">
                        <h2>{away_team}</h2>
                        <p style="color:#888">Away</p>
                        <p class="prob-text">{round(away_prob*100,1)}%</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")
                winner = home_team if prob >= 0.5 else away_team
                st.success(f"🏆 Predicted Winner: **{winner}**")
                st.progress(prob)
                st.caption("Bar shows home team win probability")

                st.markdown("---")
                st.markdown("### 📊 Stats Comparison")
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

            except Exception as e:
                st.error("Could not find stats for one of these teams: " + str(e))

# ── TAB 2: 2026 BRACKET ──
with tab2:
    st.markdown("### 🏆 2026 NBA Playoff Bracket")
    st.caption("Monte Carlo simulation — 1,000 series simulations per matchup")

    west_matchups = [("OKC","PHX"),("SAS","POR"),("DEN","MIN"),("LAL","HOU")]
    east_matchups = [("DET","ORL"),("BOS","PHI"),("NYK","TOR"),("CLE","ATL")]

    west_seeds = {"OKC":1,"SAS":2,"DEN":3,"LAL":4,"HOU":5,"MIN":6,"POR":7,"PHX":8}
    east_seeds = {"DET":1,"BOS":2,"NYK":3,"CLE":4,"ATL":5,"TOR":6,"PHI":7,"ORL":8}

    def reseed(winners, seed_dict):
        sorted_w = sorted(winners, key=lambda x: seed_dict.get(x, 9))
        if len(sorted_w) == 2:
            return [(sorted_w[0], sorted_w[1])]
        return [(sorted_w[0], sorted_w[3]), (sorted_w[1], sorted_w[2])]

    def display_bracket_round(matchups, round_name):
        st.markdown(f"#### {round_name}")
        cols = st.columns(len(matchups))
        winners = []
        for i, (home, away) in enumerate(matchups):
            with cols[i]:
                with st.spinner(f"{home} vs {away}..."):
                    winner, avg_games, win_prob, game_prob = predict_series(home, away)
                loser = away if winner == home else home
                color = "#FF6B35" if win_prob >= 0.7 else "#FFA500" if win_prob >= 0.55 else "#888"
                st.markdown(f"""
                <div class="series-card">
                    <p style="color:#888;font-size:0.8rem">{home} vs {away}</p>
                    <p style="font-size:1.3rem;font-weight:800;color:{color}">🏆 {winner}</p>
                    <p style="color:#888;font-size:0.85rem">{round(win_prob*100,1)}% series win</p>
                    <p style="color:#555;font-size:0.8rem">~{avg_games} games</p>
                    <p style="color:#444;font-size:0.75rem">❌ {loser}</p>
                </div>""", unsafe_allow_html=True)
            winners.append(winner)
        return winners

    if st.button("🎲 Simulate Full Bracket", use_container_width=True, type="primary"):
        st.markdown("---")
        col_west, col_east = st.columns(2)

        with col_west:
            st.markdown("### 🟠 Western Conference")
            w1 = display_bracket_round(west_matchups, "First Round")
            w2 = display_bracket_round(reseed(w1, west_seeds), "Semifinals")
            w3 = display_bracket_round(reseed(w2, west_seeds), "Conference Finals")
            west_champ = w3[0]
            st.markdown(f"<h3 style='color:#FF6B35;text-align:center'>🏆 West: {west_champ}</h3>", unsafe_allow_html=True)

        with col_east:
            st.markdown("### 🔵 Eastern Conference")
            e1 = display_bracket_round(east_matchups, "First Round")
            e2 = display_bracket_round(reseed(e1, east_seeds), "Semifinals")
            e3 = display_bracket_round(reseed(e2, east_seeds), "Conference Finals")
            east_champ = e3[0]
            st.markdown(f"<h3 style='color:#4A90D9;text-align:center'>🏆 East: {east_champ}</h3>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏆 NBA Finals")
        finals_winner, avg_games, win_prob, _ = predict_series(west_champ, east_champ)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1f2e,#0d1b2a);border:2px solid gold;
        border-radius:16px;padding:2rem;text-align:center;margin-top:1rem">
            <h2 style="color:gold">🏆 NBA Champion</h2>
            <h1 style="color:white;font-size:3rem">{finals_winner}</h1>
            <p style="color:#888">{west_champ} vs {east_champ} — {round(win_prob*100,1)}% series win probability</p>
        </div>""", unsafe_allow_html=True)
        st.balloons()