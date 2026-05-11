import streamlit as st
import pandas as pd
import pickle
import random

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")

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
team_ids = games[['TEAM_ID','TEAM_ABBREVIATION']].drop_duplicates()

# Current season stats only for predictions
current = regular[regular['SEASON_ID'] == '2025-26'].merge(team_ids, on='TEAM_ID')

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

teams = sorted(current['TEAM_ABBREVIATION'].tolist())

features = ['off_rtg_diff','def_rtg_diff','net_rtg_diff','pace_diff',
            'ts_pct_diff','w_pct_diff','ast_pct_diff','oreb_pct_diff',
            'dreb_pct_diff','tov_pct_diff','efg_pct_diff',
            'rest_diff','last15_win_diff','last15_pm_diff','home_court']

def get_features(home, away, home_rest, away_rest):
    h = current[current['TEAM_ABBREVIATION'] == home].iloc[0]
    a = current[current['TEAM_ABBREVIATION'] == away].iloc[0]
    hl = last15[last15['TEAM_ABBREVIATION'] == home]
    al = last15[last15['TEAM_ABBREVIATION'] == away]
    hl = hl.iloc[0] if not hl.empty else None
    al = al.iloc[0] if not al.empty else None

    return {
        'off_rtg_diff':  h['OFF_RATING']  - a['OFF_RATING'],
        'def_rtg_diff':  h['DEF_RATING']  - a['DEF_RATING'],
        'net_rtg_diff':  h['NET_RATING']  - a['NET_RATING'],
        'pace_diff':     h['PACE']        - a['PACE'],
        'ts_pct_diff':   h['TS_PCT']      - a['TS_PCT'],
        'w_pct_diff':    h['W_PCT']       - a['W_PCT'],
        'ast_pct_diff':  h['AST_PCT']     - a['AST_PCT'],
        'oreb_pct_diff': h['OREB_PCT']    - a['OREB_PCT'],
        'dreb_pct_diff': h['DREB_PCT']    - a['DREB_PCT'],
        'tov_pct_diff':  h['TM_TOV_PCT']  - a['TM_TOV_PCT'],
        'efg_pct_diff':  h['EFG_PCT']     - a['EFG_PCT'],
        'rest_diff':     home_rest        - away_rest,
        'last15_win_diff': (hl['last15_win'] - al['last15_win']) if hl is not None and al is not None else 0,
        'last15_pm_diff':  (hl['last15_pm']  - al['last15_pm'])  if hl is not None and al is not None else 0,
        'home_court': 1
    }, h, a

def predict_series(home, away, n=1000):
    try:
        game, h, a = get_features(home, away, 2, 2)
        X = pd.DataFrame([game])[features]
        home_win_prob = model.predict_proba(X)[0][1]
        home_wins = 0
        game_counts = []
        for _ in range(n):
            hw, aw, gn = 0, 0, 0
            while hw < 4 and aw < 4:
                gn += 1
                prob = home_win_prob if gn in [1,2,5,7] else 1 - home_win_prob
                if random.random() < prob:
                    hw += 1
                else:
                    aw += 1
            if hw == 4:
                home_wins += 1
            game_counts.append(hw + aw)
        series_prob = home_wins / n
        avg_games = sum(game_counts) / len(game_counts)
        winner = home if series_prob >= 0.5 else away
        win_prob = series_prob if winner == home else 1 - series_prob
        return winner, round(avg_games, 1), win_prob, home_win_prob
    except:
        return home, 6.0, 0.5, 0.5

# CSS
st.markdown("""
<style>
    .big-title { font-size: 3rem; font-weight: 800; color: #FF6B35; text-align: center; }
    .subtitle { font-size: 1.1rem; color: #888; text-align: center; margin-bottom: 2rem; }
    .prob-text { font-size: 3rem; font-weight: 800; color: #FF6B35; }
    .team-box { background: #1a1f2e; border: 2px solid #FF6B35; border-radius: 12px; padding: 2rem; text-align: center; }
    .series-card { background: #1a1f2e; border-radius: 10px; padding: 1rem; border: 1px solid #2d3748; text-align: center; margin: 0.3rem 0; }
    .champ-box { background: linear-gradient(135deg, #1a1f2e, #0d1b2a); border: 3px solid gold; border-radius: 16px; padding: 2rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🏀 NBA Playoff Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML predictions using 2025-26 season stats — advanced metrics, rolling form, Monte Carlo simulation</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 Game Predictor", "🏆 2026 Bracket"])

# ── TAB 1 ──
with tab1:
    st.markdown("### Pick your matchup")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏠 Home Team")
        home_team = st.selectbox("", teams, index=teams.index("OKC") if "OKC" in teams else 0, key="home_sel")
        home_rest = st.slider("Rest days", 0, 7, 2, key="home_rest")
    with col2:
        st.markdown("#### ✈️ Away Team")
        away_team = st.selectbox("", teams, index=teams.index("BOS") if "BOS" in teams else 1, key="away_sel")
        away_rest = st.slider("Rest days", 0, 7, 2, key="away_rest")

    if st.button("🔮 Predict Winner", use_container_width=True, type="primary"):
        if home_team == away_team:
            st.error("Please select two different teams!")
        else:
            try:
                game, h, a = get_features(home_team, away_team, home_rest, away_rest)
                X = pd.DataFrame([game])[features]
                prob = model.predict_proba(X)[0][1]
                away_prob = 1 - prob

                st.markdown("---")
                col3, col4, col5 = st.columns([2,1,2])
                with col3:
                    st.markdown(f"""<div class="team-box">
                        <h2>{home_team}</h2><p style="color:#888">Home</p>
                        <p class="prob-text">{round(prob*100,1)}%</p>
                    </div>""", unsafe_allow_html=True)
                with col4:
                    st.markdown("<br><br><h2 style='text-align:center;color:#888'>VS</h2>", unsafe_allow_html=True)
                with col5:
                    st.markdown(f"""<div class="team-box">
                        <h2>{away_team}</h2><p style="color:#888">Away</p>
                        <p class="prob-text">{round(away_prob*100,1)}%</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")
                winner = home_team if prob >= 0.5 else away_team
                st.success(f"🏆 Predicted Winner: **{winner}**")
                st.progress(prob)
                st.caption("Bar shows home team win probability")

                st.markdown("---")
                st.markdown("### 📊 2025-26 Stats Comparison")
                stats_df = pd.DataFrame({
                    'Stat': ['Off Rating','Def Rating','Net Rating','Pace','TS%','EFG%','Win %'],
                    home_team: [
                        round(h['OFF_RATING'],1), round(h['DEF_RATING'],1),
                        round(h['NET_RATING'],1), round(h['PACE'],1),
                        str(round(h['TS_PCT']*100,1))+"%",
                        str(round(h['EFG_PCT']*100,1))+"%",
                        str(round(h['W_PCT']*100,1))+"%"
                    ],
                    away_team: [
                        round(a['OFF_RATING'],1), round(a['DEF_RATING'],1),
                        round(a['NET_RATING'],1), round(a['PACE'],1),
                        str(round(a['TS_PCT']*100,1))+"%",
                        str(round(a['EFG_PCT']*100,1))+"%",
                        str(round(a['W_PCT']*100,1))+"%"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error("Error: " + str(e))

# ── TAB 2 ──
with tab2:
    st.markdown("### 🏆 2026 NBA Playoff Bracket")
    st.caption("1,000 Monte Carlo simulations per series using 2025-26 season stats")

    west_matchups = [("OKC","PHX"),("SAS","POR"),("DEN","MIN"),("LAL","HOU")]
    east_matchups = [("DET","ORL"),("BOS","PHI"),("NYK","TOR"),("CLE","ATL")]
    west_seeds = {"OKC":1,"SAS":2,"DEN":3,"LAL":4,"HOU":5,"MIN":6,"POR":7,"PHX":8}
    east_seeds = {"DET":1,"BOS":2,"NYK":3,"CLE":4,"ATL":5,"TOR":6,"PHI":7,"ORL":8}

    def reseed(winners, seed_dict):
        s = sorted(winners, key=lambda x: seed_dict.get(x, 9))
        return [(s[0], s[1])] if len(s) == 2 else [(s[0], s[3]), (s[1], s[2])]

    def show_round(matchups, label):
        st.markdown(f"#### {label}")
        cols = st.columns(len(matchups))
        winners = []
        for i, (home, away) in enumerate(matchups):
            winner, avg_games, win_prob, _ = predict_series(home, away)
            loser = away if winner == home else home
            color = "#FF6B35" if win_prob >= 0.7 else "#FFA500" if win_prob >= 0.55 else "#aaa"
            with cols[i]:
                st.markdown(f"""<div class="series-card">
                    <p style="color:#888;font-size:0.8rem">{home} vs {away}</p>
                    <p style="font-size:1.2rem;font-weight:800;color:{color}">🏆 {winner}</p>
                    <p style="color:#888;font-size:0.8rem">{round(win_prob*100,1)}% in ~{avg_games}g</p>
                    <p style="color:#555;font-size:0.75rem">❌ {loser}</p>
                </div>""", unsafe_allow_html=True)
            winners.append(winner)
        return winners

    if st.button("🎲 Simulate Full Bracket", use_container_width=True, type="primary"):
        st.markdown("---")
        col_w, col_e = st.columns(2)

        with col_w:
            st.markdown("### 🟠 West")
            w1 = show_round(west_matchups, "First Round")
            w2 = show_round(reseed(w1, west_seeds), "Semifinals")
            w3 = show_round(reseed(w2, west_seeds), "Conference Finals")
            st.markdown(f"<h3 style='color:#FF6B35;text-align:center'>🏆 {w3[0]}</h3>", unsafe_allow_html=True)

        with col_e:
            st.markdown("### 🔵 East")
            e1 = show_round(east_matchups, "First Round")
            e2 = show_round(reseed(e1, east_seeds), "Semifinals")
            e3 = show_round(reseed(e2, east_seeds), "Conference Finals")
            st.markdown(f"<h3 style='color:#4A90D9;text-align:center'>🏆 {e3[0]}</h3>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏆 NBA Finals")
        champ, avg_g, wp, _ = predict_series(w3[0], e3[0])
        st.markdown(f"""<div class="champ-box">
            <h2 style="color:gold">🏆 NBA Champion</h2>
            <h1 style="color:white;font-size:3rem">{champ}</h1>
            <p style="color:#888">{w3[0]} vs {e3[0]} — {round(wp*100,1)}% series win probability</p>
        </div>""", unsafe_allow_html=True)
        st.balloons()