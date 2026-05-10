import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelog, commonteamroster
from nba_api.stats.static import teams
import time

def get_team_id(abbreviation):
    all_teams = teams.get_teams()
    for t in all_teams:
        if t['abbreviation'] == abbreviation:
            return t['id']
    return None

def get_active_roster(team_abbr, season="2025-26"):
    team_id = get_team_id(team_abbr)
    if not team_id:
        print("Team not found: " + team_abbr)
        return []
    roster = commonteamroster.CommonTeamRoster(
        team_id=team_id,
        season=season
    )
    df = roster.get_data_frames()[0]
    time.sleep(1)
    return df[['PLAYER_ID', 'PLAYER', 'NUM', 'POSITION']].to_dict('records')

def get_player_recent_stats(player_id, season="2025-26", last_n=5):
    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Playoffs"
        )
        df = log.get_data_frames()[0]
        time.sleep(0.6)
        if len(df) == 0:
            # Fall back to regular season
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star="Regular Season"
            )
            df = log.get_data_frames()[0]
            time.sleep(0.6)
        if len(df) == 0:
            return None
        recent = df.head(last_n)
        return {
            'avg_pts': recent['PTS'].mean(),
            'avg_reb': recent['REB'].mean(),
            'avg_ast': recent['AST'].mean(),
            'avg_min': recent['MIN'].astype(float).mean() if recent['MIN'].dtype == object else recent['MIN'].mean(),
            'games': len(recent)
        }
    except:
        return None

def get_team_top_players(team_abbr, season="2025-26", top_n=5):
    print("Getting roster for " + team_abbr + "...")
    roster = get_active_roster(team_abbr, season)
    
    player_stats = []
    for player in roster:
        pid = player['PLAYER_ID']
        name = player['PLAYER']
        print("  Fetching " + name + "...")
        stats = get_player_recent_stats(pid, season)
        if stats:
            stats['name'] = name
            stats['player_id'] = pid
            player_stats.append(stats)
    
    if not player_stats:
        return []
    
    # Sort by average points and return top N
    player_stats.sort(key=lambda x: x['avg_pts'], reverse=True)
    return player_stats[:top_n]

def get_team_summary(team_abbr, season="2025-26"):
    top_players = get_team_top_players(team_abbr, season)
    if not top_players:
        return None
    
    total_pts  = sum(p['avg_pts'] for p in top_players)
    total_reb  = sum(p['avg_reb'] for p in top_players)
    total_ast  = sum(p['avg_ast'] for p in top_players)
    star_pts   = top_players[0]['avg_pts']  # best player's scoring
    
    return {
        'team': team_abbr,
        'top5_pts': total_pts,
        'top5_reb': total_reb,
        'top5_ast': total_ast,
        'star_pts': star_pts,
        'star_name': top_players[0]['name'],
        'players': top_players
    }

# Test it
if __name__ == "__main__":
    print("Testing with BOS...")
    summary = get_team_summary("BOS")
    if summary:
        print("\n" + summary['team'] + " top players (last 5 games):")
        for p in summary['players']:
            print("  " + p['name'] + " — " + str(round(p['avg_pts'],1)) + " pts, " + str(round(p['avg_reb'],1)) + " reb, " + str(round(p['avg_ast'],1)) + " ast")
        print("\nTop 5 combined pts: " + str(round(summary['top5_pts'],1)))
        print("Star player: " + summary['star_name'] + " (" + str(round(summary['star_pts'],1)) + " ppg)")