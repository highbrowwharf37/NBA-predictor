from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import time

seasons = [
    "2014-15","2015-16","2016-17","2017-18","2018-19",
    "2019-20","2020-21","2021-22","2022-23","2023-24",
    "2024-25","2025-26"
]

all_playoff = []
all_regular = []

for season in seasons:
    print("Fetching " + season + "...")
    try:
        # Playoff stats
        playoff = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Playoffs"
        )
        df_p = playoff.get_data_frames()[0]
        df_p['SEASON_ID'] = season
        all_playoff.append(df_p)
        time.sleep(1)

        # Regular season stats
        regular = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season"
        )
        df_r = regular.get_data_frames()[0]
        df_r['SEASON_ID'] = season
        all_regular.append(df_r)
        time.sleep(1)

    except Exception as e:
        print("Error on " + season + ": " + str(e))
        time.sleep(2)

pd.concat(all_playoff, ignore_index=True).to_csv("advanced_stats.csv", index=False)
pd.concat(all_regular, ignore_index=True).to_csv("regular_season_stats.csv", index=False)
print("Done!")