from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import time

seasons = [
    "2014-15","2015-16","2016-17","2017-18","2018-19",
    "2019-20","2020-21","2021-22","2022-23","2023-24",
    "2024-25","2025-26"
]

all_stats = []

for season in seasons:
    print("Fetching advanced stats for " + season + "...")
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Playoffs"
        )
        df = stats.get_data_frames()[0]
        df['SEASON_ID'] = season
        all_stats.append(df)
        time.sleep(1)
    except Exception as e:
        print("Error on " + season + ": " + str(e))
        time.sleep(2)

combined = pd.concat(all_stats, ignore_index=True)
combined.to_csv("advanced_stats.csv", index=False)
print("Done! " + str(len(combined)) + " rows saved to advanced_stats.csv")
print("Columns: " + str(combined.columns.tolist()))