from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time

seasons = [
    "2014-15","2015-16","2016-17","2017-18","2018-19",
    "2019-20","2020-21","2021-22","2022-23","2023-24", 
    "2024-25", "2025-26"
]

all_games = []

for season in seasons:
    print(f"Fetching {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Playoffs"
    )
    df = finder.get_data_frames()[0]
    all_games.append(df)
    time.sleep(1)

games = pd.concat(all_games, ignore_index=True)
games.to_csv("playoff_games.csv", index=False)
print(f"Done! {len(games)} rows saved to playoff_games.csv")