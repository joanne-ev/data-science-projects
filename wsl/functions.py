import polars as pl

def import_data(latest_season:int = 25):
    dfs =[]

    for i in range(23, (latest_season+1)):
        df = pl.read_csv(f"https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/wsl/wsl/data/wsl-20{i}-UTC.csv").with_columns(pl.lit(f"{i}/{i+1}").alias("Season"))
        dfs.append(df)

    return pl.concat(dfs)

def total_games(team:str, data:pl.DataFrame):
    games = []

    for season in data['Season'].unique().to_list():
        count = (
            data
            .filter(
                pl.col('Season').eq(season),
                (pl.col('Home Team').eq(team) | pl.col('Away Team').eq(team))
            )
            .shape[0]
        )

        games.append(count)

    return sum(games)