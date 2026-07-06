import polars as pl
from urllib.error import URLError

def import_data(latest_season:int = 25):
    dfs: list[pl.DataFrame] =[]

    try: 
        for i in range(23, (latest_season+1)):
            df: pl.DataFrame = pl.read_csv(f"https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/wsl/wsl/data/wsl-20{i}-UTC.csv").with_columns(pl.lit(f"{i}/{i+1}").alias("Season"))
            dfs.append(df)

    except URLError as e:
        print(f'Network error: {e.reason}')

    except ValueError:
        print('CSVs are not being read as Polars DataFrames')

    return pl.concat(dfs)


def total_games(team:str, data:pl.DataFrame):
    games: list[int] = []

    for season in data['Season'].unique().to_list():
        count: int = (
            data
            .filter(
                pl.col('Season').eq(season),
                (pl.col('Home Team').eq(team) | pl.col('Away Team').eq(team))
            )
            .shape[0]
        )

        games.append(count)

    return sum(games)