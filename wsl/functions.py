import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    from urllib.error import URLError

    return URLError, pl


@app.cell
def _(URLError, pl):
    def import_research_data(latest_season:int = 25):
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

    return


@app.cell
def _(URLError, pl):
    def import_modelling_player_data(latest_season:int = 25):
        dfs: list[pl.DataFrame] =[]

        try: 
            for i in range(16, (latest_season+1)):
                df: pl.DataFrame = pl.read_csv(f"https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/wsl/wsl/data/female_players_{i}.csv", infer_schema_length=0).with_columns(pl.lit(f"{i-1}/{i}").alias("FIFA/EA Season"))
                dfs.append(df)

        except URLError as e:
            print(f'Network error: {e.reason}')

        except ValueError:
            print('CSVs are not being read as Polars DataFrames')

        return pl.concat(dfs)

    return


@app.cell
def _(pl):
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

    return


if __name__ == "__main__":
    app.run()
