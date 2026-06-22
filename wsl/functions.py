import polars as pl

def import_data(latest_season=25):
    dfs =[]

    for i in range(23, (latest_season+1)):
        df = pl.read_csv(f"https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/wsl/wsl/data/wsl-20{i}-UTC.csv").with_columns(pl.lit(f"{i}/{i+1}").alias("Season"))
        dfs.append(df)

    return pl.concat(dfs)