import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import duckdb as db
    import polars.selectors as cs

    return cs, pl


@app.cell
def _(cs, pl):
    df_polars = (
        pl.read_csv('https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/superheroine-fatigue/superheroine_fatigue/dc_marvel_movie_performance.csv')
        .rename({
            "Box office gross Domestic (U.S. and Canada )": "Domestic Gross (U.S. and Canada; $)",
            "Box office gross Other territories": "Other Territories Gross ($)",
            "Box office gross Worldwide": "Worldwide Gross ($)",
            "Budget": "Budget ($)",
            "Inflation Adjusted Worldwide Gross": "Inflation Adjusted Worldwide Gross ($)",
            "Inflation Adjusted Budget": "Inflation Adjusted Budget ($)",
            "2.5x prod": "2.5 Production Costs ($)",
        })
        .with_columns(
            cs.contains('$').str.replace_all(r'[^\d]', ''),  # removes everything that's NOT a digit
            cs.contains('%').str.replace_all(r'[^\d]', ''),  # removes everything that's NOT a digit
            pl.col('Phase').replace({'NA': None})
        )
        .cast({
            # "U.S. release date": pl.Datetime,
            "Domestic Gross (U.S. and Canada; $)": pl.UInt128,
            "Other Territories Gross ($)": pl.UInt128,
            "Worldwide Gross ($)": pl.UInt128,
            "Budget ($)": pl.Int128,
            "Phase": pl.UInt8,
            "Domestic %": pl.UInt8,
            "Gross to Budget": pl.UInt8,
            "Rotten Tomatoes Critic Score": pl.UInt8,
            "Year": pl.Utf8,
            "Inflation Adjusted Worldwide Gross ($)": pl.UInt128,
            "Inflation Adjusted Budget ($)": pl.UInt128,
            "2.5 Production Costs ($)": pl.UInt128,
        })
    )

    return (df_polars,)


@app.cell
def _(df_polars):
    df_polars
    return


if __name__ == "__main__":
    app.run()
