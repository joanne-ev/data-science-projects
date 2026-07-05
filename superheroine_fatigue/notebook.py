import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import duckdb as db
    import polars.selectors as cs

    return cs, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Polars Pre-processing
    """)
    return


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
            "Gross to Budget": pl.Float16,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SQL Pre-processing
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        CREATE TABLE movie_performance AS
            SELECT * 
            FROM read_csv('https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/superheroine-fatigue/superheroine_fatigue/dc_marvel_movie_performance.csv');
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _df = mo.sql(
        f"""
        -- Rename columns
        ALTER TABLE movie_performance RENAME COLUMN "Box office gross Domestic (U.S. and Canada )" TO "Domestic Gross (U.S. and Canada; $)";

        ALTER TABLE movie_performance RENAME COLUMN "Box office gross Other territories" TO "Other Territories Gross ($)";

        ALTER TABLE movie_performance RENAME COLUMN "Box office gross Worldwide" TO "Worldwide Gross ($)";

        ALTER TABLE movie_performance RENAME COLUMN "Budget" TO "Budget ($)";

        ALTER TABLE movie_performance RENAME COLUMN "Inflation Adjusted Worldwide Gross" TO "Inflation Adjusted Worldwide Gross ($)";

        ALTER TABLE movie_performance RENAME COLUMN "Inflation Adjusted Budget" TO "Inflation Adjusted Budget ($)";

        ALTER TABLE movie_performance RENAME COLUMN "2.5x prod" TO "2.5 Production Costs ($)";
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, movie_performance):
    _df = mo.sql(
        f"""
        -- Clean string columns using regex
        UPDATE movie_performance SET "Domestic Gross (U.S. and Canada; $)" = REGEXP_REPLACE("Domestic Gross (U.S. and Canada; $)", '[^\d]', '', 'g');

        UPDATE movie_performance SET "Other Territories Gross ($)" = REGEXP_REPLACE("Other Territories Gross ($)", '[^\d]', '', 'g');

        UPDATE movie_performance SET "Worldwide Gross ($)" = REGEXP_REPLACE("Worldwide Gross ($)", '[^\d]', '', 'g');

        UPDATE movie_performance SET "Budget ($)" = REGEXP_REPLACE("Budget ($)", '[^\d]', '', 'g');

        UPDATE movie_performance SET "Domestic %" = REGEXP_REPLACE("Domestic %", '[^\d]', '', 'g');

        UPDATE movie_performance SET "Inflation Adjusted Worldwide Gross ($)" = REGEXP_REPLACE("Inflation Adjusted Worldwide Gross ($)", '[^\d]', '', 'g');

        UPDATE movie_performance SET "Inflation Adjusted Budget ($)" = REGEXP_REPLACE("Inflation Adjusted Budget ($)", '[^\d]', '', 'g');

        UPDATE movie_performance SET "2.5 Production Costs ($)" = REGEXP_REPLACE("2.5 Production Costs ($)", '[^\d]', '', 'g');
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _df = mo.sql(
        f"""
        -- Recast data type
        ALTER TABLE movie_performance ALTER COLUMN "Domestic Gross (U.S. and Canada; $)" SET DATA TYPE BIGINT USING "Domestic Gross (U.S. and Canada; $)"::BIGINT;

        ALTER TABLE movie_performance ALTER COLUMN "Other Territories Gross ($)" SET DATA TYPE BIGINT USING "Other Territories Gross ($)"::BIGINT;

        ALTER TABLE movie_performance ALTER COLUMN "Other Territories Gross ($)" SET DATA TYPE BIGINT USING "Other Territories Gross ($)"::BIGINT;

        ALTER TABLE movie_performance ALTER COLUMN "Worldwide Gross ($)" SET DATA TYPE BIGINT USING "Worldwide Gross ($)"::BIGINT;

        ALTER TABLE movie_performance ALTER COLUMN "Budget ($)" SET DATA TYPE BIGINT USING "Budget ($)"::BIGINT;

        ALTER TABLE movie_performance ALTER COLUMN "Domestic %" SET DATA TYPE BIGINT USING "Domestic %"::INT;

        ALTER TABLE movie_performance ALTER COLUMN "Inflation Adjusted Worldwide Gross ($)" SET DATA TYPE BIGINT USING "Inflation Adjusted Worldwide Gross ($)"::BIGINT;

        ALTER TABLE movie_performance ALTER COLUMN "Inflation Adjusted Budget ($)" SET DATA TYPE BIGINT USING "Inflation Adjusted Budget ($)"::BIGINT;

        ALTER TABLE movie_performance ALTER COLUMN "2.5 Production Costs ($)" SET DATA TYPE BIGINT USING "2.5 Production Costs ($)"::BIGINT;

        -- ALTER TABLE movie_performance ALTER COLUMN "Year" SET DATA TYPE VARCHAR USING "Year"::VARCHAR;
        """
    )
    return


@app.cell
def _(mo, movie_performance):
    _df = mo.sql(
        f"""
        SELECT * FROM movie_performance LIMIT 10;
        """
    )
    return


@app.cell
def _(mo, movie_performance):
    _df = mo.sql(
        f"""
        CREATE TABLE heroine_movies AS
        	SELECT *
        	FROM movie_performance
        	WHERE "Male/Female-led" = 'Female';

        CREATE TABLE hero_movies AS 
        	SELECT *
        	FROM movie_performance
        	WHERE "Male/Female-led" = 'Male';
        """
    )
    return


@app.cell
def _(heroine_movies, mo):
    _df = mo.sql(
        f"""
         WITH value_counts AS (
            SELECT 
                "Break Even", 
                COUNT(*) AS "Count",
                (COUNT(*) * 100 / (SELECT COUNT(*) FROM heroine_movies)) AS "p",
            	AVG("Domestic Gross (U.S. and Canada; $)") AS "avg_dg",
            	AVG("Worldwide Gross ($)") AS "avg_ww",
            	AVG("Budget ($)") AS "avg_budget",
            	AVG("Gross to Budget") AS "avg_gtb",
             	AVG("Rotten Tomatoes Critic Score") AS "avg_rts",
             	AVG("Year") AS "Year",
             	MODE("Franchise") AS "DC or MCU",
            FROM heroine_movies, 
            GROUP BY "Break Even"
        )

        SELECT 
            "Break Even",
            "Count",
            ROUND("p", 2) AS "Percent",
            ROUND("avg_dg") AS "Average Domestic Gross ($)",
            ROUND("avg_ww") AS "Average Worldwide Gross ($)",
            ROUND("avg_budget") AS "Average Budget ($)",
            ROUND("avg_gtb", 2) AS "Average Gross to Budget Ratio",
            ROUND("avg_rts") AS "Average RT Score",
            ROUND("Year") AS "Average Year",
            "DC or MCU"
        FROM value_counts;
        """
    )
    return


if __name__ == "__main__":
    app.run()
