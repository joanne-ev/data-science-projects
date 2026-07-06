import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ⚽ Playing around with the Women's Super League (WSL)

    ## Background

    The Women's Super League (WSL) is the top division league for professional women's football in England. It is widely regarded as one of the most popular and competitive league in women's football, attracting elite talent from across the globe. This project analyses the WSL through various visualisations of goals scored segmented by teams and regions alongside match-specific facets like kickoff timings. This project will also look at smaller research questions I was interested in knowing with the ultimate goal of developing a deep learning model to predict the next season's winner.

    ## Data Source

    [Fixture Download](https://fixturedownload.com/)

    > There is a missing game in the 24/25 season for both Chelsea and Manchester United. For these teams, data is only available for 22 instead of 23 games.

    ## Research Questions

    1. [R] Who are the Big Four of the Women's Super League?
    2. [M] Who will be the Big Four of the WSL next season (26/27)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Who will be the Big Four of the WSL next season (26/27)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Background:**

    **Problem:**

    **Purpose:**

    **Analysis:**
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from functions import import_data

    return import_data, mo, pl


@app.cell
def _(import_data):
    data_full = import_data()
    return (data_full,)


@app.cell
def _(data_full, pl):
    data = (
        data_full.clone()

        .with_columns(
            pl.col("Result").str.replace_all(" ", ""),  # remove all whitespace
            pl.col("Date").str.to_datetime("%d/%m/%Y %H:%M"),  # convert to DateTime data type
            pl.col("Location").alias("Stadium"),  # rename the column
        )

        .with_columns(
            # Get results for home and away teams
            pl.col("Result")
            .str.split("-")
            .list.get(0)
            .cast(pl.Int8)
            .alias("Home Goals"),

            pl.col("Result")
            .str.split("-")
            .list.get(1)
            .cast(pl.Int8)
            .alias("Away Goals"),

            # Separate the date into individual variables
            pl.col("Date").dt.day().alias("Day"),
            pl.col("Date").dt.month().alias("Month"),
            pl.col("Date").dt.year().alias("Year"),
            pl.col("Date").dt.hour().alias("Hour"),

            # Identify Kickoff period based on the Date
            pl.when(pl.col("Date").dt.hour().le(13))  # <=12pm for morning
            .then(pl.lit("Morning"))
            .when(pl.col("Date").dt.hour().is_between(12, 18, closed="none"))  # 13 < time < 18 for afternoon
            .then(pl.lit("Afternoon"))
            .when(pl.col("Date").dt.hour().ge(18))  # >=18 for evening
            .then(pl.lit("Evening"))
            .alias("Kickoff"),
        )

        .with_columns(
            # Calculate the number of goals scored
            (pl.col("Home Goals") + pl.col("Away Goals")).alias("Goals Scored"),

            # Determine the winners
            pl.when(pl.col("Home Goals").gt(pl.col("Away Goals")))
            .then(pl.lit("Home"))
            .when(pl.col("Away Goals").gt(pl.col("Home Goals")))
            .then(pl.lit("Away"))
            .otherwise(pl.lit("Draw"))
            .alias("Winner"),
        )

        .with_columns(
            # Identify winning teams
            pl.when(pl.col("Winner").eq("Away"))
            .then(pl.col("Away Team"))
            .when(pl.col("Winner").eq("Home"))
            .then(pl.col("Home Team"))
            .otherwise(None)
            .alias("Winning Team"),

            # Identify losing teams
            pl.when(pl.col("Winner").eq("Away"))
            .then(pl.col("Home Team"))
            .when(pl.col("Winner").eq("Home"))
            .then(pl.col("Away Team"))
            .otherwise(None)
            .alias("Losing Team"),

            # Draw (1) — first drawing team
            pl.when(pl.col("Winner").eq("Draw"))
            .then(pl.col("Home Team"))
            .otherwise(None)
            .alias("Draw (1)"),

            # Draw (2) — second drawing team
            pl.when(pl.col("Winner").eq("Draw"))
            .then(pl.col("Away Team"))
            .otherwise(None)
            .alias("Draw (2)"),
        )

        # Drop unnecessary columns
        .drop(["Location", "Date", "Result"])

        # Assign the right data type for a column - lower-precision variants are more memory-efficient
        .cast(
            {
                "Round Number": pl.UInt8,
                "Day": pl.UInt8,
                "Month": pl.UInt8,
                "Year": pl.String,
                "Hour": pl.UInt8,
                "Home Goals": pl.Int8,
                "Away Goals": pl.Int8,
                "Goals Scored": pl.UInt8,
                "Kickoff": pl.Enum(["Morning", "Afternoon", "Evening"])     # ordered categorical data type
            }
        )

        # Reorder columns 
        .select(
            [
                "Season", "Round Number",
                "Hour", "Day", "Month", "Year", "Kickoff",
                "Stadium", "Home Team", "Away Team",
                "Home Goals", "Away Goals", "Winner", 
                "Winning Team", "Losing Team", "Draw (1)", "Draw (2)",
                "Goals Scored",
            ]
        )

    )

    data
    return


if __name__ == "__main__":
    app.run()
