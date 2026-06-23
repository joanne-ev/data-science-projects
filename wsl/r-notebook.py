import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # [R] Who are the *Big Four* of the WSL?
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import altair as alt
    from functions import import_data, total_games

    return alt, cs, import_data, mo, pl, total_games


@app.cell
def _(import_data):
    data_full = import_data()
    return (data_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The processed dataset used for further analysis is shown below:
    """)
    return


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
            # Calculate the goal difference
            (pl.col("Home Goals") - pl.col("Away Goals")).alias("Goal Difference"),

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
                "Home Goals": pl.UInt8,
                "Away Goals": pl.UInt8,
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
                "Goal Difference", "Goals Scored",
            ]
        )

    )

    data
    return (data,)


@app.cell
def _(data):
    seasons = data['Season'].unique().len()
    return (seasons,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -   Ratio of games won to games lost
    -   Ratio of goals scored to goals conceeded -> Goal Difference
    -   Consistency throughout seasons
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ratio of Games Won to Games Lost
    """)
    return


@app.cell
def _(data, pl, total_games):
    wins: pl.DataFrame = data.group_by('Winning Team').agg(pl.len().alias('Wins')).rename({'Winning Team': 'Team'})
    loss: pl.DataFrame = data.group_by('Losing Team').agg(pl.len().alias('Losses')).rename({'Losing Team': 'Team'})

    win_lose: pl.DataFrame = wins.join(loss, on='Team', how='inner').with_columns(pl.col('Team').map_elements(lambda team: total_games(team=team, data=data)).alias('Total Games'))
    return (win_lose,)


@app.cell
def _(pl, win_lose: "pl.DataFrame"):
    win_lose_ratio: pl.DataFrame = (
        win_lose
        .with_columns(
            (pl.col('Wins') / pl.col('Losses')).round(1).alias('Win Ratio'),     # x games won for every game loss
        )
        .sort(by=['Win Ratio'], descending=True)
        .head(4)
        .drop('Total Games')
    )

    win_lose_ratio
    return


@app.cell(hide_code=True)
def _(data, mo, seasons, total_games):
    mo.md(f"""
    The table above shows the number of games teams have won, lost and their win ratio over the {seasons} seasons.

    The Win Ratio column can be interpreted as the number of wins a team has for every loss. The Big Four teams with the highest win ratios are: Chelsea, Arsenal, Manchester City and Manchester United. These high ratio suggest that teams are winning more games than they lose, with Chelsea having the highest ratio winning approximately 8.5 games before losing their first game.

    To visualise this ratio, the stacked bar chart below shows the proportion of wins versus losses for the Big Four teams. These teams have the largest proportion of games won with Chelsea leading the field having won 78.5% of the games played over {total_games(data=data, team='Chelsea')} games in {seasons} seasons
    """)
    return


@app.cell
def _(cs, pl, win_lose: "pl.DataFrame"):
    win_lose_percent: pl.DataFrame = (
        win_lose
        .with_columns(
            (pl.col('Wins') / pl.col('Total Games') * 100).round(1).alias('Win %'),
            (pl.col('Losses') / pl.col('Total Games') * 100).round(1).alias('Loss %'),
            ((pl.col('Total Games') - pl.sum_horizontal(['Wins', 'Losses'])) / pl.col('Total Games') * 100).round(1).alias('Draw %'),
        )
        .sort(by=['Win %'], descending=True)
        .head(4)
        .unpivot(cs.numeric(), index='Team', value_name='Percentage')
        .filter(pl.col('variable').str.contains('%'))
        .with_columns(
            pl.col('variable').replace({'Win %': 0, 'Draw %': 1, 'Loss %': 2}).cast(pl.Int8).alias('sort_order')
        )
    )

    win_lose_percent.pivot(on='variable', index='Team', values='Percentage')
    return (win_lose_percent,)


@app.cell
def _(alt, win_lose_percent: "pl.DataFrame"):
    (
        alt.Chart(win_lose_percent)
        .mark_bar().encode(
            x='Percentage',
            y=alt.Y('Team', sort='-x'),  # sort by x value descending 
            color=alt.Color(
                'variable', 
                scale=alt.Scale(domain=['Loss %', 'Draw %', 'Win %'], range=['#FBB5AE', '#CCCCCC', '#B3E2CD'])
            ), 
            order=alt.Order('sort_order', sort='ascending')
        )
        .properties(width=1000, height=400)
    )
    return


if __name__ == "__main__":
    app.run()
