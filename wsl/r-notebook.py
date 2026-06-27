import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # ⚽ Predicting Game Results in the Women's Super League (WSL)

    ## Background

    The Women's Super League (WSL) is the top division league for professional women's football in England. It is widely regarded as one of the most popular and competitive league in women's football, attracting elite talent from across the globe. This project analyses the WSL through various visualisations of goals scored segmented by teams and regions alongside match-specific facets like kickoff timings. This project will also look at smaller research questions I was interested in knowing with the ultimate goal of developing a deep learning model to predict the next season's winner.

    ## Data Source

    [Fixture Download](https://fixturedownload.com/)

    ## Research Question

    1. [R] Who are the Big Four of the Women's Super League?
    2. [M] Who will be the Big Four of the WSL next season (26/27)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # [R] Who are the *Big Four* of the WSL?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Background:** The *Big Six* is a term used to describe the six wealthiest clubs in the Premier League (i.e., top division league for professional men's football in England) that historically dominate the league standings and possess the largest global fanbases. The Big Six include: Arsenal, Chelsea, Liverpool, Manchester City, Manchester United and Tottenham Hotspur.

    **Problem:** The Big Six teams are commonly attributed to the men's teams in the Premier League. This Big Six was not officially established for women's teams in the WSL. Currently, the WSL's equivalent of the Big Six mirrors that of the men, but it is unclear whether this is true. Moreover, as time has gone on, the Big Six in the Premier League is being questioned with teams like Tottenham Hotspur struggling against relegation (25/26) and others like Manchester United (24/25) and Chelsea (25/26) not finishing in the top six. As well, the WSL is a smaller league than the Premier League with only 12 teams playing per season with plans to grow to 14 teams in the 26/27 season. Therefore, it would be more suitable for the WSL to have a Big Four rather than a Big Six, in keeping with the Premier League's 30% ratio.

    **Purpose:** This research looks to determine the current *Big Four* of the WSL based on data from previous seasons.

    **Analysis:** The WSL's Big Four will be determined by

    1. Ratio of games won to games lost
    2. Ratio of goals scored to goals conceeded -> Goal Difference
    3. Consistency throughout seasons
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import altair as alt
    from functions import import_data, total_games

    return alt, cs, import_data, mo, pl, total_games


@app.cell(hide_code=True)
def _(import_data, pl):
    data_full: pl.DataFrame = import_data()
    return (data_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The processed dataset used for further analysis is shown below:
    """)
    return


@app.cell(hide_code=True)
def _(data_full: "pl.DataFrame", pl):
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
    return (data,)


@app.cell(hide_code=True)
def _(data):
    seasons = data['Season'].unique().len()
    return (seasons,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -   Ratio of games won to games lost ✅
    -   Ratio of goals scored to goals conceeded -> Goal Difference ✅
    -   Consistency throughout seasons
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consistency throughout seasons
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consistency here refers to table placing in the league; which teams have regularly ended the season within the top four slots.
    """)
    return


@app.cell(hide_code=True)
def _(data, pl):
    points = (
        data
        .with_columns(
            pl.when(pl.col('Winning Team').is_not_null())
            .then(pl.lit(3))
            .when(pl.col('Draw (1)').is_not_null() | pl.col('Draw (2)').is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(None))
            .alias('Points')
        )
        .select(['Season', 'Winning Team', 'Draw (1)', 'Draw (2)', 'Points',])
    )
    return (points,)


@app.cell(hide_code=True)
def _(cs, pl, points):
    points_win = (
        points
        .group_by('Season', 'Winning Team')
        .agg(pl.col('Points').sum())
        .rename({'Winning Team':'Team', 'Points':'W'})
        .drop_nulls()
    )

    points_draw1 = (
        points
        .group_by('Season', 'Draw (1)')
        .agg(pl.col('Points').sum())
        .rename({'Draw (1)':'Team', 'Points':'D1'})
    )

    points_draw2 = (
        points
        .group_by('Season', 'Draw (2)')
        .agg(pl.col('Points').sum())
        .rename({'Draw (2)':'Team', 'Points':'D2'})
    )

    points_ranked = (
        points_win
        .join(points_draw1, on=['Season', 'Team'], how='full', suffix='_D1')
        .join(points_draw2, on=['Season', 'Team'], how='full', suffix='_D2')
        .drop(cs.contains('Team_', 'Season_'))
        .drop_nulls(subset=['Season'])
        .with_columns(
            pl.sum_horizontal(['W', 'D1', 'D2']).alias('Points'),
        )
        .drop(['W', 'D1', 'D2'])
        .sort(by=['Season', 'Points'], descending=[True, True])

    )
    return (points_ranked,)


@app.cell(hide_code=True)
def _(gd_calc, pl):
    pos_gd_season = gd_calc.group_by(['Season', 'Winning Team']).agg(pl.sum('Positive GD')).rename({'Winning Team': 'Team'})
    neg_gd_season = gd_calc.group_by(['Season', 'Losing Team']).agg(pl.sum('Negative GD')).rename({'Losing Team': 'Team'})

    gd_season = (
        pos_gd_season
        .join(neg_gd_season, on=['Season', 'Team'])
        .with_columns(
            (pl.col('Positive GD') + pl.col('Negative GD')).alias('Final GD'), 
        )
        .sort('Final GD', descending=True)
        .drop_nulls('Team')
        .drop(['Positive GD', 'Negative GD'])
    )
    return (gd_season,)


@app.cell(hide_code=True)
def _(gd_season, pl, points_ranked):
    final_rank = (
        points_ranked
        .join(gd_season, on=['Season', 'Team'])
        .with_columns(
            pl.col('Points').rank(method='min', descending=True).over('Season').alias('Rank')
        )
        .filter(pl.col('Rank').le(4))
        .with_columns(
            pl.struct(pl.col('Points'), pl.col('Final GD'))
            .rank(method='min', descending=True).over('Season').alias('Rank')
        )
    )

    final_rank
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ratio of Games Won to Games Lost
    """)
    return


@app.cell(hide_code=True)
def _(data, pl, total_games):
    wins: pl.DataFrame = data.group_by('Winning Team').agg(pl.len().alias('Wins')).rename({'Winning Team': 'Team'})
    loss: pl.DataFrame = data.group_by('Losing Team').agg(pl.len().alias('Losses')).rename({'Losing Team': 'Team'})

    win_lose: pl.DataFrame = (
        wins
            .join(loss, on='Team', how='inner')
            .with_columns(
                pl.col('Team').map_elements(lambda team: total_games(team=team, data=data)).alias('Total Games')
            )
    )
    return (win_lose,)


@app.cell(hide_code=True)
def _(big4_ratio, mo, pl, seasons, win_lose_ratio: "pl.DataFrame"):
    mo.md(f"""
    The table below shows the Big Four based on their win ratio over {seasons} seasons. The win ratio can be interpreted as the number of wins a team has for every loss. The teams with the highest win ratios are: {big4_ratio[0]}, {big4_ratio[1]}, {big4_ratio[2]} and {big4_ratio[3]}. These high ratios suggest that teams are winning more games than they lose, with {big4_ratio[0]} having the highest ratio winning {win_lose_ratio.select(pl.first('Win Ratio')).item()} games before losing their first game.
    """)
    return


@app.cell(hide_code=True)
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

    big4_ratio = win_lose_ratio['Team'].to_list()

    win_lose_ratio
    return big4_ratio, win_lose_ratio


@app.cell(hide_code=True)
def _(
    big4_percent,
    data,
    mo,
    pl,
    seasons,
    total_games,
    win_lose_percent: "pl.DataFrame",
):
    mo.md(f"""
    To visualise this ratio, the stacked bar chart below shows the proportion of wins versus losses for the Big Four teams. These teams have the largest proportion of games won with {big4_percent[0]} leading the field having won {win_lose_percent.select(pl.first('Win %')).item()}% of the games played over {total_games(data=data, team=big4_percent[0])} games in {seasons} seasons
    """)
    return


@app.cell(hide_code=True)
def _(pl, win_lose: "pl.DataFrame"):
    win_lose_percent: pl.DataFrame = (
        win_lose
        .with_columns(
            (pl.col('Wins') / pl.col('Total Games') * 100).round(1).alias('Win %'),
            (pl.col('Total Games') - pl.sum_horizontal(['Wins', 'Losses'])).alias('Draws'),
            (pl.col('Losses') / pl.col('Total Games') * 100).round(1).alias('Loss %'),
            ((pl.col('Total Games') - pl.sum_horizontal(['Wins', 'Losses'])) / pl.col('Total Games') * 100).round(1).alias('Draw %'),
        )
        .sort(by=['Win %'], descending=True)
        .head(4)
        .select(['Team', 'Win %', 'Loss %', 'Draw %', 'Wins', 'Losses', 'Draws', 'Total Games'])
    )

    big4_percent = win_lose_percent['Team'].to_list()


    win_lose_percent
    return big4_percent, win_lose_percent


@app.cell(hide_code=True)
def _(alt, cs, pl, win_lose_percent: "pl.DataFrame"):
    win_lose_percent_visualisation = (
        win_lose_percent
        .clone()
        .unpivot(cs.numeric(), index='Team', value_name='Percentage')
        .filter(pl.col('variable').str.contains('%'))
        .with_columns(
            pl.col('variable').replace({'Win %': 0, 'Draw %': 1, 'Loss %': 2}).cast(pl.Int8).alias('sort_order')
        )
    )

    (
        alt.Chart(win_lose_percent_visualisation)
        .mark_bar()
        .encode(
            x='Percentage',
            y=alt.Y('Team', sort='-x'),  # sort by x value descending 
            color=alt.Color(
                'variable', 
                scale=alt.Scale(domain=['Loss %', 'Draw %', 'Win %'], range=['#FBB5AE', '#CCCCCC', '#B3E2CD'])
            ), 
            order=alt.Order('sort_order', sort='ascending'),
            tooltip=['Team', 'variable', 'Percentage']
        )
        .properties(width=800, height=400)
    )
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ratio of goals scored to goals conceeded (or Goal Difference)
    """)
    return


@app.cell(hide_code=True)
def _(data, pl):
    gd_calc = (
        data
        .with_columns(
            pl.when(pl.col('Winner').eq('Away'))
            .then((pl.col('Away Goals') - pl.col('Home Goals')))
            .when(pl.col('Winner').eq('Home'))
            .then((pl.col('Home Goals') - pl.col('Away Goals')))
            .otherwise(pl.lit(0))
            .alias('Positive GD'),
        )

        .with_columns(
            (pl.col('Positive GD') * -1).alias('Negative GD')
        )

        # .select(["Season", "Winning Team", "Losing Team", "Positive GD", "Negative GD"])
    )

    positive_gd = gd_calc.group_by('Winning Team').agg(pl.sum('Positive GD')).rename({'Winning Team': 'Team'})
    negative_gd = gd_calc.group_by('Losing Team').agg(pl.sum('Negative GD')).rename({'Losing Team': 'Team'})

    goal_difference = (
        positive_gd
            .join(negative_gd, on='Team')
            .with_columns(
                (pl.col('Positive GD') + pl.col('Negative GD')).alias('Final GD'), 
            )
            .sort('Final GD', descending=True)
    )
    return gd_calc, goal_difference


@app.cell(hide_code=True)
def _(cs, data, goal_difference, pl, total_games):
    gd_fig = (
        goal_difference
        .select(['Team', 'Positive GD', 'Negative GD', 'Final GD'])
        .unpivot(cs.numeric(), index='Team')
        .with_columns(
            pl.col('Team').map_elements(lambda team: total_games(team=team, data=data)).alias('Total Games')
        )
    )
    return (gd_fig,)


@app.cell(hide_code=True)
def _(gd_fig, pl):
    gd_b4_team = gd_fig.filter(pl.col('variable').eq('Final GD')).sort(by='value', descending=True).head(4).select('Team')

    gd_best = gd_fig.filter(pl.col('variable').eq('Final GD')).sort(by='value', descending=True).head(1)
    return gd_b4_team, gd_best


@app.cell(hide_code=True)
def _(gd_b4_team, gd_best, mo, seasons):
    mo.md(f"""
    Goal difference looks at the difference between goals scored to goals conceeded. A positive goal difference sees a team scoring more goals than they conceed while a negative goal difference sees a team conceed more goals than they score. For example, if Arsenal wins a game against Chelsea, 2-1, the goal difference would be 1 with Arsenal having the positive goal difference (+1) and Chelsea would have a negative goal difference (-1). The goal difference is calculated for every game. 

    From the graph below, the top four teams with the best goal difference are {gd_b4_team[0].item()}, {gd_b4_team[1].item()}, {gd_b4_team[2].item()} and {gd_b4_team[3].item()} with {gd_best.select('Team').item()} leading having a final goal difference of {gd_best.select('value').item()} over {gd_best.select('Total Games').item()} games in {seasons} seasons.
    """)
    return


@app.cell(hide_code=True)
def _(alt, gd_fig, seasons):
    (
        alt.Chart(gd_fig)
            .mark_bar()
            .encode(
                x=alt.X('Team', sort='-y'),
                y=alt.Y('value', title='Goal Difference (GD)'),
                color=alt.Color(
                    'variable', 
                    scale=alt.Scale(domain=['Positive GD', 'Negative GD'], range=['green', 'tomato']),
                    title='GD Type'
                ),
                tooltip=[
                    alt.Tooltip('Team'),
                    alt.Tooltip('variable', title='Type'),
                    alt.Tooltip('value', title='GD'),
                    alt.Tooltip('Total Games', title='Total Games'),  # pulled from data, not encoded
                ]
            )

            .properties(
                width=600,
                height=400,
                title=f'Goal Difference by Team Over {seasons} Seasons'
            )
    )
    return


@app.cell(hide_code=True)
def _(alt, gd_fig, pl, seasons):
    total_gd_fig = gd_fig.clone().filter(pl.col('variable').eq('Final GD'))

    (
        alt.Chart(total_gd_fig)
        .mark_bar()
        .encode(
            x=alt.X('Team', sort='-y'),
            y=alt.Y('value'),
            color=alt.Color(
                    'variable', 
                    scale=alt.Scale(domain=['Final GD'], range=['skyblue']),
                    title='GD Type'
                ),
            tooltip=[
                    alt.Tooltip('Team'),
                    alt.Tooltip('value', title='Final GD'),
                    alt.Tooltip('Total Games:Q', title='Total Games'),  # pulled from data, not encoded
                ]
        )
        .properties(
            width=800,
            height=300,
            title=f'Final Goal Difference by Team Over {seasons} Seasons'
        )
    )
    return


if __name__ == "__main__":
    app.run()
