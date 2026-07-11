import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs
    import altair as alt
    import marimo as mo

    return alt, cs, mo, pl


@app.cell
def _(cs, pl):
    df = (
        pl.read_csv('https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/superheroine-fatigue/superheroine_fatigue/dc_marvel_movie_performance.csv')
        .rename({
            "Box office gross Domestic (U.S. and Canada )": "Domestic Gross ($)",
            "Box office gross Other territories": "Other Territories Gross ($)",
            "Box office gross Worldwide": "Worldwide Gross ($)",
            "Budget": "Budget ($)",
            "Inflation Adjusted Worldwide Gross": "Inflation Adjusted Worldwide Gross ($)",
            "Inflation Adjusted Budget": "Inflation Adjusted Budget ($)",
            "2.5x prod": "2.5 Production Costs ($)",
        })

        # Removes everything that's NOT a digit
        .with_columns(
            cs.contains('$').str.replace_all(r'[^\d]', ''),
            cs.contains('%').str.replace_all(r'[^\d]', ''),
            pl.col('Phase').replace({'NA': None}),
            pl.col('U.S. release date').str.strptime(pl.Date, format="%d/%m/%Y"),
        )
        .cast({
            "U.S. release date": pl.Datetime,
            cs.contains('$') : pl.UInt128,
            "Phase": pl.UInt8,
            "Domestic %": pl.UInt8,
            "Gross to Budget": pl.Float16,
            "Rotten Tomatoes Critic Score": pl.UInt8,
            "Year": pl.Utf8,
        })
    )
    return (df,)


@app.cell
def _(cs, df, pl):
    def worldwide_gross_grouped_df(franchise: bool = False, year: bool = False):

        group_by_vars: list[str] = ['Male/Female-led'] + ['Year'] * year + ['Franchise'] * franchise

        return (
            df.clone()
            .group_by(group_by_vars)
            .agg(
                pl.mean('Worldwide Gross ($)'),
                pl.mean('Inflation Adjusted Worldwide Gross ($)'),
            )
            .with_columns(
                (pl.col('Worldwide Gross ($)') / pl.sum('Worldwide Gross ($)') * 100).round_sig_figs(2).alias('Percent (%)'),
                (pl.col('Inflation Adjusted Worldwide Gross ($)') / pl.sum('Inflation Adjusted Worldwide Gross ($)') * 100).round_sig_figs(2).alias('Inflated Percent (%)'),
            )
            .cast({cs.contains('$') : pl.UInt128, cs.contains('%') : pl.Int8})
            .drop_nulls(subset=['Male/Female-led'])
        )

    return (worldwide_gross_grouped_df,)


@app.cell
def _(worldwide_gross_grouped_df):
    worldwide_gross_protagonists = worldwide_gross_grouped_df()
    worldwide_gross_protagonists_year = worldwide_gross_grouped_df(year=True)
    worldwide_gross_protagonists_franchise = worldwide_gross_grouped_df(franchise=True)
    worldwide_gross_protagonists_year_franchise = worldwide_gross_grouped_df(year=True, franchise=True)
    return (
        worldwide_gross_protagonists,
        worldwide_gross_protagonists_franchise,
        worldwide_gross_protagonists_year,
        worldwide_gross_protagonists_year_franchise,
    )


@app.cell
def _():
    colours_domain: list[str] = ['Female', 'Male', 'Co-starring']
    colours_css: list[str] = ['hotpink', 'skyblue', 'yellow']
    return colours_css, colours_domain


@app.cell
def _(mo):
    inflation_dropdown = mo.ui.dropdown(options=[True, False], value=True, label='Figures adjusted for inflation:')

    inflation_dropdown
    return (inflation_dropdown,)


@app.cell
def _(inflation_dropdown):
    inflation_selection: str = inflation_dropdown.value
    return (inflation_selection,)


@app.cell
def _(
    alt,
    colours_css: list[str],
    colours_domain: list[str],
    inflation_selection: str,
    pl,
    worldwide_gross_protagonists,
):
    def worldwide_gross_protagonists_fig(data: pl.DataFrame, inflation_adjusted: bool):
        var = ['Inflation Adjusted Worldwide Gross' if inflation_adjusted else 'Worldwide Gross'][0]

        return (
            alt.Chart(data)
            .mark_arc()
            .encode(
                theta=alt.Theta(f'{var} ($)', stack=True),
                color=alt.Color(
                    shorthand='Male/Female-led', 
                    scale=alt.Scale(domain=colours_domain, range=colours_css),
                    legend=alt.Legend(
                        title=['Protagonist Gender'],
                        titleLimit=300, # increasing the number of characters shown in the legend title
                        titleFontSize=13,
                        labelFontSize=12,

                        # To adjust location of legend
                        orient='none',
                        legendX=330,
                        legendY=100,
                        direction='vertical'
                    )

                ),
                tooltip=['Male/Female-led', 'Percent (%)']
            )
            .properties(
                width = 300,
                title=alt.TitleParams(
                    text=f"{var.title()} by Protagonist Gender",
                    fontSize=14,
                    offset=20   # space between title and pie chart
                )
            )
        )

    worldwide_gross_protagonists_fig(data=worldwide_gross_protagonists, inflation_adjusted=inflation_selection)
    return


@app.cell
def _(
    alt,
    colours_css: list[str],
    colours_domain: list[str],
    inflation_selection: str,
    pl,
    worldwide_gross_protagonists_year,
):
    def worldwide_gross_protagonists_year_fig(data: pl.DataFrame, inflation_adjusted: bool):

        var = ['Inflation Adjusted Worldwide Gross' if inflation_adjusted else 'Worldwide Gross'][0]

        return (
            alt.Chart(data)
            .mark_line()
            .encode(
                x='Year',
                y=f'{var} ($):Q',
                shape=alt.Shape('Male/Female-led'),
                color=alt.Color(shorthand='Male/Female-led',
                                scale=alt.Scale(domain=colours_domain, range=colours_css),
                                title='Protagonist Gender'),
            )
            .properties(
                width = 600,
                title=alt.TitleParams(
                    text=f"{var.title()} by Protagonist Gender between {data['Year'].min()} and {data['Year'].max()}",
                    fontSize=14,
                    offset=20   # space between title and pie chart
                )
            )
        )

    worldwide_gross_protagonists_year_fig(worldwide_gross_protagonists_year, inflation_adjusted=inflation_selection)
    return


@app.cell
def _(
    alt,
    inflation_selection: str,
    pl,
    worldwide_gross_protagonists_franchise,
):
    def worldwide_gross_protagonists_franchise_fig(data: pl.DataFrame, inflation_adjusted: bool):

        var = ['Inflation Adjusted Worldwide Gross' if inflation_adjusted else 'Worldwide Gross'][0]

        return (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X('Male/Female-led:N', axis=alt.Axis(labelAngle=0), title='Protagonist Gender'),
                y=alt.Y(f'{var} ($):Q'),
                color=alt.Color('Franchise:N',
                                scale=alt.Scale(domain=['DC', 'Marvel'], 
                                                range=['CornflowerBlue', 'coral'])),
                xOffset='Franchise:N',
                tooltip=['Male/Female-led:N', f'{var} ($):Q']
            )
            .properties(
                width = 250,
                title=alt.TitleParams(
                    text=[f"{var.title()} by", "Protagonist Gender between DC and Marvel"],
                    fontSize=14,
                    offset=20   # space between title and pie chart
                )
            )
        )

    worldwide_gross_protagonists_franchise_fig(worldwide_gross_protagonists_franchise, inflation_adjusted=inflation_selection)
    return


@app.cell
def _(
    alt,
    colours_css: list[str],
    colours_domain: list[str],
    inflation_selection: str,
    pl,
    worldwide_gross_protagonists_year_franchise,
):
    def worldwide_gross_protagonists_year_franchise_fig(data: pl.DataFrame, inflation_adjusted: bool):

        var = ['Inflation Adjusted Worldwide Gross' if inflation_adjusted else 'Worldwide Gross'][0]

        base = (
            alt.Chart(data)
            .mark_line()
            .encode(
                x='Year',
                y=f'{var} ($)',
                shape='Male/Female-led',
                color=alt.Color(
                    shorthand='Male/Female-led',
                    scale=alt.Scale(domain=colours_domain, range=colours_css),
                    title='Protagonist Gender',
                    legend=alt.Legend(
                        title=['Protagonist Gender'],
                        titleLimit=300, # increasing the number of characters shown in the legend title
                        titleFontSize=13,
                        labelFontSize=12,

                        # To adjust location of legend
                        orient='none',
                        legendX=650,
                        legendY=300,
                        direction='vertical'
                    )
                ),
            )
            .properties(width=600,)
        )

        dc = base.transform_filter(alt.FieldEqualPredicate(field='Franchise', equal='DC'))
        marvel = base.transform_filter(alt.FieldEqualPredicate(field='Franchise', equal='Marvel'))

        return (
            alt.vconcat(dc, marvel)
            .properties(title=alt.TitleParams(text=f"{var.title()} by Protagonist Gender between {data['Year'].min()} and {data['Year'].max()} for DC (top) and Marvel (bottom)", 
                        offset=25, 
                        fontSize=16))
        )

    worldwide_gross_protagonists_year_franchise_fig(worldwide_gross_protagonists_year_franchise, inflation_adjusted=inflation_selection)
    return


if __name__ == "__main__":
    app.run()
