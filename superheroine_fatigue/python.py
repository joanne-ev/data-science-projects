import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs
    import altair as alt

    return alt, cs, pl


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
    return (worldwide_gross_protagonists,)


@app.cell
def _(alt, pl):
    def worldwide_gross_protagonists_fig(data: pl.DataFrame, inflation_adjusted : bool = False):
        var = ['Inflation Adjusted Worldwide Gross' if inflation_adjusted else 'Worldwide Gross'][0]
    
        return (
            alt.Chart(data)
            .mark_arc()
            .encode(
                theta=alt.Theta(f'{var} ($)', stack=True),
                color=alt.Color(
                    'Male/Female-led', 
                    scale=alt.Scale(domain=sorted(data['Male/Female-led'].to_list()), range=['#CCCCCC', 'orange', 'brown']),
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

    return (worldwide_gross_protagonists_fig,)


@app.cell
def _(worldwide_gross_protagonists, worldwide_gross_protagonists_fig):
    worldwide_gross_protagonists_fig(data=worldwide_gross_protagonists, inflation_adjusted=False)
    return


if __name__ == "__main__":
    app.run()
