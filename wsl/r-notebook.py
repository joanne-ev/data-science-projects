import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    data_23= pl.read_csv("https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/wsl/wsl/data/wsl-2023-UTC.csv")
    data_24= pl.read_csv("https://codeberg.org/joanne_ev/data-science-projects/raw/branch/project/wsl/wsl/data/wsl-2024-UTC.csv")
    return


if __name__ == "__main__":
    app.run()
