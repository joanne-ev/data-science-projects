# Data Science Projects

## Background

This repository contains a curated collection of Python projects applying data science to my areas of interest (e.g., sports, psychology, society). These projects can be [R] research-specific where I use data to provide evidence-based answers to specific research questions or it can be [M] model-specific where I build predictive and descriptive models (ML/DL) to understand the underlying structure of complex datasets or it could be a mixture of both!

> Each project is worked on its own branch to keep finished and in-progress work separate. Only completed, projects are merged into the main branch.

## Current Projects

| Project Name                           | Description                                                                                                        | Datasets                                                                                                                                                            | Start Date | Branch Name       | Branch Created |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------- | -------------- |
| DC vs Marvel: Battle of the Box Office | [R] Which comic universe is most popular?                                                                          |                                                                                                                                                                     |            | project/dc-marvel |                |
| WSL: Big Four                          | [R] Who are the Big Four of the Women's Super League? [M] Who will be the Big Four of the WSL next season (26/27)? | [25/26](https://fixturedownload.com/results/wsl-2025), [24/25](https://fixturedownload.com/results/wsl-2024), [23/24](https://fixturedownload.com/results/wsl-2023) |            | project/wsl       | ✅              |

## How to run notebooks

[Marimo notebooks](https://marimo.io/) were used in favour of Jupyter Notebooks. Marimo notebooks are Git-friendly reproducible Python scripts `.py`) that can be edited as a notebook as shared as a web app.

[UV](https://docs.astral.sh/uv/) is used as the default Python package and project manager. UV ensures project reproducibility and consistent environment management by syncing the environment via the `pyproject.toml` file, which guarantees that the project will run for you exactly as it did for me.

### To run a Marimo notebook

1. The completed project analysis will be on the `main` branch. If the project is not in the `main` branch, it is still being worked on so will be on it’s own project branch (refer to the table above for project branch names).
    ```bash
    # Clone the repository
    git clone <repo-url>
    cd <repo-name>

    # Switch to a specific project branch
    git checkout <branch-name>
    ```

1. If you have UV:
    1. Run `uv sync` to create the virtual environment with the neccessary packages noted in the `pyproject.toml` file
    2. Run `marimo edit notebook.py` to run the actual Marimo notebook on a local host.
2. If you do not have UV:
    1. You can run the downloaded `notebook.py` in [molab](https://molab.marimo.io/notebooks), which allows you to create, run and share cloud-hosted marimo notebookarimo
