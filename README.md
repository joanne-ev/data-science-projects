# Data Science Projects

## Background

This repository contains a curated collection of Python projects applying data science to my areas of interest (e.g., sports, psychology, society). These projects can be [R] research-specific where I use data to provide evidence-based answers to specific research questions or it can be [M] model-specific where I build predictive and descriptive models (ML/DL) to understand the underlying structure of complex datasets or it could be a mixture of both!

> Each project is worked on its own branch to keep finished and in-progress work separate. Only completed, projects are merged into the main branch.

## Current Projects

| Project Name      | Description                                                                                                             | Datasets                                                         | Start Date                            | Branch Name               | Branch Created | Updates                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------- | ------------------------- | -------------- | --------------------------- |
| WSL: Big Four     | [R] Who are the Big Four of the Women's Super League? <br> [M] Who will be the Big Four of the WSL next season (26/27)? | [Fixture Download](https://fixturedownload.com/results/wsl-2025) | [R] 22 June 2026 <br> [M] 2 July 2026 | project/wsl               | ✅             | 29 June 2026: Completed [R] |
| Superhero Fatigue | [R] Are female superhero movies doomed to fail?                                                                         | [Kaggle - mdtoomey](https://www.kaggle.com/datasets/mdtoomey/box-office-of-dc-and-marvel-superhero-movies)                                          | [R] 2 July 2026                       | project/superhero-fatigue | ✅             |                             |
|                   |                                                                                                                         |                                                                  |                                       |                           |                |                             |

## How to run notebooks

[Marimo notebooks](https://marimo.io/) were used in favour of Jupyter Notebooks. Marimo notebooks are Git-friendly reproducible Python scripts `.py`) that can be edited as a notebook as shared as a web app.

[UV](https://docs.astral.sh/uv/) is used as the default Python package and project manager. UV ensures project reproducibility and consistent environment management by syncing the environment via the `pyproject.toml` file, which guarantees that the project will run for you exactly as it did for me.

### To run a Marimo notebook

1. The completed project analysis will be on the `main` branch. If the project is not in the `main` branch, it is still being worked on so will be on it’s own project branch (refer to the table above for project branch names).

   ```bash
   # Clone the repository
   git clone https://codeberg.org/joanne_ev/data-science-projects.git
   cd data-science-projects

   # Switch to a specific project branch (e.g., git checkout project/wsl)
   git checkout <branch-name>
   ```

1. If you have UV installed:
   1. Run `uv sync` to create the virtual environment with the neccessary packages noted in the `pyproject.toml` file
   2. Run `marimo run notebook.py` to run the notebook as an app on a local host.
   3. Run `marimo edit notebook.py` to view the notebook with the code accessible on a local host.
1. If you do not have UV:
   1. Using `pip` :

      ```bash
      # Create a virtual environment
      python -m venv .venv

      # Activate the virtual environment
      source .venv/bin/activate

      # Installs packages using pyproject.toml in current directory
      pip install -e .
      ```

   2. Using `conda`:

      ```bash
      # Create a new conda environment
      conda create --name uv-marimo

      # Activate the conda environment
      conda activate uv-marimo

      # Install UV
      conda install conda-forge::uv
      ```

      Follow the rules laid out in the step above as you now have UV installed in your conda environment.
