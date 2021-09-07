# starbucks_challenge - Pipelines and Data Workflows

## Installation

    pip install pipenv  # if you haven't already
    pipenv install
    pipenv run python pipelines.py

## Running code

```
    # generate interim datasets
    python -m data_processing.profile ../data/0_raw/profile.json ../data/1_interim/profile.pkl
    python -m data_processing.portfolio ../data/0_raw/portfolio.json ../data/1_interim/portfolio.pkl 
    python -m data_processing.transcript ../data/0_raw/transcript.json ../data/1_interim/transcript.pkl

    # generate combine datasets
    python -m data_processing.offer_response ../data/1_interim/transcript.pkl ../data/1_interim/profile.pkl ../data/1_interim/portfolio.pkl ../data/1_interim/offer_response.pkl

    # model building 
    python -m offer_response.model ../data/1_interim/offer_response.pkl ../output/models/offer_response.pkl
```

## Directory structure

    ├── Pipfile               <- The Pipfile for reproducing the pipelines environment
    ├── pipelines.py          <- The CLI entry point for all the pipelines
    ├── <repo_name>           <- Code for the various steps of the pipelines
    │   ├──  __init__.py
    │   ├── etl.py            <- Download, generate, and process data
    │   ├── visualize.py      <- Create exploratory and results oriented visualizations
    │   ├── features.py       <- Turn raw data into features for modeling
    │   └── train.py          <- Train and evaluate models
    └── tests
        ├── fixtures          <- Where to put example inputs and outputs
        │   ├── input.json    <- Test input data
        │   └── output.json   <- Test output data
        └── test_pipelines.py <- Integration tests for the HTTP API


Installing Pipenv
Pipenv & Virtual Environments https://pipenv-fork.readthedocs.io/en/latest/install.html

For Mac OSX user: 
export SYSTEM_VERSION_COMPAT=1
pipenv shell

jupyter-lab

Fall back to more traditional method
pipenv lock -r > requirements.txt

Installing PyTorch
✗ pipenv install --verbose "https://download.pytorch.org/whl/cpu/torch-1.9.0-cp39-none-macosx_10_9_x86_64.whl"


LailaSabar
/
Starbucks-Capstone-Challenge, https://github.com/LailaSabar/Starbucks-Capstone-Challenge, https://towardsdatascience.com/starbucks-capstone-challenge-b95b0931bab4


