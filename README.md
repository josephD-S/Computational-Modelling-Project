# Computational-Modelling-Project

Assignment for CITS4403: Computational Modelling. An agent-based model of young-adult interstate migration with rent feedback, used to estimate the subsidy a state needs to sustain net in-migration (see `Project_Proposal_Subsidy_Capitalisation.md`).

## Structure

```
src/            Main model code (agents, states, simulation)
utils/          Helper functions (data loading, plotting, metrics)
data/           Small state-level input tables
notebooks/      Analysis, experiments and demonstration notebooks
tests/          Model checks (boundary cases, conservation)
requirements.txt
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The Census data the model needs is already included in `data/`, so **no API key is required to run the project**:

- `data/initial_state_2010.csv`: population, median rent and median income for 8 states (ACS 2010, 1-year).
- `data/observed_net_migration.csv`: observed net domestic migration, 2010-11 to 2018-19 (Census population estimates), used only to check the baseline model.

To regenerate these files from the Census API (optional), get a free key from https://api.census.gov/data/key_signup.html, put `CENSUS_API_KEY=<your key>` in a `.env` file in the project root, and run `python -m utils.fetch_census`.

Usage instructions for the model will be added as it is built.
