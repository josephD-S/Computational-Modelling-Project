# Subsidy capitalisation in an agent-based migration model

CITS4403 Computational Modelling project. An agent-based model of interstate migration with rent feedback.

**Research question.** How large a subsidy must a state (here Ohio) offer its residents to reach a target population gain once rent feedback is accounted for, and how does this depend on housing supply elasticity?

**Model.** 5,000 agents live in 8 U.S. states (CA, NY, IL, OH, TX, FL, CO, NC), initialised from 2010 Census data. Each year a share of agents reconsider where to live and choose a state by logit choice over income, rent, a subsidy (Ohio only), social ties and a migration cost. Rent then responds to each state's population change through a housing supply elasticity, so the subsidy attracts people, people raise rent, and higher rent erodes the subsidy (capitalisation).

**Main findings** (within the model, not forecasts):
- The less elastic the housing supply, the more of the subsidy is absorbed by rent (about 80-90% at elasticity 0.3, about 20% at 8) and the larger the subsidy needed for a given gain.
- The required subsidy falls with diminishing returns as supply becomes more elastic. Where it levels off depends on whether the gain is measured against Ohio's initial population or against where Ohio would otherwise be.
- Because every resident is paid, most of the money goes to people who would have lived in Ohio anyway, so each extra resident costs several times the headline subsidy.
- The shape of these results is robust to the other parameters; their absolute size is not.

## Structure

```
src/            The model: parameters and initial state (params.py), agents and yearly update (model.py)
utils/          Data download and loading, outcome measures, experiments and sweeps, plots, interactive demo
data/           2010 Census inputs, observed migration, state outlines; data/results/ holds cached experiment output
notebooks/      project_walkthrough.ipynb: the full analysis and the interactive demonstration
tests/          Unit tests (pytest): model rules, boundary cases, outcome measures, data loading, demo helpers
requirements.txt
```

## Setup

Tested with Python 3.10.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Open `notebooks/project_walkthrough.ipynb` in Jupyter or VS Code and run all cells (about 20 seconds). It loads the data, runs the model, and reproduces every result and figure. Section 8 has the interactive demonstration (sliders and flow maps), which needs a live kernel.

The full experiment set takes about 4 minutes, so its results are cached in `data/results/` and the notebook reads them. To regenerate them, run this from the project root:

```
python -m utils.run_sweeps --recompute
```

or set `RECOMPUTE = True` in the notebook. Run the tests with `python -m pytest`.

## Data

The Census data the model needs is already included in `data/`, so **no API key is required to run the project**:

- `data/initial_state_2010.csv`: population, median rent and median household income for the 8 states (ACS 2010, 1-year).
- `data/observed_net_migration.csv`: observed net domestic migration, 2010-11 to 2018-19 (Census population estimates), used only to check the baseline model.
- `data/us_states_contiguous.geojson`: state outlines for the flow maps (contiguous U.S. states from the U.S. Census Bureau cartographic boundaries, public domain, via the PublicaMundi MappingAPI GeoJSON; coordinates rounded to three decimals). Used only for drawing.

To regenerate these files from the Census API (optional), get a free key from https://api.census.gov/data/key_signup.html, put `CENSUS_API_KEY=<your key>` in a `.env` file in the project root, and run `python -m utils.fetch_census`.
