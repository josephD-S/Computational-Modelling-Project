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

Usage instructions will be added as the model is built.
