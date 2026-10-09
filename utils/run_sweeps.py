"""The full experiment set for the report. Results are cached as CSVs in data/results/.

Usage (from the project root):
    python -m utils.run_sweeps            # compute anything not already cached
    python -m utils.run_sweeps --recompute

All experiments use Ohio as the policy state, 20 seeds per scenario and a 20-year horizon
unless stated. Parameter values other than the one being varied are the defaults in
``src/params.py``.
"""
import sys
from pathlib import Path

from src.params import Params
from utils.data import load_initial_state
from utils.experiments import (gain_kappa_grid, load_or_compute, required_subsidy_table,
                               sensitivity_table)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"

HORIZON = 20
N_SEEDS = 20
ELASTICITIES = [0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
SUBSIDIES = [1, 2, 3, 4, 5, 6, 8, 10, 12]
TARGETS = [0.02, 0.05, 0.10, 0.20]   # +2% is close to the real scholarship benchmark (Bartik and Sotherland 2015)
SENSITIVITY_TARGET = 0.10
SENSITIVITY_ELASTICITIES = [0.3, 0.5, 1, 2, 4, 8]
SENSITIVITY = {                       # parameter -> values tried (the default is among them)
    "migration_cost": [0.0, 6.0, 8.0, 10.0, 20.0],   # 6 and 8 match the IRS and CPS move rates; 10 was the old default
    "tie_strength": [0.0, 2.5, 5.0, 10.0, 15.0],
    "beta": [0.25, 0.5, 1.0],
    "move_fraction": [0.05, 0.1, 0.2],
}
HORIZONS = [10, 20, 40]


def compute_all(recompute: bool = False, verbose: bool = True) -> dict:
    init = load_initial_state(ROOT / "data" / "initial_state_2010.csv")
    base = Params(policy_state=init.index("Ohio"), n_years=HORIZON)
    out = {}

    def step(name, fn):
        if verbose:
            print(f"{name} ...", flush=True)
        out[name] = load_or_compute(RESULTS / f"{name}.csv", fn, recompute)

    step("gain_kappa_grid", lambda: gain_kappa_grid(base, init, ELASTICITIES, SUBSIDIES, HORIZON, N_SEEDS))
    step("required_subsidy", lambda: required_subsidy_table(base, init, ELASTICITIES, TARGETS, HORIZON, N_SEEDS))
    import pandas as pd
    for name, values in SENSITIVITY.items():
        step(f"sensitivity_{name}", lambda name=name, values=values: sensitivity_table(
            base, init, name, values, SENSITIVITY_ELASTICITIES, SENSITIVITY_TARGET, HORIZON, N_SEEDS))
    step("sensitivity_horizon", lambda: pd.concat([
        sensitivity_table(base, init, "n_years", [h], SENSITIVITY_ELASTICITIES, SENSITIVITY_TARGET, h, N_SEEDS)
        .assign(parameter="horizon", value=h) for h in HORIZONS], ignore_index=True))
    return out


if __name__ == "__main__":
    compute_all(recompute="--recompute" in sys.argv)
