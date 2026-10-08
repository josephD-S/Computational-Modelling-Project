"""Running many seeds of a scenario, and searching for the subsidy that reaches a target."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import MigrationModel
from src.params import InitialState, Params
from utils.metrics import (capitalisation_share, population_gain, population_gain_se,
                           required_subsidy)


@dataclass
class Ensemble:
    """Outputs of the same scenario run over several seeds. T = years, S = states."""

    population: np.ndarray  # (n_seeds, T+1, S)
    rent: np.ndarray        # (n_seeds, T+1, S)


def run_ensemble(params: Params, initial: InitialState, n_seeds: int = 20, base_seed: int = 0) -> Ensemble:
    """Run ``n_seeds`` replicates with seeds base_seed, base_seed+1, ...

    Using the same seeds for a treated and a control scenario pairs the runs: they share
    the same initial agent incomes, which reduces noise in the difference between them.
    """
    histories = [MigrationModel(replace(params, seed=base_seed + k), initial).run() for k in range(n_seeds)]
    return Ensemble(np.array([h.population for h in histories]), np.array([h.rent for h in histories]))


def required_subsidies_for_targets(params: Params, initial: InitialState, targets, horizon: int,
                                   n_seeds: int = 20, base_seed: int = 0, s_max: float = 30.0,
                                   tol: float = 0.05) -> dict:
    """Required subsidy for each target gain, sharing the simulation runs between targets.

    Returns {target: smallest subsidy ($k/yr) whose population gain over the no-subsidy run
    at year ``horizon`` reaches the target}. 0.0 means no subsidy is needed; ``inf`` means
    ``s_max`` is not enough. Gains are cached per subsidy so each is simulated only once.
    """
    state = params.policy_state
    control = run_ensemble(replace(params, subsidy=0.0), initial, n_seeds, base_seed)
    cache = {}

    def gain_at(s: float) -> float:
        if s not in cache:
            treated = run_ensemble(replace(params, subsidy=s), initial, n_seeds, base_seed)
            cache[s] = population_gain(treated, control, state, horizon)
        return cache[s]

    return {t: required_subsidy(gain_at, t, s_max, tol) for t in targets}


def required_subsidy_for_target(params: Params, initial: InitialState, target: float, horizon: int,
                                n_seeds: int = 20, s_max: float = 30.0, tol: float = 0.05,
                                base_seed: int = 0) -> float:
    """Smallest subsidy ($k/yr) at which the policy state's population gain over the
    no-subsidy run, at year ``horizon``, reaches ``target`` (a fraction of its initial population).

    Returns 0.0 if no subsidy is needed and ``inf`` if ``s_max`` is not enough.
    """
    return required_subsidies_for_targets(params, initial, [target], horizon, n_seeds, base_seed,
                                          s_max, tol)[target]


# ---------------------------------------------------------------------------
# Sweeps. Each returns a tidy DataFrame, so results can be cached as CSV.
# ---------------------------------------------------------------------------

def gain_kappa_grid(params: Params, initial: InitialState, elasticities, subsidies, horizon: int,
                    n_seeds: int = 20) -> pd.DataFrame:
    """Population gain (with standard error) and capitalisation share on an elasticity x subsidy grid."""
    state = params.policy_state
    rows = []
    for eps in elasticities:
        p = replace(params, elasticity=eps, n_years=max(params.n_years, horizon))
        control = run_ensemble(replace(p, subsidy=0.0), initial, n_seeds)
        for s in subsidies:
            treated = run_ensemble(replace(p, subsidy=s), initial, n_seeds)
            rows.append({"elasticity": eps, "subsidy": s,
                         "gain": population_gain(treated, control, state, horizon),
                         "gain_se": population_gain_se(treated, control, state, horizon),
                         "kappa": capitalisation_share(treated, control, state, s, horizon)})
    return pd.DataFrame(rows)


def required_subsidy_table(params: Params, initial: InitialState, elasticities, targets, horizon: int,
                           n_seeds: int = 20, n_batches: int = 5, s_max: float = 30.0) -> pd.DataFrame:
    """Required subsidy per (elasticity, target), repeated over ``n_batches`` independent seed batches.

    Columns: s_mean, s_min, s_max_batch (spread across batches, a measure of simulation noise),
    and ``n_unattainable`` (batches where ``s_max`` was not enough).
    """
    rows = []
    for eps in elasticities:
        p = replace(params, elasticity=eps, n_years=max(params.n_years, horizon))
        per_batch = [required_subsidies_for_targets(p, initial, targets, horizon, n_seeds,
                                                    base_seed=1000 * b, s_max=s_max)
                     for b in range(n_batches)]
        for t in targets:
            vals = np.array([d[t] for d in per_batch])
            finite = vals[np.isfinite(vals)]
            rows.append({"elasticity": eps, "target": t,
                         "s_mean": finite.mean() if len(finite) else np.inf,
                         "s_min": finite.min() if len(finite) else np.inf,
                         "s_max_batch": finite.max() if len(finite) else np.inf,
                         "n_unattainable": int((~np.isfinite(vals)).sum())})
    return pd.DataFrame(rows)


def sensitivity_table(params: Params, initial: InitialState, name: str, values, elasticities,
                      target: float, horizon: int, n_seeds: int = 20, s_max: float = 30.0) -> pd.DataFrame:
    """Required subsidy as one parameter (``name``) is varied, across elasticities."""
    rows = []
    for v in values:
        for eps in elasticities:
            # n_years is set last so that varying the horizon (name='n_years') cannot clash with it.
            p = replace(params, **{name: v, "elasticity": eps, "n_years": max(params.n_years, horizon)})
            s = required_subsidy_for_target(p, initial, target, horizon, n_seeds, s_max=s_max)
            rows.append({"parameter": name, "value": v, "elasticity": eps, "s_star": s})
    return pd.DataFrame(rows)


def load_or_compute(path, compute, recompute: bool = False) -> pd.DataFrame:
    """Return the CSV at ``path`` if it exists, otherwise run ``compute()`` and save it there."""
    path = Path(path)
    if path.exists() and not recompute:
        return pd.read_csv(path)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
