"""Outcome measures and comparisons with observed data."""
import numpy as np
import pandas as pd

from src.model import History
from src.params import InitialState


def simulated_net_migration_per_1000(history: History, n_agents: int) -> np.ndarray:
    """Cumulative net migration over the run for each state, per 1,000 initial residents."""
    return (history.population[-1] - history.population[0]) / n_agents * 1000


def observed_net_migration_per_1000(path: str, initial: InitialState) -> np.ndarray:
    """Cumulative observed net domestic migration per state, per 1,000 residents in the initial year."""
    obs = pd.read_csv(path).groupby("state")["net_domestic_migration"].sum()
    total = obs.reindex(initial.names).to_numpy(float)
    return total / np.asarray(initial.population, float) * 1000


# ---------------------------------------------------------------------------
# Subsidy outcome measures. A "treated" and a "control" Ensemble (see
# utils/experiments.py) are the same scenario with and without the subsidy,
# run over the same seeds.
# ---------------------------------------------------------------------------

def _check_horizon(ensemble, horizon: int) -> None:
    if not 0 <= horizon < ensemble.population.shape[1]:
        raise ValueError(f"horizon must be between 0 and {ensemble.population.shape[1] - 1}")


def population_gain(treated, control, state: int, horizon: int, relative_to: str = "initial") -> float:
    """Extra population in ``state`` at year ``horizon`` caused by the subsidy.

    Mean over seeds of (treated - control), as a fraction of the state's initial
    population, so 0.10 means +10% of the starting population. With
    ``relative_to="control"`` it is instead a fraction of the control run's population
    at ``horizon`` (0.10 = 10% more people than there would have been without the subsidy).
    """
    _check_horizon(treated, horizon)
    _check_horizon(control, horizon)
    diff = treated.population[:, horizon, state].mean() - control.population[:, horizon, state].mean()
    if relative_to == "initial":
        base = treated.population[:, 0, state].mean()
    elif relative_to == "control":
        base = control.population[:, horizon, state].mean()
    else:
        raise ValueError('relative_to must be "initial" or "control"')
    return float(diff / base)


def capitalisation_share(treated, control, state: int, subsidy: float, horizon: int) -> float:
    """Fraction of the subsidy absorbed by higher rent at year ``horizon``.

    (mean rent with subsidy - mean rent without) / subsidy, both in $k/yr.
    0 means rent did not rise; 1 means rent rose by the full subsidy.
    """
    if subsidy <= 0:
        raise ValueError("capitalisation share is undefined for a zero subsidy")
    _check_horizon(treated, horizon)
    _check_horizon(control, horizon)
    rise = treated.rent[:, horizon, state].mean() - control.rent[:, horizon, state].mean()
    return float(rise / subsidy)


def settling_change(ensemble, state: int, window: int = 5) -> float:
    """Relative change in the mean population of ``state`` over the last ``window`` years.

    Close to 0 means the run has levelled off. Large values mean it is still drifting, so
    results at the final year depend on the chosen horizon.
    """
    mean_pop = ensemble.population[:, :, state].mean(axis=0)
    if window >= len(mean_pop):
        raise ValueError("window is longer than the run")
    return float((mean_pop[-1] - mean_pop[-1 - window]) / mean_pop[-1 - window])


def required_subsidy(gain_at, target: float, s_max: float, tol: float = 0.05, max_iter: int = 40) -> float:
    """Smallest s in [0, s_max] with gain_at(s) >= target, by bisection.

    Assumes ``gain_at`` increases with s. Returns 0.0 if the target is met without a
    subsidy and ``inf`` if it is not met at ``s_max``. ``tol`` is in the units of s.
    """
    if gain_at(0.0) >= target:
        return 0.0
    if gain_at(s_max) < target:
        return float("inf")
    lo, hi = 0.0, s_max
    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if gain_at(mid) >= target else (mid, hi)
    return hi


def population_gain_se(treated, control, state: int, horizon: int) -> float:
    """Standard error (across paired seeds) of ``population_gain``."""
    _check_horizon(treated, horizon)
    _check_horizon(control, horizon)
    per_seed = (treated.population[:, horizon, state] - control.population[:, horizon, state]) \
        / treated.population[:, 0, state].mean()
    return float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))


def annual_move_rate(history: History, n_agents: int) -> float:
    """Average share of agents who change state per year over a run."""
    flows = history.flows                      # (T, S, S), flows[t, i, j] = moved i -> j
    movers = flows.sum(axis=(1, 2)) - np.trace(flows, axis1=1, axis2=2)
    return float(movers.mean() / n_agents)


def cost_per_extra_resident(treated, control, state: int, subsidy: float, horizon: int) -> float:
    """Subsidy paid in year ``horizon`` per extra resident it attracted or kept, in $k/yr.

    The subsidy goes to everyone living in ``state``, not only to the extra residents, so this is
    subsidy * treated population / (treated - control population). Equivalently
    subsidy / (1 - control/treated): the more of the payments go to people who would have lived
    there anyway, the more each extra resident costs. ``inf`` if the subsidy added nobody.
    """
    _check_horizon(treated, horizon)
    _check_horizon(control, horizon)
    t = treated.population[:, horizon, state].mean()
    extra = t - control.population[:, horizon, state].mean()
    return float(subsidy * t / extra) if extra > 0 else float("inf")
