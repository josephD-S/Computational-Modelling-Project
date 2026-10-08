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
