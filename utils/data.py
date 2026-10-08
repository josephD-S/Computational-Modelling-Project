"""Loading initial conditions for the model."""
import numpy as np
import pandas as pd

from src.params import InitialState


def load_initial_state(path: str) -> InitialState:
    """Read a CSV with columns: state, population, median_rent_annual, median_income.

    Rent and income are in dollars per year and are converted to $k/yr.
    """
    df = pd.read_csv(path)
    return InitialState(
        names=tuple(df["state"]),
        population=df["population"].to_numpy(float),
        rent=df["median_rent_annual"].to_numpy(float) / 1000.0,
        wage=df["median_income"].to_numpy(float) / 1000.0,
    )


def toy_initial_state(n_states: int = 4, identical: bool = False) -> InitialState:
    """SYNTHETIC placeholder data for development and tests. Not real census data."""
    names = tuple(f"S{i}" for i in range(n_states))
    if identical:
        return InitialState(names, np.full(n_states, 1000.0),
                            np.full(n_states, 15.0), np.full(n_states, 50.0))
    return InitialState(
        names,
        population=np.linspace(1000, 2000, n_states),
        rent=np.linspace(12.0, 20.0, n_states),
        wage=np.linspace(45.0, 60.0, n_states),
    )
