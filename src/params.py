"""Parameters and initial conditions for the migration model.

All money values are in thousands of dollars per year ($k/yr).
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    """Model parameters. Frozen so a run cannot silently change them;
    use ``dataclasses.replace(params, subsidy=5.0)`` to build sweep variants.
    """

    # --- Manipulated parameters (the four experimental variables) ---
    subsidy: float = 0.0           # annual subsidy ($k) paid to residents of the policy state
    elasticity: float = 1.0        # housing supply elasticity (> 0); high = rent barely responds
    migration_cost: float = 10.0   # utility penalty ($k-equivalent) for living somewhere new
    tie_strength: float = 5.0      # utility per unit of a state's population share (social ties)

    # --- Fixed constants (placeholders until calibrated; justify in the report) ---
    policy_state: int = 0          # index of the state offering the subsidy
    n_agents: int = 5000           # size of the simulated agent sample
    n_years: int = 20              # simulation horizon (one step = one year)
    move_fraction: float = 0.1     # share of agents who reconsider their location each year
    beta: float = 0.5              # choice sensitivity: higher = more deterministic decisions
    skill_sigma: float = 0.3       # spread (log-sd) of the agent income multiplier
    seed: int = 0

    def __post_init__(self):
        if self.elasticity <= 0:
            raise ValueError("elasticity must be > 0")
        if not 0 <= self.move_fraction <= 1:
            raise ValueError("move_fraction must be in [0, 1]")
        if self.n_agents <= 0 or self.n_years < 0:
            raise ValueError("n_agents must be > 0 and n_years >= 0")
        if self.beta < 0 or self.migration_cost < 0 or self.tie_strength < 0:
            raise ValueError("beta, migration_cost and tie_strength must be >= 0")


@dataclass(frozen=True)
class InitialState:
    """Starting values for each state (one entry per state, same order)."""

    names: Tuple[str, ...]
    population: np.ndarray   # used only to allocate agents to states in proportion
    rent: np.ndarray         # $k/yr
    wage: np.ndarray         # $k/yr, held fixed during a run

    def __post_init__(self):
        n = len(self.names)
        for field in ("population", "rent", "wage"):
            if np.asarray(getattr(self, field)).shape != (n,):
                raise ValueError(f"{field} must have one value per state ({n})")
        if np.any(np.asarray(self.rent) <= 0):
            raise ValueError("rent must be positive")
        if np.any(np.asarray(self.population) <= 0):
            raise ValueError("population must be positive")

    @property
    def n_states(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Index of a state by name, e.g. ``initial.index("Ohio")``."""
        return self.names.index(name)
