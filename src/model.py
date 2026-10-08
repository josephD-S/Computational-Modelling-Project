"""Agent-based model of interstate migration with rent feedback.

Agents are stored as NumPy arrays (one entry per agent) rather than objects,
which keeps large parameter sweeps fast. Every agent still makes its own
random choice each year, so this is still an agent-based model.

Yearly loop (see ``MigrationModel.step``):
  1. A random fraction of agents reconsider where to live.
  2. Each computes a utility for every state (income - rent + subsidy
     - migration cost + social-tie term) and picks one at random with
     softmax (logit) probabilities.
  3. State populations are recounted.
  4. Rent in each state is updated from its population change:
         rent_new = rent_old * (pop_new / pop_old) ** (1 / elasticity)
     so the new rents shape next year's decisions (the feedback loop).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.params import InitialState, Params


# ---------------------------------------------------------------------------
# Pure functions (easy to test in isolation)
# ---------------------------------------------------------------------------

def utility(skill, location, wage, rent, subsidy, pop_share, migration_cost, tie_strength):
    """Utility of every state for each agent, shape (n_agents, n_states), in $k.

    skill     (m,)  income multiplier for each agent
    location  (m,)  current state index of each agent
    wage      (S,)  base wage in each state
    rent      (S,)  rent in each state
    subsidy   (S,)  subsidy paid in each state (zero except the policy state)
    pop_share (S,)  fraction of all agents currently living in each state
    """
    n_states = len(wage)
    income = skill[:, None] * wage[None, :]                      # (m, S)
    u = income - rent[None, :] + subsidy[None, :]
    u = u + tie_strength * pop_share[None, :]                    # social ties
    is_move = np.arange(n_states)[None, :] != location[:, None]  # True where j != current
    return u - migration_cost * is_move


def choice_probabilities(util, beta):
    """Softmax over states for each agent. Rows sum to 1. beta=0 gives uniform choice."""
    z = beta * util
    z = z - z.max(axis=1, keepdims=True)  # subtract row max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sample_choices(probs, rng):
    """Draw one state per row of ``probs`` (inverse-CDF sampling)."""
    cdf = np.cumsum(probs, axis=1)
    r = rng.random(probs.shape[0])[:, None]
    choice = (r > cdf).sum(axis=1)
    return np.minimum(choice, probs.shape[1] - 1)  # guard against float rounding at cdf end


def update_rent(rent, pop_old, pop_new, elasticity):
    """Rent responds to population change: rent * (pop_new/pop_old) ** (1/elasticity)."""
    return rent * (pop_new / pop_old) ** (1.0 / elasticity)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class History:
    """Recorded output of a run. T = number of years, S = number of states."""

    population: np.ndarray  # (T+1, S) agents in each state, including the initial year
    rent: np.ndarray        # (T+1, S)
    flows: np.ndarray       # (T, S, S) flows[t, i, j] = agents who moved i -> j in year t

    @property
    def net_migration(self) -> np.ndarray:
        """(T, S) inflow minus outflow per state per year."""
        return np.diff(self.population, axis=0)


class MigrationModel:
    def __init__(self, params: Params, initial: InitialState):
        if not 0 <= params.policy_state < initial.n_states:
            raise ValueError("policy_state is not a valid state index")
        self.p = params
        self.init = initial
        self.rng = np.random.default_rng(params.seed)
        S = initial.n_states

        # Allocate agents to states in proportion to initial population.
        shares = np.asarray(initial.population, float)
        shares = shares / shares.sum()
        counts = np.floor(shares * params.n_agents).astype(int)
        counts[np.argmax(shares)] += params.n_agents - counts.sum()  # fix rounding remainder
        self.location = np.repeat(np.arange(S), counts)

        # Heterogeneous income multiplier (median 1).
        self.skill = self.rng.lognormal(0.0, params.skill_sigma, params.n_agents)

        self.wage = np.asarray(initial.wage, float)
        self.rent = np.asarray(initial.rent, float).copy()
        self.subsidy = np.zeros(S)
        self.subsidy[params.policy_state] = params.subsidy

        self._pop = [self._count()]
        self._rent = [self.rent.copy()]
        self._flows = []

    def _count(self) -> np.ndarray:
        return np.bincount(self.location, minlength=self.init.n_states)

    def step(self) -> None:
        S = self.init.n_states
        pop_old = self._count()
        pop_share = pop_old / self.p.n_agents

        movers = np.flatnonzero(self.rng.random(self.p.n_agents) < self.p.move_fraction)
        old_loc = self.location[movers]
        util = utility(self.skill[movers], old_loc, self.wage, self.rent, self.subsidy,
                       pop_share, self.p.migration_cost, self.p.tie_strength)
        new_loc = sample_choices(choice_probabilities(util, self.p.beta), self.rng)
        self.location[movers] = new_loc

        flows = np.zeros((S, S), dtype=int)
        np.add.at(flows, (old_loc, new_loc), 1)

        pop_new = self._count()
        # Floor both populations at 1 so a state that empties out (e.g. under a very large
        # subsidy elsewhere) does not cause a division by zero in the rent update.
        self.rent = update_rent(self.rent, np.maximum(pop_old, 1), np.maximum(pop_new, 1),
                                self.p.elasticity)

        self._pop.append(pop_new)
        self._rent.append(self.rent.copy())
        self._flows.append(flows)

    def run(self, n_years: Optional[int] = None) -> History:
        for _ in range(self.p.n_years if n_years is None else n_years):
            self.step()
        return self.history()

    def history(self) -> History:
        S = self.init.n_states
        flows = np.array(self._flows) if self._flows else np.zeros((0, S, S), dtype=int)
        return History(np.array(self._pop), np.array(self._rent), flows)
