"""Model checks: invariants, boundary cases and expected directions of effect."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (MigrationModel, choice_probabilities, sample_choices,
                       update_rent, utility)
from src.params import Params
from utils.data import toy_initial_state


def run(params, initial=None):
    return MigrationModel(params, initial or toy_initial_state(identical=True)).run()


def test_choice_probabilities_sum_to_one_and_beta_zero_is_uniform():
    util = np.array([[1.0, 5.0, -3.0], [0.0, 0.0, 0.0]])
    p = choice_probabilities(util, beta=0.7)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.allclose(choice_probabilities(util, beta=0.0), 1 / 3)


def test_choice_probabilities_stable_for_large_utilities():
    p = choice_probabilities(np.array([[1e4, 0.0]]), beta=1.0)
    assert np.all(np.isfinite(p)) and np.isclose(p[0, 0], 1.0)


def test_sample_choices_matches_probabilities():
    rng = np.random.default_rng(1)
    probs = np.tile([0.2, 0.5, 0.3], (50000, 1))
    freq = np.bincount(sample_choices(probs, rng), minlength=3) / 50000
    assert np.allclose(freq, [0.2, 0.5, 0.3], atol=0.01)


def test_update_rent_boundaries():
    rent, old, new = np.array([10.0]), np.array([100.0]), np.array([110.0])
    assert np.allclose(update_rent(rent, old, new, elasticity=1.0), 11.0)
    assert np.allclose(update_rent(rent, old, new, elasticity=1e9), 10.0)  # huge elasticity: rent fixed
    assert np.allclose(update_rent(rent, old, old, elasticity=0.5), 10.0)  # no pop change: no rent change


def test_utility_penalises_moving_and_rewards_subsidy():
    kwargs = dict(skill=np.array([1.0]), wage=np.array([50.0, 50.0]), rent=np.array([10.0, 10.0]),
                  pop_share=np.array([0.5, 0.5]), tie_strength=0.0)
    u = utility(location=np.array([0]), subsidy=np.array([0.0, 0.0]), migration_cost=7.0, **kwargs)
    assert np.isclose(u[0, 0] - u[0, 1], 7.0)
    u = utility(location=np.array([0]), subsidy=np.array([0.0, 3.0]), migration_cost=7.0, **kwargs)
    assert np.isclose(u[0, 1] - u[0, 0], 3.0 - 7.0)


def test_agents_conserved_and_history_shapes():
    p = Params(n_agents=1000, n_years=10)
    h = run(p, toy_initial_state())
    assert (h.population.sum(axis=1) == 1000).all()
    assert h.population.shape == (11, 4) and h.flows.shape == (10, 4, 4)
    assert (h.net_migration.sum(axis=1) == 0).all()
    assert (h.flows.sum(axis=(1, 2)) <= 1000).all()


def test_same_seed_reproducible_different_seed_differs():
    p = Params(n_agents=1000, n_years=5, subsidy=3.0)
    assert np.array_equal(run(p).population, run(p).population)
    assert not np.array_equal(run(p).population, run(replace(p, seed=1)).population)


def test_no_moves_when_move_fraction_zero():
    h = run(Params(n_agents=500, n_years=5, move_fraction=0.0, subsidy=50.0))
    assert (h.population == h.population[0]).all()


def test_identical_states_no_subsidy_has_no_systematic_drift():
    h = run(Params(n_agents=5000, n_years=20, seed=3))
    assert np.abs(h.population[-1] - h.population[0]).max() < 0.05 * 5000


def test_subsidy_attracts_migrants():
    base = run(Params(n_agents=5000, n_years=10, seed=2))
    sub = run(Params(n_agents=5000, n_years=10, seed=2, subsidy=8.0))
    assert sub.population[-1, 0] > base.population[-1, 0]


def test_low_elasticity_erodes_subsidy_effect():
    """Core hypothesis check: same subsidy attracts fewer people when rent responds strongly."""
    gain = {}
    for e in (0.2, 5.0):
        s = np.mean([run(Params(n_agents=5000, n_years=15, seed=k, subsidy=8.0, elasticity=e))
                     .population[-1, 0] for k in range(5)])
        b = np.mean([run(Params(n_agents=5000, n_years=15, seed=k, elasticity=e))
                     .population[-1, 0] for k in range(5)])
        gain[e] = s - b
    assert gain[5.0] > gain[0.2]


def test_rent_rises_in_policy_state_with_subsidy():
    h = run(Params(n_agents=5000, n_years=10, subsidy=8.0, elasticity=0.5))
    assert h.rent[-1, 0] > h.rent[0, 0]


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        Params(elasticity=0)
    with pytest.raises(ValueError):
        Params(move_fraction=1.5)


def test_initial_state_index_lookup():
    init = toy_initial_state()
    assert init.index("S2") == 2
    with pytest.raises(ValueError):
        init.index("Nowhere")


def test_simulated_net_migration_per_1000_sums_to_zero():
    from utils.metrics import simulated_net_migration_per_1000
    h = run(Params(n_agents=2000, n_years=5), toy_initial_state())
    assert abs(simulated_net_migration_per_1000(h, 2000).sum()) < 1e-9


def test_rent_stays_finite_when_a_state_empties_out():
    """A huge subsidy drains the other states completely; rent must remain finite and positive."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any divide-by-zero warning fails the test
        h = run(Params(n_agents=1000, n_years=15, subsidy=500.0, move_fraction=1.0, beta=2.0),
                toy_initial_state(identical=True))
    assert (h.population[-1, 1:] == 0).all()  # the other states really did empty
    assert np.isfinite(h.rent).all() and (h.rent > 0).all()
