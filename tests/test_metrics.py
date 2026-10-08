"""Checks for the subsidy outcome measures and the experiment helpers."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from utils.data import toy_initial_state
from utils.experiments import required_subsidy_for_target, run_ensemble
from utils.metrics import (capitalisation_share, population_gain, required_subsidy,
                           settling_change)

INIT = toy_initial_state(identical=True)
BASE = Params(n_agents=2000, n_years=10, policy_state=0, elasticity=1.0)


def test_gain_is_zero_when_treated_equals_control():
    e = run_ensemble(BASE, INIT, n_seeds=3)
    assert population_gain(e, e, state=0, horizon=10) == 0.0


def test_gain_positive_with_subsidy_and_capitalisation_positive():
    control = run_ensemble(BASE, INIT, n_seeds=6)
    treated = run_ensemble(replace(BASE, subsidy=5.0), INIT, n_seeds=6)
    assert population_gain(treated, control, 0, 10) > 0
    assert capitalisation_share(treated, control, 0, 5.0, 10) > 0


def test_capitalisation_share_larger_when_housing_supply_inelastic():
    shares = {}
    for eps in (0.3, 5.0):
        p = replace(BASE, elasticity=eps)
        c = run_ensemble(p, INIT, n_seeds=6)
        t = run_ensemble(replace(p, subsidy=5.0), INIT, n_seeds=6)
        shares[eps] = capitalisation_share(t, c, 0, 5.0, 10)
    assert shares[0.3] > shares[5.0]


def test_capitalisation_share_rejects_zero_subsidy_and_bad_horizon():
    e = run_ensemble(BASE, INIT, n_seeds=2)
    with pytest.raises(ValueError):
        capitalisation_share(e, e, 0, 0.0, 5)
    with pytest.raises(ValueError):
        population_gain(e, e, 0, horizon=99)


def test_settling_change_zero_for_constant_population():
    e = run_ensemble(replace(BASE, move_fraction=0.0), INIT, n_seeds=2)
    assert settling_change(e, state=0, window=5) == 0.0


def test_required_subsidy_on_a_known_function():
    g = lambda s: 0.1 * s
    assert required_subsidy(g, target=0.5, s_max=30, tol=0.01) == pytest.approx(5.0, abs=0.02)
    assert required_subsidy(g, target=0.0, s_max=30) == 0.0       # already met
    assert required_subsidy(g, target=10.0, s_max=30) == float("inf")  # unattainable


def test_required_subsidy_rises_with_target_in_the_model():
    kwargs = dict(horizon=10, n_seeds=6, s_max=40.0, tol=0.2)
    low = required_subsidy_for_target(BASE, INIT, target=0.05, **kwargs)
    high = required_subsidy_for_target(BASE, INIT, target=0.15, **kwargs)
    assert 0 < low < high < float("inf")


def test_shared_run_helper_matches_single_target_helper():
    from utils.experiments import required_subsidies_for_targets
    kwargs = dict(horizon=10, n_seeds=6, s_max=40.0, tol=0.2)
    both = required_subsidies_for_targets(BASE, INIT, [0.05, 0.15], **kwargs)
    assert both[0.05] == required_subsidy_for_target(BASE, INIT, 0.05, **kwargs)
    assert both[0.15] == required_subsidy_for_target(BASE, INIT, 0.15, **kwargs)


def test_gain_kappa_grid_shape_and_gain_rises_with_subsidy():
    from utils.experiments import gain_kappa_grid
    df = gain_kappa_grid(BASE, INIT, elasticities=[0.5, 2.0], subsidies=[2.0, 6.0], horizon=10, n_seeds=6)
    assert len(df) == 4 and {"gain", "gain_se", "kappa"} <= set(df.columns)
    for eps in (0.5, 2.0):
        g = df[df["elasticity"] == eps].sort_values("subsidy")["gain"].to_numpy()
        assert g[1] > g[0]


def test_sensitivity_table_rows_and_migration_cost_effect():
    from utils.experiments import sensitivity_table
    df = sensitivity_table(BASE, INIT, "migration_cost", [0.0, 20.0], [1.0], target=0.05,
                           horizon=10, n_seeds=6, s_max=60.0)
    assert len(df) == 2
    # Moving is costlier, so attracting people needs a larger subsidy.
    assert df[df["value"] == 20.0]["s_star"].iloc[0] > df[df["value"] == 0.0]["s_star"].iloc[0]


def test_load_or_compute_caches(tmp_path):
    import pandas as pd
    from utils.experiments import load_or_compute
    calls = []
    def compute():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "sub" / "x.csv"
    load_or_compute(path, compute)
    load_or_compute(path, compute)
    assert len(calls) == 1
    load_or_compute(path, compute, recompute=True)
    assert len(calls) == 2
