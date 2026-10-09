"""Checks for the demonstration helpers (no widgets needed)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.params import Params
from utils.data import toy_initial_state
from utils.demo import STATE_XY, _positions, animate_flows, plot_scenario, scenario_summary

INIT = toy_initial_state(identical=True)
BASE = Params(n_agents=800, n_years=6, policy_state=0)


def test_scenario_summary_gain_positive_with_subsidy_and_kappa_nan_without():
    with_sub = scenario_summary(Params(**{**BASE.__dict__, "subsidy": 6.0}), INIT, n_seeds=4)
    assert with_sub["gain"] > 0 and with_sub["kappa"] > 0
    no_sub = scenario_summary(BASE, INIT, n_seeds=4)
    assert no_sub["gain"] == 0.0 and np.isnan(no_sub["kappa"])


def test_plot_scenario_has_three_panels():
    p = Params(**{**BASE.__dict__, "subsidy": 4.0})
    fig = plot_scenario(scenario_summary(p, INIT, n_seeds=3), p, INIT)
    assert len(fig.axes) == 3


def test_positions_use_map_for_known_states_and_a_circle_otherwise():
    assert _positions(list(STATE_XY)).shape == (8, 2)
    xy = _positions(["a", "b", "c", "d"])
    assert xy.shape == (4, 2) and np.allclose(np.hypot(xy[:, 0], xy[:, 1]), 0.4)


def test_animate_flows_writes_a_gif(tmp_path):
    out = tmp_path / "flows.gif"
    p = Params(**{**BASE.__dict__, "n_agents": 300, "n_years": 3, "subsidy": 5.0})
    animate_flows([("test", p)], INIT, path=str(out), fps=2)
    assert out.exists() and out.stat().st_size > 1000


def test_us_map_loads_and_projects_sensibly():
    from utils import usmap
    states = usmap.load_states()
    assert {"Ohio", "California", "Texas", "Florida"} <= set(states) and len(states) >= 48
    x, y = usmap.albers([-96.0, -120.0, -75.0], [37.5, 37.0, 40.0])    # projection origin, west coast, east coast
    assert abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9 and x[1] < 0 < x[2]
    ohio, cal = usmap.centroid(states["Ohio"]), usmap.centroid(states["California"])
    assert cal[0] < ohio[0]                                            # California lies west of Ohio


def test_positions_follow_the_map_when_names_are_us_states():
    pos = _positions(["California", "Ohio", "Florida"])
    assert pos[0, 0] < pos[1, 0] and pos[2, 1] < pos[1, 1]             # CA west of OH, FL south of OH


def test_demo_widgets_update_one_image_in_place_and_button_answers():
    import pytest
    pytest.importorskip("ipywidgets")
    from utils.demo import build_demo
    d = build_demo(INIT, n_seeds=3, base=Params(**{**BASE.__dict__, "subsidy": 2.0}))
    first = d["image"].value
    assert first[:4] == b"\x89PNG"                       # drawn at construction
    d["sliders"]["subsidy"].value = 9.0
    d["sliders"]["elasticity"].value = 0.5
    assert d["image"].value != first                      # the same widget is updated, not a new one added
    assert len(d["box"].children) == 7                    # 4 sliders, target row, answer, one image
    d["target"].value = 0.05
    d["button"].click()
    assert "Required subsidy" in d["answer"].value and "per year" in d["answer"].value
