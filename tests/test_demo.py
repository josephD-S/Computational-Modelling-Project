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


def test_scenarios_for_mode_builds_the_right_comparisons():
    from utils.demo import MODES, scenarios_for_mode
    base = Params(**{**BASE.__dict__, "elasticity": 2.0})
    left, right = scenarios_for_mode(MODES[0], base, subsidy=6.0, elasticity=0.5, elasticity_b=3.0)
    assert left[1].subsidy == 0.0 and right[1].subsidy == 6.0 and left[1].elasticity == right[1].elasticity == 0.5
    a, b = scenarios_for_mode(MODES[1], base, subsidy=6.0, elasticity=0.5, elasticity_b=3.0)
    assert a[1].subsidy == b[1].subsidy == 6.0 and (a[1].elasticity, b[1].elasticity) == (0.5, 3.0)
    (only,) = scenarios_for_mode(MODES[2], base, subsidy=4.0, elasticity=1.5, elasticity_b=3.0)
    assert only[1].subsidy == 4.0 and only[1].elasticity == 1.5


def test_flow_demo_renders_every_year_and_scrubbing_swaps_the_image():
    import pytest
    pytest.importorskip("ipywidgets")
    from utils.demo import MODES, build_flow_demo
    base = Params(**{**BASE.__dict__, "n_agents": 500, "n_years": 4})
    d = build_flow_demo(INIT, base=base, dpi=40)
    assert len(d["frames"]) == 5 and d["image"].value[:4] == b"\x89PNG"      # rendered at construction
    two_panel = d["image"].value
    d["year"].value = 0
    assert d["image"].value == d["frames"][0] and d["image"].value != two_panel
    d["controls"]["mode"].value = MODES[2]                                    # single, bigger map
    d["run"].click()
    assert len(d["frames"]) == 5 and d["frames"][4] != two_panel
    assert d["controls"]["elasticity_b"].disabled and d["year"].value == 4
    assert d["play"].repeat is True and d["play"].show_repeat is False        # Play loops; no confusing toggle
    assert len(d["box"].children) == 9                                        # 6 controls, button row, play row, one image
