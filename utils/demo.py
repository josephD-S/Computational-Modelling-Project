"""Interactive demonstration: live sliders and an animation of migration flows between states.

``interactive_demo`` needs ipywidgets and a Jupyter front end. ``animate_flows`` and
``plot_scenario`` work anywhere matplotlib does.
"""
import io
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from src.model import MigrationModel
from src.params import InitialState, Params
from utils.experiments import required_subsidy_for_target, run_ensemble
from utils.metrics import capitalisation_share, population_gain
from matplotlib.collections import PolyCollection

from utils import usmap
from utils.plots import GRID, INK, INK2, SEQ, SERIES, SURFACE, _style

# Approximate state centres (longitude, latitude) for the map layout.
STATE_XY = {
    "California": (-119.5, 37.0), "New York": (-75.5, 42.9), "Illinois": (-89.2, 40.0),
    "Ohio": (-82.8, 40.3), "Texas": (-99.0, 31.5), "Florida": (-82.4, 28.6),
    "Colorado": (-105.5, 39.0), "North Carolina": (-79.4, 35.6),
}
ABBR = {"California": "CA", "New York": "NY", "Illinois": "IL", "Ohio": "OH",
        "Texas": "TX", "Florida": "FL", "Colorado": "CO", "North Carolina": "NC"}


# ---------------------------------------------------------------------------
# Live scenario comparison
# ---------------------------------------------------------------------------

def scenario_summary(params: Params, initial: InitialState, n_seeds: int = 20) -> dict:
    """Run a scenario and its no-subsidy control over the same seeds and summarise the policy state."""
    state, T = params.policy_state, params.n_years
    control = run_ensemble(replace(params, subsidy=0.0), initial, n_seeds)
    treated = run_ensemble(params, initial, n_seeds)
    return {
        "control": control, "treated": treated, "state": state,
        "gain": population_gain(treated, control, state, T),
        "kappa": capitalisation_share(treated, control, state, params.subsidy, T) if params.subsidy > 0 else np.nan,
    }


def plot_scenario(summary: dict, params: Params, initial: InitialState, pyplot: bool = True):
    """Ohio's population and rent with and without the subsidy, plus the headline numbers.

    ``pyplot=False`` builds a standalone Figure that the notebook backend never sees (used by the
    sliders, so redrawing cannot leave extra copies behind).
    """
    state = summary["state"]
    years = np.arange(params.n_years + 1)
    grid = {"width_ratios": [1, 1, 0.8]}
    if pyplot:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw=grid)
    else:
        fig = Figure(figsize=(11.5, 3.6))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3, gridspec_kw=grid)
    for ax, key, label in [(axes[0], "population", "agents"), (axes[1], "rent", "rent ($k per year)")]:
        for color, name, ens in [(SERIES[0], "with subsidy", summary["treated"]),
                                 (SERIES[1], "no subsidy", summary["control"])]:
            data = getattr(ens, key)[:, :, state]
            ax.fill_between(years, np.percentile(data, 10, axis=0), np.percentile(data, 90, axis=0),
                            color=color, alpha=0.2, lw=0)
            ax.plot(years, data.mean(axis=0), color=color, lw=2, label=name)
            ax.annotate(name, (years[-1], data.mean(axis=0)[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set(xlabel="year", ylabel=label, xlim=(0, params.n_years * 1.25))
        ax.set_title(f"{initial.names[state]}: {key}", fontsize=9, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    ax = axes[2]
    ax.axis("off")
    kappa = "n/a (no subsidy)" if np.isnan(summary["kappa"]) else f"{summary['kappa']:.2f}"
    lines = [("Population gain", f"{summary['gain']:+.0%} of initial"),
             ("Capitalisation share", kappa),
             ("Ohio agents, year 0", f"{summary['treated'].population[:, 0, state].mean():.0f}"),
             (f"Ohio agents, year {params.n_years}", f"{summary['treated'].population[:, -1, state].mean():.0f}"
              f" (control {summary['control'].population[:, -1, state].mean():.0f})")]
    for i, (a, b) in enumerate(lines):
        ax.text(0.0, 0.88 - i * 0.22, a, fontsize=8.5, color=INK2, transform=ax.transAxes)
        ax.text(0.0, 0.88 - i * 0.22 - 0.09, b, fontsize=11, color=INK, weight="bold", transform=ax.transAxes)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def figure_png(fig, dpi: int = 110) -> bytes:
    """Render a figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return buf.getvalue()


def build_demo(initial: InitialState, n_seeds: int = 20, base: Params = None):
    """Build the widgets for ``interactive_demo``. Returns a dict so the pieces can also be driven in tests.

    The plot is rendered off-screen to PNG and shown in a single Image widget whose value is replaced on
    each change, so redrawing never stacks copies of the figure.
    """
    import ipywidgets as w

    base = base or Params(policy_state=initial.index("Ohio"))
    style = {"description_width": "110px"}
    sliders = {
        "subsidy": w.FloatSlider(value=5.0, min=0.0, max=12.0, step=0.5, description="subsidy ($k/yr)",
                                 continuous_update=False, style=style),
        "elasticity": w.FloatLogSlider(value=1.0, base=10, min=np.log10(0.3), max=np.log10(8), step=0.05,
                                       description="elasticity", continuous_update=False, style=style),
        "migration_cost": w.FloatSlider(value=base.migration_cost, min=0.0, max=20.0, step=0.5,
                                        description="migration cost", continuous_update=False, style=style),
        "tie_strength": w.FloatSlider(value=base.tie_strength, min=0.0, max=15.0, step=0.5,
                                      description="tie strength", continuous_update=False, style=style),
    }
    image = w.Image(format="png", layout=w.Layout(width="100%", max_width="950px"))

    def current_params():
        return replace(base, **{k: sl.value for k, sl in sliders.items()})

    def update(_=None):
        params = current_params()
        fig = plot_scenario(scenario_summary(params, initial, n_seeds), params, initial, pyplot=False)
        image.value = figure_png(fig)

    for sl in sliders.values():
        sl.observe(update, names="value")
    update()

    target = w.Dropdown(options=[("+2%", 0.02), ("+5%", 0.05), ("+10%", 0.10), ("+20%", 0.20)], value=0.10,
                        description="target gain", style=style)
    button = w.Button(description="Find required subsidy", button_style="primary")
    answer = w.HTML("")

    def on_click(_):
        answer.value = "Searching..."
        params = current_params()
        s = required_subsidy_for_target(params, initial, target.value, params.n_years, n_seeds)
        answer.value = (f"Required subsidy for a <b>+{target.value:.0%}</b> gain at elasticity {params.elasticity:.2f}: "
                        f"<b>${s:.2f}k per year</b>" if np.isfinite(s) else "Not reachable within $30k per year.")

    button.on_click(on_click)
    box = w.VBox([*sliders.values(), w.HBox([target, button]), answer, image])
    return {"box": box, "sliders": sliders, "target": target, "button": button, "answer": answer, "image": image}


def interactive_demo(initial: InitialState, n_seeds: int = 20, base: Params = None):
    """Sliders for subsidy, elasticity, migration cost and tie strength, with live plots."""
    from IPython.display import display
    display(build_demo(initial, n_seeds, base)["box"])


# ---------------------------------------------------------------------------
# Animation of flows between states
# ---------------------------------------------------------------------------

def _positions(names):
    """Node positions: state centroids on the projected map if available, else approximate lon/lat, else a circle."""
    states = usmap.load_states()
    if all(n in states for n in names):
        return np.array([usmap.centroid(states[n]) for n in names])
    if all(n in STATE_XY for n in names):
        return np.column_stack(usmap.albers(*np.array([STATE_XY[n] for n in names]).T))
    angle = np.linspace(0, 2 * np.pi, len(names), endpoint=False)   # fallback layout for other state sets
    return np.column_stack([np.cos(angle), np.sin(angle)]) * 0.4


def _draw_map(ax, history, initial, t, rent_norm, policy_state, title, n_agents, top_flows=12):
    names, xy = initial.names, _positions(initial.names)
    states = usmap.load_states()
    on_map = all(n in states for n in names)
    ax.clear()
    ax.set_facecolor(SURFACE)
    pop, rent = history.population[t], history.rent[t]
    colours = SEQ(rent_norm(rent))

    if on_map:
        others = [r for n, rings in states.items() if n not in names for r in rings]
        ax.add_collection(PolyCollection(others, facecolors="#ecebe6", edgecolors=SURFACE, linewidths=0.6, zorder=0))
        for k, name in enumerate(names):
            rings = states[name]
            fc = colours[k].copy(); fc[3] = 0.6                          # tint the whole state by its rent
            ax.add_collection(PolyCollection(rings, facecolors=[fc], linewidths=1.6 if k == policy_state else 0.6, zorder=1,
                                             edgecolors=INK if k == policy_state else SURFACE))
        allpts = np.vstack([r for rings in states.values() for r in rings])
        pad = np.ptp(allpts, axis=0) * 0.03
        ax.set_xlim(allpts[:, 0].min() - pad[0], allpts[:, 0].max() + pad[0])
        ax.set_ylim(allpts[:, 1].min() - pad[1], allpts[:, 1].max() + pad[1])
        ax.set_aspect("equal")
        label_dy = 0.03
    else:
        pad = np.ptp(xy, axis=0) * 0.12
        ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
        ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
        label_dy = np.ptp(xy[:, 1]) * 0.17

    scale = 0.9 * (np.ptp(xy, axis=0).max() / 0.8) if on_map else 1.0   # arrow shrink scales with the map
    if t > 0:
        flows = history.flows[t - 1].astype(float)
        np.fill_diagonal(flows, 0)
        order = np.dstack(np.unravel_index(np.argsort(flows, axis=None)[::-1], flows.shape))[0][:top_flows]
        for i, j in order:
            if flows[i, j] <= 0:
                continue
            ax.add_patch(FancyArrowPatch(xy[i], xy[j], connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
                                         mutation_scale=9, lw=0.5 + 0.18 * flows[i, j], color=INK, alpha=0.5,
                                         shrinkA=(13 if on_map else 12) * scale, shrinkB=(13 if on_map else 12) * scale, zorder=3))
    if not on_map:   # without state shapes, circles show where the states are and how large they are
        ax.scatter(xy[:, 0], xy[:, 1], s=pop / n_agents * 7000, c=rent, cmap=SEQ, norm=rent_norm,
                   edgecolors=[INK if k == policy_state else SURFACE for k in range(len(names))],
                   linewidths=[2.0 if k == policy_state else 1.0 for k in range(len(names))], zorder=4)
    # Labels sit on a solid background so arrows never run through the letters.
    box = dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none", alpha=0.92)
    for k, name in enumerate(names):
        ax.text(xy[k, 0], xy[k, 1] + (0.0 if on_map else label_dy + 0.05 * np.sqrt(pop[k] / n_agents)), ABBR.get(name, name),
                ha="center", va="center" if on_map else "baseline", fontsize=8.5, weight="bold", color=INK,
                zorder=5, bbox=box)
    ax.axis("off")
    ax.set_title(f"{title}\nyear {t}, {names[policy_state]}: {pop[policy_state]} agents, "
                 f"rent ${rent[policy_state]:.1f}k", fontsize=9, loc="left", color=INK)


def animate_flows(scenarios, initial: InitialState, path=None, fps=3, seed=0):
    """Animate one run per scenario, side by side. ``scenarios`` is a list of (title, Params).

    On the US map each modelled state is tinted by its rent ($k per year), arrows are the largest
    yearly flows (thicker means more agents) and the policy state has a dark outline. Without the map
    file, circles (area = share of agents) mark the states instead.
    Saves a GIF to ``path`` when given. Returns the matplotlib animation.
    """
    runs = [(title, p, MigrationModel(replace(p, seed=seed), initial).run()) for title, p in scenarios]
    lo = min(h.rent.min() for _, _, h in runs)
    hi = max(h.rent.max() for _, _, h in runs)
    norm = Normalize(vmin=lo, vmax=hi)
    n_years = min(p.n_years for _, p, _ in runs)

    fig, axes = plt.subplots(1, len(runs), figsize=(6.6 * len(runs), 4.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=SEQ), ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("rent ($k per year)", color=INK2, fontsize=8)
    cbar.ax.tick_params(colors=INK2, labelsize=8)
    cbar.outline.set_visible(False)

    def update(t):
        for ax, (title, p, h) in zip(axes[0], runs):
            _draw_map(ax, h, initial, t, norm, p.policy_state, title, p.n_agents)
        return []

    anim = FuncAnimation(fig, update, frames=range(n_years + 1), interval=1000 // fps, blit=False)
    if path:
        anim.save(path, writer=PillowWriter(fps=fps), dpi=80, savefig_kwargs={"facecolor": SURFACE})
    return anim, fig, update
