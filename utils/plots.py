"""Figures for the report. Colours follow a validated palette: categorical slots 1-3
(blue, orange, aqua) for a few named series, and one blue ramp for ordered levels."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]                       # categorical slots 1-3
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal blue, light to dark
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def _ramp(n):
    """n ordered colours from the blue ramp (light to dark)."""
    idx = np.linspace(0, len(RAMP) - 1, n).round().astype(int)
    return [RAMP[i] for i in idx]


def _save(fig, save_path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=SURFACE)


def plot_gain_kappa(grid, save_path=None):
    """Heatmaps of population gain and capitalisation share on the elasticity x subsidy grid."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    specs = [("gain", "Population gain (fraction of Ohio's initial population)", None),
             ("kappa", "Capitalisation share (rent rise / subsidy)", (0, 1))]
    for ax, (col, title, lim) in zip(axes, specs):
        pv = grid.pivot(index="elasticity", columns="subsidy", values=col)
        kw = dict(vmin=lim[0], vmax=lim[1]) if lim else {}
        mesh = ax.pcolormesh(np.arange(pv.shape[1] + 1), np.arange(pv.shape[0] + 1), pv.values,
                             cmap=SEQ, edgecolors=SURFACE, linewidth=1.5, **kw)
        ax.set_xticks(np.arange(pv.shape[1]) + 0.5, [f"{s:g}" for s in pv.columns])
        ax.set_yticks(np.arange(pv.shape[0]) + 0.5, [f"{e:g}" for e in pv.index])
        ax.set(xlabel="subsidy ($k per year)", ylabel="housing supply elasticity")
        ax.set_title(title, fontsize=9, loc="left")
        _style(ax); ax.grid(False)
        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)
    _save(fig, save_path)
    return fig


def plot_required_subsidy(table, save_path=None):
    """Required subsidy against elasticity for each target, with the spread across seed batches."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    targets = sorted(table["target"].unique())
    # Ohio's metro areas span elasticities of about 1.0 to 3.7 (Saiz 2010, Table VI).
    ax.axvspan(1.02, 3.71, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(1.9, 0.98, "Ohio metros\n(Saiz 2010)", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7.5, color=INK2)
    for color, t in zip(_ramp(len(targets)), targets):
        d = table[table["target"] == t].sort_values("elasticity")
        ax.fill_between(d["elasticity"], d["s_min"], d["s_max_batch"], color=color, alpha=0.25, lw=0)
        ax.plot(d["elasticity"], d["s_mean"], "o-", color=color, lw=2, ms=4, label=f"+{t:.0%} target")
        ax.annotate(f"+{t:.0%}", (d["elasticity"].iloc[-1], d["s_mean"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set(xscale="log", xlabel="housing supply elasticity", ylabel="required subsidy ($k per year)",
           ylim=(0, None))
    ax.set_title("Subsidy needed for a target gain in Ohio's population at year 20", fontsize=9, loc="left")
    ax.set_xticks(sorted(table["elasticity"].unique()), [f"{e:g}" for e in sorted(table["elasticity"].unique())])
    ax.minorticks_off(); ax.set_xlim(None, ax.get_xlim()[1] * 1.15)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    _style(ax); _save(fig, save_path)
    return fig


def plot_sensitivity(tables, defaults, log_y=("migration_cost",), save_path=None):
    """Small multiples: required subsidy vs elasticity as one parameter varies.

    tables   {parameter: DataFrame with value, elasticity, s_star}
    defaults {parameter: default value}, drawn with a thicker line.
    """
    names = list(tables)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.4 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        d = tables[name]
        values = sorted(d["value"].unique())
        for color, v in zip(_ramp(len(values)), values):
            dv = d[d["value"] == v].sort_values("elasticity")
            is_default = name in defaults and np.isclose(v, defaults[name])
            ax.plot(dv["elasticity"], dv["s_star"], "o-", color=color, ms=3.5,
                    lw=2.8 if is_default else 1.6, label=f"{v:g}" + (" (default)" if is_default else ""))
        ax.set(xscale="log", xlabel="housing supply elasticity", ylabel="required subsidy ($k/yr)")
        ax.set_title(name.replace("_", " "), fontsize=9, loc="left")
        if name in log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, None)
        ax.set_xticks(sorted(d["elasticity"].unique()), [f"{e:g}" for e in sorted(d["elasticity"].unique())])
        ax.minorticks_off()
        ax.legend(frameon=False, fontsize=7, labelcolor=INK, title=None)
        _style(ax)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    _save(fig, save_path)
    return fig
