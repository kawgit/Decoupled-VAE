"""Training progress plots."""

import os

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# a single-hue ordinal ramp, light to dark: later epochs are darker
RAMP = LinearSegmentedColormap.from_list("epochs", ["#86b6ef", "#3987e5", "#184f95", "#0d366b"])
INK, MUTED, GRID = "#1d1d1f", "#6b6b70", "#e2e2df"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_depths(history, path):
    """Save a run's progress: the val mse after every code length, one line per epoch (darker is later), and the
    val mse after the full code by epoch. history holds dicts with "epoch" and "val_mse" (after 1..K tokens)."""
    if not history:
        return
    fig, (by_depth, by_epoch) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    first, last = history[0]["epoch"], history[-1]["epoch"]
    norm = Normalize(vmin=first, vmax=max(last, first + 1))
    for h in history:
        mses = h["val_mse"]
        by_depth.plot(range(1, len(mses) + 1), mses, color=RAMP(norm(h["epoch"])), linewidth=1.6,
                      marker="o" if len(mses) <= 12 else None, markersize=3)
    by_depth.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(by_depth, "Val mse by code length, one line per epoch", "tokens", "val mse")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=by_depth, pad=0.02)
    bar.locator = MaxNLocator(integer=True)
    bar.update_ticks()
    bar.set_label("epoch", color=MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED)

    xs = [h["epoch"] for h in history]
    final = [h["val_mse"][-1] for h in history]
    K = len(history[-1]["val_mse"])
    by_epoch.plot(xs, final, color="#2a78d6", linewidth=1.6, marker="o", markersize=4)
    best = min(range(len(final)), key=final.__getitem__)
    by_epoch.annotate(f"best {final[best]:.4f}", (xs[best], final[best]), textcoords="offset points",
                      xytext=(0, 10), ha="right" if best > 0.8 * len(xs) else "center", fontsize=9, color=INK)
    by_epoch.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(by_epoch, f"Val mse after {K} tokens by epoch", "epoch", "val mse")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
