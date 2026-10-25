"""Training progress plots for the encoder build."""

import os

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# a single-hue ordinal ramp, light to dark: later rounds are darker
RAMP = LinearSegmentedColormap.from_list("rounds", ["#86b6ef", "#3987e5", "#184f95", "#0d366b"])
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


def plot_build(history, path):
    """Save the build's progress: the val decompression loss at every code length, one line per round (darker is
    later), and the overall val loss by round."""
    rounds = [h for h in history if "val_loss_by_length" in h]
    if not rounds:
        return
    fig, (by_length, by_round) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    norm = Normalize(vmin=rounds[0]["round"], vmax=max(rounds[-1]["round"], rounds[0]["round"] + 1))
    for h in rounds:
        losses = h["val_loss_by_length"]
        by_length.plot(range(len(losses)), losses, color=RAMP(norm(h["round"])), linewidth=1.6,
                       marker="o" if len(losses) <= 12 else None, markersize=3)
    _style(by_length, "Val decompression loss by code length, one line per round", "code length k",
           "decompression loss")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=by_length, pad=0.02)
    bar.locator = MaxNLocator(integer=True)
    bar.update_ticks()
    bar.set_label("round", color=MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED)

    xs = [h["round"] for h in rounds]
    by_round.plot(xs, [h["val_loss"] for h in rounds], color="#2a78d6", linewidth=1.6, marker="o", markersize=4)
    best = min(range(len(rounds)), key=lambda i: rounds[i]["val_loss"])
    by_round.annotate(f"best {rounds[best]['val_loss']:.4f}", (xs[best], rounds[best]["val_loss"]),
                      textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
    by_round.set_xticks(xs if len(xs) <= 15 else xs[:: max(1, len(xs) // 10)])
    _style(by_round, "Val loss by round (averaged over code lengths)", "round", "decompression loss")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
