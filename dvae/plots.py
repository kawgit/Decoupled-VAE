"""Training progress plots, one per stage, each redrawn every epoch."""

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


def _colorbar(fig, ax, norm, label):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=ax, pad=0.02)
    bar.locator = MaxNLocator(integer=True)
    bar.update_ticks()
    bar.set_label(label, color=MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED)


def plot_depths(history, path):
    """Save an encoder run's progress (train_encoder.py): the val mse after every code length, one line
    per epoch (darker is later); the val mse after the full code by epoch; and, from the latest epoch, the val
    mse gain of every Lloyd step at every depth, one line per depth (darker is deeper). history holds dicts with
    "epoch", "val_mse" (after 1..K tokens) and optionally "step_gains" (per depth, each step's gain)."""
    if not history:
        return
    gains = history[-1].get("step_gains") or []
    panels = 3 if any(gains) else 2
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.5),
                             gridspec_kw={"width_ratios": [3, 2, 3][:panels]})
    by_depth, by_epoch = axes[0], axes[1]
    first, last = history[0]["epoch"], history[-1]["epoch"]
    norm = Normalize(vmin=first, vmax=max(last, first + 1))
    for h in history:
        mses = h["val_mse"]
        by_depth.plot(range(1, len(mses) + 1), mses, color=RAMP(norm(h["epoch"])), linewidth=1.6,
                      marker="o" if len(mses) <= 12 else None, markersize=3)
    by_depth.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(by_depth, "Val mse by code length, one line per epoch", "tokens", "val mse")
    _colorbar(fig, by_depth, norm, "epoch")

    xs = [h["epoch"] for h in history]
    final = [h["val_mse"][-1] for h in history]
    K = len(history[-1]["val_mse"])
    by_epoch.plot(xs, final, color="#2a78d6", linewidth=1.6, marker="o", markersize=4)
    best = min(range(len(final)), key=final.__getitem__)
    by_epoch.annotate(f"best {final[best]:.4f}", (xs[best], final[best]), textcoords="offset points",
                      xytext=(0, 10), ha="right" if best > 0.8 * len(xs) else "center", fontsize=9, color=INK)
    by_epoch.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(by_epoch, f"Val mse after each epoch's last token ({K} now) by epoch", "epoch", "val mse")

    if panels == 3:
        by_step = axes[2]
        deep = Normalize(vmin=0, vmax=max(len(gains) - 1, 1))
        for depth, g in enumerate(gains):
            if g:
                by_step.plot(range(1, len(g) + 1), g, color=RAMP(deep(depth)), linewidth=1.4, marker="o",
                             markersize=2.5)
        by_step.set_yscale("symlog", linthresh=1e-6)
        by_step.axhline(0, color=MUTED, linewidth=0.8)
        by_step.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style(by_step, f"Val mse gain of each Lloyd step, one line per depth (epoch {history[-1]['epoch']})",
               "Lloyd step", "val mse taken off by the step")
        _colorbar(fig, by_step, deep, "depth")
    _save(fig, path)


def _save(fig, path):
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_prior(history, edges, kept, path):
    """Save a prior run's progress (train_prior.py): each band of depths' bits per token by epoch on the val
    codes and on the fixed train subset, one line per band (darker is deeper), each band's kept epoch marked.
    history holds dicts with "epoch", "val" and "train" (bits per token, per depth); edges [bands + 1] cut the
    depths into bands, and kept [bands] is each band's kept epoch (-1 for none yet)."""
    if not history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = [h["epoch"] for h in history]
    deep = Normalize(vmin=0, vmax=max(len(edges) - 2, 1))
    for ax, split in zip(axes, ("val", "train")):
        for b, (lo, hi) in enumerate(zip(edges, edges[1:])):
            ys = [sum(h[split][lo:hi]) / (hi - lo) for h in history]
            color = RAMP(deep(b))
            ax.plot(epochs, ys, color=color, linewidth=1.6, label=f"depths {lo + 1}-{hi}")
            if kept[b] in epochs:
                at = epochs.index(kept[b])
                ax.plot([kept[b]], [ys[at]], "o", color=color, markersize=6.5, markeredgecolor="white",
                        markeredgewidth=1.5, zorder=3)
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style(ax, f"{split} codes: bits per token by band (dots: each band's kept epoch)", "epoch",
               "bits per token, mean over the band")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _save(fig, path)


def plot_decoder(history, path):
    """Save a decoder run's progress (train_decoder.py): the val and train pixel bits per
    dimension by epoch, the best val marked. history holds dicts with "epoch", "val" and "train"."""
    if not history:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = [h["epoch"] for h in history]
    for split, color in (("val", "#2a78d6"), ("train", "#86b6ef")):
        ax.plot(epochs, [h[split] for h in history], color=color, linewidth=1.6, marker="o", markersize=3,
                label=split)
    val = [h["val"] for h in history]
    best = min(range(len(val)), key=val.__getitem__)
    ax.annotate(f"best val {val[best]:.4f}", (epochs[best], val[best]), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=INK)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    _style(ax, "Pixel bits per dimension given the code", "epoch", "bits per dimension")
    _save(fig, path)
