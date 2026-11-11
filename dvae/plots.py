"""Training progress plots: the encoder's (redrawn every depth), the prior's and the decoder's (every epoch)."""

import os

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

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


def _log_y(ax):
    """A log y axis labelled in plain numbers at 1, 2, 3 and 5 times each power of ten."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _colorbar(fig, ax, norm, label):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=ax, pad=0.02)
    bar.locator = MaxNLocator(integer=True)
    bar.update_ticks()
    bar.set_label(label, color=MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED)


PRIOR = "#a8a8ad"  # the rest of a depth's seconds
COMPONENT_COLORS = dict(train="#e8743b", val="#2a78d6", test="#3aa57a")


def plot_components(records, path):
    """Save a train_encoder_components.py run's progress (its records.json): the train, val and test mse after
    every code length (the depths past the kept ones, tried for the depth patience, dashed); the val and train mse
    after the tree so far against hours; the
    cumulative val mse gain by Lloyd step at every depth (darker is deeper), each depth's chosen steps ringed;
    each depth's val and train gain against the depth threshold; the Lloyd steps chosen and tried per depth with
    the epochs its networks trained; and each depth's seconds split into training, val and the rest, with the
    GPU's utilization."""
    depths = records["depths"]
    if not depths:
        return
    args, kept, root = records["args"], records["kept"], records["root"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), gridspec_kw={"width_ratios": [3, 2, 3]})
    (by_depth, by_hour, by_step), (gain_ax, steps_ax, time_ax) = axes
    K = len(depths)
    for split, color in COMPONENT_COLORS.items():
        ys = [root[split]] + [h[split] for h in depths]
        by_depth.plot(range(0, kept + 1), ys[:kept + 1], color=color, linewidth=1.6, marker="o", markersize=3,
                      label=split)
        if K > kept:
            by_depth.plot(range(kept, K + 1), ys[kept:], color=color, linewidth=1.2, linestyle="--", marker="o",
                          markersize=2.5, alpha=0.6)
    by_depth.xaxis.set_major_locator(MaxNLocator(integer=True))
    _log_y(by_depth)
    by_depth.legend(frameon=False, labelcolor=INK)
    best = depths[kept - 1] if kept else depths[-1]
    _style(by_depth, f"Mse by code length ({kept} kept: val {best['val']:.5f}, test {best['test']:.5f})",
           "tokens", "mse")

    hours = [h["seconds"] / 3600 for h in depths]
    by_hour.plot(hours, [h["val"] for h in depths], color=COMPONENT_COLORS["val"], linewidth=1.6, marker="o",
                 markersize=3, label="this run, val")
    by_hour.plot(hours, [h["train"] for h in depths], color=COMPONENT_COLORS["train"], linewidth=1.2, marker="o",
                 markersize=2.5, label="this run, train")
    by_hour.xaxis.set_major_locator(MaxNLocator(5))
    _log_y(by_hour)
    by_hour.legend(frameon=False, labelcolor=INK, fontsize=8)
    _style(by_hour, f"Mse after the tree so far ({K} depths)", "hours", "mse")

    deep = Normalize(vmin=0, vmax=max(K - 1, 1))
    positive = [v for h in depths for v in h["gains"] if v > 0]
    floor = min(positive) / 1e6 if positive else 1e-12
    for h in depths:
        shown = [max(v, floor) for v in h["gains"]]
        color = RAMP(deep(h["d"]))
        by_step.plot(range(1, len(shown) + 1), shown, color=color, linewidth=1.4, marker="o", markersize=2.5)
        if 1 <= h["steps"] <= len(shown):
            by_step.plot([h["steps"]], [shown[h["steps"] - 1]], "o", markersize=7, markerfacecolor="none",
                         markeredgecolor=color, markeredgewidth=1.4, zorder=3)
    _log_y(by_step)
    if positive:
        by_step.set_ylim(min(positive) / 1.3, max(positive) * 1.3)
    by_step.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(by_step, "Cumulative val mse gain by Lloyd step (chosen ringed)", "Lloyd step",
           "val mse taken off by steps 1..t")
    _colorbar(fig, by_step, deep, "depth")

    xs = [h["d"] for h in depths]
    trains = [root["train"]] + [h["train"] for h in depths]
    gain_ax.bar(xs, [max(h["gain"], 1e-12) for h in depths], color=COMPONENT_COLORS["val"], width=0.8, label="val")
    gain_ax.plot(xs, [max(trains[i] - trains[i + 1], 1e-12) for i in range(K)], "o", color=COMPONENT_COLORS["train"],
                 markersize=3, label="train")
    gain_ax.axhline(args["depth_threshold"], color=INK, linestyle=":", linewidth=1.2, label="depth threshold")
    _log_y(gain_ax)
    gain_ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    _style(gain_ax, "Each depth's mse gain", "depth", "mse taken off")

    steps_ax.bar(xs, [h["tried"] for h in depths], color=GRID, width=0.8, label="tried")
    steps_ax.bar(xs, [h["steps"] for h in depths], color=COMPONENT_COLORS["val"], width=0.8, label="chosen")
    epochs: dict[int, list[float]] = {}
    for m in records["models"]:
        if not m.get("reused"):
            epochs.setdefault(m["d"], []).append(m["epochs"])
    twin = steps_ax.twinx()
    ed = sorted(epochs)
    twin.plot(ed, [sum(epochs[d]) / len(epochs[d]) for d in ed], color=COMPONENT_COLORS["train"], linewidth=1.4,
              marker="o", markersize=2.5)
    twin.set_ylabel("epochs per network trained", color=COMPONENT_COLORS["train"])
    twin.tick_params(colors=MUTED)
    for side in ("top", "right"):
        twin.spines[side].set_visible(False)
    steps_ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper left")
    _style(steps_ax, "Lloyd steps per depth, and epochs per network", "depth", "Lloyd steps")

    bottom = [0.0] * K
    for part, color in (("train", COMPONENT_COLORS["val"]), ("val", COMPONENT_COLORS["train"]), ("other", PRIOR)):
        ys = [h["timing"][part] for h in depths]
        time_ax.bar(xs, ys, bottom=bottom, color=color, width=0.8, label=part)
        bottom = [a + b for a, b in zip(bottom, ys)]
    busy = [(h["d"], h["util"]) for h in depths if h.get("util") is not None]
    if busy:
        twin = time_ax.twinx()
        twin.plot(*zip(*busy), color=INK, linewidth=1.2, marker="o", markersize=2)
        twin.set_ylim(0, 105)
        twin.set_ylabel("GPU % busy", color=MUTED)
        twin.tick_params(colors=MUTED)
    time_ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper left")
    _style(time_ax, f"Seconds per depth ({hours[-1]:.2f} h in all)", "depth", "seconds")
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
            ax.plot(epochs, ys, color=color, linewidth=1.6, marker="o", markersize=3, label=f"depths {lo + 1}-{hi}")
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


if __name__ == "__main__":  # (python -m dvae.plots RECORDS PNG: a components run's redraw, in a process of its own
    import json  # so that it never holds the training process's interpreter lock)
    import sys

    with open(sys.argv[1]) as f:
        plot_components(json.load(f), sys.argv[2])
