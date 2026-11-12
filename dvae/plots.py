"""Training progress plots: the inner encoder's (redrawn every depth) and the prior's (every epoch)."""

import math
import os

import matplotlib
import numpy as np

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


def _bars(ax, xs, ys, color, label, bottom=None, log=False):
    """A bar per x, drawn as one stepped area: with more bars than pixels, separate bars round to zero width in
    stretches and vanish. log: from just under the smallest value up (0 has no place on a log axis)."""
    if log:
        lo = [min(ys) / 2] * len(ys)
        ax.fill_between(xs, lo, ys, step="mid", color=color, linewidth=0, label=label)
        return
    lo = bottom if bottom is not None else [0.0] * len(ys)
    ax.fill_between(xs, lo, [a + b for a, b in zip(lo, ys)], step="mid", color=color, linewidth=0, label=label)


def _pair_grid(real, recon, gap=2, between=6):
    """[n, C, H, W] images over their reconstructions: originals in the top row, each one's reconstruction under
    it, [2H + gap, W', C]."""
    n, C, H, W = real.shape
    grid = np.ones((2 * H + gap, n * W + (n - 1) * between, C))
    for i in range(n):
        x = i * (W + between)
        grid[:H, x:x + W] = real[i].transpose(1, 2, 0)
        grid[H + gap:, x:x + W] = recon[i].transpose(1, 2, 0)
    return grid.clip(0, 1)


def plot_components(records, path, pairs=None):
    """Save a train_inner_components.py run's progress (its records.json): the train, val and test mse after
    every code length (the depths past the kept ones, tried for the depth patience, dashed); the val and train mse
    after the tree so far against hours; the
    cumulative val mse gain by Lloyd step at every depth (darker is deeper), each depth's chosen steps ringed;
    each depth's val and train gain against the depth threshold; the Lloyd steps chosen and tried per depth; each
    depth's seconds split into training, val and the rest, with the GPU's utilization; and, given pairs (dict of
    "real" and "recon" [n, C, H, W] arrays and "tokens"), test images beside their reconstructions so far."""
    depths = records["depths"]
    if not depths:
        return
    args, kept, root = records["args"], records["kept"], records["root"]
    fig = plt.figure(figsize=(19, 12.5 if pairs is not None else 10))
    grid = fig.add_gridspec(3 if pairs is not None else 2, 3, width_ratios=[1, 1, 1],
                            height_ratios=[1, 1, 0.42] if pairs is not None else [1, 1])
    by_depth, by_hour, by_step = (fig.add_subplot(grid[0, i]) for i in range(3))
    gain_ax, steps_ax, time_ax = (fig.add_subplot(grid[1, i]) for i in range(3))
    if pairs is not None:
        pair_ax = fig.add_subplot(grid[2, :])
        shown = _pair_grid(pairs["real"], pairs["recon"])
        pair_ax.imshow(shown[..., 0] if shown.shape[-1] == 1 else shown, cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
        pair_ax.set_axis_off()
        pair_ax.set_title(f"Test images (top) and their reconstructions after {pairs['tokens']} tokens (bottom)", loc="left",
                          fontsize=11, color=INK)
    K = len(depths)
    for split, color in COMPONENT_COLORS.items():
        ys = [root[split]] + [h[split] for h in depths]
        by_depth.plot(range(0, kept + 1), ys[:kept + 1], color=color, linewidth=1.6, marker="o", markersize=3,
                      label=split)
        if K > kept:
            by_depth.plot(range(kept, K + 1), ys[kept:], color=color, linewidth=1.2, linestyle="--", marker="o",
                          markersize=2.5, alpha=0.6)
    px = [split for split in COMPONENT_COLORS if f"{split}_px" in depths[0]]  # (on an outer run's latents: decoded)
    for split in px:
        ys = [root.get(f"{split}_px", math.nan)] + [h[f"{split}_px"] for h in depths]
        by_depth.plot(range(0, K + 1), ys, color=COMPONENT_COLORS[split], linewidth=1.4, linestyle=":",
                      label=f"{split}, decoded to pixels")
    by_depth.xaxis.set_major_locator(MaxNLocator(integer=True))
    _log_y(by_depth)
    by_depth.legend(frameon=False, labelcolor=INK)
    best = depths[kept - 1] if kept else depths[-1]
    _style(by_depth, f"Mse by code length ({kept} kept: val {best['val']:.5f}"
           + (f", in pixels {best['val_px']:.5f})" if "val_px" in best else f", test {best['test']:.5f})"), "tokens", "mse")

    hours = [h["seconds"] / 3600 for h in depths]
    for split, color in COMPONENT_COLORS.items():
        by_hour.plot(hours, [h[split] for h in depths], color=color, linewidth=1.6, label=split)
    for split in px:
        by_hour.plot(hours, [h[f"{split}_px"] for h in depths], color=COMPONENT_COLORS[split], linewidth=1.4,
                     linestyle=":", label=f"{split}, decoded to pixels")
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
    _bars(gain_ax, xs, [max(h["gain"], 1e-12) for h in depths], COMPONENT_COLORS["val"], "val", log=True)
    gain_ax.plot(xs, [max(trains[i] - trains[i + 1], 1e-12) for i in range(K)], "o", color=COMPONENT_COLORS["train"],
                 markersize=3, label="train")
    if args["depth_threshold"] > 0:  # (0, the default, has no place on a log axis)
        gain_ax.axhline(args["depth_threshold"], color=INK, linestyle=":", linewidth=1.2, label="depth threshold")
    _log_y(gain_ax)
    gain_ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    _style(gain_ax, "Each depth's mse gain", "depth", "mse taken off")

    _bars(steps_ax, xs, [h["tried"] for h in depths], GRID, "tried")
    _bars(steps_ax, xs, [h["steps"] for h in depths], COMPONENT_COLORS["val"], "chosen")
    steps_ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper left")
    _style(steps_ax, "Lloyd steps per depth", "depth", "Lloyd steps")

    bottom = [0.0] * K
    for part, color in (("train", COMPONENT_COLORS["val"]), ("val", COMPONENT_COLORS["train"]), ("other", PRIOR)):
        ys = [h["timing"][part] for h in depths]
        _bars(time_ax, xs, ys, color, part, bottom)
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


def _smooth(ys, window):
    """A centred moving average of ys (shorter at the ends)."""
    ys = np.asarray(ys, dtype=float)
    if window <= 1 or len(ys) < 3:
        return ys
    c = np.concatenate([[0.0], np.cumsum(ys)])
    lo = np.clip(np.arange(len(ys)) - window // 2, 0, len(ys))
    hi = np.clip(np.arange(len(ys)) + window // 2 + 1, 0, len(ys))
    return (c[hi] - c[lo]) / (hi - lo)


def plot_prior(records, path, samples=None):
    """Save a prior run's progress (train_prior.py): bits per token by code position (the kept epoch's train and
    val, and test once scored, beside the constant-share and encoder's-share baselines; a moving average over
    about 2% of the depths); bits per image against hours (each epoch's train and val, the baselines as lines);
    each epoch's seconds by part with the GPU's utilization; and, given samples (dict of "prior" and "own" images
    [n, numbers] and "shape"), codes sampled from the prior (top) and from the encoder's own shares (bottom)."""
    history = records["history"]
    if not history:
        return
    K, base = records["K"], records["baselines"]
    best = next((h for h in history if h["epoch"] == records.get("best_epoch")), history[-1])
    fig = plt.figure(figsize=(19, 12.5 if samples is not None else 6.5))
    grid = fig.add_gridspec(2 if samples is not None else 1, 3, height_ratios=[1, 0.5] if samples is not None else [1])
    by_depth, by_hour, time_ax = (fig.add_subplot(grid[0, i]) for i in range(3))

    window = max(1, K // 50)
    xs = np.arange(1, K + 1)
    for split, color in (("train", COMPONENT_COLORS["train"]), ("val", COMPONENT_COLORS["val"])):
        by_depth.plot(xs, _smooth(best[split], window), color=color, linewidth=1.6, label=f"{split}, prior")
    if "test" in records:
        by_depth.plot(xs, _smooth(records["test"], window), color=COMPONENT_COLORS["test"], linewidth=1.6,
                      label="test, prior")
    by_depth.plot(xs, _smooth(base["val_constant"], window), color=INK, linewidth=1.2, linestyle=":",
                  label="val, constant share per depth")
    by_depth.plot(xs, _smooth(base["val_own"], window), color=MUTED, linewidth=1.2, linestyle="--",
                  label="val, the encoder's own shares")
    by_depth.xaxis.set_major_locator(MaxNLocator(integer=True))
    by_depth.legend(frameon=False, labelcolor=INK, fontsize=8)
    _style(by_depth, f"Bits per token by code position (epoch {best['epoch']}; {window}-depth average)", "tokens",
           "bits per token")

    hours = [h["seconds"] / 3600 for h in history]
    for split, color in (("train", COMPONENT_COLORS["train"]), ("val", COMPONENT_COLORS["val"])):
        by_hour.plot(hours, [sum(h[split]) for h in history], color=color, linewidth=1.6, marker="o", markersize=3,
                     label=f"{split}, prior")
    by_hour.axhline(sum(base["val_constant"]), color=INK, linewidth=1.2, linestyle=":", label="val, constant share")
    by_hour.axhline(sum(base["val_own"]), color=MUTED, linewidth=1.2, linestyle="--", label="val, encoder's shares")
    by_hour.plot([best["seconds"] / 3600], [sum(best["val"])], "o", color=COMPONENT_COLORS["val"], markersize=8,
                 markerfacecolor="none", markeredgewidth=1.6, label="kept epoch")
    by_hour.legend(frameon=False, labelcolor=INK, fontsize=8)
    _style(by_hour, f"Bits per image (kept: val {sum(best['val']):.2f}; {K} tokens)", "hours", "bits per image")

    epochs = [h["epoch"] for h in history]
    bottom = np.zeros(len(history))
    parts = [("train", COMPONENT_COLORS["val"]), ("replay", PRIOR), ("eval", COMPONENT_COLORS["train"]),
             ("samples", COMPONENT_COLORS["test"])]
    for part, color in parts:
        ys = np.array([h["timing"].get(part, 0.0) / 60 for h in history])
        time_ax.bar(epochs, ys, bottom=bottom, color=color, width=0.8, label=part)
        bottom += ys
    busy = [(h["epoch"], h["util"]) for h in history if h.get("util") is not None]
    if busy:
        twin = time_ax.twinx()
        twin.plot(*zip(*busy), color=INK, linewidth=1.2, marker="o", markersize=3)
        twin.set_ylim(0, 105)
        twin.set_ylabel("GPU % busy", color=MUTED)
        twin.tick_params(colors=MUTED)
    time_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    time_ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper left")
    _style(time_ax, f"Minutes per epoch ({hours[-1]:.2f} h in all)", "epoch", "minutes")

    if samples is not None:
        ax = fig.add_subplot(grid[1, :])
        shape = tuple(samples["shape"])
        rows = [samples["prior"].view(-1, *shape).numpy(), samples["own"].view(-1, *shape).numpy()]
        shown = _pair_grid(rows[0], rows[1])
        ax.imshow(shown[..., 0] if shown.shape[-1] == 1 else shown, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(f"Codes sampled from the prior (top) and from the encoder's own shares (bottom), epoch "
                     f"{history[-1]['epoch']}", loc="left", fontsize=11, color=INK)
    _save(fig, path)


if __name__ == "__main__":  # (python -m dvae.plots RECORDS PNG: a components run's redraw, in a process of its own
    import json  # so that it never holds the training process's interpreter lock; test_pairs.pt beside RECORDS
    import sys  # adds its sample test images)

    import torch

    with open(sys.argv[1]) as f:
        records = json.load(f)
    pairs_path = os.path.join(os.path.dirname(sys.argv[1]), "test_pairs.pt")
    pairs = None
    if os.path.exists(pairs_path):
        saved = torch.load(pairs_path, weights_only=False)
        shape = tuple(saved["shape"])
        pairs = dict(real=saved["real"].view(-1, *shape).numpy(), recon=saved["recon"].view(-1, *shape).numpy(),
                     tokens=saved["tokens"])
    plot_components(records, sys.argv[2], pairs)
