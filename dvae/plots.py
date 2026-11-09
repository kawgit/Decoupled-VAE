"""Training progress plots, one per stage, each redrawn every epoch."""

import itertools
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


def plot_depths(history, path):
    """Save an encoder run's progress (train_encoder.py, train_centroids.py): the val mse after every code
    length, one line per epoch (darker is later); the val mse after the full code by epoch, beside the train mse
    when there is one (both on log scales); and, from the latest epoch, the cumulative val mse gain after every
    Lloyd step at every depth (what steps 1..t take off together; averaged over epochs when the run averages
    them), one line per depth (darker is deeper), on a log axis fit to the positive gains (a gain at or below zero
    is drawn far below it, off the panel), each depth's number of steps these gains choose ringed. Every line has
    a dot at each point. history holds dicts with "epoch", "val_mse" (after 1..K tokens) and optionally
    "train_mse" (after the epoch's last token, over the epoch's batches), "step_gains" (per depth, each step's
    gain), "smoothed_gains" (their running average over epochs, drawn in their place) and "chosen" (per depth,
    the steps the drawn gains choose)."""
    if not history:
        return
    gains = history[-1].get("smoothed_gains") or history[-1].get("step_gains") or []
    averaged = bool(history[-1].get("smoothed_gains"))
    panels = 3 if any(gains) else 2
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.5),
                             gridspec_kw={"width_ratios": [3, 2, 3][:panels]})
    by_depth, by_epoch = axes[0], axes[1]
    first, last = history[0]["epoch"], history[-1]["epoch"]
    norm = Normalize(vmin=first, vmax=max(last, first + 1))
    for h in history:
        mses = h["val_mse"]
        by_depth.plot(range(1, len(mses) + 1), mses, color=RAMP(norm(h["epoch"])), linewidth=1.6, marker="o",
                      markersize=3)
    by_depth.xaxis.set_major_locator(MaxNLocator(integer=True))
    _log_y(by_depth)
    _style(by_depth, "Val mse by code length, one line per epoch", "tokens", "val mse")
    _colorbar(fig, by_depth, norm, "epoch")

    xs = [h["epoch"] for h in history]
    final = [h["val_mse"][-1] for h in history]
    K = len(history[-1]["val_mse"])
    by_epoch.plot(xs, final, color="#2a78d6", linewidth=1.6, marker="o", markersize=3, label="val")
    train = [(h["epoch"], h["train_mse"]) for h in history if "train_mse" in h]
    if train:
        by_epoch.plot(*zip(*train), color="#e8743b", linewidth=1.6, marker="o", markersize=3, label="train")
        by_epoch.legend(frameon=False, labelcolor=INK)
    best = min(range(len(final)), key=final.__getitem__)
    by_epoch.annotate(f"best {final[best]:.4f}", (xs[best], final[best]), textcoords="offset points",
                      xytext=(0, 10), ha="right" if best > 0.8 * len(xs) else "center", fontsize=9, color=INK)
    by_epoch.xaxis.set_major_locator(MaxNLocator(integer=True))
    _log_y(by_epoch)
    _style(by_epoch, f"{'Val and train' if train else 'Val'} mse after the last token ({K} now)", "epoch", "mse")

    if panels == 3:
        by_step = axes[2]
        deep = Normalize(vmin=0, vmax=max(len(gains) - 1, 1))
        totals = [list(itertools.accumulate(g)) for g in gains]
        positive = [v for total in totals for v in total if v > 0]
        floor = min(positive) / 1e6 if positive else 1e-12  # where gains at or below zero go: off the panel
        stops = history[-1].get("chosen") or []
        for depth, total in enumerate(totals):
            if total:
                shown = [max(v, floor) for v in total]
                color = RAMP(deep(depth))
                by_step.plot(range(1, len(total) + 1), shown, color=color, linewidth=1.4, marker="o", markersize=2.5)
                if depth < len(stops) and 1 <= stops[depth] <= len(total):
                    by_step.plot([stops[depth]], [shown[stops[depth] - 1]], "o", markersize=7, markerfacecolor="none",
                                 markeredgecolor=color, markeredgewidth=1.4, zorder=3)
        _log_y(by_step)
        if positive:
            by_step.set_ylim(min(positive) / 1.3, max(positive) * 1.3)
        by_step.xaxis.set_major_locator(MaxNLocator(integer=True))
        what = "Averaged cumulative" if averaged else "Cumulative"  # (one line per depth: the colorbar says so)
        _style(by_step, f"{what} val mse gain by Lloyd step (epoch {history[-1]['epoch']})",
               "Lloyd step", "val mse taken off by steps 1..t")
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


def plot_rounds(history, path):
    """Save a rounds run's progress (train_rounds.py), one line per round (darker is later): the val (solid) and
    train (dashed) mse after every code length for each round's encoder; decoder_mean's final val and train mse by
    round; and each stage's val score by epoch: the encoder's val mse after all tokens, the prior's bits per image,
    decoder_mean's mse and decoder_cov's bits per pixel (when there is one). history holds a dict per round with
    "round", each stage's scores ("encoder", "prior", "decoder_mean", optionally "decoder_cov"), "encoder_out" and
    "encoder_out_train" (its encoder's val and train mse after each token) and, once the round is done,
    "decoder_mean_val" and "decoder_mean_train"."""
    if not history:
        return
    fig = plt.figure(figsize=(19, 10))
    stages = [("encoder", "Encoder: val mse after all tokens", "mse"),
              ("prior", "Prior: val bits per image", "bits per image"),
              ("decoder_mean", "decoder_mean: val mse", "mse"),
              ("decoder_cov", "decoder_cov: val bits per pixel", "bits per pixel")]
    stages = [s for s in stages if s[0] != "decoder_cov" or any(h.get("decoder_cov") for h in history)]
    grid = fig.add_gridspec(2, 12)
    by_length, by_round = fig.add_subplot(grid[0, :8]), fig.add_subplot(grid[0, 8:])
    span = 12 // len(stages)
    axes = [fig.add_subplot(grid[1, span * i:span * (i + 1)]) for i in range(len(stages))]
    norm = Normalize(vmin=0, vmax=max(history[-1]["round"], 1))
    for h in history:
        for key, style in (("encoder_out", "-"), ("encoder_out_train", "--")):
            mses = h.get(key) or []
            if mses:
                by_length.plot(range(1, len(mses) + 1), mses, color=RAMP(norm(h["round"])), linewidth=1.4,
                               linestyle=style, marker="o", markersize=3)
    by_length.xaxis.set_major_locator(MaxNLocator(integer=True))
    if any(h.get("encoder_out") for h in history):
        _log_y(by_length)
        by_length.plot([], [], color=MUTED, linestyle="-", label="val")
        by_length.plot([], [], color=MUTED, linestyle="--", label="train")
        by_length.legend(frameon=False, labelcolor=INK)
    _style(by_length, "Mse by code length, one line pair per round's encoder", "tokens", "mse")
    _colorbar(fig, by_length, norm, "round")

    for key, color, label in (("decoder_mean_val", "#2a78d6", "val"), ("decoder_mean_train", "#e8743b", "train")):
        points = [(h["round"], h[key]) for h in history if key in h]
        if points:
            by_round.plot(*zip(*points), color=color, linewidth=1.6, marker="o", markersize=3.5, label=label)
    if any("decoder_mean_val" in h for h in history):
        _log_y(by_round)
        by_round.legend(frameon=False, labelcolor=INK)
    by_round.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(by_round, "decoder_mean: final val and train mse by round", "round", "mse")

    for ax, (key, title, ylabel) in zip(axes, stages):
        for h in history:
            scores = h.get(key) or []
            if scores:
                ax.plot(range(len(scores)), scores, color=RAMP(norm(h["round"])), linewidth=1.6,
                        marker="o", markersize=3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key in ("decoder_mean", "encoder") and any(h.get(key) for h in history):
            _log_y(ax)
        _style(ax, title, "epoch of the round's stage", ylabel)
    _colorbar(fig, axes[-1], norm, "round")
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
