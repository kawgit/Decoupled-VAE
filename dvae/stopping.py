"""Early stopping's patience, shared by the training scripts. It grows with the run: a run is stagnant once it has
gone a share of the epochs it has run (--patience-frac, 10% by default) in a row without a better val score, and
never fewer than --patience epochs (10 by default)."""

import math


def add_patience_args(p):
    p.add_argument("--patience", type=int, default=10,
                   help="the fewest epochs in a row without a better val score that make a run stagnant")
    p.add_argument("--patience-frac", type=float, default=0.1,
                   help="share of the epochs run so far that must pass without a better val score to stop")


def patience(epoch, args):
    """The epochs in a row without a better val score that make a run stagnant after epoch (counted from 0)."""
    return max(args.patience, math.ceil(args.patience_frac * (epoch + 1)))


def stagnant(epoch, best_epoch, args):
    """Whether a run whose best val score came at best_epoch is stagnant after epoch."""
    return epoch - best_epoch >= patience(epoch, args)
