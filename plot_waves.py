"""Draw a train_waves.py run's progress (dvae/plots.py's plot_waves) against other encoder runs, on GPU hours:
e.g. a wave run beside the train_encoder.py run it branched from, or a control trained at the same time.

Run from the repo: python plot_waves.py artifacts/517-waves-from-510 --compare control=artifacts/510-fresh-control
(each a run's folder or its model.pt; the picture goes to the run's folder as waves.png, or --out)
"""

import argparse
import os

import torch

from dvae.persist import load_checkpoint
from dvae.plots import plot_waves


def model_path(path):
    return os.path.join(path, "model.pt") if os.path.isdir(path) else path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="the wave run's folder or model.pt")
    p.add_argument("--compare", nargs="*", default=[], metavar="LABEL=RUN",
                   help="other runs to draw beside it (the first is the one the depths are measured against)")
    p.add_argument("--out", default=None, help="the picture (default: waves.png in the run's folder)")
    args = p.parse_args()
    run = model_path(args.run)
    ckpt = load_checkpoint(run, "cpu")
    stages = (ckpt.get("waves") or {}).get("stages", [])
    compare = []
    for item in args.compare:
        label, _, path = item.partition("=")
        other = torch.load(model_path(path), map_location="cpu", weights_only=False, mmap=True)
        compare.append((label, other["metrics"]))
    out = args.out or os.path.join(os.path.dirname(run), "waves.png")
    plot_waves(ckpt["metrics"], stages, out, compare)
    print(f"{out}: {run} after its epoch {ckpt['epoch']}, {len(stages)} stages"
          + "".join(f", beside {label}" for label, _ in compare))


if __name__ == "__main__":
    main()
