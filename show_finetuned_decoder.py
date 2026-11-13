"""Side-by-side test images for a finetune_outer_decoder.py run: each row one test image, its columns the original,
then the outer decoder's output before and after fine-tuning on the encoder's exact latents and on the tree's
reconstructions at depth K, K / 2 and K / 4 (the fine-tuning run's marks).

What the run keeps (artifacts/<job>/): run.log, grid.png (the images at their own size) and grid_x2.png (doubled).

Run from the repo: python show_finetuned_decoder.py --finetune artifacts/<finetune_outer_decoder.py job>
"""

import argparse
import json
import os
import time

import torch

from dvae.artifacts import artifact
from dvae.data import get_device, load_images
from dvae.logs import RunLog
from dvae.outer import load_outer
from dvae.viz import save_grid
from dvae.vreplay import VTree


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finetune", required=True, metavar="RUN", help="a finetune_outer_decoder.py run")
    p.add_argument("--n", type=int, default=8, help="test images shown (picked at random)")
    p.add_argument("--fp16", action="store_true", help="float16 autocast (V100s), as the fine-tuning ran")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== show_finetuned_decoder.py {time.strftime('%Y-%m-%d')} {vars(args)}")

    with open(os.path.join(args.finetune, "records.json")) as f:
        rec = json.load(f)
    ft = rec["args"]
    K, marks = rec["depth"], rec["marks"]
    model, lat = load_outer(ft["outer"], device)
    C, h, w = lat["shape"]
    mean, std = lat["mean"].to(device)[None, :, None, None], lat["std"].to(device)[None, :, None, None]
    test = lat["test"][:ft["limit_test"]]
    idx = torch.randperm(len(test), generator=torch.Generator().manual_seed(args.seed))[:args.n]
    Z = test[idx].float().flatten(1).to(device)
    images = (load_images(lat["dataset"], "test", args.data_root)[idx].to(device).float() + 0.5) / 256
    log(f"test images {idx.tolist()} of {len(test)}; tree {ft['tree']} coded to depth {K}, shown at {marks}")

    tree = VTree(ft["tree"], device, K, fp16=args.fp16, node_batch=ft["node_batch"])
    g, shots = tree.code(Z, K, marks)
    log(f"coded: latent mse at depth {K} {g.R.square().mean().item():.6f} on these images "
        f"(the tree's test {rec['tree_record']['test']:.6f})")
    inputs = {"exact": Z, **{f"d{k}": shots[k].to(device).float() for k in marks}}

    @torch.no_grad()
    def decode(z):
        x = z.view(-1, C, h, w) / lat["scale"] * std + mean
        with torch.autocast(device.type, dtype=torch.float16, enabled=args.fp16):
            return model.decoder(x).float().clamp(0, 1)

    shown = {}
    for when in ("before", "after"):
        if when == "after":
            saved = torch.load(os.path.join(args.finetune, f"decoder_d{K}.pt"), map_location=device,
                               weights_only=False)
            model.decoder.load_state_dict(saved["state_dict"])
            model.decoder.eval()
        for name, z in inputs.items():
            shown[name, when] = decode(z)
    cols = [("original", images)]
    for name in ["exact"] + [f"d{k}" for k in sorted(marks, reverse=True)]:
        for when in ("before", "after"):
            y = shown[name, when]
            cols.append((f"{name} {when}", y))
            log(f"{name} {when}: pixel mse {(y - images).square().mean().item():.6f} on these images")
    log("columns: " + " | ".join(c for c, _ in cols))
    grid = torch.stack([y for _, y in cols], 1).flatten(0, 1)  # (row-major: image by image)
    save_grid(grid, os.path.join(out, "grid.png"), nrow=len(cols), scale=1)
    save_grid(grid, os.path.join(out, "grid_x2.png"), nrow=len(cols), scale=2)
    log(f"done; {out}")


if __name__ == "__main__":
    main()
