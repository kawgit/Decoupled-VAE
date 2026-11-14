"""Can a prior network learn a token from geometry it is shown? Synthetic codes built as a V-way tree's are: at each of
--depths depths the cells are made by plus and minus V / 2 random orthonormal directions of that depth (scaled by the
remainders' rms along each, as pca_start's), each image's token is its remainder's nearest, and the code moves by
0.8 of it. An oracle prior is shown each node's remainder and centroids, so the token is a fixed rule of what it
sees; trained on --images images, it should learn the rule at every depth. NodePrior (each vector embedded by a
linear map to the width, then compared) against GeoPrior (attention biased by every pair's geometry in the vectors'
own dimension).

What the run keeps (artifacts/<job>/): run.log, records.json (each network's loss and accuracy by step).

Run from the repo: python check_prior_geometry.py --depths 512
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from dvae.artifacts import artifact
from dvae.data import get_device
from dvae.logs import RunLog
from dvae.vprior import CandStore, make_net


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--depths", type=int, default=512)
    p.add_argument("--V", type=int, default=8)
    p.add_argument("--images", type=int, default=2000)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nets", default="node,geo")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== check_prior_geometry.py {time.strftime('%Y-%m-%d')} {vars(args)}")
    torch.manual_seed(args.seed)
    n, K, V, D, W = args.images, args.depths, args.V, args.D, args.window
    st = CandStore(n, K, V, D, device, torch.zeros(D), parts=True, images=True)
    X = torch.randn(n, D, device=device) / math.sqrt(D)
    rows = torch.arange(n, device=device)
    put = st.writer(rows, X)
    R = X.clone()
    for d in range(K):
        U = torch.linalg.qr(torch.randn(D, V // 2, device=device))[0].T
        s = (R @ U.T).square().mean(0).sqrt()
        q = torch.cat([U * s[:, None], -U * s[:, None]]).expand(n, -1, -1)
        t = (q.square().sum(-1) - 2 * torch.einsum("nd,nvd->nv", R, q)).argmin(-1)
        put(d, q * 0.8, t, torch.full((n, V), 1 / V, device=device), q)
        R = R - st.cand[rows, d, t].float() * st.scale[rows, d][:, None]
    log(f"{n} images coded through {K} depths ({st.nbytes() / 2**30:.2f} GiB); remainder rms {R.square().mean():.3g} "
        f"from {X.square().mean():.3g}")
    records = dict(args=vars(args), runs={})
    for name in args.nets.split(","):
        torch.manual_seed(args.seed + 1)
        config = dict(kind=f"{name}_prior", depths=K, V=V, D=D, window=W, width=args.width, layers=args.layers,
                      heads=args.heads, oracle=True)
        if name == "node":
            config["parts"] = True
        net = make_net(config).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        hist = []
        t0 = time.time()
        for step in range(1, args.steps + 1):
            i = torch.randint(0, n, (args.batch,), device=device)
            d = torch.randint(0, K, (args.batch,), device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = net(st.inputs(i, d, W)).float()
            loss = F.cross_entropy(logits, st.tok[i, d].long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 500 == 0:
                acc = (logits.argmax(-1) == st.tok[i, d].long()).float().mean().item()
                hist.append(dict(step=step, loss=loss.item(), acc=acc))
                log(f"{name}: step {step}: loss {loss.item():.4f} nats (uniform {math.log(V):.4f}), batch accuracy "
                    f"{acc * 100:.1f}%; {time.time() - t0:.0f}s")
        records["runs"][name] = hist
        with open(os.path.join(out, "records.json"), "w") as f:
            json.dump(records, f)
    log(f"done; {out}")


if __name__ == "__main__":
    main()
