"""How well a train_vprior.py prior learned its own buffer, for a run with a fixed buffer (--rotate 0): the buffer is
rebuilt (the same pool order, the same images), and the kept prior (prior.pt) is read on it token by token (teacher
forced: bits and how often its most likely token is the code's, by depth), then sampled. Each sampled code is matched
to the buffer code it follows longest; where it first leaves it says whether the prior's slips come early (the image
lost) or late (detail lost). Sampled at temperature 1 (as the run's QA), at 0.5, and greedily after the first --lead
tokens of each buffer code (does the prior finish an image it was started on?).

What the run keeps (artifacts/<job>/): run.log; records.json; run.png (teacher-forced bits and accuracy by depth, and
where samples first leave the buffer); samples.png (each sampling: rows of samples above the buffer image each
follows longest, decoded).

Run from the repo: python check_prior_memory.py --run artifacts/<train_vprior.py job>
"""

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.plots import COMPONENT_COLORS, _style  # noqa: E402
from dvae.viz import save_grid  # noqa: E402
from dvae.vprior import CandStore, CodeQA, make_net  # noqa: E402
from dvae.vreplay import VTree  # noqa: E402

LN2 = math.log(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a train_vprior.py run with --rotate 0")
    p.add_argument("--n", type=int, default=64, help="codes sampled per sampling")
    p.add_argument("--lead", type=int, default=8, help="buffer tokens given before greedy continuation")
    p.add_argument("--shown", type=int, default=10, help="samples shown per sampling")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.dirname(artifact("run.log"))
    os.makedirs(out, exist_ok=True)
    device = get_device(args.device)
    log = RunLog(os.path.join(out, "run.log"), device)
    log(f"=== check_prior_memory.py {time.strftime('%Y-%m-%d')} {vars(args)}")

    with open(os.path.join(args.run, "records.json")) as f:
        ra = json.load(f)["args"]
    assert ra["rotate"] == 0, "the run's buffer rotated: its images are not the ones the prior last saw"
    tree = VTree(ra["tree"], device, ra["depths"], fp16=ra["fp16"], node_batch=ra["node_batch"])
    K, V = tree.K, tree.args["V"]
    saved = torch.load(os.path.join(args.run, "prior.pt"), map_location=device, weights_only=False)
    prior = make_net(saved["config"]).to(device).eval()
    prior.load_state_dict(saved["state_dict"])
    W = prior.window

    def amp():
        dtype = torch.float16 if ra["fp16"] else torch.bfloat16
        return torch.autocast(device.type, dtype=dtype, enabled=ra["fp16"] or ra["bf16"])

    qa = CodeQA(tree, 1, ra["data_root"], ra["seed"])
    pool = qa.lat["train"].flatten(1)
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(ra["seed"] + 2))
    n_buf = ra["buffer"]
    Z = pool[order[:n_buf]].to(device).float()
    buf = CandStore(n_buf, K, V, Z.shape[1], device, tree.mean)
    rows = torch.arange(n_buf, device=device)
    g, _ = tree.code(Z, K, visit=buf.writer(rows))
    buf_img = qa.decode(Z - g.R)
    log(f"run {args.run}: prior of epoch {saved['epoch']}; buffer of {n_buf} train images rebuilt; {K} depths, V={V}")

    # teacher forced, every (image, depth)
    bits = torch.zeros(n_buf, K, device=device)
    hit = torch.zeros(n_buf, K, device=device)
    with torch.no_grad():
        for d in range(K):
            dd = torch.full((n_buf,), d, dtype=torch.long, device=device)
            with amp():
                lp = prior(buf.inputs(rows, dd, W)).float().log_softmax(-1)
            t = buf.tok[:, d].long()
            bits[:, d] = -lp.gather(-1, t[:, None])[:, 0] / LN2
            hit[:, d] = (lp.argmax(-1) == t).float()
    per_image = bits.sum(1)
    log(f"teacher forced: bits per image mean {per_image.mean():.2f} (median {per_image.median():.2f}, max "
        f"{per_image.max():.2f}; floor log2({n_buf}) = {math.log2(n_buf):.2f}); most likely token right "
        f"{hit.mean() * 100:.3f}% of tokens, {(hit.min(1).values == 1).sum().item()} of {n_buf} codes right at "
        f"every depth")
    for lo, hi in ((0, 8), (8, 32), (32, 128), (128, 256), (256, K)):
        log(f"  depths {lo}-{hi - 1}: {bits[:, lo:hi].sum(1).mean():.2f} bits per image, "
            f"{bits[:, lo:hi].mean() * 1000:.2f} millibits per token; most likely right "
            f"{hit[:, lo:hi].mean() * 100:.3f}%")

    def sample(n, temp, lead_codes=None):
        """n codes from the prior at temperature temp (0: greedy), each given lead_codes' first --lead tokens."""
        store = CandStore(n, K, V, Z.shape[1], device, tree.mean)
        r = torch.arange(n, device=device)
        put = store.writer(r)
        gen = torch.Generator(device).manual_seed(ra["seed"] + 7)

        def choose(d, cands, shares):
            if lead_codes is not None and d < args.lead:
                token = lead_codes[:, d]
            else:
                dd = torch.full((n,), d, dtype=torch.long, device=device)
                with amp():
                    logits = prior(store.inputs(r, dd, W, cands.float())).float()
                token = logits.argmax(-1) if temp == 0 else \
                    torch.multinomial((logits / temp).softmax(-1), 1, generator=gen)[:, 0]
            put(d, cands, token, shares)
            return token
        toks, z = tree.sample(n, choose, K)
        return toks, z

    btok = buf.tok.long()
    results, panels, deviations = {}, [], {}
    for name, temp, lead in (("T=1", 1.0, False), ("T=0.5", 0.5, False), (f"greedy after {args.lead}", 0.0, True)):
        n = n_buf if lead else args.n
        toks, z = sample(n, temp, btok[:n] if lead else None)
        if lead:  # (matched to the code it was started on)
            which = torch.arange(n, device=device)
            follow = (toks == btok[:n]).long().cumprod(-1).sum(-1)
        else:  # (common prefix lengths with every buffer code [n, n_buf]; the longest)
            follow, which = (toks[:, None, :] == btok[None]).long().cumprod(-1).sum(-1).max(1)
        mse = (z - (Z - g.R)[which]).square().mean(1)
        results[name] = dict(exact=(follow == K).sum().item(), n=n, first_leave=follow.tolist(),
                             latent_mse_to_followed=mse.mean().item())
        deviations[name] = follow.cpu()
        log(f"{name}: {results[name]['exact']} of {n} samples exactly a buffer code; first leaves the code it "
            f"follows longest at depth median {follow.float().median():.0f} (quartiles "
            f"{follow.float().quantile(0.25):.0f}, {follow.float().quantile(0.75):.0f}); latent mse to that code's "
            f"reconstruction {mse.mean():.6f} (the tree's own at depth {K}: {g.R.square().mean():.6f})")
        k = min(args.shown, n)
        panels += [qa.decode(z[:k]), buf_img[which[:k].cpu()]]
    save_grid(torch.cat(panels), os.path.join(out, "samples.png"), nrow=min(args.shown, args.n), scale=2)
    log("samples.png: row pairs (sample, the buffer image it follows longest) for " + ", ".join(results))

    with open(os.path.join(out, "records.json"), "w") as f:
        json.dump(dict(args=vars(args), run_args=ra, epoch=saved["epoch"], bits_by_depth=bits.mean(0).tolist(),
                       hit_by_depth=hit.mean(0).tolist(), bits_per_image=per_image.tolist(), samples=results), f)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    xs = list(range(K))
    axes[0].plot(xs, (bits.mean(0) * 1000).tolist(), color=COMPONENT_COLORS["train"], linewidth=1.2)
    axes[0].set_yscale("log")
    _style(axes[0], "teacher forced: millibits per token on the buffer", "depth", "millibits")
    axes[1].plot(xs, ((1 - hit.mean(0)) * 100).tolist(), color=COMPONENT_COLORS["val"], linewidth=1.2)
    _style(axes[1], "most likely token wrong, % of the buffer", "depth", "%")
    for (name, f_), color in zip(deviations.items(), ("#e8743b", "#2a78d6", "#3aa57a")):
        axes[2].hist(f_.numpy(), bins=32, range=(0, K + 1), histtype="step", linewidth=1.6, color=color, label=name)
    _style(axes[2], "samples: depth where each first leaves its buffer code", "depth", "samples")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "run.png"), dpi=130)
    plt.close(fig)
    log(f"done; {out}")


if __name__ == "__main__":
    main()
