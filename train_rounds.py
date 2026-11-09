"""Rounds through a generator (dvae/rounds.py). Round 0's generator is the training images themselves (their
empirical distribution), and every round
1. trains the encoder on images from the generator: round 0 on the training images, each epoch all of them again;
   later rounds on images the last round's generator makes, --draws fresh ones every epoch (codes drawn from its
   prior down its encoder's tree, each with an image from its decoder at the leaf, dvae/rounds.py's sample_codes),
   so no image is seen twice. Early stopping on the val images' mse after all --depths tokens down its tree (the
   k-means objective on --val-draws held-out generated images is logged beside it). The k-means encoder
   (CodeEncoder) sends each image to its nearest centroid at every depth and trains on their squared errors
   (dvae/rounds.py's kmeans_sse); with --encoder lloyd, the Lloyd encoder (LloydEncoder, dvae/deltatree.py's) runs
   --lloyd-steps Lloyd steps at every node and trains on the actual side means of the batch's images there
   (train_encoder.py's train_step), all --depths depths at once, each depth's start tracked from its remainders;
   --init-encoder takes round 0's from a train_encoder.py checkpoint instead, which is that round's training;
2. sends the train and val images down the encoder's tree;
3. trains the prior on their codes (cross-entropy of each token, early stopping on val bits per image);
4. trains the decoder on their codes, decoder = (decoder_mean, decoder_cov): decoder_mean on each image's squared
   error from its leaf's mean (early stopping on val mse), then with --cov-rank decoder_cov, a Gaussian around each
   leaf's mean (held fixed), on each image's negative log-likelihood, with its own early stopping, on val bits per
   pixel: the likelihood of the 8-bit values, the density times each value's bin, 1/256;
and the generator becomes (encoder, prior, decoder): its images are the decoder's means at the sampled leaves, or
with --cov-rank draws from decoder_cov's Gaussians. Every model carries its weights from one round to the next
(the encoder unless --fresh-encoder), each stage with a fresh optimizer, and a stage that starts from carried
weights trains at --warm-lr, not --lr, so its warmup does not knock it off the weights it has. Each stage's val
score is logged every epoch, every round's generator is saved (round_<r>.pt), and the progress plot
(dvae/plots.py's plot_rounds) is redrawn every epoch: each round's encoder's val mse after every code length, and
each stage's val score by epoch, one line per round.

--resume continues after any finished round (each saves round_<r>.pt), its generator making the next round's
images, with the code and settings of the time. The run stops once --round-patience rounds in a row made no
better encoder (val mse after all --depths tokens), or after --rounds.

Run from the repo: python train_rounds.py --depths 24
"""

import argparse
import copy
import math
import os
import time
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # ~6.5 GB: fail instead of swapping
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dvae.artifacts import artifact  # noqa: E402
from dvae.data import get_device, load_mnist  # noqa: E402
from dvae.deltatree import D, pixels, top_direction  # noqa: E402
from dvae.logs import RunLog  # noqa: E402
from dvae.plots import plot_rounds  # noqa: E402
from dvae.rounds import (CodeEncoder, CodePrior, Decoder, DecoderCov, DecoderMean, LloydEncoder, Tree,  # noqa: E402
                         kmeans_sse, paths, route, sample_codes)
from dvae.stopping import add_patience_args, stagnant  # noqa: E402
from train_encoder import refresh, train_step  # noqa: E402

LN2 = math.log(2)


def code_batch(tree: Tree, idx):
    """The paths of images idx to their leaves, as dvae/rounds.py's paths gives them: tokens [B, K + 1], each
    position's choice's centroids [B, K + 1, V, D] and the nodes [B, K + 1]."""
    return paths(tree, tree.K, tree.node[tree.K][idx])


def candidates(tree: Tree, anc):
    """The centroids [B, K, V, D] of the nodes anc [B, K + 1] above the leaves."""
    return torch.stack([tree.C[j][anc[:, j]].float() for j in range(tree.K)], 1)


def prior_bits(prior, tree, idx):
    """Each image's bits [B] under the prior, images idx."""
    tok, prev, anc = code_batch(tree, idx)
    K = tree.K
    logits = prior.logits(tok[:, :K], prev[:, :K], candidates(tree, anc))
    return F.cross_entropy(logits.flatten(0, 1).float(), tok[:, 1:].flatten(), reduction="none").view(len(idx), K) \
        .sum(1) / LN2


def decoder_sse(decoder_mean, tree, X, idx):
    """Each image's squared error [B] from its leaf's mean under decoder_mean."""
    tok, prev, _ = code_batch(tree, idx)
    return (decoder_mean.mean_image(tok, prev) - X[idx]).square().sum(1)


def decoder_means(decoder_mean, tree, idx):
    """Each image's leaf mean [B, D] under decoder_mean."""
    tok, prev, _ = code_batch(tree, idx)
    return decoder_mean.mean_image(tok, prev)


def cov_nll(decoder_cov, tree, mus, X, idx):
    """Each image's negative log-likelihood [B] (nats) under decoder_cov's Gaussian at its leaf, around its leaf's
    mean mus[idx]."""
    tok, prev, _ = code_batch(tree, idx)
    return decoder_cov.nll(tok, prev, mus[idx], X[idx])


def chunked(fn, idx, size=1024):
    """fn(idx) [B], computed size indices at a time."""
    return torch.cat([fn(i) for i in idx.split(size)])


def train_stage(name, model, loss, val, N, args, log, gen, plot, draw=None, bf16=False, batch=None, lr=None,
                patience=None, backward=True, after_step=None):
    """Train model on N examples, a batch's loss(idx) its mean loss, early stopping on val() (lower is better; a
    score and a description) scaled by (1 + --epoch-penalty) ** epoch, so a later epoch counts as better only if
    it beats an earlier one by that share per epoch between them; the weights kept are the best unscaled epoch's.
    With draw, it is called before every epoch (to draw the epoch's examples). With bf16, the training passes run
    in bfloat16 autocast on CUDA (val in float32). Batches of batch (default --batch) examples, at learning rate lr
    (default --lr) after the warmup, stopping after patience epochs without a better scaled score at the least
    (default --patience). With backward=False, loss runs its own backward pass (returning the loss detached);
    after_step(step), if given, runs after every optimizer step. Returns the val score per epoch."""
    if patience is not None:
        args = copy.copy(args)
        args.patience = patience
    opt = torch.optim.AdamW(model.parameters(), lr=lr or args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    best, state, scores = (math.inf, -1), None, []
    stop = (math.inf, -1)  # the best scaled score and its epoch
    step = 0
    for epoch in range(args.epochs):
        began = time.time()
        if draw is not None:
            draw()
        drawn = time.time() - began
        model.train()
        running = 0.0
        for idx in torch.randperm(N, generator=gen).split(batch or args.batch):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(args.device.type, torch.bfloat16, enabled=bf16 and args.device.type == "cuda"):
                out = loss(idx.to(args.device)).float()
            if backward:
                out.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if after_step is not None:
                after_step(step)
            running += out.item() * len(idx)
        trained = time.time() - began
        model.eval()
        with torch.no_grad():
            score, detail = val()
        scores.append(score)
        improved = score < best[0]
        if improved:
            best = (score, epoch)
            state = copy.deepcopy(model.state_dict())
        if args.epoch_penalty:
            assert score > 0, f"{name}: the epoch penalty needs a positive val score (0 = perfect), not {score}"
        scaled = score * (1 + args.epoch_penalty) ** epoch
        if scaled < stop[0]:
            stop = (scaled, epoch)
        log(f"  {name} epoch {epoch}: {detail}; training loss {running / N:.6f} ("
            + (f"{drawn:.0f}s draw, " if draw is not None else "")
            + f"{trained - drawn:.0f}s train, {time.time() - began - trained:.0f}s val){'  *' if improved else ''}")
        plot(scores)
        if stagnant(epoch, stop[1], args):
            break
    assert state is not None
    model.load_state_dict(state)
    log(f"  {name}: kept epoch {best[1]} ({best[0]:.6f})"
        + (f"; best scaled score at epoch {stop[1]}" if args.epoch_penalty else "")
        + f", stopped after epoch {len(scores) - 1}")
    return scores


def lloyd_from_checkpoint(path, steps, device):
    """A LloydEncoder from a train_encoder.py checkpoint (its network, data mean and tracked starts)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["shared"]["config"]
    assert not (cfg.get("seeded") or cfg.get("node_mean") or cfg.get("scaled")), \
        "the Lloyd encoder takes an unseeded, unscaled checkpoint without --node-mean"
    enc = LloydEncoder(cfg["depths"], steps, cfg["width"], cfg["layers"], cfg["heads"]).to(device)
    enc.net.load_state_dict(ckpt["shared"]["state_dict"])
    enc.mean.copy_(ckpt["mean"])
    assert ckpt.get("grams") is not None and len(ckpt["grams"]) >= cfg["depths"], \
        "the checkpoint must hold its depths' tracked starts (trained with --start depth)"
    enc.grams.copy_(ckpt["grams"][:cfg["depths"]])
    enc.tracked.copy_(ckpt["tracked"][:cfg["depths"]])
    return enc


def new_encoder(args, dims, mean, X):
    """A fresh encoder: the k-means one, or with --encoder lloyd the Lloyd one, every depth's start the training
    remainders' top direction until tracked."""
    if args.encoder == "lloyd":
        enc = LloydEncoder(args.depths, args.lloyd_steps, args.width, args.layers, args.heads).to(args.device)
        R = X - mean
        top = top_direction(R)
        with torch.no_grad():
            enc.mean.copy_(mean)
            enc.grams.copy_((R.T @ R / len(R)).expand(args.depths, D, D))
            enc.tracked.copy_(top.expand(args.depths, D))
            enc.net.starts.copy_(enc.tracked)
        return enc
    enc = CodeEncoder(**dims, direct=args.direct_centroids, centered=args.centered_offsets).to(args.device)
    enc.mean.copy_(mean)
    return enc


def saved(model):
    return None if model is None else dict(config=model.config, state_dict=model.state_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=int, default=24, help="tokens per code")
    p.add_argument("--V", type=int, default=2, help="centroids per node, the tokens' alphabet")
    p.add_argument("--rounds", type=int, default=1000, help="the most rounds")
    p.add_argument("--round-patience", type=int, default=3,
                   help="stop once this many rounds in a row made no better encoder (val mse after all tokens)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1, help="in the prior and the decoder (and k-means encoder)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr", type=float, default=None,
                   help="the encoder's learning rate when it starts afresh (default: --lr)")
    p.add_argument("--warm-lr", type=float, default=1e-4,
                   help="the learning rate of a stage whose model carries last round's weights")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200, help="steps of linear learning-rate warmup in each stage")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000, help="the most epochs of each stage")
    add_patience_args(p)
    p.set_defaults(patience=5)
    p.add_argument("--encoder-patience", type=int, default=None,
                   help="the encoder's least patience, in epochs (default: --patience)")
    p.add_argument("--epoch-penalty", type=float, default=0.0,
                   help="early stopping scales each epoch's val score by (1 + this) ** epoch")
    p.add_argument("--encoder", choices=("kmeans", "lloyd"), default="kmeans",
                   help="the k-means encoder (CodeEncoder) or the Lloyd encoder (LloydEncoder, V = 2)")
    p.add_argument("--init-encoder", default=None, metavar="CKPT",
                   help="with --encoder lloyd: round 0's encoder from a train_encoder.py checkpoint, not trained here")
    p.add_argument("--lloyd-steps", type=int, default=16, help="the Lloyd encoder's steps per node")
    p.add_argument("--max-nodes", type=int, default=1024,
                   help="nodes per forward pass in the Lloyd encoder's training (more run in pieces)")
    p.add_argument("--ema", type=float, default=0.95,
                   help="the Lloyd encoder's tracked starts: their running statistics' decay per step")
    p.add_argument("--refresh", type=int, default=50, help="steps between the Lloyd encoder's start updates")
    p.add_argument("--draws", type=int, default=None,
                   help="images generated for each epoch of the encoder (default: as many as training images)")
    p.add_argument("--val-draws", type=int, default=5000,
                   help="images generated once a round and held out, the encoder's k-means objective on them "
                        "logged each epoch (0: none)")
    p.add_argument("--encoder-batch", type=int, default=1024, help="images per step of the encoder")
    p.add_argument("--attraction", type=float, default=0.0,
                   help="the k-means encoder's loss also pulls every centroid toward all of its node's images, this "
                        "much (dvae/rounds.py's kmeans_sse), so a centroid that wins none still moves")
    p.add_argument("--sample-batch", type=int, default=32768,
                   help="codes sampled at once, sharing their prefixes' work (memory grows with it)")
    p.add_argument("--sample-fp32", action="store_true", help="sample in float32 (default: bfloat16 on CUDA)")
    p.add_argument("--train-fp32", action="store_true",
                   help="train the k-means encoder in float32 (default: bfloat16 autocast on CUDA; the prior and "
                        "decoder, too small to gain, and the Lloyd encoder always train in float32)")
    p.add_argument("--cov-rank", type=int, default=0,
                   help="the decoder's covariance (decoder_cov): a Gaussian around each leaf's mean, covariance "
                        "F F^T + diag(var) with F of this rank, fit by likelihood with its own early stopping (val "
                        "bits per pixel); the generator's images are drawn from it, not the leaves' means (0: none)")
    p.add_argument("--var-floor", type=float, default=1e-4, help="the least per-pixel variance, with --cov-rank")
    p.add_argument("--fresh-encoder", action="store_true",
                   help="start every round's encoder from new random weights, not last round's")
    p.add_argument("--centered-offsets", action="store_true",
                   help="the k-means encoder's V offsets at a node are centred (their mean taken off), so its "
                        "centroids average to the centroid chosen last (the data's mean at the root): for V = 2, "
                        "c = base +- r")
    p.add_argument("--direct-centroids", action="store_true",
                   help="the k-means encoder gives each node's centroids as the data's mean plus its output, not as "
                        "offsets from the centroid chosen last (which, trained by k-means, repeats the root's split "
                        "below it)")
    p.add_argument("--no-channels-last", action="store_true",
                   help="run the image heads' convolutions in the default layout, not channels_last")
    p.add_argument("--node-batch", type=int, default=2048, help="nodes whose centroids are computed at once")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--log", default=None, help="summary log (default: run.log in the run's artifacts folder)")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="continue after a finished round: its round_<r>.pt, or a run's folder (its latest round); the "
                        "generator (its models' architecture), data mean and history come from it, every setting "
                        "from now")
    args = p.parse_args()
    args.log = args.log or artifact("run.log")
    args.device = get_device(args.device)
    device, K, V = args.device, args.depths, args.V
    assert args.encoder == "kmeans" or V == 2, "the Lloyd encoder has V = 2"
    assert args.init_encoder is None or args.encoder == "lloyd", "--init-encoder takes a Lloyd encoder"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    # (not cudnn.benchmark: routing makes new shapes every epoch, and tuning each one cost more than it saved)
    log = RunLog(args.log, device)
    log(f"=== {time.strftime('%Y-%m-%d')} train_rounds.py {vars(args)}")

    X = pixels(load_mnist("train", args.data_root)[:args.limit_train]).to(device)
    Xv = pixels(load_mnist("val", args.data_root)[:args.limit_val]).to(device)
    XA, N = torch.cat([X, Xv]), len(X)
    val_idx = torch.arange(N, len(XA), device=device)
    mean = X.mean(0)
    gen = torch.Generator().manual_seed(args.seed)
    dims = dict(K=K, V=V, width=args.width, layers=args.layers, heads=args.heads, dropout=args.dropout,
                channels_last=not args.no_channels_last)
    history = []  # per round: each stage's val score per epoch
    best_round = (math.inf, -1)  # the best encoder's val mse after all tokens, and its round
    first = 0
    encoder: CodeEncoder | LloydEncoder | None = None  # the generator's, none while it is the training images
    if args.resume:
        path = args.resume
        if os.path.isdir(path):
            rounds = [f for f in os.listdir(path) if f.startswith("round_") and f.endswith(".pt")]
            path = os.path.join(path, max(rounds, key=lambda f: int(f[6:-3])))
        ckpt = torch.load(path, map_location=device, weights_only=False)
        assert "decoder_mean" in ckpt, f"{path} is from before the generator rounds and cannot be resumed"

        def restore(cls, entry):
            model = cls(**entry["config"]).to(device)
            model.load_state_dict(entry["state_dict"])
            return model

        encoder = restore(LloydEncoder if ckpt["encoder"]["kind"] == "lloyd" else CodeEncoder, ckpt["encoder"])
        prior, decoder_mean = restore(CodePrior, ckpt["prior"]), restore(DecoderMean, ckpt["decoder_mean"])
        decoder_cov = restore(DecoderCov, ckpt["decoder_cov"]) if ckpt.get("decoder_cov") else \
            DecoderCov(**dims, rank=args.cov_rank, var_floor=args.var_floor).to(device) if args.cov_rank else None
        mean, history, first = ckpt["mean"].to(device), ckpt["history"], ckpt["round"] + 1
        for h in history:
            if h.get("encoder_out") and h["encoder_out"][-1] < best_round[0]:
                best_round = (h["encoder_out"][-1], h["round"])
        log(f"resumed {path} after round {ckpt['round']}; the best encoder so far is round {best_round[1]}'s "
            f"({best_round[0]:.6f})")
    else:
        prior, decoder_mean = CodePrior(**dims).to(device), DecoderMean(**dims).to(device)
        decoder_cov = DecoderCov(**dims, rank=args.cov_rank, var_floor=args.var_floor).to(device) \
            if args.cov_rank else None
    decoder = Decoder(decoder_mean, decoder_cov)
    log(f"models: prior {sum(q.numel() for q in prior.parameters()) / 1e6:.1f}M, decoder_mean "
        f"{sum(q.numel() for q in decoder_mean.parameters()) / 1e6:.1f}M"
        + (f", decoder_cov {sum(q.numel() for q in decoder_cov.parameters()) / 1e6:.1f}M"
           if decoder_cov is not None else "")
        + f" parameters; {N} train and {len(Xv)} val images")

    plot_path = os.path.splitext(args.log)[0] + ".png"
    for r in range(first, args.rounds):
        began = time.time()
        h: dict = dict(round=r, encoder=[], prior=[], decoder_mean=[], decoder_cov=[])
        history.append(h)

        def plot(stage):
            def draw(scores):
                h[stage] = list(scores)
                plot_rounds(history, plot_path)
            return draw

        # 1. the encoder, on images from the generator
        if encoder is None and args.init_encoder:
            next_encoder = lloyd_from_checkpoint(args.init_encoder, args.lloyd_steps, device)
            mean = next_encoder.mean.clone()
            log(f"round {r}: the encoder is {args.init_encoder}'s")
        else:
            warm = encoder is not None and not args.fresh_encoder
            next_encoder = copy.deepcopy(encoder) if warm else new_encoder(args, dims, mean, X)
            assert next_encoder is not None
            images: dict[str, torch.Tensor] = {}
            held_out, n_images = None, N
            if encoder is None:
                images["train"] = X
                draw = None
                log(f"round {r}: training the encoder on the {N} training images")
            else:
                n_images = args.draws or N

                def draw_images(frozen=encoder):
                    images["train"] = sample_codes(frozen, prior.eval(), decoder.eval(), mean, n_images, K, V,
                                                   args.sample_batch, not args.sample_fp32).leaf

                draw = draw_images

                if args.val_draws:
                    held_out = sample_codes(encoder, prior.eval(), decoder.eval(), mean, args.val_draws, K, V,
                                            args.sample_batch, not args.sample_fp32).leaf
                log(f"round {r}: training the encoder on {n_images} images an epoch from round {r - 1}'s generator")

            def encoder_val(enc=next_encoder, held_out=held_out):
                _, mses = route(enc, Xv, mean, K, V, args.node_batch)
                detail = f"val mse after {K} tokens {mses[-1]:.6f} (per depth " + " ".join(f"{m:.5f}" for m in mses) \
                    + ")"
                if held_out is not None:
                    _, objective = route(enc, held_out, mean, K, V, args.node_batch)
                    detail += f"; k-means objective {sum(objective) / K:.6f} on {len(held_out)} held-out generated " \
                              "images"
                return mses[-1], detail

            stage = dict(lr=args.warm_lr if warm else args.encoder_lr, patience=args.encoder_patience,
                         batch=args.encoder_batch, draw=draw)
            if isinstance(next_encoder, LloydEncoder):
                enc = next_encoder
                lloyd_args = SimpleNamespace(ema=args.ema, max_nodes=args.max_nodes, tol=0.0, dir_weight=1.0,
                                             batch=args.encoder_batch)
                caps = [enc.steps] * K

                def lloyd_loss(idx, enc=enc, lloyd_args=lloyd_args, caps=caps):
                    stats, _, _ = train_step(enc.net, images["train"][idx], enc.mean, enc.grams, lloyd_args, caps)
                    return stats[0].sum() / (len(idx) * K * D)

                def after_step(step, enc=enc):
                    if step % args.refresh == 0:
                        refresh(enc.net, enc.grams, enc.tracked)

                h["encoder"] = train_stage(f"round {r} encoder", enc, lloyd_loss, encoder_val, n_images, args, log,
                                           gen, plot("encoder"), backward=False, after_step=after_step, **stage)
            else:
                assert isinstance(next_encoder, CodeEncoder)
                h["encoder"] = train_stage(
                    f"round {r} encoder", next_encoder,
                    lambda idx, enc=next_encoder: kmeans_sse(enc, images["train"][idx], args.attraction)
                    / (len(idx) * K * D),
                    encoder_val, n_images, args, log, gen, plot("encoder"), bf16=not args.train_fp32, **stage)
            del images
        encoder = next_encoder.eval()

        # 2. the train and val images down its tree
        tree = route(encoder, XA, mean, K, V, args.node_batch)[0]
        h["encoder_out"], h["encoder_out_train"] = tree.mses(slice(N, None)), tree.mses(slice(0, N))
        sizes = " ".join(str(len(t)) for t in tree.token[1:])
        log(f"round {r}: the encoder's val mse after {K} tokens {h['encoder_out'][-1]:.6f} (train "
            f"{h['encoder_out_train'][-1]:.6f}; val per depth " + " ".join(f"{m:.5f}" for m in h["encoder_out"])
            + f"); nodes per depth {sizes}")
        plot_rounds(history, plot_path)
        warm_lr = args.warm_lr if r > 0 else None  # (the prior and decoder carry last round's weights)

        # 3. the prior
        prior_scores = train_stage(
            f"round {r} prior", prior, lambda idx: prior_bits(prior, tree, idx).mean() * LN2 / K,
            lambda: (lambda b: (b, f"val {b:.3f} bits per image"))(
                chunked(lambda i: prior_bits(prior, tree, i), val_idx).mean().item()),
            N, args, log, gen, plot("prior"), lr=warm_lr)

        # 4. the decoder: its mean, then its covariance
        train_stage(
            f"round {r} decoder_mean", decoder_mean,
            lambda idx: decoder_sse(decoder_mean, tree, XA, idx).mean() / D,
            lambda: (lambda m: (m, f"val mse {m:.6f}"))(
                chunked(lambda i: decoder_sse(decoder_mean, tree, XA, i), val_idx).mean().item() / D),
            N, args, log, gen, plot("decoder_mean"), lr=warm_lr)
        decoder_mean.eval()
        with torch.no_grad():
            h["decoder_mean_train"] = chunked(lambda i: decoder_sse(decoder_mean, tree, XA, i),
                                              torch.arange(N, device=device)).mean().item() / D
            h["decoder_mean_val"] = chunked(lambda i: decoder_sse(decoder_mean, tree, XA, i),
                                            val_idx).mean().item() / D
        cov_scores = []
        if decoder_cov is not None:
            with torch.no_grad():  # every image's leaf mean, which decoder_cov reads and centres on
                mus = chunked(lambda i: decoder_means(decoder_mean, tree, i), torch.arange(len(XA), device=device))

            def cov_val(cov=decoder_cov):
                nats = chunked(lambda i: cov_nll(cov, tree, mus, XA, i), val_idx).mean().item() / D
                bits = (nats + math.log(256)) / LN2  # each 8-bit pixel's probability, density times the bin width
                return bits, f"val {bits:.4f} bits per pixel (nll of the 8-bit values)"

            cov_scores = train_stage(
                f"round {r} decoder_cov", decoder_cov,
                lambda idx, cov=decoder_cov: cov_nll(cov, tree, mus, XA, idx).mean() / D,
                cov_val, N, args, log, gen, plot("decoder_cov"), lr=warm_lr)
            decoder_cov.eval()
            del mus
        del tree

        # the generator (encoder, prior, decoder), saved
        torch.save(dict(args=vars(args), round=r, mean=mean.cpu(), history=history,
                        encoder=dict(kind="lloyd" if isinstance(encoder, LloydEncoder) else "kmeans",
                                     config=encoder.config, state_dict=encoder.state_dict()),
                        prior=saved(prior), decoder_mean=saved(decoder_mean), decoder_cov=saved(decoder_cov)),
                   artifact(f"round_{r}.pt"))
        improved = h["encoder_out"][-1] < best_round[0]
        if improved:
            best_round = (h["encoder_out"][-1], r)
        log(f"round {r} done ({(time.time() - began) / 60:.0f} min): encoder val mse after {K} tokens "
            f"{h['encoder_out'][-1]:.6f} (train {h['encoder_out_train'][-1]:.6f}){'  *' if improved else ''}, prior "
            f"{min(prior_scores):.3f} bits per image, decoder_mean val mse {h['decoder_mean_val']:.6f} (train "
            f"{h['decoder_mean_train']:.6f})"
            + (f", decoder_cov {min(cov_scores):.4f} bits per pixel" if cov_scores else ""))
        if r - best_round[1] >= args.round_patience:
            log(f"{args.round_patience} rounds without a better encoder: stopping; the best is round "
                f"{best_round[1]}'s ({best_round[0]:.6f}, round_{best_round[1]}.pt)")
            break


if __name__ == "__main__":
    main()
