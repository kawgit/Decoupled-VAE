# Decoupled VAE — MNIST MVP

A discrete VAE with a 1-D autoregressive latent of `k_max` tokens over a vocab of `V`, trained in
three separate stages. The encoder is trained alone, against an ideal decoder. Then the decoder and
the prior are each trained on the frozen encoder's codes.

Default latent: `V=2`, `k_max=10`, i.e. 1024 codes. `analyze_encoder.py` and `benchmark.py`
enumerate every code, so they need a code space that small; training works at any size.

## Encoder objective

```
L(φ) = E_x Σ_k p(k) E_{z~q(z_{1:k}|x)} F_k(x, z)
F_k(x, z) = E_{y~p(y|z_{1:k})} d(x, y),     p(y | z_{1:k}) ∝ q(z_{1:k} | y) p_data(y)
```

`d` is mean squared pixel distance and `p(k)` is set by `--k-dist uniform|geometric|custom`.

Each step takes `B = --batch-size` anchors and a pool of `D = --num-refs` reference images: the
anchors themselves plus D − B extras. Each anchor is left out of its own posterior. For each anchor
and k, the loss is Σₙ ωₙ Fₙ over a set of codes n:

- **Enumerated**, when `--num-samples` ≥ Vᵏᵐᵃˣ: every prefix, with ω = q(prefix | x). Exact given the
  pool; cost D × (Vᵏᵐᵃˣ − 1)/(V − 1) tree positions.
- **Sampled**, otherwise: S = `--num-samples` codes per anchor (default 16), drawn from q′ (q at
  `--temperature`, default 1), with ω = (q / q′)(prefix) / S. Unbiased; cost B × S × D × k_max
  positions, independent of V.

There are two proposals, each corrected with importance weights:

- **q′:** what codes are drawn from.
- **posterior′:** what the extra references are drawn from, chosen by `--ref-ranking`:
  - `uniform` (default);
  - `two-towers`: a learned score, image vector · prefix vector, with one prefix vector per prefix
    length. posterior′ is the softmax over training images, mixed over k by p(k), so it matches the
    posterior's breadth at every prefix length. The towers train alongside the encoder, against the
    posteriors each step already computes, so their scores track log q(z₁:ₖ | y) and the softmax needs
    no extra temperature.

  `--ref-uniform-frac` (default 0.25) of the extras stay uniform so every image stays reachable. The
  posterior is self-normalized with weights p_data / s, where s is the mixture the pool was drawn
  from. That makes it consistent, with a bias that shrinks as the number of effective references grows.

`--estimator` picks the gradient terms:

| | own-choice (REINFORCE) term | pathwise term |
|---|---|---|
| `full` (default) | ✓ | ✓ |
| `reinforce` | ✓ | ✗ (reference scoring runs without gradient, much less memory) |
| `pathwise` | ✗ | ✓ |

In sampled mode the REINFORCE term uses a leave-one-out baseline over the S samples
(`--baseline loo|none`).

- **Batch size:** if `--batch-size` is unset, B is the largest that fits in `--max-mem-gb`. When
  enumerating with uniform references, B = D, which is the original exact setup.
- **Validation:** always uses fixed pools of D uniform val images and codes drawn from q, so runs
  with different settings are comparable.
- **Val log:** reports `per_k` (the expected distance at each prefix length), `k0` (the distance with
  no code) and `ess` (the effective number of references behind each posterior, per k).
  `train_logs` holds the same for the training pools.

See `dvae/objective.py` and `dvae/references.py`.

## Setup

```bash
source .venv/bin/activate   # torch + torchvision; MNIST downloads to data/ on first run
```

The device is picked automatically (cuda → mps → cpu); override it with `--device`.

## Pipeline

```bash
python train_encoder.py                                  # -> checkpoints/encoder.pt
python train_decoder.py                                  # -> checkpoints/decoder_gray.pt   (8-bit pixels)
python train_decoder.py --likelihood binary              # -> checkpoints/decoder_binary.pt (binarized)
python train_prior.py                                    # -> checkpoints/prior.pt
python qa_generate.py -n 64                              # -> outputs/generate.png
python qa_compress.py -n 16 -k 1 2 4 10 -m 8             # -> outputs/compress/k01.png ...
python benchmark.py --decoder checkpoints/decoder_gray.pt checkpoints/decoder_binary.pt \
                    --encoder checkpoints/encoder.pt     # -> results/benchmark.{json,md}
python analyze_encoder.py --encoder checkpoints/encoder.pt # -> results/usage_encoder.json
```

- **Training:** every training script early-stops on its val loss (a fixed 5k split of the MNIST
  train set) and keeps the best checkpoint at `--out`. Set the learning rate with `--lr`.
- **Stopping and resuming:**
  - Ctrl-C stops at the end of the current step and saves `*_latest.pt` next to `--out`. A second
    Ctrl-C aborts without saving. `*_latest.pt` is also rewritten after every epoch.
  - Both checkpoints hold the full training state: model, optimizer, scheduler, counters, RNG and
    shuffle state, and the original args. A resumed run matches an uninterrupted one step for step,
    even mid-epoch.
  - To resume, pass the checkpoint alone: `python train_encoder.py --resume checkpoints/encoder_latest.pt`.
  - Flags you add when resuming override the saved ones, e.g. `--epochs 50 --lr 1e-4`. Don't change
    architecture flags.
- **Token sampling:** `--encoder-mode` and `--prior-mode` pick how tokens are drawn:
  - `actual`: the model's own distribution;
  - `greedy`: the most likely token at each step;
  - `topp`: nucleus sampling, with `--*-top-p`.
- **Compression QA:** `qa_compress.py` encodes greedily by default, so it acts as a deterministic
  codec. Each output row is the original image followed by `m` decodings, each with the rest of
  the code filled in by the prior.
- **Benchmark:** `benchmark.py` scores each decoder on the data it was trained for: 8-bit MNIST, or
  the test set binarized with a fixed seed. It reports exact NLL, plus −ELBO and rate when you pass
  `--encoder`, in both nats/image and bits/dim.

- **Latent usage:** `analyze_encoder.py` reports, for each prefix length k, how many of the
  k·log₂V bits carry information about the image, as I(x; z₁:ₖ). It also shows how much each extra
  token adds, how many codes are used, and how many bits about the digit label each prefix
  carries. Everything is computed exactly over a split; use `--split test` for final numbers.

## Layout

```
dvae/data.py        MNIST tensors, train/val split, binarization
dvae/models.py      Encoder, Prior, Decoder; tree enumeration of all code prefixes
dvae/objective.py   decoupled encoder objective (leave-one-out, numerically stable)
dvae/common.py      device, checkpoints, CLI flags, early-stopping loop
dvae/viz.py         image grids
```

## Speed (M3 Pro, MPS)

- **Enumerated** at `V=2`, `k_max=10`, `D=256` (`--num-samples 1024`): about 1.1 s/step, or about
  4 min/epoch, using about 5.5 GB.
- **Sampled:** the cost scales with B × S × D × k_max. The defaults pick B to fit `--max-mem-gb`.
- Decoder and prior epochs take about a minute.
