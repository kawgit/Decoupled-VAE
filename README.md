# Decoupled VAE — MNIST MVP

A discrete VAE with a 1-D autoregressive latent of `k_max` tokens over a vocab of `V`, trained in
three separate stages. The encoder is trained alone, against an ideal decoder. Then the decoder and
the prior are each trained on the frozen encoder's codes.

Default latent: `V=2`, `k_max=10`, i.e. 1024 codes. That's small enough to enumerate, so:

- the encoder objective and its gradient are exact given the reference pool (no REINFORCE);
- the benchmark reports the exact test log-likelihood, not just a bound.

Larger code spaces (e.g. `V=8`, `k_max=8`) train with a sampled estimator instead; see below.
`analyze_encoder.py` and `benchmark.py` still require a code space small enough to enumerate.

## Encoder objective

```
L(φ) = E_x E_{k~p(k)} E_{z~q(z_{1:k}|x)} E_{y~p(y|z,k)} [ d(x, y) ]
p(y | z, k) ∝ q(z_{1:k} | y) p_data(y)
```

- `d` is mean squared pixel distance.
- The posterior is estimated over `D = --num-refs` reference images, with the anchor image itself
  left out.
- `p(k)` is set by `--k-dist uniform|geometric|custom`.

`--estimator` picks how the expectation over codes is computed. The default, `auto`, uses exact
whenever its estimated memory fits under `--max-mem-gb`:

| | `exact` | `sampled` |
|---|---|---|
| per step | a pool of D images; the first `B = --batch-size` (default: all D) are anchors | B anchors (default 32) and D separate random references |
| codes | every prefix of every code, weighted by q | `S = --samples` codes drawn from each anchor's q |
| gradient | exact, by backprop | backprop through the posterior + REINFORCE for the sampling (reward-to-go per token, leave-one-out baseline over the S samples) |
| cost | D × (Vᵏᵐᵃˣ − 1)/(V − 1) tree positions | B × S × D × k_max positions, independent of V |
- The val log reports `per_k`, the expected distance at each prefix length, next to `k0`, the
  distance with no code.

See `dvae/objective.py`.

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

- **Exact**, at the defaults (`V=2`, `k_max=10`, `D=256`, `d_model=64`, 2 token layers): about
  1.1 s/step, or about 4 min/epoch, using about 5.5 GB.
- **Sampled**, at `V=8`, `k_max=8`, `B=32`, `S=4`, `D=256`: about 0.5 s/step, or about 15 min/epoch
  (each training image is an anchor once per epoch), using about 5.5 GB.
- Decoder and prior epochs take about a minute.
