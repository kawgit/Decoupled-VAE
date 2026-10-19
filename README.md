# Decoupled VAE — MNIST MVP

A discrete VAE with a 1-D autoregressive latent of `k_max` tokens over a vocab of `V`, trained in
three separate stages. The encoder is trained alone, against an ideal decoder. Then the decoder and
the prior are each trained on the frozen encoder's codes.

Default latent: `V=2`, `k_max=10`, i.e. 1024 codes. That's small enough to enumerate, so:

- the encoder objective and its gradient are exact given the reference pool (no REINFORCE);
- the benchmark reports the exact test log-likelihood, not just a bound.

## Encoder objective

```
L(φ) = E_x E_{k~p(k)} E_{z~q(z_{1:k}|x)} E_{y~p(y|z,k)} [ d(x, y) ]
p(y | z, k) ∝ q(z_{1:k} | y) p_data(y)
```

- `d` is mean squared pixel distance.
- Each step draws a pool of `D = --num-refs` training images. The posterior is estimated over the
  pool, with the anchor image itself left out.
- The first `B = --batch-size` pool images (default: all of them) are the anchors.
- `p(k)` is set by `--k-dist uniform|geometric|custom`.
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
```

- **Training:** every training script early-stops on its val loss (a fixed 5k split of the MNIST
  train set) and keeps the best checkpoint.
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

## Layout

```
dvae/data.py        MNIST tensors, train/val split, binarization
dvae/models.py      Encoder, Prior, Decoder; tree enumeration of all code prefixes
dvae/objective.py   decoupled encoder objective (leave-one-out, numerically stable)
dvae/common.py      device, checkpoints, CLI flags, early-stopping loop
dvae/viz.py         image grids
```

## Speed (M3 Pro, MPS)

Encoder training at the defaults (`D=256`, `d_model=64`, 2 token layers) is about 1.1 s/step, or
about 4 min/epoch, using about 5.5 GB. The cost scales with D × (number of tree nodes). Decoder and
prior epochs take about a minute.
