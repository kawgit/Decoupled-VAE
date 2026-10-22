# Decoupled VAE — MNIST MVP (partition-encoder branch)

A discrete VAE with a 1-D autoregressive latent of `k_max` tokens over a vocab of `V`, trained in
three separate stages. The encoder is built first, against an ideal decoder. Then the decoder and
the prior are each trained on the frozen encoder's codes.

Default latent: `V=2`, `k_max=10`, i.e. 1024 codes. `analyze_encoder.py` and `benchmark.py`
enumerate every code, so they need a code space that small.

## The objective

```
L = E_x Σ_k p(k) E_{z~q(z_{1:k}|x)} F_k(x, z)
F_k(x, z) = E_{y~p(y|z_{1:k})} d(x, y),     p(y | z_{1:k}) ∝ q(z_{1:k} | y) p_data(y)
```

`d` is mean squared pixel distance.

## How the encoder is built

On this branch the encoder isn't trained by descending L directly. That objective is non-convex in the
encoder's outputs, much like k-means. Instead `train_encoder.py` builds the code one token at a time. At
each position it constructs candidate partitions, trains the encoder to reproduce all of them, and
learns which one to use for each code.

**Second definition of decompression loss.** For a partition of a weighted set of (image, code) pairs into
V classes, this is the (first-definition) loss of the new token for an encoder that exactly reproduces
(image, code + class).

At position k (prefix length t = k−1):

1. **Partitions** (`dvae/partition.py`):
   - Sample `--anchors` P anchor images and `--references` R reference images, each with one
     out-of-fold code.
   - For every anchor, truncate each reference's code to s+1 tokens, where s is the length of the
     prefix it shares with the anchor's code (the whole code when it matches). References are
     thereby grouped finely around the anchor's code and more coarsely further from it.
   - Within each group, Hartigan's method (from a random start, with batched moves that are kept only
     if the total drops) assigns references to V classes, strictly lowering the second-definition loss
     until no single move helps.
   - A partition labels any image with the class whose loss rises least if the image joins it.
     Codes whose group has no references get class 0.
2. **Encoders.**
   - Train `--folds` + 1 `StageNet`s with one head per partition, by cross-entropy against the
     partitions' labels: fold model f on the training images outside fold f, and a final model on
     all of them.
   - Each minibatch draws one fresh out-of-fold code per image (memoized, `--memo-cap`), so each
     position learns from the mistakes earlier positions really make on unseen images.
   - All copies share the epoch count, chosen by the pooled held-out cross-entropy.
3. **Costs.** For every training image, with its held-out encoder, the new token's decompression loss
   under each partition. It's exact, with all training images as references.
4. **Selectors.** Train `--folds` + 1 `SelectorNet`s to predict those costs from the code alone (MSE),
   with the same fold split, early-stopped on held-out MSE. The encoder uses the partition with the
   lowest predicted cost, so the new token means the same thing for every image with the same code.
5. **Report.** The new token's loss on the val split: with the selector's choices, with the single
   best partition, and with no code.

Every step logs when it starts, its progress and its duration.

## Setup

```bash
source .venv/bin/activate   # torch + torchvision; MNIST downloads to data/ on first run
```

The device is picked automatically (cuda → mps → cpu); override it with `--device`.

## Pipeline

```bash
python train_encoder.py                                  # -> checkpoints/encoder.pt (+ encoder_build.pt)
python train_decoder.py                                  # -> checkpoints/decoder_gray.pt   (8-bit pixels)
python train_decoder.py --likelihood binary              # -> checkpoints/decoder_binary.pt (binarized)
python train_prior.py                                    # -> checkpoints/prior.pt
python qa_generate.py -n 64                              # -> outputs/generate.png
python qa_compress.py -n 16 -k 1 2 4 10 -m 8             # -> outputs/compress/k01.png ...
python benchmark.py --decoder checkpoints/decoder_gray.pt checkpoints/decoder_binary.pt \
                    --encoder checkpoints/encoder.pt     # -> results/benchmark.{json,md}
python analyze_encoder.py --encoder checkpoints/encoder.pt # -> results/usage_encoder.json
```

- **Building the encoder:**
  - `checkpoints/encoder.pt` is rewritten after every position, so the other scripts can use a
    partly built encoder.
  - `checkpoints/encoder_build.pt` keeps every fold model. Continue with `--resume` (and a larger
    `--k-max` to add positions). Ctrl-C loses only the position in progress.
  - For quick tests, `--limit-train` and `--limit-val` use fewer images.
- **Decoder and prior training:**
  - They early-stop on their val loss (a fixed 5k split of the MNIST train set) and keep the best
    checkpoint at `--out`. Set the learning rate with `--lr`.
  - Ctrl-C saves `*_latest.pt`. Continue with `--resume checkpoints/<name>_latest.pt`; flags you add
    override the saved ones.
- **Token sampling:** `--encoder-mode` and `--prior-mode` pick how tokens are drawn:
  - `actual`: the model's own distribution;
  - `greedy`: the most likely token at each step;
  - `topp`: nucleus sampling, with `--*-top-p`.
- **Compression QA:** `qa_compress.py` encodes greedily by default. Each output row is the original
  image followed by `m` decodings, each with the rest of the code filled in by the prior.
- **Benchmark:** `benchmark.py` scores each decoder on the data it was trained for: 8-bit MNIST, or
  the test set binarized with a fixed seed. It reports exact NLL, plus −ELBO and rate when you pass
  `--encoder`.
- **Latent usage:** `analyze_encoder.py` reports, for each prefix length k, how many bits carry
  information about the image and about the digit label, and how many codes are used.

## Layout

```
train_encoder.py    builds the staged encoder (anchor partitions, folded encoders and selectors)
dvae/staged.py      StageNet and SelectorNet (one each per position), StagedEncoder (sample, log_prob, tree)
dvae/partition.py   anchor partitions (truncated codes, Hartigan, label rule), exact per-prefix objective
dvae/models.py      Prior, Decoder; tree enumeration of all code prefixes
dvae/data.py        MNIST tensors, train/val split, labels, binarization
dvae/common.py      device, checkpoints, CLI flags, early-stopping loop
dvae/viz.py         image grids
```
