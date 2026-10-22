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
encoder's outputs, much like k-means. Instead `train_encoder.py` builds the code one token at a time,
and at each position it *chooses* a partition and trains a classifier to reproduce it:

1. **Prefixes.** For each training image, draw `--prefix-samples` (S, default 4) *distinct* prefixes
   z₁..z_t without replacement from its out-of-fold encoder distribution, i.e. using the models of
   earlier positions on images they weren't trained on. The method is stochastic beam search. Each
   prefix gets an importance weight q(z)/π(z), so weighted sums estimate sums over the image's whole
   prefix distribution, and are exact once S covers every prefix. Cost is fixed per image, whatever
   the code-space or dataset size.
2. **Candidates** (`dvae/partition.py`).
   - **Generation:** draw `--num-candidates` (default 64) candidates. Each picks a sharing level m
     uniformly from 0..t, groups the prefixes by their first m tokens, and fits one weighted V-means
     codebook per group to the residuals (image minus its full prefix group's mean). m = t gives every
     prefix its own codebook, and m = 0 gives one codebook for all. Repeated m values act as restarts.
   - **Labels:** an image's label is its residual's nearest codeword. Codewords are labelled by rank
     along one shared direction (`--labels aligned`), a pixel-space heuristic; `--labels random`
     shuffles them per group instead.
   - **Pruning:** every candidate is scored by its exact in-sample objective at this position, on
     training images only. Near-duplicates (weighted per-prefix ARI > `--dedup-ari`) are dropped in
     favour of the better-scoring one, and the best `--keep-candidates` (default 32) are kept.
3. **Training.** Train `--folds` + 1 copies of a `StageNet` with one head per kept candidate, by
   cross-entropy: fold model f on the images outside fold f, and a final model on all of them.
   Each batch draws one prefix per image from the out-of-fold models (a fresh draw every epoch), so each position learns from the
   mistakes earlier positions really make on unseen images. Those next-token distributions are
   memoized (`--memo-cap`), since earlier positions are frozen.
4. **Scoring and selection.** After every epoch, every head is scored on each fold's held-out images
   with the decompression objective, over its weighted prefix entries. The score is exact, with all
   training entries as references, from per-prefix sums. The best head is picked *per prefix*, and the epoch with the best total is
   kept. The fold models and the final model share that epoch count and that selection.
5. **The stage.** The final model plus its per-prefix head selector becomes this position. The log
   reports its honest score on the separate val split (S distinct prefixes per val image, through the
   final models), for the per-prefix selection and for the single best head.

The ground truth for every choice is the decompression objective of a trained model on held-out images.
Pixel-space criteria (k-means fit, alignment) only generate the candidates.

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
train_encoder.py    builds the staged encoder (folds, multi-head training, per-prefix head selection)
dvae/staged.py      StageNet (one per position) and StagedEncoder (sample, log_prob, tree enumeration)
dvae/partition.py   weighted prefix entries, candidate codebooks, pruning/dedup, exact per-prefix objective
dvae/models.py      Prior, Decoder; tree enumeration of all code prefixes
dvae/data.py        MNIST tensors, train/val split, labels, binarization
dvae/common.py      device, checkpoints, CLI flags, early-stopping loop
dvae/viz.py         image grids
```
