# Decoupled VAE — MNIST MVP (partition-encoder branch)

A discrete VAE with a 1-D autoregressive latent of `k_max` tokens over a vocab of `V`, trained in
three separate stages. The encoder is built first, against an ideal decoder. Then the decoder and
the prior are each trained on the frozen encoder's codes.

Default latent: `V=2`, `k_max=10`, i.e. 1024 codes. `analyze_encoder.py` and `benchmark.py`
enumerate every code, so they need a code space that small.

## The objective

$d$ is mean squared pixel distance throughout. In every posterior below, an image or pair is left out
of its own posterior.

**Decompression loss** of an encoder $q$ on an image dataset $x_1, \dots, x_N$:

```math
L \;=\; \frac{1}{N} \sum_i \sum_k p(k)\, \mathbb{E}_{z \sim q(z_{1:k} \mid x_i)}\, F_k(x_i, z),
\qquad
F_k(x_i, z) \;=\; \mathbb{E}_{j \sim \pi(j \mid z_{1:k}),\; j \ne i}\; d(x_i, x_j),
\qquad
\pi(j \mid z_{1:k}) \;\propto\; q(z_{1:k} \mid x_j).
```

**Marginal decompression loss** of a next-token encoder $q(c \mid x, z)$ on a weighted (image, code)
dataset $D = \{(x_i, z_i, w_i)\}$, with all codes of one length $t$. It measures what one more token adds
on top of the given codes:

```math
M(D, q) \;=\; \sum_i w_i\, \ell_i,
\qquad
\ell_i \;=\; \sum_c q(c \mid x_i, z_i) \sum_{j \ne i} \pi(j \mid z_i, c)\, d(x_i, x_j),
\qquad
\pi(j \mid z, c) \;\propto\; w_j\, \mathbf{1}[z_j = z]\, q(c \mid x_j, z_j).
```

The decompression loss is a $p(k)$-weighted average of marginal ones:

```math
L \;=\; \sum_k p(k)\, M(D_{k-1}, q_k),
\qquad
D_{k-1} \;=\; \Big\{\big(x_i,\, z,\, \tfrac{1}{N} q(z \mid x_i)\big) : z \in \{1, \dots, V\}^{k-1}\Big\},
\qquad
q_k \;=\; q(z_k \mid x, z_{1:k-1}).
```

Here $\pi(j \mid z, c) \propto q(z \mid x_j)\, q(c \mid x_j, z) = q(z c \mid x_j)$, which is the
posterior in $F_k$.

**A partition's implied marginal decompression loss.** A partition $\lambda$ gives each pair in $D$ one
of $V$ classes. Its implied marginal decompression loss is $M(D, q_\lambda)$ with
$q_\lambda(c \mid x_i, z_i) = \mathbf{1}[c = \lambda_i]$: the marginal decompression loss of an encoder
that reproduces each pair's class exactly. Each pair's
term is its mean distance to the other pairs with the same code and class, so the whole loss follows
from per-cluster sums.

## How the encoder is built

On this branch the encoder isn't trained by descending L directly. That objective is non-convex in the
encoder's outputs, much like k-means. Instead `train_encoder.py` builds the code one token at a time. At
each position it constructs candidate partitions, trains the encoder to reproduce all of them, and
learns which one to use for each code.

At each position the build works with the marginal decompression loss on the training set. D is every
training image with one sampled out-of-fold code of the current length and weight 1/N, a one-sample
estimate of D_t. Partitions are built and compared by their implied marginal decompression loss.

At position k (prefix length t = k−1):

1. **Partitions** (`dvae/partition.py`):
   - Sample `--anchors` P anchor images, each with one out-of-fold code.
   - Give every anchor `--references` R reference draws, split evenly over its t+1 groups by shared
     prefix length s = 0 … t (s = t is the full match). An image's mass in group s is its exact
     out-of-fold probability of landing there, q(a₁..aₛ | y) · (1 − q(aₛ₊₁ | y, a₁..aₛ)). Draws are
     with replacement and proportional to that mass. In a mismatch group, the token at position s+1 is
     drawn from q among the non-anchor tokens. Each draw weighs the group's mass divided by its draws.
   - So deep groups get as many references as shallow ones, however few images land there. The weights
     are equal within a group, so they don't change the partition; they only put the logged loss on a
     whole-training-set scale.
   - For every anchor, truncate each reference's code to s+1 tokens, where s is the length of the
     prefix it shares with the anchor's code (the whole code when it matches). References are
     thereby grouped finely around the anchor's code and more coarsely further from it.
   - Within each group, Hartigan's method (from a random start, with batched moves that are kept only
     if the total drops) assigns references to V classes, strictly lowering the partition's implied
     marginal decompression loss until no single move helps.
   - A partition labels any image with the class whose loss rises least if the image joins it.
     Codes whose group has no references get class 0.
2. **Encoders.**
   - Train `--folds` + 1 `StageNet`s with one head per partition, by cross-entropy against the
     partitions' labels: fold model f on the training images outside fold f, and a final model on
     all of them.
   - Each minibatch draws one fresh out-of-fold code per image (memoized, `--memo-cap`), so each
     position learns from the mistakes earlier positions really make on unseen images.
   - All copies share the epoch count, chosen by the pooled held-out cross-entropy.
3. **Costs.** For every training image, with its held-out encoder, its term ℓ_i of the marginal
   decompression loss under each partition. It's exact, with all training images as references.
4. **Selectors.** Train `--folds` + 1 `SelectorNet`s to predict those costs from the code alone (MSE),
   with the same fold split, early-stopped on the held-out marginal decompression loss of the fold
   copies' choices (the cost of the partition each picks, pooled over images). Then every copy uses the
   final selector, so the new token means the same thing for every image with the same code, in every
   copy. The encoder uses the partition with the lowest predicted cost.
5. **Report.** The marginal decompression loss on the val split: with the selector's choices, with the
   single best partition, and with no code.

Every step logs when it starts, its progress and its duration.

## Encoder training in pseudo-code

**Notation.**
- Training images $x_1, \dots, x_N$, each in a fold $f(i) \in \{1, \dots, F\}$.
- Model copies $m \in \{1, \dots, F, \star\}$: fold copy $m$ never sees fold $m$; $\star$ is the final copy.
- Position $k$ has encoder copies $e_k^m(x, z) \in \mathbb{R}^{P \times V}$ (logits, one row per partition)
  and selector copies $s_k^m(z) \in \mathbb{R}^P$ (predicted loss per partition). Once position $k$ is
  built, every copy codes with the final selector $s_k^\star$.
- $d(x, y) = \tfrac{1}{D} \lVert x - y \rVert^2$, with $D$ pixels.

**Codes.** A code through the first $t$ positions, using copy $m$:

```math
\begin{array}{l}
\textbf{function } \mathrm{code}^m(x, t): \\
\quad z \leftarrow () \\
\quad \textbf{for } j = 1, \dots, t: \\
\qquad p \leftarrow \arg\min_{p'} \, s_j^\star(z)_{p'} \qquad \text{(one choice per code, shared by every copy)} \\
\qquad z_j \sim \mathrm{softmax}\big(e_j^m(x, z)_p\big) \\
\quad \textbf{return } z
\end{array}
```

A training image's *out-of-fold* code is $\mathrm{code}^{f(i)}(x_i, t)$, memoized per (image, prefix).

**Marginal decompression loss, as computed.** Each image has a code $z_i$ (weight 1/N) and a
distribution $q_i$ over the next token $c$. Then

```math
\ell_i \;=\; \sum_{c} q_i(c) \sum_{j \ne i} \pi(j \mid z_i, c)\, d(x_i, x_j),
\qquad
\pi(j \mid z, c) \;\propto\; \mathbf{1}[z_j = z]\; q_j(c).
```

If no other image carries $(z, c)$, the posterior falls back to the images with code $z$. This is exact
over all images, from per-code sums of weights, pixels and squared norms.

**A partition's implied marginal decompression loss.** For a partition with classes $c_j$, it is
$\sum_j w_j \ell_j$ with $q_j = \mathrm{onehot}(c_j)$: the loss of an encoder that reproduces
(image, code + class) exactly.

**Truncation against an anchor.** For an anchor code $a$ and a code $z$ (both of length $t$), let
$s(z, a)$ be the length of their shared prefix. Then

```math
\hat z_a \;=\; z_{1 : \min(s(z, a) + 1,\; t)} .
```

**All positions.**

```math
\begin{array}{l}
\textbf{for } k = 1, \dots, K: \qquad (t = k - 1) \\[4pt]
\quad \text{// 1. partitions} \\
\quad \text{sample anchors } a_1, \dots, a_P \text{ from the training images}, \quad z_{a_p} \leftarrow \text{out-of-fold code} \\
\quad \textbf{for } p = 1, \dots, P: \qquad (a = z_{a_p}) \\
\qquad \textbf{for } s = 0, \dots, t: \qquad \text{(group } s \text{: shares exactly } s \text{ tokens with } a \text{; } s = t \text{ is the full match)} \\
\qquad\quad m_s(y) = q(a_{1:s} \mid y)\,\big(1 - q(a_{s+1} \mid y, a_{1:s})\big) \;\; (s < t), \qquad m_t(y) = q(a_{1:t} \mid y) \\
\qquad\quad \text{draw } R/(t+1) \text{ references } y_j \propto m_s \text{ with replacement}, \quad
        \text{weight } w_j = \textstyle\sum_y m_s(y) \,/\, (R/(t+1)) \\
\qquad\quad z_{y_j} = a_{1:s} \text{ then } u \propto q(u \mid y_j, a_{1:s}),\; u \ne a_{s+1} \;\; (s < t), \qquad z_{y_j} = a \;\; (s = t) \\
\qquad \text{group every reference by } \hat z_{y_j} \text{ truncated against } z_{a_p} \\
\qquad c_j \sim \mathrm{Uniform}\{1, \dots, V\} \text{ for every reference} \\
\qquad \textbf{repeat}: \qquad \text{(Hartigan, batched)} \\
\qquad\quad \Delta_{j, c} \leftarrow \text{exact change in the implied marginal decompression loss from moving } y_j \text{ to class } c \\
\qquad\quad M \leftarrow \text{references with } \min_c \Delta_{j,c} < 0, \text{ best first} \\
\qquad\quad \textbf{if } M = \emptyset: \textbf{ break} \\
\qquad\quad \text{apply the } |M| \text{ best moves; halve the batch until the total loss drops} \\
\qquad \lambda_p(x, z) \;=\; \arg\min_c \,\big[\text{loss with } (x, \hat z, c) \text{ added}\big]
        \quad (\text{class } 1 \text{ if no reference shares } \hat z) \\[4pt]
\quad \text{// 2. encoders: } F + 1 \text{ copies, lockstep epochs} \\
\quad \textbf{for each epoch}: \\
\qquad \textbf{for } m \in \{1, \dots, F, \star\}: \\
\qquad\quad \textbf{for each minibatch } B \text{ of training images with } f(i) \ne m: \\
\qquad\qquad z_i \leftarrow \text{out-of-fold code (fresh)} \\
\qquad\qquad \mathcal{L} = \frac{1}{|B| P} \sum_{i \in B} \sum_{p} \mathrm{CE}\big(e_k^m(x_i, z_i)_p,\; \lambda_p(x_i, z_i)\big);
          \quad \text{step } e_k^m \\
\qquad \text{held-out CE} = \frac{1}{NP} \sum_i \sum_p \mathrm{CE}\big(e_k^{f(i)}(x_i, \bar z_i)_p,\; \lambda_p(x_i, \bar z_i)\big)
        \quad (\bar z_i \text{ fixed per position}) \\
\quad \text{keep the epoch with the lowest held-out CE (all copies); stop after a patience of epochs} \\[4pt]
\quad \text{// 3. costs} \\
\quad \textbf{for } p = 1, \dots, P: \quad
        q_i^{(p)} = \mathrm{softmax}\big(e_k^{f(i)}(x_i, \bar z_i)_p\big), \qquad
        C_{ip} = \ell_i \text{ with codes } \bar z \text{ and } q^{(p)} \\[4pt]
\quad \text{// 4. selectors: } F + 1 \text{ copies, lockstep epochs} \\
\quad \textbf{for each epoch}: \\
\qquad \textbf{for } m \in \{1, \dots, F, \star\}: \quad
        \text{minimize } \tfrac{1}{P} \lVert s_k^m(\bar z_i) - C_{i,:} \rVert^2 \text{ over images with } f(i) \ne m \\
\qquad \text{held-out loss of choices} = \frac{1}{N} \sum_i C_{i, \hat p_i}, \qquad
        \hat p_i = \arg\min_p s_k^{f(i)}(\bar z_i)_p \\
\quad \text{keep the epoch with the lowest held-out loss of choices (all copies); stop after a patience of epochs} \\
\quad s_k^m \leftarrow s_k^\star \text{ for every fold copy } m \quad \text{(fold selectors only chose the epoch)} \\[4pt]
\quad \text{// 5. the position} \\
\quad \text{position } k \text{ of the encoder} = \big(e_k^\star, s_k^\star\big); \quad
        \text{report } \ell \text{ on the val split with the selector's choices}
\end{array}
```

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
dvae/partition.py   anchor partitions (truncated codes, Hartigan, label rule), exact marginal decompression loss
dvae/models.py      Prior, Decoder; tree enumeration of all code prefixes
dvae/data.py        MNIST tensors, train/val split, labels, binarization
dvae/common.py      device, checkpoints, CLI flags, early-stopping loop
dvae/viz.py         image grids
```
