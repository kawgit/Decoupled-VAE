# Decoupled VAE — MNIST MVP (partition-encoder branch)

A discrete VAE with a 1-D autoregressive latent of `k_max` tokens over a vocab of `V`, trained in
three separate stages. The encoder is built first, against an ideal decoder. Then the decoder and
the prior are each trained on the frozen encoder's codes.

Default latent: `V=2`, `k_max=10`, i.e. 1024 codes. `analyze_encoder.py` and `benchmark.py`
enumerate every code, so they need a code space that small.

## The objective

**Notation.**
- An encoder $q$ maps an image $x$ to a code $z = (z_1, \dots, z_K)$ of $K$ (`k_max`) tokens, each in
  $\{1, \dots, V\}$, autoregressively: $q(z_{1:k} \mid x) = \prod_{u=1}^{k} q(z_u \mid x, z_{1:u-1})$, where
  $z_{1:k}$ is the length-$k$ prefix and $z_{1:0}$ the empty one. $zc$ is the code $z$ with token $c$ appended.
- $d(x, y) = \tfrac{1}{n_{\mathrm{px}}} \lVert x - y \rVert^2$ is mean squared pixel distance, with
  $n_{\mathrm{px}}$ pixels per image.
- $p$ is a distribution over code lengths $k \in \{1, \dots, K\}$: how many tokens a decoder gets to see
  (for example uniform). The build below never needs it, since it lowers each position's term in turn.
- $\mathbf{1}[\cdot]$ is 1 when its condition holds and 0 otherwise. $\mathrm{U}\{1, \dots, N\}$ is
  uniform, and $\mathbb{E}_i$ without a distribution means $i$ uniform over the images.
- In every posterior below, an image or pair is left out of its own posterior.

**Decompression loss** of an encoder $q$ on an image dataset $x_1, \dots, x_N$:

```math
L \;=\; \mathbb{E}_{i \sim \mathrm{U}\{1, \dots, N\}}\; \mathbb{E}_{k \sim p}\; \mathbb{E}_{z \sim q(z_{1:k} \mid x_i)}\, F_k(x_i, z),
\qquad
F_k(x_i, z) \;=\; \mathbb{E}_{j \sim \pi(j \mid z_{1:k}),\; j \ne i}\; d(x_i, x_j),
\qquad
\pi(j \mid z_{1:k}) \;\propto\; q(z_{1:k} \mid x_j).
```

$F_k(x_i, z)$ is the expected distance from $x_i$ to another image $x_j$ drawn from the posterior $\pi$ of
the images that could have produced the prefix $z_{1:k}$.

**Marginal decompression loss** of a next-token encoder $q(c \mid x, z)$, a distribution over the next
token $c \in \{1, \dots, V\}$, on a weighted (image, code) dataset $D = \{(x_i, z_i, w_i)\}$: every code
$z_i$ has the same length $t$, and the weights $w_i \ge 0$ sum to 1. It measures what one more token adds
on top of the given codes:

```math
M(D, q) \;=\; \mathbb{E}_{i \sim w}\, \ell_i,
\qquad
\ell_i \;=\; \mathbb{E}_{c \sim q(c \mid x_i, z_i)}\; \mathbb{E}_{j \sim \pi(j \mid z_i, c),\; j \ne i}\; d(x_i, x_j),
\qquad
\pi(j \mid z, c) \;\propto\; w_j\, \mathbf{1}[z_j = z]\, q(c \mid x_j, z_j).
```

$\ell_i$ is pair $i$'s term, and $i \sim w$ draws pair $i$ with probability $w_i$.

The decompression loss is an average of marginal ones over $k \sim p$:

```math
L \;=\; \mathbb{E}_{k \sim p}\, M(D_{k-1}, q_k),
\qquad
D_{k-1} \;=\; \Big\{\big(x_i,\, z,\, \tfrac{1}{N} q(z \mid x_i)\big) : i \in \{1, \dots, N\},\; z \in \{1, \dots, V\}^{k-1}\Big\},
\qquad
q_k(c \mid x, z) \;=\; q(z_k = c \mid x, z_{1:k-1} = z).
```

Here $\pi(j \mid z, c) \propto q(z \mid x_j)\, q(c \mid x_j, z) = q(zc \mid x_j)$, which is the posterior
in $F_k$.

**A partition's implied marginal decompression loss.** A partition $\lambda$ gives each pair $i$ in $D$
one of $V$ classes, $\lambda_i$. Its implied marginal decompression loss is $M(D, q_\lambda)$ with
$q_\lambda(c \mid x_i, z_i) = \mathbf{1}[c = \lambda_i]$: the marginal decompression loss of an encoder
that reproduces each pair's class exactly. Each pair's term is its mean distance to the other pairs with
the same code and class, so the whole loss follows from per-cluster sums.

## How the encoder is built

On this branch the encoder isn't trained by descending L directly. That objective is non-convex in the
encoder's outputs, much like k-means. Instead `train_encoder.py` builds the code one token at a time. At
each position it constructs candidate partitions, trains the encoder to reproduce all of them, and
learns which one to use for each code.

At each position the build works with the marginal decompression loss on the training set. $D$ is every
training image with one sampled out-of-fold code of the current length and weight $1/N$, a one-sample
estimate of $D_t$. Partitions are built and compared by their implied marginal decompression loss.

At position $k$ (prefix length $t = k - 1$):

1. **Partitions** (`dvae/partition.py`):
   - Sample `--anchors` $P$ anchor images, each with one out-of-fold code $a = (a_1, \dots, a_t)$.
   - Give every anchor `--references` $R$ reference draws, split evenly over its $t+1$ groups by shared
     prefix length $s = 0, \dots, t$ ($s = t$ is the full match). A training image $y$'s mass in group
     $s$ is its exact out-of-fold probability of landing there,
     $\mu_s(y) = q(a_{1:s} \mid y)\,\big(1 - q(a_{s+1} \mid y, a_{1:s})\big)$ for $s < t$ and
     $\mu_t(y) = q(a \mid y)$. Draws are with replacement and proportional to that mass. In a mismatch
     group, the token at position $s+1$ is drawn from $q$ among the non-anchor tokens. Each draw weighs
     the group's mass $\sum_y \mu_s(y)$ divided by its draws.
   - So deep groups get as many references as shallow ones, however few images land there. The weights
     are equal within a group, so they don't change the partition; they only put the logged loss on a
     whole-training-set scale.
   - For every anchor, truncate each reference's code to $s+1$ tokens, where $s$ is the length of the
     prefix it shares with the anchor's code (the whole code when it matches). References are
     thereby grouped finely around the anchor's code and more coarsely further from it.
   - Within each group, Hartigan's method (from a random start, with batched moves that are kept only
     if the total drops) assigns references to $V$ classes, strictly lowering the partition's implied
     marginal decompression loss until no single move helps.
   - A partition labels any image with the class whose loss rises least if the image joins it.
     Codes whose group has no references get the first class.
2. **Encoders.**
   - Train `--folds` + 1 `StageNet`s with one head per partition, by cross-entropy against the
     partitions' labels: fold model $f$ on the training images outside fold $f$, and a final model on
     all of them.
   - Each minibatch draws one fresh out-of-fold code per image (memoized, `--memo-cap`), so each
     position learns from the mistakes earlier positions really make on unseen images.
   - All copies share the epoch count, chosen by the pooled held-out cross-entropy.
3. **Costs.** For every training image, with its held-out encoder, its term $\ell_i$ of the marginal
   decompression loss under each partition. It's exact, with all training images as references.
4. **Selectors.** Train `--folds` + 1 `SelectorNet`s to predict those costs from the code alone (MSE),
   with the same fold split, early-stopped on the held-out marginal decompression loss of the fold
   copies' choices (the cost of the partition each picks, pooled over images). Then every copy uses the
   final selector, so the new token means the same thing for every image with the same code, in every
   copy. The encoder uses the partition with the lowest predicted cost.
5. **Report.** The marginal decompression loss on the val split: with the selector's choices, with the
   single best partition, and with no code.

Every step logs when it starts, its progress and its duration.

## Encoder training

**Notation** (on top of the objective's).
- Training images $x_1, \dots, x_N$, each in one of $F$ (`--folds`) folds, $f(i) \in \{1, \dots, F\}$.
- Model copies $m \in \{1, \dots, F, \star\}$: fold copy $m$ never sees fold $m$; $\star$ is the final copy.
- $P$ (`--anchors`) partitions per position, indexed by $h \in \{1, \dots, P\}$, and $R$
  (`--references`) reference draws per partition.
- Position $k$ has encoder copies $e_k^m(x, z) \in \mathbb{R}^{P \times V}$ (logits: row $h$ is partition
  $h$'s head) and selector copies $\sigma_k^m(z) \in \mathbb{R}^P$ (predicted loss per partition). Once
  position $k$ is built, every copy codes with the final selector $\sigma_k^\star$.
- $\mathrm{CE}(\text{logits}, c)$ is cross-entropy against class $c$.

**Truncation against an anchor.** For an anchor code $a$ and a code $z$ (both of length $t$), let
$s(z, a)$ be the length of their shared prefix. Truncation keeps one token past it:

```math
\hat z_a \;=\; \tau_a(z) \;=\; z_{1 : \min(s(z, a) + 1,\; t)} .
```

Two versions follow: a concise one in equations, and the detailed pseudo-code the code follows.

### In equations

At position $k$ (prefix length $t = k-1$), positions $1, \dots, t$ are built. Copy $m$ of the encoder so
far, $q^m$, uses copy $m$'s encoder nets and the shared final selectors:

```math
q^m(z_u \mid x, z_{1:u-1}) \;=\; \mathrm{softmax}\big(e_u^m(x, z_{1:u-1})_{\,\eta_u(z_{1:u-1})}\big),
\qquad
\eta_u(z) \;=\; \arg\min_h \sigma_u^\star(z)_h .
```

A training image's out-of-fold code is $z \sim q^{f(i)}(\cdot \mid x_i)$. Each position fixes one,
$\bar z_i$, per image, giving the dataset $\bar D = \{(x_i, \bar z_i, \tfrac{1}{N})\}$. Write
$\operatorname{fit}_\theta\, A \,\|\, B$ for training $\theta$ on loss $A$ by minibatch steps and keeping the
epoch with the lowest held-out score $B$, with one epoch count shared by all $F+1$ copies.

**1. Partitions.** For each $h$, draw a training image $\alpha_h$ and its anchor code
$a^h \sim q^{f(\alpha_h)}(\cdot \mid x_{\alpha_h})$. Then

```math
\lambda_h \;=\; \operatorname{Hartigan}_\lambda\; M\big(D^h, q_\lambda\big),
\qquad
D^h \;=\; \bigcup_{s=0}^{t} \Big\{\big(y_j,\, \tau_{a^h}(z_j),\, \tfrac{\rho_s}{R/(t+1)}\big)\Big\}_{j=1}^{R/(t+1)},
\qquad
(y_j, z_j) \sim \mathrm{U}(y)\, q^{f(y)}(z \mid y) \;\big|\; s(z, a^h) = s,
```

drawn per stratum $s$ (codes sharing exactly $s$ tokens with $a^h$), where
$\rho_s = \mathbb{E}_y\, q^{f(y)}\big(s(z, a^h) = s \mid y\big)$ is the stratum's probability, so the
weights sum to 1.
$\operatorname{Hartigan}_\lambda$ is a local minimum over class assignments $\lambda$ of $D^h$, found by
Hartigan's method from a random start. A partition labels any $(x, z)$ by the class whose implied loss
rises least when $(x, \tau_{a^h}(z))$ joins $D^h$ as one more draw with that class (the first class if
no draw shares $\tau_{a^h}(z)$):

```math
\lambda_h(x, z) \;=\; \arg\min_c \; M\big(D^h + (x, \tau_{a^h}(z)),\; q_{\lambda_h + c}\big).
```

**2. Encoders**, one head per partition:

```math
e_k^m \;=\; \operatorname{fit}_e\;
  \mathbb{E}_{i : f(i) \ne m}\; \mathbb{E}_{z \sim q^{f(i)}(\cdot \mid x_i)}\; \mathbb{E}_h\, \mathrm{CE}\big(e(x_i, z)_h,\, \lambda_h(x_i, z)\big)
\;\big\|\;
  \mathbb{E}_i\, \mathbb{E}_h\, \mathrm{CE}\big(e_k^{f(i)}(x_i, \bar z_i)_h,\, \lambda_h(x_i, \bar z_i)\big).
```

**3. Costs**: each image's held-out term under each head,

```math
C_{ih} \;=\; \ell_i \text{ of } M\big(\bar D, q^{(h)}\big),
\qquad
q^{(h)}(c \mid x_i, \bar z_i) \;=\; \mathrm{softmax}\big(e_k^{f(i)}(x_i, \bar z_i)_h\big)_c .
```

**4. Selectors**, scored by the held-out loss of their choices, then shared:

```math
\sigma_k^m \;=\; \operatorname{fit}_\sigma\;
  \mathbb{E}_{i : f(i) \ne m}\; \mathbb{E}_h\, \big(\sigma(\bar z_i)_h - C_{ih}\big)^2
\;\big\|\;
  \mathbb{E}_i\, C_{i,\, \arg\min_h \sigma_k^{f(i)}(\bar z_i)_h},
\qquad
\sigma_k^m \leftarrow \sigma_k^\star \;\; \text{for all } m .
```

**5. Position $k$** of the encoder is $q_k(c \mid x, z) = \mathrm{softmax}\big(e_k^\star(x, z)_{\,\eta_k(z)}\big)_c$;
the fold copies $e_k^m$ are kept for out-of-fold codes at later positions.

### In pseudo-code

**Codes.** A code through the first $t$ positions, using copy $m$:

```math
\begin{array}{l}
\textbf{function } \mathrm{code}^m(x, t): \\
\quad z \leftarrow () \\
\quad \textbf{for } u = 1, \dots, t: \\
\qquad h \leftarrow \arg\min_{h'} \, \sigma_u^\star(z)_{h'} \qquad \text{(one choice per code, shared by every copy)} \\
\qquad z_u \sim \mathrm{softmax}\big(e_u^m(x, z)_h\big) \\
\quad \textbf{return } z
\end{array}
```

A training image's *out-of-fold* code is $\mathrm{code}^{f(i)}(x_i, t)$, memoized per (image, prefix).

**Marginal decompression loss, as computed.** Each training image has a code $z_i$ (weight $1/N$) and a
distribution $q_i$ over the next token $c$. Then

```math
\ell_i \;=\; \mathbb{E}_{c \sim q_i}\; \mathbb{E}_{j \sim \pi(j \mid z_i, c),\; j \ne i}\; d(x_i, x_j),
\qquad
\pi(j \mid z, c) \;\propto\; \mathbf{1}[z_j = z]\; q_j(c).
```

If no other image carries $(z, c)$, the posterior falls back to the images with code $z$. This is exact
over all images, from per-code sums of weights, pixels and squared norms.

**A partition's implied marginal decompression loss.** For a partition with classes $c_j$, it is
$\mathbb{E}_{j \sim w}\, \ell_j$ with $q_j = \mathrm{onehot}(c_j)$, the one-hot distribution on $c_j$: the
loss of an encoder that reproduces (image, code + class) exactly.

**All positions.**

```math
\begin{array}{l}
\textbf{for } k = 1, \dots, K: \qquad (t = k - 1) \\[4pt]
\quad \text{// 1. partitions} \\
\quad \text{sample anchor images } x_{\alpha_1}, \dots, x_{\alpha_P}, \quad z_{\alpha_h} \leftarrow \text{out-of-fold code} \\
\quad \textbf{for } h = 1, \dots, P: \qquad (a = z_{\alpha_h}) \\
\qquad \textbf{for } s = 0, \dots, t: \qquad \text{(group } s \text{: shares exactly } s \text{ tokens with } a \text{; } s = t \text{ is the full match)} \\
\qquad\quad \mu_s(y) = q(a_{1:s} \mid y)\,\big(1 - q(a_{s+1} \mid y, a_{1:s})\big) \;\; (s < t), \qquad \mu_t(y) = q(a_{1:t} \mid y)
        \qquad (y \text{ a training image, out-of-fold } q) \\
\qquad\quad \text{draw } R/(t+1) \text{ references } y_j \propto \mu_s \text{ with replacement}, \quad
        \text{weight } w_j = \textstyle\sum_y \mu_s(y) \,/\, (R/(t+1)) \\
\qquad\quad z_{y_j} = a_{1:s} \text{ then } v \propto q(v \mid y_j, a_{1:s}),\; v \ne a_{s+1} \;\; (s < t), \qquad z_{y_j} = a \;\; (s = t) \\
\qquad \text{group every reference by } \hat z_{y_j} \text{, its code truncated against } a \\
\qquad c_j \sim \mathrm{U}\{1, \dots, V\} \text{ for every reference (its class)} \\
\qquad \textbf{repeat}: \qquad \text{(Hartigan, batched)} \\
\qquad\quad \Delta_{j, c} \leftarrow \text{exact change in the implied marginal decompression loss from moving } y_j \text{ to class } c \\
\qquad\quad J \leftarrow \text{references with } \min_c \Delta_{j,c} < 0, \text{ best first} \\
\qquad\quad \textbf{if } J = \emptyset: \textbf{ break} \\
\qquad\quad \text{apply the } |J| \text{ best moves; halve the batch until the total loss drops} \\
\qquad \lambda_h(x, z) \;=\; \arg\min_c \,\big[\text{loss with } (x, \hat z_a, c) \text{ added}\big]
        \quad (\text{class } 1 \text{ if no reference shares } \hat z_a) \\[4pt]
\quad \text{// 2. encoders: } F + 1 \text{ copies, lockstep epochs} \\
\quad \textbf{for each epoch}: \\
\qquad \textbf{for } m \in \{1, \dots, F, \star\}: \\
\qquad\quad \textbf{for each minibatch } B \text{ of training images with } f(i) \ne m: \\
\qquad\qquad z_i \leftarrow \text{out-of-fold code (fresh)} \\
\qquad\qquad \mathcal{L} = \mathbb{E}_{i \sim B}\, \mathbb{E}_{h}\, \mathrm{CE}\big(e_k^m(x_i, z_i)_h,\; \lambda_h(x_i, z_i)\big);
          \quad \text{step } e_k^m \\
\qquad \text{held-out CE} = \mathbb{E}_{i}\, \mathbb{E}_{h}\, \mathrm{CE}\big(e_k^{f(i)}(x_i, \bar z_i)_h,\; \lambda_h(x_i, \bar z_i)\big)
        \quad (\bar z_i \text{: one out-of-fold code per image, drawn once per position}) \\
\quad \text{keep the epoch with the lowest held-out CE (all copies); stop after a patience of epochs} \\[4pt]
\quad \text{// 3. costs} \\
\quad \textbf{for } h = 1, \dots, P: \quad
        q_i^{(h)} = \mathrm{softmax}\big(e_k^{f(i)}(x_i, \bar z_i)_h\big), \qquad
        C_{ih} = \ell_i \text{ with codes } \bar z \text{ and } q^{(h)} \\[4pt]
\quad \text{// 4. selectors: } F + 1 \text{ copies, lockstep epochs} \\
\quad \textbf{for each epoch}: \\
\qquad \textbf{for } m \in \{1, \dots, F, \star\}: \quad
        \text{minimize } \mathbb{E}_{i : f(i) \ne m}\, \mathbb{E}_{h}\, \big(\sigma_k^m(\bar z_i)_h - C_{ih}\big)^2 \\
\qquad \text{held-out loss of choices} = \mathbb{E}_{i}\, C_{i, \hat h_i}, \qquad
        \hat h_i = \arg\min_h \sigma_k^{f(i)}(\bar z_i)_h \\
\quad \text{keep the epoch with the lowest held-out loss of choices (all copies); stop after a patience of epochs} \\
\quad \sigma_k^m \leftarrow \sigma_k^\star \text{ for every fold copy } m \quad \text{(fold selectors only chose the epoch)} \\[4pt]
\quad \text{// 5. the position} \\
\quad \text{position } k \text{ of the encoder} = \big(e_k^\star, \sigma_k^\star\big); \quad
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
python qa_compress_analytically.py -n 16 -m 8            # -> outputs/compress_analytically/k01.png ...
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
  `qa_compress_analytically.py` needs only the encoder: it decompresses by drawing `m` training images
  from the ideal decoder's posterior $\pi(y \mid z_{1:k}) \propto q(z_{1:k} \mid y)$ (`--posterior-mode`
  picks the $q$ used), and reports how many training images the posterior spreads over.
- **Benchmark:** `benchmark.py` scores each decoder on the data it was trained for: 8-bit MNIST, or
  the test set binarized with a fixed seed. It reports exact NLL, plus −ELBO and rate when you pass
  `--encoder`. When the code space is too large to enumerate (more than 2¹⁶ codes), it uses the encoder
  (default `checkpoints/encoder.pt`): the top `--beam` codes per image under q are summed exactly, and
  `--samples` more are drawn from q outside them. That gives an unbiased estimate of p(x), reported as
  `nll_is` (an upper bound on the NLL on average), plus a guaranteed upper bound from the top codes
  alone (`nll_top_bound`) and the −ELBO, distortion and rate. The decoder runs once per distinct code.
  Use `--limit` for a quicker partial run.
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
