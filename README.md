# Decoupled VAE: delta trees

A compressor and generative model for MNIST whose code is a path down a binary tree. Each token answers one
yes/no question about the image: which side of a hyperplane it lies on. Every prefix of a code is a node, and a
node decodes to a predicted mean image, so a code can be cut after any token and still decode.

The model is built in four stages, each a script that trains on the stage before's output and redraws a progress
plot every epoch, with a QA script for each trained model:

| Stage | Script | What it learns | QA |
|---|---|---|---|
| 1 | `train_encoder.py` | the tree: where each node splits, and each node's mean image | `qa_encoder.py` |
| 2 | `dump_encoder.py` | (no training) every image's code and the splits along it | |
| 3 | `train_prior.py` | which codes are valid: p(each token's side given the ones before) | `qa_prior.py` |
| 4 | `train_decoder.py` | p(image given its code), 256 grey levels per pixel | `qa_decoder.py` |

## Motivation

Each major choice in the design follows from the one before it.

### A learned prior, after the fact

A VAE fixes the prior, usually a standard Gaussian, and trains the encoder to fit it. VQ-VAE
([van den Oord et al., 2017](https://arxiv.org/abs/1711.00937)) dropped that constraint: the encoder and decoder
train first, the latent space takes whatever shape suits them, and a separate prior learns that shape
afterwards. This model does the same. The encoder is free to build whatever space is easiest to compress into,
and the prior is fit to its codes later.

### A one-dimensional sequence of tokens

TiTok ([Yu et al., 2024](https://arxiv.org/abs/2406.07550)) showed that an image's latent can be a short,
one-dimensional sequence of discrete tokens rather than a two-dimensional grid. Here a code is such a sequence,
of binary tokens.

### One fixed order

The encoder and the prior both decide tokens one at a time, each given the tokens before it, in the same order
every time. If tokens are decided one at a time, a fixed order is the simplest choice:

- **It costs no expressiveness.** By the chain rule, any distribution over codes factors in a fixed order.
- **It narrows what the models must learn.** They only ever see the situations that one order produces.
- **It makes every prefix of a code a code in its own right.** So the same models give a compression pipeline
  alongside the generation one.

![Generation, compression and decompression pipelines](docs/pipelines.png)

- **Generation:** the prior samples every token, and the decoder draws an image from the code.
- **Compression:** the encoder writes the image's tokens, and the first $k$ are kept. With an entropy coder, each
  token costs about its bits under the prior.
- **Decompression:** the prior fills in the dropped tokens given the first $k$, and the decoder draws an image
  consistent with all of them. (The current QA scripts decode a $k$-token prefix as its node's predicted mean
  instead.)

### The encoder's objective: decompression loss

A node holds every image whose code begins with its prefix. Decompressing it means producing an image consistent
with it, and the error is set by how spread out the node's images are. So the loss of a node $\pi$ is the
expected distance between two of its images drawn independently:

```math
D(\pi) = \mathbb{E}_{x, y \sim \pi}\, d(x, y).
```

A decoder that samples from the node errs by exactly $D$ on average, and one that outputs the node's mean errs by
$D/2$. With $d$ the mean squared distance over $n = 784$ pixels, $D$ is twice the node's total variance per pixel:

```math
D(\pi) = \frac{2}{n} \operatorname{tr} \operatorname{Cov}_\pi(x).
```

**At one node this is the k-means objective.** Splitting a node into $V$ children $v$ with shares $p_v$ and
means $\mu_v$, the loss after the token is the share-weighted sum of the children's losses:

```math
\sum_v p_v\, D(\pi_v) = \frac{2}{n}\, \mathbb{E}_{x \sim \pi} \lVert x - \mu_{v(x)} \rVert^2 ,
```

the within-cluster sum of squares. So the best split of a node is a k-means clustering of its images, with
$k = V$. Splitting greedily, one node at a time, keeps the problem tractable.

**With $V = 2$, each token is the side of a hyperplane.** Two-means assigns each point to the nearer of two
centroids. The boundary is the set of points equidistant from both, which is the hyperplane bisecting the
segment between them. So a binary token answers which side of a hyperplane the image lies on. Binary is the
simplest split, and any $V$-way decision can be built from binary ones.

### Predict means, not hyperplanes

The ideal encoder would output the best hyperplane for any node. But the two-means objective is not convex in
the hyperplane. It has many separate good optima, and swapping the two sides gives the same split. A network
trained to regress the best hyperplane would average between optima and land on a split that is good for none of
them.

What is well defined for any hyperplane is the mean of the images on each side. That is a conditional
expectation, smooth in the hyperplane, and regression learns it well. So the encoder predicts the two side means
for a given node and hyperplane, and Lloyd's algorithm does the search: move the hyperplane to the bisector of the
predicted means, and repeat. That is k-means' own algorithm, with the network standing in for the node's images.
So it also works at deep nodes, which hold one training image or none.

### Why a separate prior

The encoder already implies a share for every split: its $m$, with $p = m / (1 + m)$. With unlimited data and
capacity, that share and a separately trained prior would converge to the same answer. In practice both must
compromise, and they compromise differently.

**The encoder's $m$ is trained to place a mean, not to predict a share.** Its loss only asks that the common
side's predicted mean, $-m\,r$, land on the actual common-side mean $\bar c$. The best $m$ is then the projection

```math
m = -\frac{\langle \bar c, r \rangle}{\lVert r \rVert^2},
```

which equals the share ratio $p / (1 - p)$ only when $p\,r + (1 - p)\,\bar c = 0$. That holds when the node's
images really are centred on its predicted mean. Any error left by earlier tokens breaks it. For example, along
$r$'s direction, with the node's predicted mean at 0:

| side | actual share | actual mean |
|---|---|---|
| rare | 0.25 | +2.0 |
| common | 0.75 | -0.2 |

The node's images are centred at $0.25 \cdot 2 - 0.75 \cdot 0.2 = 0.35$, not 0.
- **The encoder's fit:** $r = 2$ and $m = 0.1$, which put both side means exactly right. That implies a share of
  $0.1 / 1.1 \approx 0.09$.
- **The cost to compression:** pricing the token at 0.09 costs 0.968 bits on average. The true share of 0.25,
  which a prior trained on cross-entropy learns, costs 0.811. That wastes 0.16 bits on this token.
- **The cost to sampling:** a sampler using the encoder's share picks the rare side about a third as often as
  real images go there.

Errors build up with depth, so this gap should grow with depth, and it does. At 119 tokens, the encoder's own
shares cost about 1.00 bit per token at depths 65-119, while the prior's cost 0.90.

Two more differences:
- **A cap:** $m \le 1$ caps the encoder's share at one half. It can never say that the rare side holds most of a
  node's images, and the prior can.
- **The right loss:** cross-entropy is the loss whose compromise compression and sampling actually pay for.

(When the encoder merely cannot tell two situations apart, but each has its images centred on its predicted mean,
both models settle on the same pooled share. It takes error in the means to pull them apart.)

So the two predictions serve different purposes:
- **The encoder's is to produce a space with an easy-to-learn prior.**
- **The prior's is to learn which codes in that space are valid,** correcting the encoder's shares where they
  are wrong.

### Why a separate decoder

For a code, the encoder outputs a single image: its node's predicted mean. That is the best single guess under
squared error. But a node still holds many images, so the mean is a blur of them, and a point estimate gives no
likelihood to score.

The decoder models $p(x \mid \text{code})$: a distribution over each pixel's 256 grey levels, given the code and
its mean image.
- **Likelihoods:** the whole model can then report bits per dimension.
- **Sharp samples:** samples, or each pixel's most likely level, have sharp strokes where the mean is blurry.

Keeping it separate leaves the codes alone. The encoder chooses its splits for the decompression loss, and the
decoder is fit afterwards to the frozen codes, without pulling them toward whatever is easiest to model one pixel
at a time. It does treat pixels as independent given the code, which is why its single draws look speckled.

## How it works

### Stage 1: the encoder

**Nodes and remainders.** Pixels are grey levels' bin centers, (level + 0.5) / 256. The root's predicted mean is
the training mean; each token adds a delta, so a node's predicted mean is the training mean plus its deltas so
far. An image's remainder at a node is the image minus that predicted mean.

**One network splits every node.** Given a node's history (its deltas, sides and shares so far) and a hyperplane
$(n, b)$, the network predicts the mean delta $r$ of the hyperplane's rare side and the ratio $m \in [0, 1]$ of
the rare side's share to the common side's. The common side's mean delta is then $-m\,r$, so the two predicted
side means always average back to the node's own (the tower property):

```math
p\,r + (1 - p)(-m\,r) = 0 \quad\text{when}\quad m = \frac{p}{1 - p}.
```

**Lloyd's algorithm through the network.** A node's hyperplane starts from its depth's start (the training
remainders' top principal direction at that depth) and each Lloyd step moves it to the bisector of the two
predicted side means: normal $r / \lvert r\rvert$, bias $\lvert r\rvert (1 - m) / 2$. After a fixed number of
steps (`--lloyd-steps`), the image goes to the side it lies on, and the side's predicted delta extends its
history. Every query depends only on the code, never on the image, so a code alone replays the encoder's choices
exactly and maps back to its node's predicted mean image.

**Seeing the node.** With `--node-mean`, the network also reads each node's predicted mean image, through a
small convolutional net, as one more memory token beside the history of deltas. It adds no information (the mean
follows from the deltas) but shows the network what the node looks like, which is what made the prior work.

**Training.** Batches walk down the tree depth by depth. At every node the network is trained on the actual
side means of the batch's images there (each side weighted by its image count, so every image counts once per
depth), averaged over its Lloyd steps. The network starts with one depth and grows one more every
`--grow-every` epochs, up to `--depths` (0: no cap). Early stopping watches the held-out (val) images' mse after
the full code.

### Stage 2: the dump

`dump_encoder.py` sends every train, val and test image down the frozen encoder once and keeps, at every
depth, its side and the split there ($r$ and $m$, half precision), plus the labels and the training mean. Splits
depend only on the code, so later stages never need the encoder's network: a node's predicted mean is the
training mean plus the deltas $r$ or $-m\,r$ so far.

### Stage 3: the prior

`train_prior.py` trains a causal transformer that reads a code's splits in order, one token per depth, and
predicts each token's side. Token $k$ sees its own split ($r$'s direction and length, and $m$), the side taken
before it and, with `--images`, **the two mean images it chooses between** (the node's mean plus $r$ and minus
$m\,r$), read by a small convolutional net. Those images add no information, but they put it where strokes are
local, as a person comparing the two would see them, and they are what made the prior work (see Results).

Causal attention lets token $k$ see everything before it but not its own side, so one pass over a code trains
every depth. Depths need different amounts of training, so early stopping runs separately in `--bands` bands of
depths: each band keeps the weights of its own best val epoch, and the kept prior answers each depth with its
band's weights.

### Stage 4: the decoder

`train_decoder.py` trains $p(x \mid \text{code})$: 256 logits per pixel from the code's history and its
node's predicted mean image. It adds what the encoder's mean leaves out, each pixel's spread around it.
Test images are then scored as $-\log_2 p(\text{code}) - \log_2 p(x \mid \text{code})$, the one-code upper bound
on $-\log_2 p(x)$, with the code priced by the prior (`--prior`) or else by the encoder's own shares.

## Results so far

Encoder `344-fixed8-nocap` at epoch 96: 97 tokens, 8 Lloyd steps per node, still growing one depth per epoch.

| | train images | test images |
|---|---|---|
| mse per pixel after 97 tokens (1,000 images each) | 0.0091 | 0.0092 |

Reconstruction generalizes: test images, whose codes no training image shares past about 16 tokens, decode as
well as training images.

Bits per image for the same 97-token codes on the test set (lower is better; a fair coin per token is 97):

| codes priced by | bits per image | bits per token at depths 65-97 |
|---|---|---|
| the encoder's own shares | 96.97 | 0.999 |
| the best constant share per depth | 96.93 | 0.999 |
| prior, 256 wide x 4 layers | 95.65 | 0.979 |
| prior with side-mean images (`--images`) | **90.84** | **0.894** |

Samples from the image prior are visibly cleaner than from the encoder's own shares (about 7 garbled digits
in 64, against about 12).

The decoder on the same codes (4.5M parameters, run `371-categorical-fixed8`) brings the whole model
to **1.2355 bits per dimension on the test set** (671 nats per image): 90.84 bits for the code under the image
prior, then 1.1196 bits per pixel given it. Its most likely level per pixel turns the blurry mean images into
crisp strokes.

## Running it

```bash
source .venv/bin/activate   # torch, torchvision, matplotlib; MNIST downloads to data/ on first run
```

The device is picked automatically (CUDA, then Apple's MPS, then the CPU); `--device` overrides it. Each run's
files go to `artifacts/<job>/`, named by `$DVAE_JOB` or else by the script and the time.

```bash
python train_encoder.py --depths 0 --lloyd-steps 8 --lr 1e-4 --max-nodes 1024 --tf32
python dump_encoder.py --ckpt artifacts/<encoder job>/model.pt
python train_prior.py --codes artifacts/<dump job> --images --tf32
python train_decoder.py --codes artifacts/<dump job> --prior artifacts/<prior job>/prior.pt --tf32

python qa_encoder.py --ckpt artifacts/<encoder job>/model.pt
python qa_prior.py --prior artifacts/<prior job>/prior.pt
python qa_decoder.py --decoder artifacts/<decoder job>/decoder.pt --prior artifacts/<prior job>/prior.pt
python benchmark.py --decoder artifacts/<decoder job>/decoder.pt --prior artifacts/<prior job>/prior.pt
```

`train_encoder.py --resume <model.pt>` continues a run from its checkpoint, which holds the whole training
state. The QA scripts write:

- `qa_encoder.py`: `compress_train.png` and `compress_test.png` (images, then their reconstructions after a
  spread of token counts), and `mse.png` (mse by code length, training against test images).
- `qa_prior.py`: `samples_t<T>.png` (8 x 8 samples per temperature), `samples_own.png` (the same random draws
  with the encoder's own shares), `generate.png` (samples token by token) and `real.png`.
- `qa_decoder.py`: from the same sampled codes, `samples_means.png` (the encoder's mean images),
  `samples_pixels.png` (one draw from $p(x \mid \text{code})$), `samples_mode.png` and `real.png`.

`benchmark.py` scores the whole model's test bits per dimension, each image's
$-\log_2 p(\text{code}) - \log_2 p(x \mid \text{code})$, with the code priced by the prior and by the
encoder's own shares side by side (`benchmark.md` and `.json`).

`browse.py` serves a local web page for walking through an encoder's tree one node at a time.

Type checking (pyright's basic mode) must end with 0 errors and 0 warnings:

```bash
npx --yes pyright
```

## On BU's SCC

The repo lives at `/projectnb/cs599bg/kawgit56/Decoupled-VAE`, with the venv beside it. GPUs are held by
*holders* (`scc/hold.sh`), jobs that each keep one GPU and run experiments from a queue folder as they arrive.

- `scc/sync.sh` copies the repo (uncommitted edits and new files included) to SCC.
- An experiment is a file of shell commands in `../queue/pending` (or `../queue/pools/<pool>` for the holders
  started with `POOL=<pool>`), run from the repo with `$PY` the venv's python and `$DVAE_JOB` the file's name.
  Its files, a copy of its commands, its stdout, stderr and exit code go to `artifacts/<name>/`.
- `qsub scc/hold.sh` holds a V100 for 48 hours; `qsub -N hold-l40s -l gpu_type=L40S -v POOL=l40s scc/hold.sh`
  holds an L40S (about twice as fast) for the `l40s` pool.
- `scc/gpus.sh` shows every holder, what it runs or how long it has been idle, and what waits;
  `scc/cancel.sh <name>` stops an experiment; `scc/ssh.sh` opens a shell there, `scc/vscode.sh` VS Code.

## Repository

| | |
|---|---|
| `dvae/deltatree.py` | the encoder's network, Lloyd's algorithm, frontiers and splits, the decoder's network |
| `dvae/prior.py` | the prior's network, banded priors, and code costs under a prior |
| `dvae/codes.py` | loading dumps, deltas and bit costs |
| `dvae/decoder.py` | the decoder's data, code and pixel bits |
| `dvae/sampling.py` | sampling codes down the tree, from the encoder's shares or a prior |
| `dvae/plots.py` | each stage's progress plot |
| `dvae/data.py`, `dvae/artifacts.py`, `dvae/logs.py`, `dvae/viz.py` | data, output folders, logs, image grids |
| `scc/` | holders, the queue and helpers for BU's SCC |

Earlier designs and their READMEs are in the history: `git show c61b6eb:README.md` (a density-based plan that was
never built) and `git show 0579b1b:README.md` (a selector over a bank of learned questions).
