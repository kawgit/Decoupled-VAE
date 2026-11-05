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

### Two share predictions, two purposes

The encoder already predicts each split's shares (its $m$), and the prior predicts them again. They answer to
different objectives:

- **The encoder's purpose is to produce a space with an easy-to-learn prior.** Its $m$ is trained only so that
  its side means come out right.
- **The prior's purpose is to learn which codes in that space are valid.** It is trained on cross-entropy alone,
  so it fixes the encoder's mistakes in exactly the quantity that sampling and compression depend on.

With unlimited data and model capacity both would converge to the same shares. With finite ones they compromise
in different places, and the prior's compromise is the one that matters for generation.

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
