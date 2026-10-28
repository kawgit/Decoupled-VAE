# Decoupled VAE: slice trees

A compressor and generative model for MNIST whose code is a path down a binary tree: each token is the
answer to a yes/no question about the image, which side of a hyperplane (a *slice*) it lies on. The encoder
is built against an ideal decoder, and one network per round ends up playing the encoder, prior and decoder.

This README is the argument for the design, one step at a time, with the reason each step is needed. It
describes where the project is going. None of it is **implemented yet**: see [Current code](#current-code).

**Contents**
1. [The problem: decompression loss](#1-the-problem-decompression-loss)
2. [The greedy split is V-means, and for V = 2 a slice](#2-the-greedy-split-is-v-means-and-for-v--2-a-slice)
3. [Deep nodes run out of data](#3-deep-nodes-run-out-of-data)
4. [Working with a density instead of the samples](#4-working-with-a-density-instead-of-the-samples)
5. [Slicing a density analytically](#5-slicing-a-density-analytically)
6. [The half-space model](#6-the-half-space-model)
7. [Training it on real images and the density](#7-training-it-on-real-images-and-the-density)
8. [The loop](#8-the-loop)
9. [Measuring progress](#9-measuring-progress)
10. [Cost](#10-cost)
11. [Pseudocode](#11-pseudocode)
12. [Open questions](#12-open-questions)
13. [Related work](#13-related-work)
14. [Current code](#current-code)

## 1. The problem: decompression loss

**Codes are nodes of a tree.** An encoder $q$ maps an image $x$ to a code: a tuple of tokens, each in
$\{1, \dots, V\}$. Every prefix of a code is a node of a $V$-ary tree, the empty code is the root, and a
node's depth is its number of tokens. Compressing an image means keeping the first $k$ tokens of its code:
the node at depth $k$ on its path.

**A node's posterior** is the distribution of the images that reach it:

```math
\pi_z(x) \;\propto\; p_{\text{data}}(x)\, q(z \mid x),
\qquad
P(z) \;=\; \mathbb{E}_{x \sim p_{\text{data}}}\, q(z \mid x).
```

**Decompression loss of a posterior** is the mean distance between two independent images drawn from it.
With mean squared pixel distance, over $n_{\text{px}} = 784$ pixels, that is twice its total variance:

```math
D(\pi) \;=\; \mathbb{E}_{x \sim \pi,\; y \sim \pi}\; d(x, y) \;=\; \frac{2}{n_{\text{px}}} \operatorname{tr} \operatorname{Cov}_\pi(x),
\qquad
d(x, y) \;=\; \frac{1}{n_{\text{px}}} \lVert x - y \rVert^2 .
```

**Why this is the quantity to minimize.** Decompressing a node means producing an image consistent with it.
- An ideal decoder that *samples* draws an image from the posterior. Its distance from the original is
  the distance between two independent draws, so its expected error is exactly $D$.
- An ideal decoder that outputs *one fixed image* per node does best with the posterior mean, and its
  expected error is $D/2$. The mean averages every image the node could hold, so it is blurry, and its
  blur is the posterior's total variance.

Either way, the error is set by the spread of the images the decoder has to cover. A wide posterior is
what makes reconstructions blurry, or wrong. The encoder's job is to make posteriors narrow.

**Decompression loss of an encoder.** A decoder may get any number of tokens. Each further token is read
with probability $\rho$, so depth $k$ is read with probability $(1 - \rho)\rho^{k-1}$:

```math
L \;=\; \sum_{k \ge 1} (1 - \rho)\, \rho^{k-1}\, L_k,
\qquad
L_k \;=\; \sum_{|z| = k} P(z)\, D(\pi_z).
```

**Gain of a split.** Splitting node $z$ into children $zc$ lowers the loss there by

```math
G(z) \;=\; D(\pi_z) \;-\; \sum_c P(c \mid z)\, D(\pi_{zc}),
\qquad
P(c \mid z) \;=\; \frac{P(zc)}{P(z)}.
```

The loss telescopes into gains, each weighted by the chance that a decoder reads that token:

```math
L \;=\; D(p_{\text{data}}) \;-\; \sum_{k \ge 1} \rho^{k-1}\, \bar G_k,
\qquad
\bar G_k \;=\; \sum_{|z| = k-1} P(z)\, G(z).
```

## 2. The greedy split is V-means, and for V = 2 a slice

**Greedy strategy.** At each node, choose the split with the highest gain $G(z)$, without looking ahead to
how it changes the gains available below. Shallow gains are read more often (the $\rho^{k-1}$ weights), so
they matter most, and this is how tree-structured quantizers are usually grown. Whether lookahead would
pay is an open question (§12).

**The best split is V-means.** By the law of total variance, the spread of a node is the children's
expected spread plus the spread of their means, so the gain is the between-children variance:

```math
G(z) \;=\; \frac{2}{n_{\text{px}}} \sum_c P(c \mid z)\, \big\lVert \mu_{zc} - \mu_z \big\rVert^2,
\qquad
\mu_z \;=\; \mathbb{E}_{x \sim \pi_z}\, x .
```

Maximizing it is minimizing the children's within-class squared error, which is the $k$-means objective
with $V$ clusters on the node's posterior. Two consequences:
- **The best split is hard and Voronoi.** Given the children's means, sending every image to its nearest
  mean can only lower the within-class error, and recomputing the means lowers it again. So an optimal
  split assigns each image to its nearest of $V$ centroids, and is fully described by those centroids.
- **With samples, it can be found directly.** Replace $\pi_z$ by the training images that reach the node and
  run Lloyd's or Hartigan's algorithm on them. As the number of images grows, the empirical centroids
  converge to the population ones (Pollard 1981), so given enough images per node, relative to $V$ and
  the dimension, the right strategy can be computed rather than learned.

**For V = 2, a split is a hyperplane: a *slice*.** Two centroids $\mu_\pm$ split the image space along their
perpendicular bisector. The token is which side the image is on, $a \in \{+1, -1\}$:

```math
a(x) \;=\; \operatorname{sign}\big(\hat n^\top x - b\big),
\qquad
\hat n \;=\; \frac{\mu_+ - \mu_-}{\lVert \mu_+ - \mu_- \rVert},
\qquad
b \;=\; \hat n^\top \frac{\mu_+ + \mu_-}{2},
```

and the gain is

```math
G \;=\; \frac{2}{n_{\text{px}}}\; P_+\, P_-\, \big\lVert \mu_+ - \mu_- \big\rVert^2 .
```

$\hat n$ is a unit vector over the 784 pixels, a template that can be viewed as an image, and $b$ is a
threshold: the slice asks whether the image matches the template more than $b$. Lloyd's form makes the
template the difference between the two sides' mean images.

We use $V = 2$ from here on. The reasons:
- a slice is the simplest question there is, and every cell of a $V$-way Voronoi split is an intersection
  of half-spaces;
- its gain has a closed form under Gaussian densities (§5), which makes it cheap to optimize;
- a node at depth $k$ is a polytope: the intersection of the $k$ half-spaces its tokens chose.

The cost is that a $V$-way split made from several slices in sequence is chosen greedily, slice by slice,
and can be slightly worse than the best joint $V$-means split.

From here on, a **code** is a tuple of tokens, each $+1$ or $-1$. The slices along it are computed from its
tokens (§6), so they are never part of the code.

## 3. Deep nodes run out of data

With balanced slices, a node at depth $k$ holds about $N 2^{-k}$ of the $N = 55{,}000$ training images.
Three problems follow, at increasing depth.

**1. The data stops pinning down the slice (from depth 6 or 7).** Any $n \le n_{\text{px}} + 1$ points in
general position can be split by a hyperplane in every one of the $2^n$ possible ways. MNIST nodes fall below
785 images by depth 6 or 7 ($55{,}000 / 2^6 \approx 860$, $55{,}000 / 2^7 \approx 430$). Past that, many
slices split a node's images identically but treat unseen images differently. Choosing the bisector of the
empirical half-means is then an assumption about where unseen images lie, not something the data shows.

**2. Free cuts.** The empirical spread of a node is the spread of its few training images. A slice threaded
between two of them removes all of it, so on the training set it looks perfect, while on new images it
can gain almost nothing. With point masses the encoder learns to isolate training images.

**3. Nothing to work with (from depth 16).** By $2^{16} > 55{,}000$, the typical node holds one training image
or none.

**Measuring the loss hits the same wall.** Estimating $D(\pi_z)$ needs two independent held-out images in
node $z$, and the images that trained the encoder don't count, since the encoder was shaped around them.
- Averaging over every held-out pair that lands in the same node gives an unbiased estimate of
  $\sum_z P(z)^2 D(\pi_z)$: the right ingredients with an extra factor $P(z)$, which all but erases rare nodes.
- Correcting it needs $m$ more held-out images. Weighting each colliding pair by $(m+1)/(h+1)$, where $h$
  of the extras also land in its node, estimates

```math
\sum_z P(z)\, D(\pi_z)\, \big(1 - (1 - P(z))^{m+1}\big)
```

  without bias. Nodes with mass well above $1/m$ are measured; nodes below it still vanish.
- No finite sample does better. A quantity can be estimated without bias from $N$ samples only if it is a
  polynomial of degree at most $N$ in the distribution (Halmos 1946), and dividing by $P(z)$ is not.

**Conclusion.** Below a certain depth, the data alone can neither choose slices nor score them. Something
has to say where the unseen images lie. We make that assumption explicit by working with a **probability
density** over image space instead of the training images. A density can be sampled, and its mass measured,
inside any node however deep. Its assumptions can be tested at the depths where the data still speaks.

## 4. Working with a density instead of the samples

The density must allow three things:
1. **evaluating held-out likelihood**, so that densities can be compared (§9);
2. **closed-form mass, mean and second moment inside a half-space**, so that slices can be scored (§5);
3. **sampling anywhere**, so that no node runs out of images.

Mixtures of Gaussians do all three.

**Round 0: the training images, smoothed.** Put a Gaussian on every training image, with a covariance
predicted from the image (non-local manifold Parzen windows, Bengio, Larochelle & Vincent 2006):

```math
\hat p_0(y) \;=\; \frac{1}{N} \sum_{i=1}^{N} \mathcal{N}\big(y;\; x_i,\; \Sigma(x_i)\big),
\qquad
\Sigma(x) \;=\; \operatorname{diag}\big(d(x)\big) + U(x)\, U(x)^\top .
```

- **Input:** one training image, as 784 pixels.
- **Output:** $d(x)$, 784 positive per-pixel variances, and $U(x)$, a $784 \times r$ matrix ($r$ about 8 to 16)
  whose columns are directions the image can move in cheaply. A small CNN with two heads.
- **Training loss:** leave-one-out log-likelihood. Each training image is scored by every other image's
  Gaussian; leaving out its own is essential, or every Gaussian collapses onto its own image:

```math
\mathcal{L}_{\text{train}} \;=\; -\frac{1}{N} \sum_{j=1}^{N} \log \frac{1}{N - 1} \sum_{i \ne j} \mathcal{N}\big(x_j;\; x_i,\; \Sigma(x_i)\big).
```

  The inner sum runs over the 64 or so nearest training images, which hold almost all of it.
- **Validation loss:** the same on held-out images, with no exclusion, in bits per pixel. It must beat
  plain Parzen windows (one shared bandwidth, tuned on validation) and manifold Parzen windows (each $U$ the
  local principal directions of the image's nearest neighbours, Vincent & Bengio 2003).
- **Dequantize** (add uniform noise of one grey level to the pixels) and floor $d$. Pixels are quantized and
  border pixels are always 0, so without this the variances along them shrink to 0 and the likelihood
  diverges.

**Why a low-rank covariance.** A training image's neighbours mostly differ from it by small shifts,
rotations and changes in stroke width, so $U$ learns those directions. The density then spreads mass
along them. A slice that cuts across them slices through mass and gains little, so the encoder stops
spending tokens on small shifts: a soft form of the invariance pixel distance lacks.

This density fixes free cuts (§3, problem 2): every slice now cuts through mass, so a node with two images
still has real spread, and its best split follows the directions its covariance says the data varies in.

## 5. Slicing a density analytically

**Cutting one Gaussian.** Take a component $\mathcal{N}(m, \Sigma)$ and side $a$ of slice $(\hat n, b)$. With

```math
s \;=\; \sqrt{\hat n^\top \Sigma\, \hat n},
\qquad
\alpha \;=\; \frac{a\,(\hat n^\top m - b)}{s},
\qquad
\lambda \;=\; \frac{\varphi(\alpha)}{\Phi(\alpha)},
```

the kept mass, mean and covariance are

```math
\text{mass} \;=\; \Phi(\alpha),
\qquad
m' \;=\; m + \frac{a \lambda}{s}\, \Sigma \hat n,
\qquad
\Sigma' \;=\; \Sigma - \frac{\lambda (\lambda + \alpha)}{s^2}\, \Sigma \hat n\, \hat n^\top \Sigma .
```

**A node is a mixture of cut components.** Every component carries its mass inside the node, its mean and
its covariance. A Gaussian cut by a second slice at an angle to the first has no closed-form mass, so after
each token the truncated component is replaced by the Gaussian with the same mean and covariance
(*assumed density filtering*): the formulas above, applied once per token, each adding a rank-1 correction
to the covariance. Components whose mass falls below a cutoff are dropped.

The approximation leaks mass back past earlier cuts: a standard normal cut at 0 has mean 0.80 and variance
0.36, and the Gaussian with those moments puts about 9% of its mass below 0. A later slice is only affected
if it is correlated with an earlier one: when $\hat n_2^\top \Sigma \hat n_1 = 0$, cutting along $\hat n_1$
leaves the projection onto $\hat n_2$ exactly Gaussian.

**The gain of a slice has a closed form.** Write $W$ and $M$ for the node's total mass and first moment, and
$W_+$ and $M_+$ for the positive side's:

```math
W_+ \;=\; \sum_i w_i\, \Phi(\alpha_i),
\qquad
M_+ \;=\; \sum_i w_i \Big( \Phi(\alpha_i)\, m_i + \varphi(\alpha_i)\, \frac{\Sigma_i \hat n}{s_i} \Big),
\qquad
G \;=\; \frac{2}{n_{\text{px}}}\; \frac{\big\lVert W M_+ - W_+ M \big\rVert^2}{W^2\, W_+\, W_-} .
```

It is smooth in $(\hat n, b)$: smoothing the data, not the answers, gives the slice a gradient.

**Finding the best slice: Lloyd's algorithm on the density.** Start from any slice and repeat:
1. compute the two sides' means,

```math
\mu_+ \;=\; \frac{M_+}{W_+},
\qquad
\mu_- \;=\; \frac{M - M_+}{W - W_+};
```

2. move the slice to their perpendicular bisector (§2).

Each step can only lower the within-side error (as in $k$-means), so it converges to a local optimum, with
no learning rate.

**How many local optima a node has** depends on how lumpy its density is. Each optimum counts once up to
sign, since $(\hat n, b)$ and $(-\hat n, -b)$ give the same split.
- *A single Gaussian has exactly one.* Every stationary split passes through the mean, perpendicular to an
  eigenvector of $\Sigma$, and only the top eigenvector is a maximum. Lloyd's step there is power iteration,
  $\hat n \leftarrow \Sigma \hat n / \lVert \Sigma \hat n \rVert$, and the gain is
  $(2/n_{\text{px}})(2/\pi)\lambda_1$.
- *A node holding $M$ well-separated clusters* can have up to about $2^{M-1}$: roughly every grouping of the
  clusters into two that a hyperplane separates and whose bisector still separates them.
- *A finite sample of $n$ images* admits up to

```math
2 \sum_{i=0}^{n_{\text{px}}} \binom{n-1}{i}
```

  hyperplane splits (Cover 1965), and sampling noise makes many of them local optima. Smoothing merges most
  of them.

On MNIST we expect many optima at shallow nodes (digit classes and styles) and about one at deep nodes,
which look increasingly like a single Gaussian. So the number of Lloyd starts scales with depth: many at
the few shallow nodes, a handful at the many deep ones. Starts are seeded from the node's code, so the
sender and the receiver find the same slice:
- the node's top principal directions (power iteration on its covariance);
- differences of component means, sampled the way $k$-means++ seeds;
- random combinations of the top few principal directions.

**Limit.** Moment matching treats a Gaussian cut by $k$ half-spaces as Gaussian, and its error grows with
$k$ (Cunningham, Hennig & Lacoste-Julien 2011 study exactly this case). How much it matters at depth 32 is
an open question (§12).

## 6. The half-space model

Slicing the density exactly costs about $10^9$ operations per node (§10), and the density cannot travel
with the code. So each round trains one network that answers, for any node and any query slice, what the
density and the real images say about the slice's side.

**Inputs:**
- the *history*: the node's slices and tokens, as a set of $(\hat n, b, a)$, each a linear embedding of its
  786 numbers. The set has no order, since order carries no information;
- a *query slice* $(\hat n, b)$.

**Outputs**, for the query's positive side within the node:
- $p_+$, its share of the node's mass;
- $\mu_+$, its mean;
- $\Sigma_+$, its covariance, as diagonal plus low rank like §4's.

The other side is the flipped query $(-\hat n, -b)$. Its share should be $1 - p_+$, and the two sides' moments
should mix back to the node's; both serve as diagnostics.

**Everything else is computed from it:**
- **A node's Gaussian.** Node $z$ with last token $a$ is side $a$ of the slice asked at its parent: the
  parent's history queried with $(a\hat n, a b)$. The root's is the training images' mean and covariance.
- **The gain of any slice.** The node's mean $\mu$ is the two sides' mass-weighted average, so the gain of
  §2 can be written with one side alone:

```math
G \;=\; \frac{2}{n_{\text{px}}}\; \frac{p_+}{1 - p_+}\; \big\lVert \mu_+ - \mu \big\rVert^2 .
```

- **A Lloyd step.** The bisector of $\mu_+$ and $\mu_-$, each one query.
- **The encoder.** At each node, run Lloyd on the model from the seeded starts and keep the result with the
  highest predicted gain. The image's side of that slice is the token. Every query is computed from the
  history directly, so there is no running state to drift down the path.
- **The prior.** The probability of token $a$ is the model's share of side $a$.
- **The decoder.** The leaf's Gaussian: its mean as a point reconstruction, or a sample from it.
- **The density for the next round** (§8): the model's node Gaussians, weighted by its shares.

So there is no separate encoder, prior or decoder: one model per round plays all three.

**Why the target is a Lloyd step and not the best slice.** A network mapping a node straight to its best
slice has no fixed target where the node has several local optima, and one mapping a start to the optimum it
converges to has a target that jumps at the boundaries between basins. A blend of two good slices is
usually a bad one. A side's moments are a smooth function of the query slice, so Lloyd runs on the model
the way it would run on the density.

## 7. Training it on real images and the density

Every output can be learned from two sources:
- **real images:** a training image $x$ that reaches node $z$ is on one side of any query slice, so it is a
  sample of that side's share and of that side's distribution;
- **the round's density:** its exact share, mean and covariance of the side, from the cut formulas of §5.

Real images are what bring information about $p_{\text{data}}$ into each round; without them, each round
would only learn the previous round's model and the loop would stall or drift. The density is what fills
deep nodes, where real images run out.

**Combining them: the density is worth $\alpha_z$ images in node $z$.** Treat the density as a prior and the
real images as data. The combined target is then a single distribution: the $n_z$ real images in the node,
mixed with $\alpha_z$ pseudo-images drawn from the density inside it. On the query's positive side, with
the density's share $p_d$, mean $\mu_d$ and covariance $\Sigma_d$, and the $n_+$ real images there:

```math
\bar p_+ \;=\; \frac{\alpha_z\, p_d + n_+}{\alpha_z + n_z},
\qquad
\bar\mu_+ \;=\; \frac{\alpha_z\, p_d\, \mu_d + \sum_{x \in +} x}{\alpha_z\, p_d + n_+},
```

```math
\bar\Sigma_+ \;=\; \frac{\alpha_z\, p_d \big(\Sigma_d + \mu_d \mu_d^\top\big) + \sum_{x \in +} x x^\top}{\alpha_z\, p_d + n_+} \;-\; \bar\mu_+ \bar\mu_+^\top .
```

The share rule is a Pólya tree's posterior with the density as its base distribution (Lavine 1992). One
pseudo-count for every output keeps the targets consistent: they are all moments of the same
distribution, so the two sides' shares sum to 1 and their moments mix back to the node's. $\bar\Sigma_+$ is
never formed as a $784 \times 784$ matrix: it is a sum of low-rank pieces.

**How much the density is worth.** The density was fit to all the training images, so what it knows about
node $z$ beyond the $n_z$ images inside comes from the $N - n_z$ images outside, carried over by its
inductive bias. An outside image says less about the node than an inside one, so

```math
\alpha_z \;=\; \gamma\, (N - n_z),
\qquad
\text{real share at depth } k \;\approx\; \frac{1}{1 + \gamma\, 2^{k}} .
```

- $\gamma$ is how much an outside image is worth relative to an inside one. With $\gamma = 1$, the real
  share at a node is just its fraction of the training set.
- The hand-over from real images to the density happens near depth $\log_2(1/\gamma)$: with
  $\gamma = 10^{-3}$, depth 10 is half and half.
- The root uses only real images, and so does any node holding the whole training set.
- $\gamma$ is chosen by empirical Bayes before each round's training: the value under which held-out images'
  tokens are most likely under $\bar p_+$. That is a one-dimensional search that involves no network.
  It should rise from round to round as the density improves.

**The loss.** Cross-entropy of the model's outputs against the combined targets, for both sides of each
query:

```math
\ell(z, s) \;=\; \mathrm{BCE}\big(p_+, \bar p_+\big)
\;+\; \bar p_+\; \mathbb{E}_{x \sim \bar\pi_+}\big[-\log \mathcal{N}(x;\ \mu_+,\ \Sigma_+)\big]
\;+\; (\text{the same for the negative side}),
```

```math
\mathbb{E}_{x \sim \bar\pi_+}\big[-\log \mathcal{N}(x;\ \mu_+,\ \Sigma_+)\big]
\;=\; \tfrac12 \Big[\log\lvert\Sigma_+\rvert + \operatorname{tr}\big(\Sigma_+^{-1}\bar\Sigma_+\big) + (\bar\mu_+ - \mu_+)^\top \Sigma_+^{-1} (\bar\mu_+ - \mu_+)\Big] + \text{const},
```

computed with the Woodbury identity. The best Gaussian under it is exactly the moment-matched one. Each node
is weighted by its importance in the decompression loss, $\rho^{|z|} \bar P(z)$, with $\bar P(z)$ the product
of the combined shares along its path. The pseudo-count only sets the mix inside a node, not the node's
weight; otherwise the million deep nodes would swamp the shallow ones.

**Which query slices.** The ones Lloyd visits when it runs on the model: its starts and iterates at the
nodes the encoder reaches, plus perturbed copies for coverage. These depend on the model being trained, so
training alternates, as in DAgger (Ross et al. 2011): run the encoder on the training images, record its
nodes and queries, label them with combined targets, train, and repeat. Re-encoding also updates which
real images reach each node.

## 8. The loop

Round $t$'s model is trained against round $t$'s density, and defines round $t+1$'s:

```math
\hat p_t \;\xrightarrow{\ \S 7,\ \text{with real images}\ }\; \text{model}_t \;\longrightarrow\; \hat p_{t+1} .
```

**The next density is the model's tree.** A node of round $t$'s tree is one Gaussian component: the model's
Gaussian for that node, weighted by the product of its shares along the path. Round $t+1$ needs far more
than $N$ components inside its deep nodes, and there are $2^{32}$ leaves, so the tree is expanded only where
it is needed:
- round $t+1$ starts from round $t$'s root, a single coarse component;
- whenever a node of round $t+1$ holds components too coarse for it (their own variance is a large share
  of the node's spread), each is replaced by its two children in round $t$'s tree. The children's slice
  comes from Lloyd on $\text{model}_t$, their weights from its shares, their Gaussians from its outputs.
  Each child is then cut again by every slice along round $t+1$'s code.

Only the paths round $t+1$ visits get expanded, so the cost follows the visited nodes, not $2^{32}$.
Expanding asks $\text{model}_t$, never round $t$'s density, so nothing reaches back further than one
round. Round 0's density is the smoothed training images, which have nothing to expand: their components
are the finest structure they have.

**This is the structure of wake-sleep** (Hinton, Dayan, Frey & Neal 1995): the model learns from real data
and from the previous round's model. What to expect, and what to watch:
- **Rounds change mostly deep nodes.** Shallow nodes are set by real images in every round.
- **Real data enters every round**, through every output, so the loop is not training on its own samples
  (which would compound errors, "model collapse", Shumailov et al. 2024).
- **There is no shared objective.** The encoder maximizes gain, the model fits the combined targets, and
  neither step is guaranteed to improve the other. We expect a large improvement from round 0 to round 1,
  then small ones, settling at a fixed point.
- **Leaf shapes are inherited.** Real images set the shares everywhere and the shapes where they are
  plentiful, but deep inside leaves the shapes come from earlier rounds, back to §4. With narrow leaves,
  shares carry most of the information.
- **Lock-in.** If a round's model underestimates how much images vary along some direction inside deep
  nodes, the encoder never slices along it, and nothing later corrects it. Random starts counter this.
- **Spikes.** A Gaussian likelihood on single real images can pull deep covariances onto those images, and
  the next encoder gets free cuts again. The density term, the variance floor and the held-out judge are the
  defences.

A round is kept only if it improves on the last (§9), and the loop stops when one doesn't.

## 9. Measuring progress

The goal is decompression loss under the true distribution. §3 shows it cannot be measured deep in the
tree, so we measure what can be.

**Held-out bits per pixel of each round's density.** It checks how well the density fills the space
between images, which is exactly what deep slicing relies on.
- Round 0: the mixture's exact log-likelihood on held-out images.
- Later rounds: the code is deterministic, so the model's shares and leaf Gaussian give a bound:

```math
-\log \hat p(x) \;\le\; -\sum_{k} \log p_{a_k} \;-\; \log \mathcal{N}\big(x;\; \mu(z),\; \Sigma(z)\big),
```

  with $z$ the code of $x$, $a_k$ its tokens and $p_{a_k}$ their shares. It is an upper bound on bits, so
  comparing it with round 0's exact value favours round 0.

**Held-out decompression loss at shallow depths** ($k \le 8$, say), with the reweighted pair estimate of §3
on images that trained nothing. It checks that a better likelihood gives better codes where that can still
be measured, and it reports, per depth, the share of held-out images with no other held-out image in their
node: how much of the loss at that depth is invisible.

**Diagnostics, per round and per depth:**
- the fitted $\gamma$;
- the model's error against the exact combined targets along its own Lloyd paths;
- the gain lost by Lloyd on the model compared with Lloyd on the density, at a sample of held-out nodes;
- the consistency checks of §6;
- the number of distinct local optima at a sample of nodes, and how often the top principal direction
  alone finds the best.

Three splits: training images (fit everything), held-out images (judge each round and fit $\gamma$) and a
final test split, used once.

## 10. Cost

Symbols: $N = 55{,}000$, $K = 32$, $n_{\text{px}} = 784$, covariance rank $r \approx 16$, $C \approx 100$
components per node, $S$ starts of $I$ Lloyd iterations.

**Nodes per round.** Every training image passes through one node per depth, shared above depth
$\log_2 N \approx 16$ and alone below it:

```math
\#\text{nodes} \;\approx\; \sum_{k \le K} \min(2^k, N) \;\approx\; N (K - \log_2 N + 1) \;\approx\; 10^6 .
```

**Per node, per training pass:**
- the density inside it: cutting the parent's $C$ components and expanding coarse ones through the previous
  round's model, about $C/2$ model queries;
- exact targets for each query slice: about $C\, n_{\text{px}} (r + k)$, roughly $10^6$ operations;
- the model's forward and backward pass for each query.

Exact Lloyd on the density would cost $S I C n_{\text{px}} (r + k) \approx 10^9$ operations per node, so
it is run only at a sample of nodes, to measure the model's lost gain.

**Memory, not compute, is the constraint.** A component stores $n_{\text{px}}(r + k + 2)$ floats, about
160 KB, so a node's density is about 16 MB, and a million of them do not fit. The tree is walked depth
first, with the training images sorted by code, so only the current path's densities are held: about
$K \times 16$ MB. Each node is still computed once.

Rank-1 corrections pile up with depth, making deep nodes $(r + K)/r$ times as expensive as shallow ones.
Compressing each component's covariance back to rank $r$ every few cuts removes that, at a small loss of
accuracy.

**Encoding, decoding or generating one image** walks one path of $K$ nodes. At each, Lloyd on the model
takes starts × iterations forward passes, about 10 to 20 at deep nodes, more at the few shallow ones, so
about 500 small passes per image, all easy to batch. The shallow tree can be cached outright: depth 12 or
less is about 8,000 slices of 785 numbers, 25 MB.

## 11. Pseudocode

The rounds are indexed $\text{model}_t$ trained against $\hat p_t$, and $\hat p_{t+1}$ is
$\text{model}_t$'s tree. Slices are computed from codes and cached under (round, code).

```python
# ===== data =====
X, H, T = split(mnist)                          # fit; judge each round and fit gamma; test once at the end
X, H, T = map(dequantize, (X, H, T))

# ===== round 0 density (§4) =====
def fit_covariance_net(X, H):
    net = CNN(x -> d[784] > floor, U[784, r])
    for batch j in X:
        nb = nearest(X, j, 64, exclude_self=True)                 # precomputed
        loss = -mean_j logmeanexp_{i in nb[j]} logN(X[j]; X[i], Sigma(net(X[i])))
        step(loss)
        early_stop(mixture_bits(smoothed_images(net), H))
    return net

# ===== components and cuts (§5) =====
class Component:
    base_w, base_m, base_S      # weight and Gaussian before this round's cuts
    old_code                    # its node in the previous round's tree; None for a smoothed training image
    w, m, S                     # the same after this round's cuts along the current code

    def cut(self, n, b, a):     # side a of slice (n, b), moment-matched
        Sn = self.S @ n                                           # diag(d) n + U (U^T n) + rank-1 corrections
        s = sqrt(n @ Sn); alpha = a * (n @ self.m - b) / s; lam = phi(alpha) / Phi(alpha)
        return replace(self, w=self.w * Phi(alpha),
                       m=self.m + a * lam / s * Sn,
                       S=self.S - lam * (lam + alpha) / s**2 * outer(Sn, Sn))

def smoothed_images(net):
    return [Component(1 / len(X), x, Sigma(net(x)), old_code=None) for x in X]

def exact_side(comps, n, b):    # the density's share, mean and covariance (low-rank pieces) of side +
    ...

# ===== the half-space model (§6) =====
# model[t](history, (n, b)) -> (p_plus, mu_plus, Sigma_plus); history: list of (n, b, a)

def history(t, code):           # the slices and tokens along a code
    return [(*encoder(t, code[:k]), code[k]) for k in range(len(code))]

def node_gaussian(t, code):
    if code == ():
        return mean(X), cov(X)
    n, b = encoder(t, code[:-1]); a = code[-1]
    _, mu, Sig = model[t](history(t, code[:-1]), (a * n, a * b))
    return mu, Sig

def share(t, code, a):          # the model's probability of token a after code
    n, b = encoder(t, code)
    return model[t](history(t, code), (a * n, a * b))[0]

@cache
def encoder(t, code):
    # in:  round t; a code
    # out: the slice (n, b) asked after it; the next token is +1 if n @ x >= b, else -1
    h, (mu, _) = history(t, code), node_gaussian(t, code)
    best = None
    for n, b in seeded_starts(t, code, count=starts_for_depth(len(code))):
        for _ in range(iters):                                    # Lloyd on the model
            _, mu_p, _ = model[t](h, (n, b)); _, mu_m, _ = model[t](h, (-n, -b))
            n = normalize(mu_p - mu_m); b = n @ (mu_p + mu_m) / 2
        p, mu_p, _ = model[t](h, (n, b))
        best = max(best, (2 / n_px * p / (1 - p) * norm(mu_p - mu)**2, n, b))
    return best.n, best.b

def encode(t, x, K):
    code = ()
    for _ in range(K):
        n, b = encoder(t, code)
        code += (+1 if n @ x >= b else -1,)
    return code

def generate(t, K):
    code = ()
    for _ in range(K):
        code += (+1 if random() < share(t, code, +1) else -1,)
    mu, Sig = node_gaussian(t, code)
    return sample(N(mu, Sig))                                     # or mu, as a point reconstruction

# ===== round t's density inside a node (§8) =====
def density_node(t, code):      # walked depth first; only the current path is kept
    if code == ():
        comps = p0 if t == 0 else [Component(1, *node_gaussian(t - 1, ()), old_code=())]
    else:
        n, b = encoder(t, code[:-1])                              # the slice that drew the last token
        comps = prune([c.cut(n, b, code[-1]) for c in density_node(t, code[:-1])])
    return expand(t, comps, code)

def expand(t, comps, code):
    while len(comps) < max_components:
        coarse = [c for c in comps
                  if c.old_code is not None and len(c.old_code) < K
                  and c.w * trace(c.S) > eps * spread(comps) * mass(comps) / len(comps)]
        if not coarse:
            break
        c = max(coarse, key=lambda c: c.w * trace(c.S))
        comps.remove(c)
        for a0 in (+1, -1):                                       # its children in round t-1's tree
            old = c.old_code + (a0,)
            child = Component(c.base_w * share(t - 1, c.old_code, a0),
                              *node_gaussian(t - 1, old), old_code=old)
            for k, a in enumerate(code):                          # reapply this round's cuts
                child = child.cut(*encoder(t, code[:k]), a)
            comps.append(child)
    return comps

# ===== combined targets and training (§7) =====
def targets(t, code, query, real, gamma):
    d = exact_side(density_node(t, code), *query)                 # p_d, mu_d, Sigma_d
    alpha = gamma * (N - len(real))
    plus = [x for x in real if query.n @ x >= query.b]
    p = (alpha * d.p + len(plus)) / (alpha + len(real))
    mu, Sig = mixture_moments((alpha * d.p, d.mu, d.Sigma), plus)
    return p, mu, Sig

def fit_gamma(t, H):            # empirical Bayes: held-out tokens most likely under the combined shares
    return argmax_gamma sum(log p_bar(token) for h in H for each node on h's path)

def train_model(t):
    model[t] = copy(model[t - 1]) if t > 0 else fresh_model()
    gamma = fit_gamma(t, H)
    for _ in range(alternations):                                 # DAgger-style
        encoder.cache_clear()
        codes = [encode(t, x, K) for x in X]                      # records each node's Lloyd queries
        for code in depth_first(codes):
            real = [x for x, c in zip(X, codes) if c[:len(code)] == code]
            for query in recorded_queries(t, code) + perturbed(...):
                for q in (query, flip(query)):
                    p, mu, Sig = targets(t, code, q, real, gamma)
                    weight = rho**len(code) * path_mass(code)
                    loss += weight * (bce(model_p, p) + p * gauss_ce(mu, Sig, model_mu, model_Sig))
            step(loss)

# ===== judges (§9) =====
def mixture_bits(comps, H): ...                 # exact held-out bits per pixel of a mixture
def model_bits(t, H):                           # the code bound
    return mean(-sum(log share(t, code[:k], code[k]) for k in range(K))
                - log N(h; *node_gaussian(t, code))
                for h in H for code in [encode(t, h, K)]) / (n_px * ln2)
def shallow_loss(t, H, k_max=8): ...            # reweighted held-out pairs, plus the no-partner share

# ===== the loop (§8) =====
p0 = smoothed_images(fit_covariance_net(X, H))
best = mixture_bits(p0, H)
for t in range(rounds):
    train_model(t)
    bits = model_bits(t, H)
    log(t, bits, shallow_loss(t, H))
    if bits >= best:
        break                                   # keep a round only if it helps
    best, kept = bits, t
report(model_bits(kept, T), shallow_loss(kept, T))
```

## 12. Open questions

- **Greedy or lookahead?** The best next slice need not be the first of the best $k$ slices.
- **Moment matching at depth.** How far a node's moment-matched mixture drifts from the true cut density
  by depth 32. Check against samples drawn inside the node.
- **The form of $\alpha_z$.** Whether $\gamma (N - n_z)$ fits, or the trust in the density wants another
  shape with depth (a Pólya tree's usual $c k^2$ is the alternative). Fitting $\gamma$ per depth would show it.
- **Does likelihood track decompression?** A density can score better in bits while placing slices no
  better. The shallow decompression loss (§9) is there to catch it.
- **Does Lloyd on the model stay where the model is accurate,** or wander into queries it rarely saw?
- **Inherited leaf shapes, lock-in and spikes** (§8).
- **The distance.** Pixel distance charges a two-pixel shift like a different image. The low-rank
  covariance softens this; another option is to slice an embedding whose Euclidean distance is the
  distance we want, such as a coarse-to-fine pyramid or linear optimal transport (Wang et al. 2013).
- **$V = 2$** against $V$-way Voronoi splits.

## 13. Related work

Each piece has been done before; we know of no work that combines them. Not yet checked by a literature
search.
- **Splitting trees:** tree-structured vector quantization (Buzo, Gray, Gray & Markel 1980), greedy tree
  growing (Riskin & Gray 1991), bisecting $k$-means (Steinbach, Karypis & Kumar 2000), principal direction
  divisive partitioning (Boley 1998), random projection trees (Dasgupta & Freund 2008), oblique decision
  trees (Murthy, Kasif & Salzberg 1994); successive refinement (Equitz & Cover 1991).
- **$k$-means and hyperplanes:** consistency (Pollard 1981); Lloyd–Max quantizers for continuous densities
  (Lloyd 1957/1982); centroidal Voronoi tessellations (Du, Faber & Gunzburger 1999); counting hyperplane
  splits (Cover 1965).
- **Densities on partition trees:** Pólya trees (Lavine 1992; Mauldin, Sudderth & Williams 1992), optional
  Pólya trees with adaptive partitions (Wong & Ma 2010), Bayesian sequential partitioning (Lu, Jiang &
  Wong 2013); trees with Gaussian leaves, such as hierarchical mixtures of experts (Jordan & Jacobs 1994).
- **The smoothed density:** manifold Parzen windows (Vincent & Bengio 2003) and their non-local version
  (Bengio, Larochelle & Vincent 2006); pitfalls of likelihood in high dimension (Theis, van den Oord &
  Bethge 2016).
- **Gaussians cut by half-spaces:** assumed density filtering and expectation propagation (Minka 2001);
  Gaussian probabilities in polytopes (Cunningham, Hennig & Lacoste-Julien 2011).
- **Learning from a search procedure:** DAgger (Ross, Gordon & Bagnell 2011), expert iteration (Anthony,
  Tian & Barber 2017).
- **The loop:** wake-sleep (Hinton, Dayan, Frey & Neal 1995), reweighted wake-sleep (Bornschein & Bengio
  2015); model collapse (Shumailov et al. 2024).
- **Asking questions:** twenty questions with noise (Jedynak, Frazier & Sznitman 2012).
- **Unbiased estimation:** Halmos 1946.

What looks new is the combination: one network per round answering half-space queries in place of an
encoder, prior and decoder; Lloyd's algorithm running on its answers; and a Pólya-tree mix of real images
with the previous round's model as the base distribution, re-fitted every round.

## Current code

None of this is implemented yet. The repository holds the previous design: an encoder built from a
discrete bank of learned questions with a selector network, empirical posteriors over the training images,
and a prior and decoder trained on its codes. Its README, with setup and every script's usage, is in the
history:

```bash
git show 0579b1b:README.md
```
