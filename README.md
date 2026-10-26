# Decoupled VAE — MNIST MVP (blind-knight branch)

A discrete VAE with a 1-D autoregressive latent, a code of tokens over a vocab of `V`, trained in
three separate stages. The encoder is built first, against an ideal decoder. Then the decoder and
the prior are each trained on the frozen encoder's codes.

On this branch the encoder is a **blind knight encoder**. A SliceNet offers many ways to slice the
images, and a SelectorNet picks one slice per token from the code so far, never seeing the image. Lowering
the expected decompression loss on unseen images is like slicing at an opponent you can't see.

Default vocab: `V=16`. The encoder has no code length of its own: it asks one slice per token, and once
every slice is used the token is always 0. Each script that consumes codes picks a length, `--k-max`: the
decoder and prior fix theirs at training time (default 8, i.e. 16⁸ ≈ 4.3 billion codes), and the scripts that
use them read it from their checkpoints. Beyond 2¹⁶ codes, `analyze_encoder.py` analyzes only the prefixes
it can enumerate, and `benchmark.py` estimates its numbers from the encoder's top and sampled codes.

## The objective

**Notation.**
- $p_{\mathrm{data}}$ is the true image distribution. In practice it is approximated by the training images.
- An encoder $q$ maps an image $x$ to a code $z = (z_1, z_2, \dots)$ of tokens, each in
  $\{1, \dots, V\}$, autoregressively: $q(z_{1:t} \mid x) = \prod_{u=1}^{t} q(z_u \mid x, z_{1:u-1})$, where
  $z_{1:t}$ is the length-$t$ prefix (the whole code, if it is shorter) and $z_{1:0}$ the empty one. $zc$ is
  the code $z$ with token $c$ appended.
- $d(x, y) = \tfrac{1}{n_{\mathrm{px}}} \lVert x - y \rVert^2 + \lambda\, \mathbf{1}[\ell(x) \ne \ell(y)]$ is mean squared
  pixel distance, with $n_{\mathrm{px}}$ pixels per image, plus $\lambda$ (`--label-weight`, default 0)
  when the two images show different digits $\ell$. Two one-hot vectors that differ are $\sqrt 2$ apart,
  so appending each image's one-hot digit, scaled by $\sqrt{\lambda n_{\mathrm{px}} / 2}$, to its pixels
  turns $d$ into a plain squared distance, and everything below holds for it unchanged. The encoder never
  sees a label: labels only enter the objective, through the training images in a posterior.
- $p_t$ is a geometric distribution over code lengths $t \in \{1, 2, \dots\}$: how many tokens a decoder
  gets to see. Each further token is read with probability $\rho$ (`--k-decay`, default 0.8), so
  $p_t(t) = (1 - \rho)\,\rho^{t-1}$ and $P_{t \sim p_t}(t \ge u) = \rho^{u-1}$. It doesn't depend on the
  encoder, so losses stay comparable as the encoder grows.
- $\mathbf{1}[\cdot]$ is 1 when its condition holds and 0 otherwise.
- $j$ indexes training images, $v$ val images, and $d$ the build's samples (below).

**In plain terms.** Compress an image by keeping only the first $t$ tokens of its code. To decompress,
an ideal prior fills in the remaining tokens the way they plausibly continue, and an ideal decoder turns
the full code back into an image. Together they sample one of the images that could have produced those
$t$ tokens. The decompression loss is how far, on average, that decompressed image lands from the
original. With few tokens, many very different images fit and the sample usually lands far away.
Each extra token rules some of them out.

**Decompression loss of a posterior** $\pi$, a distribution over images: the mean distance between two
independent samples from it. With L2 pixel distance, that is twice its total variance:

```math
D(\pi) \;=\; \mathbb{E}_{x \sim \pi,\; y \sim \pi}\; d(x, y) \;=\; \tfrac{2}{n_{\mathrm{px}}} \operatorname{tr} \operatorname{Cov}_{\pi}(x).
```

**Decompression loss of an encoder.** Sample a length $t \sim p_t$ and a full code $z$ from the prior the
encoder implies, and take the decompression loss of the posterior of its length-$t$ prefix:

```math
L \;=\; \mathbb{E}_{t \sim p_t}\; \mathbb{E}_{z \sim p_q}\; D\big(\pi_q(\cdot \mid z_{1:t})\big),
\qquad
p_q(z) \;=\; \mathbb{E}_{x \sim p_{\mathrm{data}}}\, q(z \mid x),
\qquad
\pi_q(x \mid z_{1:t}) \;\propto\; p_{\mathrm{data}}(x)\, q(z_{1:t} \mid x).
```

Since $p_q(z)\,\pi_q(x \mid z) = p_{\mathrm{data}}(x)\, q(z \mid x)$, this is also the expected distance from
an image $x \sim p_{\mathrm{data}}$ to an ideal decoder's sample $y \sim \pi_q(\cdot \mid z)$ from the
posterior of its code $z \sim q(\cdot \mid x)$.

**Gain of a token.** A *slice* $r(c \mid x)$ is a distribution over the next token $c$ given the image.
Its gain at a prefix $z$ is how much appending its token lowers the decompression loss there:

```math
G(z, r) \;=\; D\big(\pi(\cdot \mid z)\big) \;-\; \mathbb{E}_{c \sim P(\cdot \mid z)}\; D\big(\pi(\cdot \mid zc)\big),
\qquad
P(c \mid z) \;=\; \mathbb{E}_{x \sim \pi(\cdot \mid z)}\, r(c \mid x),
\qquad
\pi(x \mid zc) \;\propto\; \pi(x \mid z)\, r(c \mid x).
```

By the law of total variance, $G$ is twice the between-token variance, so it is never negative and a
token is a soft k-means split of its prefix's posterior:

```math
G(z, r) \;=\; \tfrac{2}{n_{\mathrm{px}}} \sum_c P(c \mid z)\, \big\lVert \mu(zc) - \mu(z) \big\rVert^2,
\qquad
\mu(z) \;=\; \mathbb{E}_{x \sim \pi(\cdot \mid z)}\, x .
```

Write $L_t$ for the loss at length $t$ and $q_u$ for the encoder's slice at position $u$. The loss then
telescopes into gains, each weighted by the chance that a decoder sees that token:

```math
L \;=\; L_0 \;-\; \sum_{u=1}^{\infty} P_{t \sim p_t}(t \ge u)\; \bar G_u \;=\; L_0 \;-\; \sum_{u=1}^{\infty} \rho^{u-1}\, \bar G_u,
\qquad
\bar G_u \;=\; \mathbb{E}_{z \sim p_q}\; G\big(z_{1:u-1},\, q_u(\cdot \mid \cdot, z_{1:u-1})\big),
```

where $L_0 = D(p_{\mathrm{data}})$ is the loss with no code, and $\bar G_u = 0$ past the end of the code.

### Empirical estimate

With $p_{\mathrm{data}}$ replaced by the $N$ training images, a code's posterior is a weighting of the
training images, each by its chance of producing the code:

```math
w_j(z) \;=\; q(z \mid x_j),
\qquad
\hat\pi(j \mid z) \;=\; \frac{w_j(z)}{\sum_{j'} w_{j'}(z)} .
```

Slices see only the image, so $q(z \mid x_j)$ is a product of slice outputs. With every slice's output on
every training image computed once, any code's posterior over all $N$ images is a lookup and a product.
In the build, each image's slice outputs come from the copy that never trained on it (out-of-fold, below),
so its weights behave like a fresh image's. Weights below `--weight-cutoff` $\epsilon$ are set to 0, so a
code only touches the images that can plausibly produce it.

**Spread of a posterior.** Two independent samples from an empirical posterior can be the same image, so a
code only one image reaches would score $D = 0$. The estimate therefore averages over pairs of *different*
images. For a weighting $w$ of the training images, with the sums

```math
W = \sum_j w_j,
\qquad
M = \sum_j w_j\, x_j,
\qquad
Q = \sum_j w_j \lVert x_j \rVert^2,
\qquad
W_2 = \sum_j w_j^2,
```

its spread is

```math
\hat D(w) \;=\; \frac{\sum_{j \ne j'} w_j w_{j'}\, d(x_j, x_{j'})}{\sum_{j \ne j'} w_j w_{j'}}
\;=\; \frac{2\,\big(W Q - \lVert M \rVert^2\big)}{n_{\mathrm{px}}\,\big(W^2 - W_2\big)} .
```

It needs at least two images with weight.

**Gain of a slice at a code.** Slice $r$'s token splits the weighting $w(z)$ into one weighting per token,
$w_j(zc) = w_j(z)\, r(c \mid x_j)$, so

```math
\hat G(z, r) \;=\; \hat D\big(w(z)\big) \;-\; \sum_c \hat P(c \mid z)\, \hat D\big(w(zc)\big),
\qquad
\hat P(c \mid z) \;=\; \frac{\sum_j w_j(zc)}{\sum_j w_j(z)} .
```

A token with fewer than two images of weight adds nothing: its $\hat D$ falls back to $\hat D(w(z))$. A
partition $\lambda$, which gives every training image one of $V$ classes $\lambda_j$, has the implied gain
$\hat G(z, \lambda)$ of the slice that reproduces it exactly, $r(c \mid x_j) = \mathbf{1}[c = \lambda_j]$.

**Which codes.** There are far too many codes to enumerate, so the selector is fit on a random subset.
The build samples $C$ (`--codes`) training images without replacement, and an out-of-fold code for each. $Z_u$ is
the set of their distinct length-$u$ prefixes, and $n_z$ is how many samples share prefix $z$. Every
sample's code comes from the implied prior $p_q$, so count-weighted averages estimate prior expectations,
and each distinct prefix is computed once:

```math
\mathbb{E}_{z \sim p_q}\, f(z_{1:u}) \;\approx\; \frac{1}{C} \sum_{z \in Z_u} n_z\, f(z) .
```

**Val loss.** A val image $x_v$ is a fresh sample. With the first $t$ tokens of its code, its loss is its
mean distance to the training images under that prefix's posterior:

```math
\hat L_t \;=\; \mathbb{E}_v \sum_j \hat\pi(j \mid z_{v,1:t})\, d(x_v, x_j),
\qquad
\hat L \;=\; \mathbb{E}_{t \sim p_t}\, \hat L_t .
```

If no training image has weight on the prefix, the longest prefix that has some is used.

## The blind knight encoder

- **SliceNet** maps an image to logits $s(x) \in \mathbb{R}^{S \times V}$ for its $S$ slices. Slice $h$
  is $r_h(c \mid x) = \mathrm{softmax}\big(s(x)_h\big)_c$. It sees only the image.
- **Committed nodes.** Some codes $z$ have a slice built for them, $\kappa(z)$, which they always ask. The
  committed codes form a tree from the root: every committed node's parent is committed too.
- **SelectorNet** maps a prefix $z$ of any length to $\sigma(z) \in \mathbb{R}^S$, the predicted gain of
  each slice there. It sees only the code, and it picks wherever no slice is committed.
- **Encoding.** At each position the knight asks the committed slice, or else the unused slice with the
  highest predicted gain, and samples the token from it:

```math
q(z_u \mid x, z_{1:u-1}) \;=\; r_{\eta(z_{1:u-1})}(z_u \mid x),
\qquad
\eta(z) \;=\; \begin{cases} \kappa(z) & z \text{ committed,} \\ \arg\max_{h \notin \mathrm{used}(z)} \sigma(z)_h & \text{otherwise,} \end{cases}
\qquad
\mathrm{used}(z) \;=\; \big\{ \eta(z_{1:u}) : 0 \le u < |z| \big\}.
```

Every image with the same prefix uses the same slice, so a token means the same thing for all of them.
One slice can be asked at many prefixes and depths: the slices are a shared bank of questions, and a
code is a path down a decision tree whose questions come from the bank. A slice sees only the image, so
asking it twice along one code gets the same answer again; $\mathrm{used}(z)$, which follows from $z$
alone, rules that out. Once the bank is used up, after $S$ tokens, every further token is 0.

**The decoder and prior read the slices too.** At every position they're told which slice is asked
there (a separate index once the bank is used up). That index follows from the code's earlier tokens
through the commits and the selector, so the encoder supplies it: when they train, when the prior
samples, and when the benchmark enumerates codes. A token means nothing apart from the slice that drew
it, so each (slice, token) pair has its own embedding, $V(S+1)$ in all. The decoder reads one pair per
position. The prior reads the previous pair plus the slice it is asked to predict for.

**SliceNet grows by columns.** Each build round adds one column holding up to $B$ (`--slices-per-round`)
new slices. A column is a small conv stack with one head per slice. Its layer $l$ reads its own layer
$l-1$ plus a learned projection of every earlier column's layer $l-1$ (progressive-network laterals).
Earlier columns are frozen, so adding a column never changes an earlier slice's output.

**SelectorNet is rebuilt every round**: its output grows with the bank, and a fresh net keeps it plastic.

## How the encoder is built

`train_encoder.py` grows a committed tree top down, like divisive hierarchical $V$-means: each round
splits the $B$ nodes whose children hold the most loss, each with a slice built for it, then retrains the selector
to fill in everywhere else. Throughout, a table holds every slice's held-out output on every training
image, so any code's posterior is a lookup away.

1. **Frontier.** The uncommitted children of committed nodes, or the root before anything is committed.
   Each is scored by the loss its children would still hold under the question asked there now (the
   selector's pick, or none before any slice exists), weighted by the chance a decoder reads the token after
   it: $\rho^{|z|}\, W(z)\, \big(\hat D(z) - G_{z,\eta(z)}\big)$, with $W(z)$ the images its posterior weighs and
   $\hat D(z)$ its spread. A node the bank already splits well ranks low. Frontier nodes' ancestors are all
   committed, so their posteriors are exact products of table entries. The $B$ highest are this round's
   targets.
2. **Partitions.** For each target, split its posterior as well as possible: within one code, a
   partition's implied loss is k-means' objective with $V$ clusters, with images weighted by the posterior,
   up to the different-pairs correction.
   - The search starts as k-means++ does, from the target's own images: $V$ seeds sampled one at a time,
     each with probability proportional to its posterior weight times its squared distance to the nearest
     seed so far. Then Hartigan's method with batched moves, kept only if the gain rises. It stops when no
     move helps, or when the last 5 rounds raised the gain by less than `--hartigan-tol` of it per round.
     The best of `--restarts` starts is kept.
   - Then every training image takes its nearest class centroid, a class's posterior-weighted mean image
     at the target. That keeps the classes consistent, and a nearest-centroid rule is linear in the
     image, so a column can learn it.
3. **Column.** Train `--folds` + 1 copies of the new column by cross-entropy against the partitions'
   classes, each image weighted by its posterior weight in the target: fold copy $f$ on the training
   images outside fold $f$, a final copy on all. All copies share the epoch count, chosen by the pooled
   held-out cross-entropy. Slices see only images, so no codes are needed here. The new slices' held-out
   outputs join the table, and each target commits to its new slice. The build logs each new slice's
   realized gain at its target, next to the best gain any older slice gets there.
4. **Selector.** A fresh SelectorNet, fit one depth at a time on $C$ (`--codes`) sampled images, since
   codes at depth $u$ depend on the choices above it. For $u = 0, 1, \dots$:
   - extend the sampled codes by one token, with the commits and the selector so far; stop, with $T = u$, if
     the bank is used up or every depth-$u$ code is reached by a single image;
   - at every distinct depth-$u$ prefix, the realized gain of every slice, used ones included: asking a
     slice again gains about nothing, and the selector has to know that in case its choices above change;
   - fit `--folds` + 1 copies by MSE on those gains at depths $0, \dots, u$, early-stopped on the held-out
     gain of their choices at uncommitted prefixes (only unused slices can be chosen), weighted by depth.
     A sample belongs to its image's fold. Then every copy uses the final selector.

   Fitting a deeper depth can change choices at shallower ones after their codes were rolled out; the
   build logs how many changed, and per depth the share of committed prefixes and how often the final
   selector picks the unused slice with the highest realized gain elsewhere.
5. **Report.** On the val split: $\hat L_t$ for every length $t \le T$, and $\hat L$ (lengths past $T$
   score the same as $T$), with the label term when $\lambda > 0$; the wrong-digit rate at length $T$, the
   posterior's weight on training images of another digit, which compares runs with different $\lambda$.

The build stops after `--rounds` rounds, when no frontier node can be split, or when the val loss hasn't
improved for `--round-patience` rounds. Every step logs one line with its result and duration.

## Encoder training

**Notation** (on top of the objective's).
- Training images $x_1, \dots, x_N$, each in one of $F$ (`--folds`) folds, $f(j) \in \{1, \dots, F\}$.
- Model copies $m \in \{1, \dots, F, \star\}$: fold copy $m$ never sees fold $m$; $\star$ is the final copy.
  A val image $v$ goes through the final copy, $f(v) = \star$, which never trained on it either.
- Copy $m$ of the SliceNet has slices $r^m_h$, $h \in \{1, \dots, S\}$. There is one selector,
  $\sigma^\star$, and one commit table $\kappa$, which give $\eta$ as above. Copy $m$ of the encoder is
  $q^m(z_u \mid x, z_{1:u-1}) = r^m_{\eta(z_{1:u-1})}(z_u \mid x)$.
- The table holds $R_{jhc} = r^{f(j)}_h(c \mid x_j)$, every training image's held-out slice outputs. A
  code's posterior weights are held out too: $w_j(z) = q^{f(j)}(z \mid x_j)$.
- $\mathcal F$ is the frontier: the uncommitted children of committed nodes (the root while nothing is
  committed). $W(z) = \sum_j w_j(z)$ and $\hat D(z) = \hat D(w(z))$.
- Samples $d = 1, \dots, C$: a training image $j_d$, sampled without replacement, and its out-of-fold code
  $z_d$, rolled out to depth $T$. $Z_u$ holds their distinct length-$u$ prefixes.
- $T$ is the depth the codes stop at: $T = S$ when the bank is used up, or the first depth at which every
  sampled code is reached by a single training image.
- $\omega_u = P_{t \sim p_t}(t \ge u + 1) = \rho^u$ weighs depth $u$, whose token is position $u+1$.
- $G_{zh} = \hat G(z, R_{\cdot h \cdot})$ is slice $h$'s realized gain at code $z$, and $\hat G(z, \lambda)$
  a partition's implied gain.
- $\mathrm{CE}(\text{logits}, c)$ is cross-entropy against class $c$. $\operatorname{fit}_\theta\, A \,\|\, B$
  trains $\theta$ on loss $A$ by minibatch steps and keeps the epoch with the best held-out score $B$, with
  one epoch count shared by all $F+1$ copies.

Two versions follow: a concise one in equations, and a step-by-step one in pseudo-code.

### In equations

**One round**, with $S$ slices in the bank:

**1. Targets**: the $B$ frontier nodes $z_1, \dots, z_B \in \mathcal F$ with the highest

```math
\mathrm{score}(z) \;=\; \omega_{|z|}\; W(z)\; \big(\hat D(z) - G_{z,\eta(z)}\big),
```

the loss its children would hold under the question asked there now ($G = 0$ while the bank is empty).

**2. Partitions**, one per target:

```math
\lambda_b \;=\; \operatorname{nearestCentroid}_{z_b}\Big(\operatorname{Hartigan}_\lambda\, \hat G(z_b, \lambda)\Big),
```

the best of `--restarts` k-means++ starts among $z_b$'s images.

**3. Column**: slices $S+1, \dots, S+B$, for every copy $m$, each image weighted by its posterior weight
in the target; then $\kappa(z_b) \leftarrow S + b$:

```math
s^m_{\mathrm{new}} \;=\; \operatorname{fit}_s\;
  \frac{\mathbb{E}_{j : f(j) \ne m}\, \sum_b w_j(z_b)\, \mathrm{CE}\big(s(x_j)_{S+b},\, \lambda_{b,j}\big)}
       {\mathbb{E}_{j : f(j) \ne m}\, \sum_b w_j(z_b)}
\;\big\|\;
  \text{the same over all } j \text{ with } s^{f(j)}_{\mathrm{new}} .
```

**4. Selector**: reinitialized, with $C$ fresh samples. For $u = 0, 1, \dots$: extend every $z_d$ by one
token with $\kappa$ and the current $\sigma^\star$; stop with $T = u$ if the bank is used up or every code in
$Z_u$ is reached by a single image; else compute $G_{zh}$ for every $z \in Z_u$ and every slice $h$, used or
not, and

```math
\sigma^m \;=\; \operatorname{fit}_\sigma\;
  \mathbb{E}_{d : f(j_d) \ne m}\, \sum_{u' \le u}\, \mathbb{E}_h\, \big(\sigma(z_{d,1:u'})_h - G_{z_{d,1:u'} h}\big)^2
\;\big\|\;
  \frac{1}{C} \sum_d \sum_{u' \le u} \mathbf{1}[z_{d,1:u'} \text{ uncommitted}]\; \omega_{u'}\, G_{z_{d,1:u'} \hat h},
\qquad
\hat h \;=\; \arg\max_{h \notin \mathrm{used}} \sigma^{f(j_d)}(z_{d,1:u'})_h ,
```

with $\mathrm{used} = \mathrm{used}(z_{d,1:u'})$, warm-started from the previous depth's fit. Then
$\sigma^m \leftarrow \sigma^\star$ for every copy.

**5. Report** $\hat L$ on the val split, with val images coded by $q^\star$.

### In pseudo-code

The build is split into functions, each opening with comments that give its inputs, its outputs, what
they mean and their shapes. All of them read the shared state listed first, and the build at the end
calls them.

**Shared state, codes and gains.**

```math
\begin{array}{l}
\text{// shared state} \\
\text{// } x_j \text{: the training images (shape } N \times n_{\mathrm{px}} \text{); } f(j) \text{: each image's fold (shape } N \text{)} \\
\text{// the bank: } F + 1 \text{ SliceNet copies with } S \text{ slices each; copy } m \text{ answers slice } h \text{ with } r^m_h(\cdot \mid x) \text{ (shape } V \text{)} \\
\text{// the table: } R_{jhc} = r^{f(j)}_h(c \mid x_j) \text{, every training image's held-out slice outputs (shape } N \times S \times V \text{)} \\
\text{// } \kappa \text{: the commit table, node } \to \text{ the slice built for it; } \sigma^\star \text{: the selector} \\
\text{// the samples: images } j_1, \dots, j_C \text{ (shape } C \text{) and their codes } z_1, \dots, z_C \text{ (shape } C \times T \text{); and } T \\
\text{// the sampled codes: } Z_u \text{, the distinct length-} u \text{ prefixes (shape } |Z_u| \times u \text{)} \\[12pt]
\textbf{function } \mathrm{ask}(z): \\
\quad \text{// in: a code } z \text{ (shape } |z| \text{)} \\
\quad \text{// out: the slice asked after it: its committed slice, or the selector's best unused one (one index)} \\
\quad \textbf{return } \kappa(z) \text{ if } z \text{ is committed, else } \arg\max_{h \notin \mathrm{used}(z)} \sigma^\star(z)_h \\[12pt]
\textbf{function } \mathrm{sampleCode}(j, t): \\
\quad \text{// in: an image } j \text{, training or val; a length } t \le S \\
\quad \text{// out: } z \text{, image } j \text{'s answers to the knight's first } t \text{ questions (shape } t \text{, each token in } \{1, \dots, V\} \text{)} \\
\quad z \leftarrow () \\
\quad \textbf{for } u = 0, \dots, t-1: \\
\qquad z_{u+1} \sim r^{f(j)}_{\mathrm{ask}(z)}(\cdot \mid x_j) \qquad \text{// the image answers, through the copy that never trained on it} \\
\quad \textbf{return } z \\[12pt]
\textbf{function } \mathrm{weights}(z): \\
\quad \text{// in: a code } z \text{ (shape } |z| \text{)} \\
\quad \text{// out: } w \text{, each training image's held-out chance of producing } z \text{: the code's posterior, unnormalized (shape } N \text{, mostly 0)} \\
\quad w_j \leftarrow \textstyle\prod_{u=1}^{|z|} R_{j h_u z_u} \text{, where } h_u = \mathrm{ask}(z_{1:u-1}) \\
\quad w_j \leftarrow 0 \text{ wherever } w_j < \epsilon \\
\quad \textbf{return } w \\[12pt]
\textbf{function } \mathrm{spread}(w): \\
\quad \text{// in: } w \text{, a weighting of the training images with at least two images of weight (shape } N \text{)} \\
\quad \text{// out: } \hat D \text{, the mean distance between two different images sampled from it (one number)} \\
\quad W, M, Q, W_2 \leftarrow \textstyle\sum_j w_j,\ \sum_j w_j x_j,\ \sum_j w_j \lVert x_j \rVert^2,\ \sum_j w_j^2 \\
\quad \textbf{return } 2\big(W Q - \lVert M \rVert^2\big) \big/ \big(n_{\mathrm{px}} (W^2 - W_2)\big) \\[12pt]
\textbf{function } \mathrm{gain}(z, r): \\
\quad \text{// in: a code } z \text{ (shape } |z| \text{); } r \text{, a next-token distribution for every training image (shape } N \times V \text{)} \\
\quad \text{// out: } \hat G \text{, how much } r \text{'s token after } z \text{ lowers the decompression loss of } z \text{'s posterior (one number)} \\
\quad w \leftarrow \mathrm{weights}(z) \\
\quad \textbf{if } \text{fewer than two images have weight in } w: \textbf{ return } 0 \qquad \text{// a single image: nothing left to split} \\
\quad \textbf{for } c = 1, \dots, V: \\
\qquad w^c_j \leftarrow w_j\, r_j(c) \text{ for every image } j \qquad \text{// the posterior of } zc \\
\qquad D_c \leftarrow \mathrm{spread}(w^c) \text{ if at least two images have weight in } w^c \text{, else } \mathrm{spread}(w) \\
\quad \textbf{return } \mathrm{spread}(w) - \textstyle\sum_c \big(\sum_j w^c_j \big/ \sum_j w_j\big)\, D_c \\[12pt]
\textbf{function } \mathrm{sliceGains}(z): \\
\quad \text{// in: a code } z \text{ (shape } |z| \text{)} \\
\quad \text{// out: } G_{z \cdot} \text{, the realized gain at } z \text{ of every slice (shape } S \text{); a slice already used on } z \text{ gains about nothing} \\
\quad \textbf{return } G_{zh} = \mathrm{gain}(z, R_{\cdot h \cdot}) \text{ for every slice } h \\[12pt]
\textbf{function } \mathrm{frontier}(): \\
\quad \text{// out: the uncommitted children of committed nodes (just the root if nothing is committed), each with its} \\
\quad \text{// score: the loss its children would hold under the question asked there now; highest first (} |\mathcal F| \text{ codes)} \\
\quad \mathcal F \leftarrow \{()\} \text{ if } () \text{ is uncommitted, else } \{ zc : z \text{ committed},\ zc \text{ uncommitted} \} \\
\quad \textbf{for each } z \in \mathcal F: \quad w \leftarrow \mathrm{weights}(z); \quad
        \mathrm{score}(z) \leftarrow \omega_{|z|} \textstyle\sum_j w_j \cdot \big(\mathrm{spread}(w) - \mathrm{sliceGains}(z)_{\mathrm{ask}(z)}\big) \\
\quad \textbf{return } \mathcal F \text{ sorted by score, highest first}
\end{array}
```

**Partitions.**

```math
\begin{array}{l}
\textbf{function } \mathrm{kmeansppStart}(w): \\
\quad \text{// in: } w \text{, a weight for every training image (shape } N \text{)} \\
\quad \text{// out: } \lambda \text{, a starting class for every image (shape } N \text{): the nearest of } V \text{ well-spread seed images} \\
\quad y_1 \leftarrow \text{a training image } x_j \text{ sampled with probability} \propto w_j \\
\quad \textbf{for } c = 2, \dots, V: \quad y_c \leftarrow \text{a training image } x_j \text{ sampled with probability} \propto w_j \min_{c' < c} \lVert x_j - y_{c'} \rVert^2 \\
\quad \textbf{return } \lambda \text{ with } \lambda_j = \arg\min_c \lVert x_j - y_c \rVert^2 \\[12pt]
\textbf{function } \mathrm{hartigan}(\lambda, z): \\
\quad \text{// in: } \lambda \text{, a starting class per image (shape } N \text{); a code } z \text{ (shape } |z| \text{)} \\
\quad \text{// out: } \lambda \text{, moved image by image until no single move raises } \mathrm{gain}\big(z, \mathrm{onehot}(\lambda)\big) \text{ (shape } N \text{),} \\
\quad \text{// or until it stalls} \\
\quad \textbf{repeat at most } \texttt{hartigan-rounds} \text{ times}: \\
\qquad \Delta_{jc} \leftarrow \text{the exact change in the gain from moving image } j \text{ to class } c \text{ (shape } N \times V \text{; 0 off } z \text{'s posterior),} \\
\qquad\qquad \text{from per-class sums of } w,\ w\, x,\ w \lVert x \rVert^2,\ w^2 \text{ over } z \text{'s posterior} \\
\qquad \mathcal M \leftarrow \text{the moves } \big(j, \arg\max_c \Delta_{jc}\big) \text{ with } \max_c \Delta_{jc} > 0 \text{, largest first} \\
\qquad \textbf{if } \mathcal M = \emptyset: \textbf{ break} \\
\qquad n \leftarrow |\mathcal M| \\
\qquad \textbf{loop}: \text{ apply the first } n \text{ moves; } \textbf{if } \text{the gain rose}: \textbf{ break}; \textbf{ else } \text{undo them, } n \leftarrow \lceil n / 2 \rceil \\
\qquad \textbf{if } \text{the last 5 rounds raised the gain by less than } 5 \cdot \texttt{hartigan-tol} \text{ of it}: \textbf{ break} \\
\quad \textbf{return } \lambda \\[12pt]
\textbf{function } \mathrm{nearestCentroid}(\lambda, z): \\
\quad \text{// in: } \lambda \text{, a class per image (shape } N \text{); a code } z \text{ (shape } |z| \text{)} \\
\quad \text{// out: every image's nearest class centroid at } z \text{ (shape } N \text{): the same classes, extended to all images} \\
\quad w \leftarrow \mathrm{weights}(z); \quad \mu_c \leftarrow \textstyle\sum_{j : \lambda_j = c} w_j x_j \big/ \sum_{j : \lambda_j = c} w_j \text{ for every class with weight (shape } V \times n_{\mathrm{px}} \text{)} \\
\quad \textbf{return } \lambda' \text{ with } \lambda'_j = \arg\min_c \lVert x_j - \mu_c \rVert^2 \\[12pt]
\textbf{function } \mathrm{split}(z): \\
\quad \text{// in: a target code } z \text{ (shape } |z| \text{)} \\
\quad \text{// out: } \lambda \text{, the best } V \text{-way split of } z \text{'s posterior, extended to every image (shape } N \text{)} \\
\quad \lambda \leftarrow \text{the } \mathrm{hartigan}\big(\mathrm{kmeansppStart}(\mathrm{weights}(z)), z\big) \text{ with the highest gain at } z \text{, over } \texttt{restarts} \text{ starts} \\
\quad \textbf{return } \mathrm{nearestCentroid}(\lambda, z)
\end{array}
```

**Training the column and the selector, and scoring.**

```math
\begin{array}{l}
\textbf{function } \mathrm{fitCopies}(\theta, \mathrm{loss}, \mathrm{score}, \mathrm{patience}): \\
\quad \text{// in: } \theta^m \text{, the trainable weights of copy } m \in \{1, \dots, F, \star\} \text{ (} F + 1 \text{ sets); } \mathrm{loss}(\theta^m, \mathcal B) \text{, a minibatch loss;} \\
\quad \text{// } \mathrm{score}(\theta^1, \dots, \theta^F) \text{, a held-out score (higher is better) that scores each example through} \\
\quad \text{// the copy that never trained on its image; and a patience in epochs} \\
\quad \text{// out: every copy trained in lockstep epochs, keeping the weights from the epoch with the best score} \\
\quad \textbf{for } \text{epoch} = 1, 2, \dots: \\
\qquad \textbf{for } m \in \{1, \dots, F, \star\}: \\
\qquad\quad \textbf{for each minibatch } \mathcal B \text{ of the examples whose image has } f \ne m \text{ (all examples for } \star \text{)}: \\
\qquad\qquad \text{step } \theta^m \text{ on } \mathrm{loss}(\theta^m, \mathcal B) \\
\qquad \textbf{if } \text{no higher } \mathrm{score} \text{ for } \mathrm{patience} \text{ epochs}: \textbf{ break} \\
\quad \text{every copy takes back its weights from the epoch with the highest score} \\[12pt]
\textbf{function } \mathrm{trainColumn}(z_{1:B}, \lambda): \\
\quad \text{// in: the targets } z_1, \dots, z_B \text{ and their partitions } \lambda_1, \dots, \lambda_B \text{ (shape } B \times N \text{)} \\
\quad \text{// out: nothing returned; every copy gains } B \text{ frozen slices, so its SliceNet maps an image to logits of shape } (S+B) \times V \text{;} \\
\quad \text{// the table grows to shape } N \times (S+B) \times V \text{; and each target commits to its new slice} \\
\quad \text{add a column with heads } S+1, \dots, S+B \text{ to every copy; earlier columns stay frozen} \\
\quad w_{jb} \leftarrow \mathrm{weights}(z_b)_j \qquad \text{// each image's posterior weight in each target (shape } N \times B \text{)} \\
\quad \mathrm{CE}_w(s, \mathcal B) = \textstyle\sum_{j \in \mathcal B} \sum_b w_{jb}\, \mathrm{CE}\big(s(x_j)_{S+b},\, \lambda_{b,j}\big) \Big/ \sum_{j \in \mathcal B} \sum_b w_{jb} \\
\quad \mathrm{fitCopies}\big(\text{the new columns},\ \mathrm{CE}_w,\ -\mathrm{CE}_w \text{ over all images, each through copy } f(j),\ \texttt{patience}\big) \\
\quad \text{freeze the new columns; } R_{j, S+b, c} \leftarrow r^{f(j)}_{S+b}(c \mid x_j) \text{ for every image } j \text{, new slice } b \text{ and token } c \\
\quad \kappa(z_b) \leftarrow S + b \text{ for every target; } S \leftarrow S + B \\[12pt]
\textbf{function } \mathrm{fitSelector}(): \\
\quad \text{// out: a fresh } \sigma^\star \text{, the knight's guess of how much each question would help after any code} \\
\quad \text{// (maps a prefix of shape } u \text{ to gains of shape } S \text{); } T \text{, the depth the codes now stop at;} \\
\quad \text{// and the drift, how many earlier choices later fits changed (shape } T \text{, one fraction per depth)} \\
\quad \sigma^m \leftarrow \text{freshly initialized, for every copy } m \\
\quad j_1, \dots, j_C \leftarrow C \text{ training images sampled without replacement; } z_d \leftarrow () \text{ for every sample } d; \quad \mathcal D \leftarrow \emptyset \\
\quad \textbf{for } u = 0, 1, \dots: \\
\qquad \textbf{if } u \ge 1: \quad z_{d,u} \sim r^{f(j_d)}_{\mathrm{ask}(z_{d,1:u-1})}(\cdot \mid x_{j_d}) \text{ for every sample } d
        \qquad \text{// one more token, with the selector fit so far} \\
\qquad Z_u \leftarrow \text{the distinct prefixes } z_{d,1:u} \\
\qquad \textbf{if } u = S \text{ or every code in } Z_u \text{ is reached by a single image}: \quad T \leftarrow u; \textbf{ break}
        \qquad \text{// nothing deeper can be asked or scored} \\
\qquad G_{z \cdot} \leftarrow \mathrm{sliceGains}(z) \text{ for every } z \in Z_u; \quad \mathcal D \leftarrow \mathcal D \cup \big\{ (d, u) : d = 1, \dots, C \big\} \\
\qquad \text{// fit every copy on depths } 0, \dots, u \text{, warm-started from the last depth; a sample belongs to its image's fold} \\
\qquad \mathrm{MSE}(\sigma, \mathcal B) = \mathbb{E}_{(d, u') \in \mathcal B}\, \mathbb{E}_h \big( \sigma(z_{d,1:u'})_h - G_{z_{d,1:u'} h} \big)^2
        \qquad \text{// every slice, used ones included} \\
\qquad \mathrm{choices}(\sigma^1, \dots, \sigma^F) = \frac{1}{C} \textstyle\sum_{u' \le u} \omega_{u'} \sum_{d :\, z_{d,1:u'} \text{ uncommitted}} G_{z_{d,1:u'} \hat h},
        \quad \hat h = \arg\max_{h \notin \mathrm{used}(z_{d,1:u'})} \sigma^{f(j_d)}(z_{d,1:u'})_h \\
\qquad \mathrm{fitCopies}\big(\sigma,\ \mathrm{MSE},\ \mathrm{choices},\ \texttt{selector-patience}\big) \\
\qquad \sigma^m \leftarrow \sigma^\star \text{ for every fold copy } m \qquad \text{// } \mathrm{ask} \text{ now uses this fit} \\
\qquad \text{log the drift: the fraction of } (d, u' < u) \text{ whose } \mathrm{ask}(z_{d,1:u'}) \text{ is no longer the slice that drew } z_{d,u'+1} \\
\quad \text{log, per depth, the share of committed prefixes and how often } \sigma^\star \text{ picks the best unused slice elsewhere} \\
\quad \textbf{return } \sigma^\star, T \\[12pt]
\textbf{function } \mathrm{evaluate}(): \\
\quad \text{// out: } \hat L_t \text{, how far a val image's decompressed version lands from it on average, given } t \text{ tokens (shape } T \text{);} \\
\quad \text{// and } \hat L \text{, their } p_t \text{-weighted mean (one number)} \\
\quad z_v \leftarrow \mathrm{sampleCode}(v, T) \text{ for every val image } v \qquad \text{// shape } N_{\mathrm{val}} \times T \\
\quad \textbf{for } t = 1, \dots, T: \\
\qquad w \leftarrow \mathrm{weights}(z_{v,1:t}) \text{, or the longest prefix's with some weight, for every val image } v \\
\qquad \hat L_t \leftarrow \text{the mean over val images } v \text{ of } \textstyle\sum_j w_j\, d(x_v, x_j) \big/ \sum_j w_j \qquad \text{// from } W, M, Q \text{ of } w \\
\quad \hat L \leftarrow \textstyle\sum_{t=1}^{T} (1 - \rho)\, \rho^{t-1}\, \hat L_t \;+\; \rho^{T}\, \hat L_T
        \qquad \text{// lengths past } T \text{ score the same as } T \\
\quad \textbf{return } (\hat L_1, \dots, \hat L_T),\ \hat L
\end{array}
```

**The build.**

```math
\begin{array}{l}
\text{split the training images into } F \text{ folds at random} \\
S \leftarrow 0; \quad T \leftarrow 0; \quad \kappa \leftarrow \text{empty} \qquad \text{// no slices yet: every code is the empty one} \\
\textbf{for } \text{round} = 1, 2, \dots, \texttt{rounds}: \\
\quad z_{1:B} \leftarrow \text{the first } B \text{ of } \mathrm{frontier}() \text{ that can be split} \qquad \text{// A. the nodes whose children hold the most loss} \\
\quad \lambda_b \leftarrow \mathrm{split}(z_b) \text{ for every target} \qquad \text{// B. the best split of each} \\
\quad \mathrm{trainColumn}(z_{1:B}, \lambda) \qquad \text{// C. a new column learns them, and each target commits to its slice} \\
\quad \text{log each new slice's gain at its target, and the best older slice's there} \\
\quad (\sigma^\star, T) \leftarrow \mathrm{fitSelector}() \qquad \text{// D. the knight learns what to ask everywhere else} \\
\quad (\hat L_{1:T}, \hat L) \leftarrow \mathrm{evaluate}() \qquad \text{// E. score on held-out images} \\
\quad \textbf{if } \hat L \text{ has not improved for } \texttt{round-patience} \text{ rounds}: \textbf{ stop}
\end{array}
```

Notes:
- In `hartigan`, a single move with $\Delta_{jc} > 0$ always raises the gain, so the halving loop ends.
- In `fitSelector`, the fold copies of the selector exist only so that a held-out score can pick the
  epoch. After each depth, every copy is the final one.
- A code's posterior is exact over all training images. Only which codes the selector is fit on is sampled.

## Setup

```bash
source .venv/bin/activate   # torch + torchvision + matplotlib; MNIST downloads to data/ on first run
```

The device is picked automatically (cuda → mps → cpu); override it with `--device`.

## Pipeline

```bash
python train_encoder.py                                  # -> checkpoints/encoder.pt (+ encoder_build.pt)
python train_decoder.py --k-max 32                       # -> checkpoints/decoder_gray.pt   (8-bit pixels)
python train_decoder.py --k-max 32 --likelihood binary   # -> checkpoints/decoder_binary.pt (binarized)
python train_prior.py --k-max 32                         # -> checkpoints/prior.pt
python qa_generate.py -n 64                              # -> outputs/generate.png
python qa_compress.py -n 16 -k 1 2 4 10 -m 8             # -> outputs/compress/k01.png ...
python qa_compress_analytically.py -n 16 -m 8            # -> outputs/compress_analytically/k01.png ...
python qa_latent_space.py                                # -> http://127.0.0.1:8000/
python benchmark.py --decoder checkpoints/decoder_gray.pt checkpoints/decoder_binary.pt \
                    --encoder checkpoints/encoder.pt     # -> results/benchmark.{json,md}
python analyze_encoder.py --encoder checkpoints/encoder.pt # -> results/usage_encoder.json
```

- **Building the encoder:**
  - `checkpoints/encoder.pt` is rewritten after every round, so the other scripts can use a partly
    built encoder.
  - `checkpoints/encoder_build.pt` keeps every fold copy. Continue with `--resume` (and a larger
    `--rounds`). Ctrl-C loses only the round in progress.
  - `outputs/encoder_build.png` (`--plot`) is redrawn after every round. On the left, one line per round
    shows the val decompression loss at every code length, darker for later rounds. On the right is the
    val loss by round.
  - For quick tests, `--limit-train` and `--limit-val` use fewer images.
- **Decoder and prior training:**
  - Train them with the same `--k-max`; the scripts that use both check that they match each other and
    the encoder's bank.
  - They early-stop on their val loss (a fixed 5k split of the MNIST train set) and keep the best
    checkpoint at `--out`. Set the learning rate with `--lr`.
  - Ctrl-C saves `*_latest.pt`. Continue with `--resume checkpoints/<name>_latest.pt`; flags you add
    override the saved ones.
- **Token sampling:** `--encoder-mode` and `--prior-mode` pick how tokens are sampled:
  - `actual`: the model's own distribution;
  - `greedy`: the most likely token at each step;
  - `topp`: nucleus sampling, with `--*-top-p`.
- **Compression QA:** `qa_compress.py` encodes greedily by default. Each output row is the original
  image followed by `m` decodings, each with the rest of the code filled in by the prior.
  `qa_compress_analytically.py` needs only the encoder: it decompresses by sampling `m` training images
  from the ideal decoder's posterior $\pi(y \mid z_{1:t}) \propto q(z_{1:t} \mid y)$ (`--posterior-mode`
  picks the $q$ used), and reports how many training images the posterior spreads over.
- **Latent space browser:** `qa_latent_space.py` serves a local page (and opens it) for walking the
  encoder's code tree one node at a time: the node and its children on the left, each with the mean image
  of its posterior and its share of the parent's images, and the node's posterior on the right, heaviest
  training images first. Every node is computed on request with the real selector, so any depth can be
  browsed; children are fetched ahead of time, so stepping down is instant. Stop it with Ctrl-C.
- **Benchmark:** `benchmark.py` scores each decoder on the data it was trained for: 8-bit MNIST, or
  the test set binarized with a fixed seed. It always loads the encoder (`--encoder`, default
  `checkpoints/encoder.pt`), which supplies the slices, and reports exact NLL, −ELBO and rate. When the code
  space is too large to enumerate (more than 2¹⁶ codes), it estimates them with the encoder: the top `--beam` codes per image under q are summed exactly, and
  `--samples` more are sampled from q outside them. That gives an unbiased estimate of p(x), reported as
  `nll_is` (an upper bound on the NLL on average), plus a guaranteed upper bound from the top codes
  alone (`nll_top_bound`) and the −ELBO, distortion and rate. The decoder runs once per distinct code.
  Use `--limit` for a quicker partial run.
- **Latent usage:** `analyze_encoder.py` reports, for each prefix length k up to its `--k-max`, how many
  bits carry information about the image and about the digit label, and how many codes are used.

## Layout

```
train_encoder.py    builds the blind knight encoder (bank partitions, SliceNet columns, selector)
dvae/knight.py      SliceNet (growing columns), SelectorNet, BlindKnightEncoder (sample, log_prob, tree)
dvae/partition.py   bank-improvement partitions (Hartigan), exact gain terms
dvae/models.py      Prior, Decoder; tree enumeration of all code prefixes
dvae/data.py        MNIST tensors, train/val split, labels, binarization
dvae/common.py      device, checkpoints, CLI flags, early-stopping loop
dvae/viz.py         image grids
```
