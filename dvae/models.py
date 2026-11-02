"""Prior p(z) and decoder p(x|z) for codes of answers over a vocab of V (encoder: dvae/questions.py).

A code is a sequence of answers, each to the question the encoder's selector asked at that position (n_questions
means "no question", past the end of the bank: the answer is always 0). What a code says about an image is the set
of its (question, answer) pairs, whatever their order, so both models read sets: every pair has its own row
question * V + answer of a joint embedding table, no position embeddings are used, and pairs past the bank carry
nothing and are left out.

Codes are indexed as base-V numbers with z_1 as the most significant digit, so the children of
prefix node j at depth t are nodes j*V .. j*V+V-1 at depth t+1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ("actual", "greedy", "topp")
TREE_CHUNK = 4096  # prefix nodes per prior call when enumerating the code tree


def apply_mode(logits, mode="actual", top_p=0.9):
    """Next-answer logits -> log-probs of the distribution we actually sample from.

    actual: the model's distribution. greedy: point mass on the argmax. topp: nucleus-truncated and
    renormalized. Applied identically when sampling and when enumerating the tree, so the enumerated
    probabilities are exactly the sampling distribution.
    """
    logp = logits.float().log_softmax(-1)
    if mode == "actual":
        return logp
    if mode == "greedy":
        return torch.full_like(logp, -math.inf).scatter(-1, logp.argmax(-1, keepdim=True), 0.0)
    if mode == "topp":
        sorted_logp, order = logp.sort(-1, descending=True)
        sorted_p = sorted_logp.exp()
        keep_sorted = (sorted_p.cumsum(-1) - sorted_p) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        return logp.masked_fill(~keep, -math.inf).log_softmax(-1)
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


def all_prefixes(depth, vocab_size, device):
    """Every length-depth prefix, in node order -> [V**depth, depth]."""
    idx = torch.arange(vocab_size**depth, device=device)
    powers = vocab_size ** torch.arange(depth - 1, -1, -1, device=device)
    return (idx[:, None] // powers) % vocab_size


def pair_rows(answers, questions, vocab_size, n_questions):
    """Row of each (question, answer) pair in a joint table, and whether it's a real pair (not past the bank)."""
    valid = questions < n_questions
    return torch.where(valid, questions * vocab_size + answers, 0), valid


class ReadLayer(nn.Module):
    """Queries cross-attend to a set's items, then an MLP (pre-norm, residual)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.ln_q = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def _heads(self, t):
        return t.unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)

    def forward(self, x, items, mask):
        q = self._heads(self.q(self.ln_q(x)))
        k, v = (self._heads(t) for t in self.kv(items).chunk(2, -1))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(out.transpose(1, 2).flatten(-2))
        return x + self.mlp(self.ln_mlp(x))


class SetReader(nn.Module):
    """Queries read a set: n_layers rounds of cross-attention from the queries to the set's items, each followed by
    an MLP. The items never attend to each other, so the result doesn't depend on their order. A learned null item
    is always readable, so an empty set, or a query allowed to read nothing, is fine."""

    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.ln_items = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(ReadLayer(d_model, n_heads) for _ in range(n_layers))

    def forward(self, queries, items, allowed):
        """queries [N, U, d], items [N, P, d], allowed [N, U, P] (True: that query may read that item) -> [N, U, d]."""
        N, U, _ = queries.shape
        items = self.ln_items(torch.cat([self.null.expand(N, 1, -1), items], 1))
        mask = torch.cat([allowed.new_ones(N, U, 1), allowed], 2)[:, None]  # [N, 1, U, P + 1]
        x = queries
        for layer in self.layers:
            x = layer(x, items, mask)
        return x


class Prior(nn.Module):
    """p(z): the answer at each position, given the question asked there and the set of earlier pairs.

    The question at each position follows from the earlier answers through the encoder it was trained with, so
    sample() and prefix_logprobs() take that encoder. A query per position (the question asked there plus how many
    pairs came before it) reads the earlier pairs; since pairs never read each other, every position is scored in
    one pass and order never matters.
    """

    def __init__(self, vocab_size=2, k_max=10, n_questions=1, d_model=128, n_layers=3, n_heads=4):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, k_max=k_max, n_questions=n_questions, d_model=d_model,
                           n_layers=n_layers, n_heads=n_heads)
        self.V, self.k_max, self.S = vocab_size, k_max, n_questions
        self.pairs = nn.Embedding(vocab_size * n_questions, d_model)
        self.asked = nn.Embedding(n_questions + 1, d_model)  # the question being answered (n_questions: none left)
        self.count = nn.Embedding(k_max + 1, d_model)  # how many pairs came before it
        self.reader = SetReader(d_model, n_heads, n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def logits(self, answers, questions, asked, start=0):
        """Logits [N, U, V] of the answer to each question in asked [N, U]; query u reads the pairs (answers,
        questions [N, T]) before position start + u."""
        T, U, device = answers.shape[1], asked.shape[1], answers.device
        rows, valid = pair_rows(answers, questions, self.V, self.S)
        before = torch.arange(T, device=device) < (start + torch.arange(U, device=device))[:, None]  # [U, T]
        allowed = before & valid[:, None, :]  # [N, U, T]
        x = self.asked(asked) + self.count(allowed.sum(-1).clamp_max(self.k_max))
        return self.head(self.ln_f(self.reader(x, self.pairs(rows), allowed)))

    def log_prob(self, codes, questions, mode="actual", top_p=0.9):
        """Per-answer log-probs of codes [N, k] with the question asked at each position [N, k] -> [N, k]."""
        logp = apply_mode(self.logits(codes, questions, questions), mode, top_p)
        return logp.gather(-1, codes.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, n, encoder, prefix=None, mode="actual", top_p=0.9, generator=None, temperature=1.0):
        """Complete prefix [n, T] (or start empty) to full codes [n, k_max]. Logits are divided by temperature."""
        device = self.head.weight.device
        codes = prefix if prefix is not None else torch.zeros(n, 0, dtype=torch.long, device=device)
        questions = encoder.question_path(codes)
        while codes.shape[1] < self.k_max:
            nxt = encoder.next_questions(codes)
            logits = self.logits(codes, questions, nxt[:, None], start=codes.shape[1])[:, 0]
            logp = apply_mode(logits / temperature, mode, top_p)
            codes = torch.cat([codes, torch.multinomial(logp.exp(), 1, generator=generator)], 1)
            questions = torch.cat([questions, nxt[:, None]], 1)
        return codes

    @torch.no_grad()
    def prefix_logprobs(self, encoder, mode="actual", top_p=0.9):
        """Log-probs of every prefix at every depth up to k_max, each [V**t]."""
        device = self.head.weight.device
        tree = encoder.tree_questions(self.k_max, device)  # tree[t]: the question asked after each length-t prefix
        out = [torch.zeros(1, device=device)]
        for t in range(self.k_max):
            nodes = all_prefixes(t, self.V, device)
            j = torch.arange(len(nodes), device=device)
            path = torch.stack([tree[s][j // self.V ** (t - s)] for s in range(t)], 1) if t else nodes
            logp = torch.cat([apply_mode(self.logits(nodes[i:i + TREE_CHUNK], path[i:i + TREE_CHUNK],
                                                     tree[t][i:i + TREE_CHUNK, None], start=t)[:, 0], mode, top_p)
                              for i in range(0, len(nodes), TREE_CHUNK)])
            out.append((out[-1][:, None] + logp).flatten())
        return out


class Decoder(nn.Module):
    """p(x|z), factorized over pixels, from the set of a code's (question, answer) pairs.

    A set transformer (self-attention without positions) contextualizes the pairs; 49 learned queries read them
    into a 7x7 map, which transposed convs bring to 28x28. Reads any number of pairs; k_max only records the code
    length it was trained on.

    likelihood="gray": 256-way categorical per pixel (8-bit MNIST, scored in bits/dim).
    likelihood="binary": Bernoulli per pixel (binarized MNIST, scored in nats).
    """

    def __init__(self, vocab_size=2, k_max=10, n_questions=1, likelihood="gray", d_model=128, n_layers=2, n_heads=4,
                 channels=128, n_read_layers=2):
        super().__init__()
        assert likelihood in ("gray", "binary")
        self.config = dict(vocab_size=vocab_size, k_max=k_max, n_questions=n_questions, likelihood=likelihood,
                           d_model=d_model, n_layers=n_layers, n_heads=n_heads, channels=channels,
                           n_read_layers=n_read_layers)
        self.V, self.S = vocab_size, n_questions
        self.likelihood = likelihood
        self.pairs = nn.Embedding(vocab_size * n_questions, d_model)
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))  # always present, so no set is empty
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(49, d_model) * 0.02)
        self.reader = SetReader(d_model, n_heads, n_read_layers)
        self.to_map = nn.Conv2d(d_model, channels, 1)
        out_ch = 256 if likelihood == "gray" else 1
        self.up = nn.Sequential(
            nn.GELU(), nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1),  # 14x14
            nn.GELU(), nn.ConvTranspose2d(channels // 2, channels // 4, 4, stride=2, padding=1),  # 28x28
            nn.GELU(), nn.Conv2d(channels // 4, out_ch, 3, padding=1),
        )

    def forward(self, codes, questions):
        """codes [N, k] and the question each answers [N, k] -> logits [N, 784, 256] (gray) or [N, 784] (binary)."""
        N = len(codes)
        rows, valid = pair_rows(codes, questions, self.V, self.S)
        items = torch.cat([self.null.expand(N, 1, -1), self.pairs(rows)], 1)
        present = torch.cat([valid.new_ones(N, 1), valid], 1)
        h = self.tf(items, src_key_padding_mask=~present)
        z = self.reader(self.queries.expand(N, -1, -1), h, present[:, None, :].expand(-1, 49, -1))
        out = self.up(self.to_map(z.transpose(1, 2).reshape(N, -1, 7, 7)))
        if self.likelihood == "gray":
            return out.flatten(2).transpose(1, 2)
        return out.flatten(1)

    def log_prob(self, logits, target):
        """Sum over pixels of log p(target | z). target: long levels (gray) or {0,1} floats (binary)."""
        if self.likelihood == "gray":
            return logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(-1)
        return -F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)

    @torch.no_grad()
    def render(self, logits, how="mean"):
        """Images [N, 1, 28, 28] in [0, 1]: the per-pixel mean, a sample, or the per-pixel mode."""
        if self.likelihood == "gray":
            levels = torch.arange(256, device=logits.device, dtype=torch.float) / 255
            if how == "mean":
                img = logits.softmax(-1) @ levels
            elif how == "sample":
                img = levels[torch.multinomial(logits.softmax(-1).flatten(0, 1), 1).view(logits.shape[:2])]
            else:
                img = levels[logits.argmax(-1)]
        else:
            p = torch.sigmoid(logits)
            img = p if how == "mean" else torch.bernoulli(p) if how == "sample" else (p > 0.5).float()
        return img.view(-1, 1, 28, 28)


MODELS = {"prior": Prior, "decoder": Decoder}  # the encoder lives in dvae/questions.py
