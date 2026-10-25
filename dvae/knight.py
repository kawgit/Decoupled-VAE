"""The blind knight encoder: a bank of slices that see only the image, and a selector that sees only the code.

At every position the selector picks the unused slice with the highest predicted gain from the code so far,
and the image answers it with a token from that slice. A slice's answer doesn't depend on the code, so
asking it twice along one code adds nothing; a code ends when every slice is used. Built by train_encoder.py.

Codes are indexed as base-V numbers with z_1 most significant, so node j at depth t has children
j*V .. j*V+V-1 at depth t+1.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import apply_mode

ACTIVATION_BUDGET = 2**26  # floats of column activations held at once by one SliceNet forward pass


def all_prefixes(depth, vocab_size, device):
    """Every length-depth prefix, in node order -> [V**depth, depth]."""
    idx = torch.arange(vocab_size**depth, device=device)
    powers = vocab_size ** torch.arange(depth - 1, -1, -1, device=device)
    return (idx[:, None] // powers) % vocab_size


class Column(nn.Module):
    """One SliceNet column: a small conv stack with n_slices heads of V logits.

    Each layer after the first also reads a learned projection of every earlier column's previous layer
    (progressive-network laterals). Earlier columns are frozen by the caller, so adding a column never
    changes an earlier slice.
    """

    def __init__(self, vocab_size, n_slices, n_prev, width=32, hidden=64):
        super().__init__()
        self.V, self.n_slices = vocab_size, n_slices
        c1, c2 = width // 2, width
        self.layers = nn.ModuleList([
            nn.Conv2d(1, c1, 3, padding=1),                   # -> c1 x 28 x 28
            nn.Conv2d(c1, c2, 4, stride=2, padding=1),        # -> c2 x 14 x 14
            nn.Conv2d(c2, c2, 4, stride=2, padding=1),        # -> c2 x 7 x 7
            nn.Linear(c2 * 49, hidden),                       # -> hidden
            nn.Linear(hidden, n_slices * vocab_size),         # -> logits
        ])
        # lateral l feeds layer l + 1 from every earlier column's layer l (1x1 convs on the feature maps)
        ins = [c1, c2, c2, hidden]
        self.laterals = nn.ModuleList(
            (nn.Conv2d(n_prev * c, c, 1) if i < 3 else nn.Linear(n_prev * c, c)) if n_prev else nn.Identity()
            for i, c in enumerate(ins))
        self.n_prev = n_prev

    def forward(self, x, prev):
        """x [n, 1, 28, 28]; prev: every earlier column's activations, one list of 4 per column.
        -> (logits [n, n_slices, V], this column's 4 activations)."""
        acts = []
        h = x - 0.5
        for i, layer in enumerate(self.layers):
            if i > 0 and self.n_prev:
                h = h + self.laterals[i - 1](torch.cat([p[i - 1] for p in prev], 1))
            if i == 3:
                h = h.flatten(1)
            h = layer(h)
            if i < 4:
                h = F.gelu(h)
                acts.append(h)
        return h.view(len(x), self.n_slices, self.V), acts


class SliceNet(nn.Module):
    """Image -> logits for every slice [n, S, V], from a list of columns (configs: one dict per column)."""

    def __init__(self, vocab_size, columns=(), width=32, hidden=64):
        super().__init__()
        self.V, self.width, self.hidden = vocab_size, width, hidden
        self.columns = nn.ModuleList()
        for cfg in columns:
            self.add_column(cfg["n_slices"])

    def add_column(self, n_slices, device=None):
        column = Column(self.V, n_slices, len(self.columns), self.width, self.hidden).to(device)
        self.columns.append(column)
        return column

    @property
    def n_slices(self):
        return sum(cast(Column, c).n_slices for c in self.columns)

    @property
    def config(self):
        return [{"n_slices": cast(Column, c).n_slices} for c in self.columns]

    def forward(self, x):
        if not len(self.columns):
            return x.new_zeros(len(x), 0, self.V)
        # every column keeps its activations for the later columns' laterals, so memory grows with the
        # number of columns: split the batch to stay within the budget
        c1, c2 = self.width // 2, self.width
        per_image = len(self.columns) * 2 * (c1 * 784 + c2 * 196 + c2 * 49 + self.hidden)
        rows = max(1, ACTIVATION_BUDGET // per_image)
        if len(x) > rows:
            return torch.cat([self(x[i:i + rows]) for i in range(0, len(x), rows)])
        prev, out = [], []
        for column in self.columns:
            logits, acts = column(x, prev)
            prev.append(acts)
            out.append(logits)
        return torch.cat(out, 1)


class SelectorNet(nn.Module):
    """Predicts, from a prefix of any length, the gain of each of n_slices slices there (GRU over the tokens)."""

    def __init__(self, vocab_size, n_slices, width=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(width))
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_slices))

    @property
    def out(self) -> nn.Linear:
        return cast(nn.Linear, self.mlp[-1])

    def forward(self, prefix):
        """prefix [n, t] -> predicted gain per slice [n, n_slices]."""
        n = prefix.shape[0]
        h = self.h0.expand(n, -1)
        if prefix.shape[1] > 0:
            _, h = self.gru(self.emb(prefix), h[None].contiguous())
            h = h[0]
        return self.mlp(h)

    def pick(self, prefix, used):
        """The unused slice with the highest predicted gain after each prefix [n, t]; used [n, t] holds the
        slices already asked along it -> [n]."""
        scores = self(prefix)
        if used.shape[1]:
            scores = scores.scatter(1, used, -torch.inf)
        return scores.argmax(1)

    def path(self, codes):
        """The slice asked at every position of each code [n, t] -> [n, t]."""
        used = codes.new_zeros(len(codes), 0)
        for u in range(codes.shape[1]):
            used = torch.cat([used, self.pick(codes[:, :u], used)[:, None]], 1)
        return used


class BlindKnightEncoder(nn.Module):
    """q(z|x): the selector picks a slice from the code so far, the image answers it. Codes are k_max long.

    columns: one dict per SliceNet column (n_slices). Same interface the other scripts use.
    """

    def __init__(self, vocab_size=2, k_max=0, columns=(), width=32, hidden=64, selector_width=128):
        super().__init__()
        columns = list(columns)
        self.config = dict(vocab_size=vocab_size, k_max=k_max, columns=columns, width=width, hidden=hidden,
                           selector_width=selector_width)
        self.V, self.k_max = vocab_size, k_max
        self.slices = SliceNet(vocab_size, columns, width, hidden)
        self.selector = SelectorNet(vocab_size, self.slices.n_slices, selector_width)
        assert k_max <= self.slices.n_slices, "a code can't be longer than the bank"

    def features(self, x):
        """Every slice's logits for every image [n, S, V]; shared by all prefixes."""
        return self.slices(x)

    def stage_logits(self, t, feats, prefix):
        """Logits of the slice asked after each prefix [n, t] -> [n, V]."""
        nxt = self.selector.pick(prefix, self.selector.path(prefix))
        return feats[torch.arange(len(feats), device=feats.device), nxt]

    def log_prob(self, x, codes, mode="actual", top_p=0.9):
        """Per-token log-probs of codes [N, k_max] -> [N, k_max]."""
        logp = apply_mode(self.features(x), mode, top_p)  # [N, S, V]
        path = self.selector.path(codes)
        rows = torch.arange(len(x), device=x.device)[:, None]
        return logp[rows, path, codes]

    @torch.no_grad()
    def sample(self, x, mode="actual", top_p=0.9, generator=None):
        logp = apply_mode(self.features(x), mode, top_p)
        rows = torch.arange(len(x), device=x.device)
        codes = torch.zeros(len(x), 0, dtype=torch.long, device=x.device)
        used = codes.clone()
        for _ in range(self.k_max):
            h = self.selector.pick(codes, used)
            nxt = torch.multinomial(logp[rows, h].exp(), 1, generator=generator)
            codes, used = torch.cat([codes, nxt], 1), torch.cat([used, h[:, None]], 1)
        return codes

    def prefix_logprobs(self, x, mode="actual", top_p=0.9, chunk=65536, depth=None):
        """Enumerate the code tree: entry t is [N, V**t], the log-prob of every length-t prefix, for t up to
        depth (default k_max)."""
        logp = apply_mode(self.features(x), mode, top_p)  # [N, S, V]
        out = [torch.zeros(len(x), 1, device=x.device)]
        used = torch.zeros(1, 0, dtype=torch.long, device=x.device)  # slices asked along every node so far
        for t in range(self.k_max if depth is None else depth):
            prefixes = all_prefixes(t, self.V, x.device)
            nxt = torch.cat([self.selector.pick(prefixes[i:i + chunk], used[i:i + chunk])
                             for i in range(0, len(prefixes), chunk)])
            out.append((out[-1][:, :, None] + logp[:, nxt]).flatten(1))
            used = torch.cat([used, nxt[:, None]], 1).repeat_interleave(self.V, 0)
        return out
