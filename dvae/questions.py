"""The question-answer encoder: a bank of question networks that see only the image, and a selector that sees only
the code.

At every position the selector picks the unasked question with the highest predicted gain from the (question,
answer) pairs so far, and the image answers it: the question's network gives a distribution over V answers. An
answer doesn't depend on the code, so asking a question twice along one code adds nothing; once every question is
asked, the answer is always 0. Built by train_encoder.py.

Codes are indexed as base-V numbers with z_1 most significant, so node j at depth t has children
j*V .. j*V+V-1 at depth t+1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import SetReader, all_prefixes, apply_mode

ACTIVATION_BUDGET = 2**26  # floats of question-network activations held at once by one forward pass


class QuestionNets(nn.Module):
    """G independent question networks, run together: each is the same small conv stack (no shared weights) ending
    in V answer logits.

    Every parameter has the networks along its first dim, so networks are added, split and fused by concatenating
    and slicing, and one pass runs them all with grouped convs and batched matmuls. Images come in streams: each
    stream feeds G / streams consecutive networks. One stream: every network answers for the same images (the
    encoder's bank). Several: each fold copy's networks see that copy's own images (training copies together).
    """

    def __init__(self, n_nets, vocab_size, width=32, hidden=64):
        super().__init__()
        self.V, self.width, self.hidden = vocab_size, width, hidden
        c1, c2, G = width // 2, width, n_nets
        shapes = dict(w1=(G, c1, 1, 3, 3), b1=(G, c1), w2=(G, c2, c1, 4, 4), b2=(G, c2), w3=(G, c2, c2, 4, 4),
                      b3=(G, c2), w4=(G, c2 * 49, hidden), b4=(G, hidden), w5=(G, hidden, vocab_size),
                      b5=(G, vocab_size))
        fan_in = dict(w1=9, b1=9, w2=c1 * 16, b2=c1 * 16, w3=c2 * 16, b3=c2 * 16, w4=c2 * 49, b4=c2 * 49,
                      w5=hidden, b5=hidden)
        for name, shape in shapes.items():  # PyTorch's default conv/linear init, network by network
            bound = 1 / math.sqrt(fan_in[name])
            setattr(self, name, nn.Parameter(torch.empty(shape).uniform_(-bound, bound)))

    w1: nn.Parameter
    b1: nn.Parameter
    w2: nn.Parameter
    b2: nn.Parameter
    w3: nn.Parameter
    b3: nn.Parameter
    w4: nn.Parameter
    b4: nn.Parameter
    w5: nn.Parameter
    b5: nn.Parameter
    NAMES = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5")

    @property
    def n_nets(self):
        return self.w1.shape[0]

    def _params(self, lo, hi):
        return [getattr(self, name)[lo:hi] for name in self.NAMES]

    def append(self, other):
        """Add other's networks after these (copies of its weights; requires_grad follows these)."""
        for name in self.NAMES:
            mine = getattr(self, name)
            joined = torch.cat([mine.detach(), getattr(other, name).detach().to(mine.device)])
            setattr(self, name, nn.Parameter(joined, requires_grad=mine.requires_grad))

    def split(self, parts):
        """parts equal groups of consecutive networks, each as its own QuestionNets (copies of the weights)."""
        size = self.n_nets // parts
        out = []
        for p in range(parts):
            nets = QuestionNets(0, self.V, self.width, self.hidden).to(self.w1.device)
            for name, value in zip(self.NAMES, self._params(p * size, (p + 1) * size)):
                setattr(nets, name, nn.Parameter(value.detach().clone()))
            out.append(nets)
        return out

    def _run(self, x, params, streams):
        """Logits [n, g, V] of g networks (params) for images x [n, streams, 28, 28]."""
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = params
        n, g = len(x), w1.shape[0]
        h = F.gelu(F.conv2d(x - 0.5, w1.flatten(0, 1), b1.flatten(), padding=1, groups=streams))
        h = F.gelu(F.conv2d(h, w2.flatten(0, 1), b2.flatten(), stride=2, padding=1, groups=g))
        h = F.gelu(F.conv2d(h, w3.flatten(0, 1), b3.flatten(), stride=2, padding=1, groups=g))
        h = F.gelu(torch.einsum("ngi,gio->ngo", h.view(n, g, -1), w4) + b4)
        return torch.einsum("ngi,gio->ngo", h, w5) + b5

    def forward(self, x, streams=1):
        """Logits of every network: x [n, 1, 28, 28] -> [n, G, V] with one stream, or x [streams, n, 1, 28, 28] ->
        [streams, n, G / streams, V]. Large calls are split by networks (one stream) and by images to stay
        within the activation budget."""
        G, c1, c2 = self.n_nets, self.width // 2, self.width
        per_net = 2 * (c1 * 784 + c2 * 196 + c2 * 49 + self.hidden)  # floats per image per network
        if streams == 1:
            x = x.view(len(x), 1, 28, 28)
            if G == 0:
                return x.new_zeros(len(x), 0, self.V)
            nets = max(1, min(G, ACTIVATION_BUDGET // (per_net * min(len(x), 256))))
            rows = max(1, ACTIVATION_BUDGET // (per_net * nets))
            return torch.cat([torch.cat([self._run(x[i:i + rows], self._params(lo, lo + nets), 1)
                                         for i in range(0, len(x), rows)])
                              for lo in range(0, G, nets)], 1)
        x = x.view(streams, -1, 28, 28).transpose(0, 1)  # [n, streams, 28, 28]
        rows = max(1, ACTIVATION_BUDGET // (per_net * G))
        out = torch.cat([self._run(x[i:i + rows], self._params(0, G), streams) for i in range(0, len(x), rows)])
        return out.view(len(x), streams, G // streams, self.V).transpose(0, 1)


class SelectorNet(nn.Module):
    """Predicts, from the set of (question, answer) pairs so far, the gain of asking each of n_questions questions
    next. A learned query reads the pairs (dvae.models.SetReader), so their order doesn't matter."""

    def __init__(self, vocab_size, n_questions, width=128, n_layers=2, n_heads=4):
        super().__init__()
        self.V = vocab_size
        self.pairs = nn.Embedding(vocab_size * n_questions, width)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.reader = SetReader(width, n_heads, n_layers)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_questions))

    def forward(self, prefix, used):
        """prefix [n, t] answers to the questions used [n, t] -> predicted gain per question [n, n_questions]."""
        n, t = prefix.shape
        allowed = torch.ones(n, 1, t, dtype=torch.bool, device=prefix.device)
        h = self.reader(self.query.expand(n, -1, -1), self.pairs(used * self.V + prefix), allowed)
        return self.mlp(h[:, 0])

    def pick(self, prefix, used):
        """The unasked question with the highest predicted gain after each prefix [n, t] (used [n, t]: the
        questions asked along it) -> [n]."""
        scores = self(prefix, used)
        if used.shape[1]:
            scores = scores.scatter(1, used, -torch.inf)
        return scores.argmax(1)


def commit_picks(commits, prefix, picks):
    """Override picks [n] for the prefixes [n, t] that have a committed question (commits: code tuple -> question)."""
    t = prefix.shape[1]
    if not commits or all(len(code) != t for code in commits):
        return picks
    fixed = [(i, commits[key]) for i, key in enumerate(map(tuple, prefix.tolist())) if key in commits]
    if not fixed:
        return picks
    rows, questions = zip(*fixed)
    picks = picks.clone()
    picks[list(rows)] = torch.tensor(questions, dtype=picks.dtype, device=picks.device)
    return picks


def selector_pick(selector, commits, prefix, used):
    """The question asked after each prefix [n, t] (used [n, t]: questions already asked along it): the committed
    question where there is one, else the selector's pick -> [n]."""
    return commit_picks(commits, prefix, selector.pick(prefix, used))


def question_path(selector, commits, codes):
    """The question asked at every position of each code [n, t] -> [n, t]."""
    used = codes.new_zeros(len(codes), 0)
    for u in range(codes.shape[1]):
        used = torch.cat([used, selector_pick(selector, commits, codes[:, :u], used)[:, None]], 1)
    return used


def with_none(t):
    """Append the "no question" row to every image's per-question logits or log-probs [n, S, V] -> [n, S + 1, V]:
    once the bank is used up, the answer is always 0."""
    none = torch.full_like(t[:, :1], -torch.inf)
    none[..., 0] = 0.0
    return torch.cat([t, none], 1)


class QuestionAnswerEncoder(nn.Module):
    """q(z|x): the selector picks a question from the code so far, the image answers it. Once every question is
    asked, the answer is always 0. Codes have no fixed length: callers choose it.

    Question indices run over 0..S-1, and S stands for "no question" past the end of the bank. commits: [code,
    question] pairs, the nodes whose question is fixed (the one built for them); the selector picks everywhere else.
    """

    def __init__(self, vocab_size=2, n_questions=0, width=32, hidden=64, selector_width=128, selector_layers=2,
                 selector_heads=4, commits=()):
        super().__init__()
        commits = [[list(code), int(h)] for code, h in commits]
        self.config = dict(vocab_size=vocab_size, n_questions=n_questions, width=width, hidden=hidden,
                           selector_width=selector_width, selector_layers=selector_layers,
                           selector_heads=selector_heads, commits=commits)
        self.commits = {tuple(code): h for code, h in commits}
        self.V = vocab_size
        self.questions = QuestionNets(n_questions, vocab_size, width, hidden)
        self.selector = SelectorNet(vocab_size, n_questions, selector_width, selector_layers, selector_heads)

    @property
    def n_questions(self):
        return self.questions.n_nets

    def features(self, x):
        """Every question's answer logits for every image [n, S, V]; shared by all prefixes."""
        return self.questions(x)

    def answers(self, x, mode="actual", top_p=0.9):
        """Log-probs of every question's answer for every image, plus the "no question" row [n, S + 1, V]."""
        return with_none(apply_mode(self.features(x), mode, top_p))

    def pick(self, prefix, used):
        """The question asked after each prefix [n, t], within the bank (used [n, t]: questions asked along it)."""
        return selector_pick(self.selector, self.commits, prefix, used)

    def question_path(self, codes):
        """The question asked at every position of each code [n, t] -> [n, t] (S past the end of the bank)."""
        S, t = self.n_questions, codes.shape[1]
        path = question_path(self.selector, self.commits, codes[:, :S])
        return torch.cat([path, path.new_full((len(codes), max(t - S, 0)), S)], 1)

    def next_questions(self, prefix):
        """The question asked after each prefix [n, t] -> [n] (S once the bank is used up)."""
        if prefix.shape[1] >= self.n_questions:
            return prefix.new_full((len(prefix),), self.n_questions)
        return self.pick(prefix, question_path(self.selector, self.commits, prefix))

    def tree_questions(self, depth, device, chunk=65536):
        """The question asked after every prefix of the code tree: entry t is [V**t], for t below depth."""
        out, used = [], torch.zeros(1, 0, dtype=torch.long, device=device)
        for t in range(depth):
            if t >= self.n_questions:
                out.append(torch.full((self.V**t,), self.n_questions, dtype=torch.long, device=device))
                continue
            prefixes = all_prefixes(t, self.V, device)
            nxt = torch.cat([self.pick(prefixes[i:i + chunk], used[i:i + chunk])
                             for i in range(0, len(prefixes), chunk)])
            out.append(nxt)
            used = torch.cat([used, nxt[:, None]], 1).repeat_interleave(self.V, 0)
        return out

    def stage_logits(self, t, feats, prefix):
        """Logits of the question asked after each prefix [n, t] -> [n, V]; feats from features()."""
        return with_none(feats)[torch.arange(len(feats), device=feats.device), self.next_questions(prefix)]

    def log_prob(self, x, codes, mode="actual", top_p=0.9):
        """Per-answer log-probs of codes [N, t] -> [N, t]."""
        rows = torch.arange(len(x), device=x.device)[:, None]
        return self.answers(x, mode, top_p)[rows, self.question_path(codes), codes]

    @torch.no_grad()
    def sample(self, x, length, mode="actual", top_p=0.9, generator=None):
        """Codes [N, length] for images x."""
        logp = self.answers(x, mode, top_p)
        rows = torch.arange(len(x), device=x.device)
        codes = torch.zeros(len(x), 0, dtype=torch.long, device=x.device)
        used = codes.clone()
        for u in range(length):
            h = self.pick(codes, used) if u < self.n_questions else codes.new_full((len(x),), self.n_questions)
            nxt = torch.multinomial(logp[rows, h].exp(), 1, generator=generator)
            codes, used = torch.cat([codes, nxt], 1), torch.cat([used, h[:, None]], 1)
        return codes

    def prefix_logprobs(self, x, depth, mode="actual", top_p=0.9):
        """Enumerate the code tree: entry t is [N, V**t], the log-prob of every length-t prefix, for t up to depth."""
        logp = self.answers(x, mode, top_p)  # [N, S + 1, V]
        out = [torch.zeros(len(x), 1, device=x.device)]
        for nxt in self.tree_questions(depth, x.device):
            out.append((out[-1][:, :, None] + logp[:, nxt]).flatten(1))
        return out
