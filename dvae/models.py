"""Encoder q(z|x), prior p(z) and decoder p(x|z) for a discrete 1-D latent of k_max tokens over a vocab of V.

Codes are indexed as base-V numbers with z_1 as the most significant digit, so the children of
prefix node j at depth t are nodes j*V .. j*V+V-1 at depth t+1.
"""

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ("actual", "greedy", "topp")


def apply_mode(logits, mode="actual", top_p=0.9):
    """Next-token logits -> log-probs of the distribution we actually sample from.

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


class Block(nn.Module):
    """Pre-norm transformer block: causal self-attention, optional cross-attention, MLP.

    Runs either on ordinary sequences (forward) or one depth of the code tree at a time with
    the ancestors' keys/values cached (forward_tree).
    """

    def __init__(self, d_model, n_heads, cross):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.cross = cross
        if cross:
            self.ln2 = nn.LayerNorm(d_model)
            self.q_x = nn.Linear(d_model, d_model)
            self.kv_x = nn.Linear(d_model, 2 * d_model)
            self.proj_x = nn.Linear(d_model, d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def _heads(self, t):
        return t.unflatten(-1, (self.n_heads, self.d_head))

    def memory_kv(self, memory):
        """Cross-attention keys/values for image features [N, M, d] -> two [N, H, M, dh]."""
        k, v = self.kv_x(memory).chunk(2, -1)
        return self._heads(k).transpose(1, 2), self._heads(v).transpose(1, 2)

    def _cross_and_mlp(self, x, memory_kv):
        if self.cross:
            q = self._heads(self.q_x(self.ln2(x))).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, *memory_kv)
            x = x + self.proj_x(out.transpose(1, 2).flatten(-2))
        return x + self.mlp(self.ln3(x))

    def forward(self, x, memory_kv=None):
        q, k, v = (self._heads(t).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, -1))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(out.transpose(1, 2).flatten(-2))
        return self._cross_and_mlp(x, memory_kv)

    def forward_tree(self, x, cache, memory_kv=None):
        """x: [N, n, d], every node at depth t = len(cache). cache[s] = (k, v) of depth s, [N, V**s, H, dh].

        Node j's ancestor at depth s is j // V**(t-s), so nodes sharing an ancestor are contiguous and
        we can broadcast against the cache instead of gathering it.
        """
        N, n, _ = x.shape
        H, dh = self.n_heads, self.d_head
        # contiguous so the per-depth reshapes below are views, not copies autograd would keep
        q, k, v = (self._heads(t).contiguous() for t in self.qkv(self.ln1(x)).chunk(3, -1))
        scores = []
        for ks, _ in cache:
            width = ks.shape[1]
            qs = q.reshape(N, width, n // width, H, dh)
            scores.append((qs * ks.unsqueeze(2)).sum(-1).reshape(N, n, H))
        scores.append((q * k).sum(-1))
        att = (torch.stack(scores, -1) / math.sqrt(dh)).softmax(-1)  # [N, n, H, t+1]
        out = att[..., -1:] * v
        for s, (_, vs) in enumerate(cache):
            width = vs.shape[1]
            a = att[..., s].reshape(N, width, n // width, H, 1)
            out = out + (a * vs.unsqueeze(2)).reshape(N, n, H, dh)
        x = x + self.proj(out.flatten(-2))
        return self._cross_and_mlp(x, memory_kv), k, v


class TokenTransformer(nn.Module):
    """Autoregressive model over k_max tokens, optionally cross-attending to image features."""

    def __init__(self, vocab_size, k_max, d_model, n_layers, n_heads, cross):
        super().__init__()
        self.V, self.k_max, self.bos = vocab_size, k_max, vocab_size
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(k_max, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, cross) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    @property
    def block_list(self) -> list[Block]:
        return cast(list[Block], list(self.blocks))

    def memory_kvs(self, memory):
        return [b.memory_kv(memory) if memory is not None else None for b in self.block_list]

    def logits(self, prefix, memory_kvs):
        """Next-token logits after BOS and each token of prefix [N, T], T < k_max -> [N, T+1, V]."""
        bos = prefix.new_full((prefix.shape[0], 1), self.bos)
        tokens = torch.cat([bos, prefix], 1)
        x = self.tok_emb(tokens) + self.pos_emb[: tokens.shape[1]]
        for block, kv in zip(self.blocks, memory_kvs):
            x = block(x, kv)
        return self.head(self.ln_f(x))

    def log_prob(self, codes, memory=None, mode="actual", top_p=0.9):
        """Per-token log-probs of full codes [N, k_max] -> [N, k_max]."""
        logp = apply_mode(self.logits(codes[:, :-1], self.memory_kvs(memory)), mode, top_p)
        return logp.gather(-1, codes.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, n, memory=None, prefix=None, mode="actual", top_p=0.9, generator=None):
        """Complete prefix [n, T] (or start empty) to full codes [n, k_max]."""
        device = self.pos_emb.device
        codes = prefix if prefix is not None else torch.zeros(n, 0, dtype=torch.long, device=device)
        kvs = self.memory_kvs(memory)
        while codes.shape[1] < self.k_max:
            logp = apply_mode(self.logits(codes, kvs)[:, -1], mode, top_p)
            nxt = torch.multinomial(logp.exp(), 1, generator=generator)
            codes = torch.cat([codes, nxt], 1)
        return codes

    def prefix_logprobs(self, n, memory=None, mode="actual", top_p=0.9):
        """Enumerate the whole code tree.

        Returns a list of k_max+1 tensors; entry t is [n, V**t], the log-prob of every length-t prefix.
        Costs one transformer position per tree node (V**k_max - 1 internal nodes / V - 1).
        """
        device = self.pos_emb.device
        kvs = self.memory_kvs(memory)
        caches = [[] for _ in self.blocks]
        out = [torch.zeros(n, 1, device=device)]
        for t in range(self.k_max):
            width = self.V**t
            tok = torch.full((1,), self.bos, device=device) if t == 0 else torch.arange(width, device=device) % self.V
            x = (self.tok_emb(tok) + self.pos_emb[t]).expand(n, width, -1)
            for block, cache, kv in zip(self.block_list, caches, kvs):
                x, k, v = block.forward_tree(x, cache, kv)
                cache.append((k, v))
            logp = apply_mode(self.head(self.ln_f(x)), mode, top_p)  # [n, width, V]
            out.append((out[-1].unsqueeze(-1) + logp).flatten(1))
        return out


class Encoder(nn.Module):
    """q(z|x): CNN + transformer over 7x7 image patches, then an autoregressive token transformer."""

    def __init__(self, vocab_size=2, k_max=10, d_model=64, n_img_layers=2, n_tok_layers=2, n_heads=4):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, k_max=k_max, d_model=d_model, n_img_layers=n_img_layers,
                           n_tok_layers=n_tok_layers, n_heads=n_heads)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GELU(),  # 14x14
            nn.Conv2d(64, d_model, 4, stride=2, padding=1),  # 7x7
        )
        self.img_pos = nn.Parameter(torch.randn(49, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.img_tf = nn.TransformerEncoder(layer, n_img_layers, enable_nested_tensor=False)
        self.img_ln = nn.LayerNorm(d_model)
        self.tokens = TokenTransformer(vocab_size, k_max, d_model, n_tok_layers, n_heads, cross=True)

    def memory(self, x):
        """x: float [N, 1, 28, 28] in [0, 1] -> patch features [N, 49, d]."""
        h = self.stem(x - 0.5).flatten(2).transpose(1, 2) + self.img_pos
        return self.img_ln(self.img_tf(h))

    def prefix_logprobs(self, x, mode="actual", top_p=0.9):
        return self.tokens.prefix_logprobs(x.shape[0], self.memory(x), mode, top_p)

    def log_prob(self, x, codes, mode="actual", top_p=0.9):
        return self.tokens.log_prob(codes, self.memory(x), mode, top_p)

    @torch.no_grad()
    def sample(self, x, mode="actual", top_p=0.9, generator=None):
        return self.tokens.sample(x.shape[0], self.memory(x), mode=mode, top_p=top_p, generator=generator)


class Prior(nn.Module):
    """p(z): unconditional autoregressive token transformer."""

    def __init__(self, vocab_size=2, k_max=10, d_model=128, n_layers=3, n_heads=4):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, k_max=k_max, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.tokens = TokenTransformer(vocab_size, k_max, d_model, n_layers, n_heads, cross=False)

    def prefix_logprobs(self, mode="actual", top_p=0.9):
        """Log-probs of every prefix at every depth, each [V**t] (no batch dim)."""
        return [lp[0] for lp in self.tokens.prefix_logprobs(1, None, mode, top_p)]

    def log_prob(self, codes, mode="actual", top_p=0.9):
        return self.tokens.log_prob(codes, None, mode, top_p)

    @torch.no_grad()
    def sample(self, n, prefix=None, mode="actual", top_p=0.9, generator=None):
        return self.tokens.sample(n, None, prefix, mode, top_p, generator)


class Decoder(nn.Module):
    """p(x|z), factorized over pixels.

    likelihood="gray": 256-way categorical per pixel (8-bit MNIST, scored in bits/dim).
    likelihood="binary": Bernoulli per pixel (binarized MNIST, scored in nats).
    """

    def __init__(self, vocab_size=2, k_max=10, likelihood="gray", d_model=128, n_layers=2, n_heads=4, channels=128):
        super().__init__()
        assert likelihood in ("gray", "binary")
        self.config = dict(vocab_size=vocab_size, k_max=k_max, likelihood=likelihood, d_model=d_model,
                           n_layers=n_layers, n_heads=n_heads, channels=channels)
        self.likelihood = likelihood
        self.channels = channels
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(k_max, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(k_max * d_model, channels * 49)
        out_ch = 256 if likelihood == "gray" else 1
        self.up = nn.Sequential(
            nn.GELU(), nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1),  # 14x14
            nn.GELU(), nn.ConvTranspose2d(channels // 2, channels // 4, 4, stride=2, padding=1),  # 28x28
            nn.GELU(), nn.Conv2d(channels // 4, out_ch, 3, padding=1),
        )

    def forward(self, codes):
        """codes [N, k_max] -> logits [N, 784, 256] (gray) or [N, 784] (binary)."""
        h = self.tf(self.tok_emb(codes) + self.pos_emb).flatten(1)
        out = self.up(self.fc(h).view(-1, self.channels, 7, 7))
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


MODELS = {"encoder": Encoder, "prior": Prior, "decoder": Decoder}
