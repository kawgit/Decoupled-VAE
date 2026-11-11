"""What train_encoder.py keeps on disk between jobs, in three parts beside its --out:

- model.pt: what using the encoder needs (its network, the data mean, Lloyd steps per depth, the run's args and
  metrics); dump_encoder.py, qa_encoder.py and the like read only this.
- state.pt: what training on needs besides (the optimizer, the step and threshold schedules' state, the cost
  model, the gradient budget's statistics, the tracked starts), saved with each model.pt.
- grams.bin: the running Gram matrices of the tracked starts, raw float32 (their shape in state.pt): gigabytes
  for large images at many depths, so a file of their own rather than part of a pickle.

All three are written to a temporary name and renamed into place (grams.bin first, model.pt last), so a job
killed mid-save leaves the last complete set. load_checkpoint reads them back as one dict, as a single-file
checkpoint from before the split also reads.

And PrefixCache, --cache-depths' table of every training image's path, as memory-mapped .npy files in a
folder (np.load(path, mmap_mode="r") opens any of them; view_cache.py draws images' cached paths), so the
operating system's page cache holds what fits in memory and the table outlives the job.
"""

import atexit
import json
import os
import signal
import socket
import sys
import time

import numpy as np
import torch
from torch import Tensor

from dvae.deltatree import Frontier

MODEL_KEYS = ("args", "mean", "epoch", "step", "metrics", "best", "shared", "caps")


def _atomic(path: str, write):
    tmp = f"{path}.tmp{os.getpid()}"
    write(tmp)
    os.replace(tmp, path)


def save_checkpoint(out: str, ckpt: dict, grams: Tensor | None):
    """ckpt (everything train_encoder.py resumes from) split into out (model.pt: MODEL_KEYS), state.pt and
    grams.bin beside it."""
    folder = os.path.dirname(out) or "."
    os.makedirs(folder, exist_ok=True)
    state = {k: v for k, v in ckpt.items() if k not in MODEL_KEYS}
    state["epoch"] = ckpt["epoch"]  # (pairs it with its model.pt)
    state["grams_shape"] = None if grams is None else list(grams.shape)
    if grams is not None:
        array = grams.detach().float().cpu().numpy()
        _atomic(os.path.join(folder, "grams.bin"), lambda tmp: array.tofile(tmp))
    _atomic(os.path.join(folder, "state.pt"), lambda tmp: torch.save(state, tmp))
    _atomic(out, lambda tmp: torch.save({k: ckpt[k] for k in MODEL_KEYS if k in ckpt}, tmp))


def load_checkpoint(path: str, device, log=print) -> dict:
    """A checkpoint as one dict: path's model.pt with its folder's state.pt (when it belongs to the same epoch)
    and grams.bin (as "grams") merged in; a single-file checkpoint as it is."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    folder = os.path.dirname(path) or "."
    state_path = os.path.join(folder, "state.pt")
    if "optimizer" in ckpt or not os.path.exists(state_path):
        return ckpt  # (a single-file checkpoint, or a model alone)
    state = torch.load(state_path, map_location=device, weights_only=False)
    if state.get("epoch") != ckpt.get("epoch"):
        log(f"{state_path} is from epoch {state.get('epoch')}, {path} from {ckpt.get('epoch')}: training state "
            "not resumed")
        return ckpt
    shape = state.pop("grams_shape", None)
    grams_path = os.path.join(folder, "grams.bin")
    if shape is not None and os.path.exists(grams_path):
        state["grams"] = torch.from_numpy(np.fromfile(grams_path, dtype=np.float32).reshape(shape))
    state.pop("epoch")
    return {**ckpt, **state}


_OPEN: list["PrefixCache"] = []  # caches holding a lock in this process


def _on_term(signum, frame):
    """Free every cache's lock at once (a holder's cancel may kill the job before a long flush ends; the rows
    are written already, a flush only hastens them to disk), then exit so atexit handlers run and flush."""
    for cache in _OPEN:
        cache.unlock()
    sys.exit(128 + signum)


class PrefixCache:
    """--cache-depths: each training image's path through the shallowest depths (all of them with no cap) as last
    trained: its node's key at every depth to there, and the delta, side and m of every split on the way, so a
    batch can start at a deeper depth from those instead of recomputing the ones above it. The images a batch
    puts in one node take on one history: its member written last (the most up to date), whose prefix the others
    then copy.

    With a folder, every array is a memory-mapped file there (one set per depth, added as the paths first reach
    it), so the table persists across jobs and the page cache keeps what fits in memory; a lock file (its time
    renewed each batch) keeps a second job from writing it at once, and a batch's images are marked uncached while
    their rows change, so a job killed mid-write leaves no half-written path."""

    STALE = 600  # seconds after which an unrenewed lock is taken to be a dead job's

    def __init__(self, N: int, depths: float, D: int, folder: str | None = None, log=print):
        self.N, self.D, self.depths, self.folder = N, D, depths, folder
        self.maps: list[np.memmap] = []
        self.deltas: list[Tensor] = []  # per depth [N, D] half
        self.sides: list[Tensor] = []  # per depth [N]
        self.ms: list[Tensor] = []  # per depth [N]
        self.keys: list[Tensor] = []  # per depth from the root's [N] int64
        self.rep = torch.zeros(0, dtype=torch.long)
        self.lock: str | None = None  # (the lock file this job holds)
        if folder is None:
            self.keys.append(torch.zeros(N, dtype=torch.int64))
            self.valid = torch.zeros(N, dtype=torch.long)  # depths cached for each image
            self.stamp = torch.zeros(N, dtype=torch.long)  # the batch that wrote it last
            self.clock = 0
            return
        os.makedirs(folder, exist_ok=True)
        self._lock()
        meta_path = os.path.join(folder, "meta.json")
        fresh = not os.path.exists(meta_path)
        if not fresh:
            with open(meta_path) as f:
                meta = json.load(f)
            assert (meta["N"], meta["D"]) == (N, D), \
                f"the cache in {folder} holds {meta['N']} images of {meta['D']} pixels, not {N} of {D}"
        else:
            with open(meta_path, "w") as f:
                json.dump(dict(N=N, D=D), f)
        self.valid = self._map("valid", (N,), np.int64, fresh)
        self.stamp = self._map("stamp", (N,), np.int64, fresh)
        self.keys.append(self._map("key_0", (N,), np.int64, fresh))
        while os.path.exists(os.path.join(folder, f"delta_{len(self.deltas)}.npy")):
            self._add_depth(fresh=False)
        self.valid.clamp_(max=len(self.deltas))  # (a depth whose files never came to be)
        self.clock = int(self.stamp.max()) + 1
        log(f"prefix cache {folder}: {len(self.deltas)} depths on disk, "
            f"{int((self.valid > 0).sum())} of {N} images cached (to depth {int(self.valid.max())} at most)")
        if not fresh:
            began, size = time.time(), self.warm()
            log(f"prefix cache read into the page cache: {size / 1e9:.1f} GB in {time.time() - began:.0f}s")
        _OPEN.append(self)
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, _on_term)

    def _lock(self):
        assert self.folder is not None
        path = os.path.join(self.folder, "lock")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < self.STALE:
            with open(path) as f:
                raise RuntimeError(f"the cache in {self.folder} is in use ({f.read().strip()}); pass another "
                                   f"--cache-dir, or remove {path} if that job is dead")
        with open(path, "w") as f:
            f.write(f"{socket.gethostname()} pid {os.getpid()} job {os.environ.get('JOB_ID', '-')}")
        self.lock = path

    def _map(self, name: str, shape: tuple, dtype, fresh: bool) -> Tensor:
        assert self.folder is not None
        path = os.path.join(self.folder, f"{name}.npy")
        if fresh or not os.path.exists(path):
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        else:
            array = np.load(path, mmap_mode="r+")
            assert array.shape == shape and array.dtype == dtype, f"{path}: {array.shape} {array.dtype}"
        self.maps.append(array)
        return torch.from_numpy(array)

    def _add_depth(self, fresh=True):
        d, N = len(self.deltas), self.N
        if self.folder is None:
            self.deltas.append(torch.zeros(N, self.D, dtype=torch.half))
            self.sides.append(torch.zeros(N))
            self.ms.append(torch.zeros(N))
            self.keys.append(torch.zeros(N, dtype=torch.int64))
            return
        self.deltas.append(self._map(f"delta_{d}", (N, self.D), np.float16, fresh))
        self.sides.append(self._map(f"side_{d}", (N,), np.float32, fresh))
        self.ms.append(self._map(f"m_{d}", (N,), np.float32, fresh))
        self.keys.append(self._map(f"key_{d + 1}", (N,), np.int64, fresh))

    def warm(self) -> int:
        """Read every file of the cache once, in order, so the operating system's page cache holds it: a batch
        reads its images' rows at random, and a cold file on a network disk answers each such read slowly (a
        resumed job ran at 0% GPU for minutes). Returns the bytes read."""
        assert self.folder is not None
        size, buffer = 0, bytearray(1 << 24)
        for name in sorted(os.listdir(self.folder)):
            if name.endswith(".npy"):
                with open(os.path.join(self.folder, name), "rb", buffering=0) as f:
                    while n := f.readinto(buffer):
                        size += n
        return size

    def flush(self):
        for array in self.maps:
            array.flush()

    def unlock(self):
        lock, self.lock = self.lock, None
        if lock is not None and os.path.exists(lock):
            os.remove(lock)

    def touch(self):
        """Renew the lock's time (still in use), for a job that goes long between writes."""
        if self.lock is not None:
            os.utime(self.lock)

    def close(self):
        """Flush to disk and free the lock (at exit, a SIGTERM included)."""
        if self.folder is None:
            return
        self.flush()
        self.unlock()

    def frontier(self, idx: Tensor, X: Tensor, mean: Tensor, k: int) -> Frontier:
        """The batch idx [B] (CPU), images X [B, D], at depth k <= every one's valid depths, resident on X's
        device; also records each node's representative (into idx) for write."""
        bits, node = torch.unique(self.keys[k][idx], return_inverse=True)
        B = len(idx)
        score = self.stamp[idx] * B + torch.arange(B)
        best = torch.full((len(bits),), -1, dtype=torch.long).scatter_reduce(0, node, score, "amax")
        self.rep = rep = idx[best % B]
        device = X.device
        deltas = torch.stack([d[rep] for d in self.deltas[:k]], 1).to(device)
        R = X - mean - deltas.float().sum(1)[node.to(device)]  # (summed in single precision)
        return Frontier(R=R, node=node.to(device), bits=bits.to(device), deltas=deltas,
                        sides=torch.stack([d[rep] for d in self.sides[:k]], 1).to(device),
                        ms=torch.stack([d[rep] for d in self.ms[:k]], 1).to(device))

    def write(self, idx: Tensor, k: int, node: Tensor | None, rows: list[tuple[Tensor, Tensor, Tensor, Tensor]]):
        """The batch idx [B] after training from depth k: each image's prefix to k from its node's
        representative (node [B]: its node at k; None when k = 0), then rows[j] = (delta [B, D], side [B], m [B],
        key [B]) of its split at depth k + j, for the cached depths."""
        self.valid[idx] = 0  # (uncached while their rows change)
        if k > 0:
            assert node is not None
            src = self.rep[node.cpu()]
            for j in range(k):
                self.deltas[j][idx] = self.deltas[j][src]
                self.sides[j][idx], self.ms[j][idx] = self.sides[j][src], self.ms[j][src]
                self.keys[j + 1][idx] = self.keys[j + 1][src]
        for j, (delta, side, m, key) in enumerate(rows):
            d = k + j
            if d == len(self.deltas):  # (a depth the paths reach for the first time)
                self._add_depth()
            self.deltas[d][idx] = delta.half().cpu()
            self.sides[d][idx], self.ms[d][idx] = side.float().cpu(), m.float().cpu()
            self.keys[d + 1][idx] = key.cpu()
        self.stamp[idx] = self.clock
        self.clock += 1
        self.valid[idx] = k + len(rows)
        if self.lock is not None:
            os.utime(self.lock)  # (still in use)

    def write_row(self, idx: Tensor, d: int, delta: Tensor, side: Tensor, m: Tensor, key: Tensor):
        """Depth d's split (delta [B, D], side [B], m [B], key [B]) for the images idx [B] (CPU), whose cached
        rows above d are current (train_waves.py's refresh): their deeper rows
        no longer follow from it, so they count as cached to d + 1, no deeper."""
        self.valid[idx] = torch.clamp(self.valid[idx], max=d)  # (uncached past d while the row changes)
        if d == len(self.deltas):
            self._add_depth()
        self.deltas[d][idx] = delta.half().cpu()
        self.sides[d][idx], self.ms[d][idx] = side.float().cpu(), m.float().cpu()
        self.keys[d + 1][idx] = key.cpu()
        self.stamp[idx] = self.clock
        self.clock += 1
        self.valid[idx] = d + 1
        if self.lock is not None:
            os.utime(self.lock)
