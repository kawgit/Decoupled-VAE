"""MNIST (and CIFAR-10) as in-memory uint8 tensors, with a fixed train/val split."""

import os

import torch
from torchvision.datasets import CIFAR10, MNIST

NUM_PIXELS = 28 * 28
SHAPES = {"mnist": (1, 28, 28), "cifar10": (3, 32, 32)}  # each dataset's images: channels, height, width
NUM_VAL = 5000
SPLIT_SEED = 0  # fixed so every stage sees the same train/val split


def get_device(name=None):
    """The named device, else the best available: CUDA, then Apple's MPS, then the CPU. With $DVAE_REQUIRE_CUDA set
    (SCC's holders set it), no CUDA is an error, not a fall back to the CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("DVAE_REQUIRE_CUDA"):
        raise RuntimeError("no CUDA device, and $DVAE_REQUIRE_CUDA is set")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _load(split, root):
    dataset = MNIST(root, train=split != "test", download=True)
    images, labels = dataset.data, dataset.targets
    if split != "test":
        perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(SPLIT_SEED))
        keep = perm[NUM_VAL:] if split == "train" else perm[:NUM_VAL]
        images, labels = images[keep], labels[keep]
    return images, labels


def _load_cifar(split, root):
    dataset = CIFAR10(root, train=split != "test", download=True)
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()  # [N, 3, 32, 32]
    labels = torch.tensor(dataset.targets)
    if split != "test":
        perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(SPLIT_SEED))
        keep = perm[NUM_VAL:] if split == "train" else perm[:NUM_VAL]
        images, labels = images[keep], labels[keep]
    return images, labels


def load_images(dataset, split, root="data", device: str | torch.device = "cpu"):
    """uint8 images of dataset ("mnist": [N, 28, 28]; "cifar10": [N, 3, 32, 32], channels first) for split in
    {"train", "val", "test"} (val: a fixed 5000 of the training set)."""
    if dataset == "mnist":
        return load_mnist(split, root, device)
    assert dataset == "cifar10", dataset
    return _load_cifar(split, root)[0].to(device)


def load_mnist(split, root="data", device: str | torch.device = "cpu"):
    """Return uint8 images [N, 28, 28] for split in {"train", "val", "test"}."""
    return _load(split, root)[0].to(device)


def load_labels(split, root="data", device: str | torch.device = "cpu"):
    """Digit labels [N] aligned with load_mnist(split)."""
    return _load(split, root)[1].to(device)


def to_float(images):
    """uint8 [N, 28, 28] -> float [N, 1, 28, 28] in [0, 1] (what the encoder sees)."""
    return images.float().div(255).unsqueeze(1)


def gray_target(images):
    """uint8 [N, 28, 28] -> long [N, 784] pixel levels in 0..255."""
    return images.long().flatten(1)


def binary_target(images, generator=None):
    """Dynamic binarization: each pixel is 1 with probability intensity / 255."""
    probs = images.float().div(255).flatten(1)
    return torch.bernoulli(probs, generator=generator)


def batches(n, batch_size, shuffle, device, generator=None, drop_last=True):
    """Yield index tensors covering range(n)."""
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    stop = n - n % batch_size if drop_last else n
    for start in range(0, stop, batch_size):
        yield order[start:start + batch_size].to(device)
