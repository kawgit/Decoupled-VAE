"""MNIST, CIFAR-10 and ImageNet at 128 x 128 as in-memory uint8 tensors, with fixed train/val splits."""

import os

import torch
from torchvision.datasets import CIFAR10, MNIST

# each dataset's images: channels, height, width
SHAPES = {"mnist": (1, 28, 28), "cifar10": (3, 32, 32), "imagenet128": (3, 128, 128)}
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
    """uint8 images of dataset ("mnist": [N, 28, 28]; "cifar10": [N, 3, 32, 32] and "imagenet128": [N, 3, 128, 128],
    channels first) for split in {"train", "val", "test"} (val: a fixed 5000 of the training set; for imagenet128,
    prepare_imagenet128.py's cache: 100k of ImageNet's training set, 10k of its validation set, 10k of its test set)."""
    if dataset == "mnist":
        return _load(split, root)[0].to(device)
    if dataset == "imagenet128":
        path = os.path.join(root, "imagenet128", f"{split}.pt")
        assert os.path.exists(path), f"{path} missing: run prepare_imagenet128.py"
        return torch.load(path, weights_only=False)["images"].to(device)
    assert dataset == "cifar10", dataset
    return _load_cifar(split, root)[0].to(device)
