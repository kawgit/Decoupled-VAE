"""Image grids for human QA."""

import os

import torch.nn.functional as F
from torchvision.utils import save_image


def save_grid(images, path, nrow, scale=3):
    """images [N, 1, 28, 28] in [0, 1], row-major."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = F.interpolate(images.float().cpu(), scale_factor=scale, mode="nearest")
    save_image(images, path, nrow=nrow, padding=2, pad_value=1.0)


def code_str(code):
    return "".join(str(t) for t in code.tolist())
