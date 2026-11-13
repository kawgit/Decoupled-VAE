"""Every image of ImageNet's training set (benjamin-paine/imagenet-1k-128x128's train parquet files, which
prepare_imagenet128.py's caches draw 100k of) through a train_outer.py run's encoder, standardized as that run's
latents.pt is (its per-channel mean and std, times its scale): the pool train_inner_vway.py --pool draws its rolling
buffer of training images from. Written to <run>/latents_train_full.pt: train [N, C, h, w] (float16, in the files'
order, which is sorted by class), labels [N], and the run's mean, std, scale and shape.

JPEGs are decoded by --workers processes while the GPU encodes. ~10.5 GB for the 16 x 16 x 16 latents.

Run from the repo: python encode_imagenet128_train.py --outer artifacts/<outer job>/outer --bf16
"""

import argparse
import glob
import io
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from dvae.data import get_device
from dvae.outer import load_outer


def decode(blob: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(blob)).convert("RGB")
    if image.size != (128, 128):
        image = image.resize((128, 128), Image.Resampling.BICUBIC)
    return np.asarray(image).transpose(2, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outer", required=True, metavar="RUN", help="a train_outer.py run's folder (its latents.pt)")
    p.add_argument("--raw", default="data/imagenet128/raw")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=None, help="only the first this many images (a test)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = get_device(args.device)
    model, lat = load_outer(args.outer, device)
    C, h, w = lat["shape"]
    mean, std = lat["mean"].to(device)[None, :, None, None], lat["std"].to(device)[None, :, None, None]
    files = sorted(glob.glob(os.path.join(args.raw, "train-*.parquet")))
    assert files, f"no train files in {args.raw}"
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if args.limit is not None:
        total = min(total, args.limit)
    out = torch.empty(total, C, h, w, dtype=torch.half)
    labels = torch.full((total,), -1, dtype=torch.long)
    print(f"{time.strftime('%H:%M:%S')} {total} images from {len(files)} files -> {C} x {h} x {w} latents "
          f"({out.numel() * 2 / 2**30:.1f} GB); {args.workers} decoding workers", flush=True)
    began, done = time.time(), 0
    with Pool(args.workers) as pool:
        for f in files:
            for part in pq.ParquetFile(f).iter_batches(batch_size=args.batch, columns=["image", "label"]):
                if done >= total:
                    break
                blobs = [cell["bytes"] for cell in part.column("image").to_pylist()][:total - done]
                images = np.stack(pool.map(decode, blobs, chunksize=max(1, len(blobs) // (4 * args.workers))))
                x = torch.from_numpy(images).to(device).float().add_(0.5).div_(256)  # (dvae.deltatree.pixels)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    z = model.encoder(x).float()
                out[done:done + len(z)] = ((z - mean) / std * lat["scale"]).half().cpu()
                labels[done:done + len(z)] = torch.tensor(part.column("label").to_pylist()[:len(z)])
                done += len(z)
                if done % (100 * args.batch) < args.batch or done == total:
                    rate = done / (time.time() - began)
                    print(f"{time.strftime('%H:%M:%S')} {done} of {total} ({os.path.basename(f)}); {rate:.0f} "
                          f"images/s, {(total - done) / rate / 60:.0f} min left", flush=True)
            if done >= total:
                break
    # (the check: prepare_imagenet128.py's training cache names each of its images' rows in these files, and the
    # run's latents.pt holds their latents; the same encoder should give the same numbers, up to bfloat16)
    cache = os.path.join(os.path.dirname(args.raw.rstrip("/")), "train.pt")
    if os.path.exists(cache) and len(lat["train"]):
        rows = torch.as_tensor(torch.load(cache, map_location="cpu", weights_only=False, mmap=True)["rows"]).long()
        here = rows < done
        a, b = out[rows[here]].float(), lat["train"][:len(rows)][here].float()
        print(f"check against latents.pt: {int(here.sum())} images, rms difference "
              f"{(a - b).square().mean().sqrt():.2e} (latent rms {b.square().mean().sqrt():.2e})", flush=True)
    path = os.path.join(args.outer, "latents_train_full.pt")
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(dict(train=out, labels=labels, channels=C, shape=(C, h, w), mean=lat["mean"], std=lat["std"],
                    scale=lat["scale"], dataset=lat["dataset"], source=[os.path.basename(f) for f in files]), tmp)
    os.replace(tmp, path)
    print(f"{time.strftime('%H:%M:%S')} {done} latents -> {path} ({os.path.getsize(path) / 2**30:.1f} GB); "
          f"{(time.time() - began) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
