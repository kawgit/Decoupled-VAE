"""Sample EMA diffusion checkpoints and optionally compute Clean-FID against an explicit CIFAR-10 split.

Also accepts --images DIR to score tree samples with exactly the same reference/preprocessing protocol.
FID here uses real split images, not published 50k training statistics; report the reference and both counts.
Optional dependency: python -m pip install clean-fid . Inception weights may download on first use.
"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import time

from PIL import Image
import torch
from torchvision.utils import save_image

from dvae.data import get_device, load_images
from dvae.diffusion import Diffusion, UNet


def save_individual(images, folder, offset=0):
    values = images.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for i, value in enumerate(values):
        Image.fromarray(value).save(folder / f"{offset + i:06d}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", help="best.pt, latest.pt or final.pt (EMA weights are used)")
    source.add_argument("--images", type=Path, help="existing individual 32x32 RGB PNGs, e.g. tree samples; no grids")
    p.add_argument("--out", type=Path, required=True, help="new evaluation directory; never overwrites results")
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    p.add_argument("--sampling-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    p.add_argument("--device", default=None)
    p.add_argument("--fid", action="store_true", help="requires optional clean-fid and downloaded Inception weights")
    p.add_argument("--sample-metadata", type=Path, help="optional provenance JSON for an existing --images folder")
    p.add_argument("--reference", choices=("val", "test"), default="test")
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    if args.count < 2 or args.batch < 1:
        p.error("count must be >= 2 and batch >= 1")
    if args.images and not args.fid:
        p.error("--images requires --fid")
    fid = None
    if args.fid:
        try:
            fid = importlib.import_module("cleanfid.fid")
        except ImportError:
            p.error("install optional metric dependency first: python -m pip install clean-fid")
        # Fail before expensive generation if the optional metric stack is incompatible.
        numpy = importlib.import_module("numpy")
        try:
            sanity = fid.frechet_distance(numpy.zeros(2), numpy.eye(2), numpy.zeros(2), numpy.eye(2))
        except TypeError as error:
            p.error(f"Clean-FID/SciPy compatibility check failed: {error}; use compatible metric versions")
        if abs(sanity) > 1e-6:
            p.error("FID self-distance check failed")
    if args.out.exists():
        p.error("--out already exists; choose a new evaluation name")
    device = get_device(args.device)
    if args.precision != "fp32" and device.type != "cuda" and args.ckpt:
        p.error("use --precision fp32 on CPU/MPS")
    if args.ckpt and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("GPU does not support bf16; use fp16 or fp32")
    args.out.mkdir(parents=True)
    metadata = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "torch": str(torch.__version__), "generated_count": args.count,
                "warning": "FID protocol uses explicit split folders, not published train-statistics FID"}
    if args.sample_metadata:
        metadata["sample_provenance"] = json.loads(args.sample_metadata.read_text())
    started = time.monotonic()
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if ckpt.get("format") != "cifar10-ddpm-v1":
            p.error("not a train_diffusion.py checkpoint")
        model = UNet(**ckpt["config"]).to(device).eval()
        model.load_state_dict(ckpt["ema"])
        diffusion = Diffusion(ckpt["args"]["diffusion_steps"], ckpt["args"]["schedule"], device)
        with open(args.ckpt, "rb") as stream:
            metadata["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata.update(training_step=ckpt["step"], model=ckpt["config"], training_args=ckpt["args"])
        del ckpt
        generated = args.out / "generated"
        generated.mkdir()
        gen = torch.Generator(device=device).manual_seed(args.seed)

        def amp():
            return nullcontext() if args.precision == "fp32" else torch.autocast(
                "cuda", dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16)

        previews = []
        for offset in range(0, args.count, args.batch):
            images = diffusion.sample(model, min(args.batch, args.count - offset), gen,
                                      args.sampler, args.sampling_steps, amp).cpu()
            save_individual(images, generated, offset)
            if sum(len(v) for v in previews) < 64:
                previews.append(images[:64 - sum(len(v) for v in previews)])
            print(f"generated {min(offset + args.batch, args.count)}/{args.count}", flush=True)
        save_image(torch.cat(previews), args.out / "grid.png", nrow=8)
        del model, diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        generated = args.images
        assert generated is not None
        files = sorted(generated.glob("*.png"))
        if len(files) != args.count:
            p.error(f"--images has {len(files)} top-level PNGs; expected --count {args.count}")
        if any(path.is_dir() for path in generated.iterdir()):
            p.error("--images must be a flat folder of individual samples")
        for file in files:
            with Image.open(file) as image:
                if image.size != (32, 32) or image.mode != "RGB":
                    p.error(f"{file}: expected an individual RGB 32x32 image, not a grid")
    metadata["sampling_seconds"] = time.monotonic() - started
    metadata["status"] = "samples_complete"
    (args.out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if fid is not None:
        real = load_images("cifar10", args.reference, args.data_root)
        reference = args.out / "reference"
        reference.mkdir()
        save_individual(real.float() / 255, reference)
        metadata.update(reference_count=len(real), reference_split=args.reference,
                        clean_fid_version=importlib.metadata.version("clean-fid"), fid_mode="clean")
        metadata["metric_versions"] = {name: str(importlib.import_module(name).__version__)
                                       for name in ("scipy", "numpy", "torchvision")}
        try:
            metadata["fid"] = float(fid.compute_fid(str(generated), str(reference), mode="clean", device=device,
                                                  num_workers=0, batch_size=args.batch, use_dataparallel=False))
        except Exception as error:
            metadata.update(status="fid_failed", error=repr(error))
            (args.out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
            raise
    metadata["status"] = "complete"
    metadata["total_seconds"] = time.monotonic() - started
    (args.out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
