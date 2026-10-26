"""Archive existing checkpoints, outputs, metrics and source without overwriting earlier experiments."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

import torch

from dvae.experiments import file_hash

ROOT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="unique experiment folder name")
    parser.add_argument("--notes", required=True, help="architecture/change, hypothesis and known provenance gaps")
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--encoder", type=Path, default=Path("checkpoints/encoder.pt"))
    parser.add_argument("--decoder", type=Path, default=Path("checkpoints/decoder_gray.pt"))
    parser.add_argument("--prior", type=Path, default=Path("checkpoints/prior.pt"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--include-build-state", type=Path, help="optional encoder build checkpoint for resuming")
    args = parser.parse_args()
    if args.name in (".", "..") or Path(args.name).name != args.name:
        parser.error("--name must be a single folder name")
    destination = args.root / args.name
    if destination.exists():
        parser.error(f"{destination} already exists; choose a new name")
    sources = {kind: getattr(args, kind) for kind in ("encoder", "decoder", "prior")}
    if not any(source.is_file() for source in sources.values()):
        parser.error("no checkpoints found; supply checkpoint paths")
    if args.include_build_state and not args.include_build_state.is_file():
        parser.error("--include-build-state must point to an existing checkpoint")

    destination.mkdir(parents=True)
    manifest = {
        "name": args.name, "created_utc": datetime.now(timezone.utc).isoformat(), "notes": args.notes,
        "source_provenance": "Source captured at archive time; may differ from the code used to train these checkpoints.",
        "git": {"commit": git("rev-parse", "HEAD").strip(),
                "branch": git("branch", "--show-current").strip(), "status": git("status", "--short")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": str(torch.__version__), "torchvision": importlib.metadata.version("torchvision")},
        "checkpoints": {}, "missing": [],
    }
    checkpoint_info = {}
    for kind, source in sources.items():
        if not source.is_file():
            manifest["missing"].append(str(source))
            continue
        target = destination / "checkpoints" / (kind + ".pt")
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        # Inspect tensors directly, so old checkpoints can be archived after an architecture edit.
        saved = torch.load(target, map_location="cpu", weights_only=False)
        checkpoint_info[kind] = {
            key: saved[key] for key in ("kind", "config", "args", "metrics", "history") if key in saved
        }
        weights = saved.get("state_dict", {})
        checkpoint_info[kind]["parameter_count"] = sum(t.numel() for t in weights.values())
        checkpoint_info[kind]["tensor_shapes"] = {key: list(t.shape) for key, t in weights.items()}
        manifest["checkpoints"][kind] = {
            "source": str(source.resolve()), "archive": str(target.relative_to(destination)),
            "sha256": file_hash(target),
        }
        del saved
    if args.include_build_state:
        shutil.copy2(args.include_build_state, destination / "checkpoints" / "encoder_build.pt")
    for label in ("outputs", "results"):
        source = getattr(args, label)
        if source.is_dir():
            if destination.resolve().is_relative_to(source.resolve()):
                parser.error(f"archive destination must not be inside {source}")
            shutil.copytree(source, destination / label)
        else:
            manifest["missing"].append(str(source))

    for name in git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0"):
        source = ROOT / name
        if source.suffix in (".py", ".md", ".json") and source.is_file() and not source.is_symlink():
            target = destination / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (destination / "changes.patch").write_text(git("diff", "HEAD", "--binary"))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "training.json").write_text(json.dumps(checkpoint_info, indent=2, default=str) + "\n")
    (destination / "NOTES.md").write_text(
        f"# {args.name}\n\n{args.notes}\n\n"
        "Source was captured at archive time, not necessarily at training time.\n\n"
        "Existing images are copied as-is. If no QA metadata exists, their seed, sampling settings and "
        "checkpoint association cannot be recovered automatically.\n\n"
        "Compare outputs/generate.png and matching outputs/compress/k*.png side by side. "
        "Quantitative evaluations, if available, are under results/.\n"
    )
    print(f"Archived experiment to {destination}")
    if manifest["missing"]:
        print("Not available: " + ", ".join(manifest["missing"]))


if __name__ == "__main__":
    main()
