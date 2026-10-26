"""Provenance for visual evaluations; no model reconstruction is needed."""

import hashlib
import json
from pathlib import Path


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_qa_metadata(path, args, codes, **extra):
    checkpoints = {}
    for kind in ("encoder", "prior", "decoder"):
        source = getattr(args, kind, None)
        if source is not None:
            resolved = Path(source)
            if not resolved.exists():
                resolved = resolved.with_name(resolved.stem + "_latest" + resolved.suffix)
            checkpoints[kind] = {"path": str(resolved), "sha256": file_hash(resolved)}
    Path(path).write_text(json.dumps({
        "settings": vars(args), "checkpoints": checkpoints,
        "codes": codes.cpu().tolist(), **extra,
    }, indent=2) + "\n")
