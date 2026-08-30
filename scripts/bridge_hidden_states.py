#!/usr/bin/env python3
"""Convert vLLM-extracted Freaksterz frames to DFlash2's native basis.

The input files are the ``*.safetensors`` records produced by Speculators'
offline extractor.  Every H-wide frame is multiplied by R, including the final
verifier frame.  This is deliberately a data-only operation: it never changes
the source traces and is safe to resume.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_adapter(path: Path) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as file:
        rotation = file.get_tensor("rotation").float().contiguous()
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"invalid rotation shape {tuple(rotation.shape)}")
    return rotation


def convert(source: Path, destination: Path, rotation: torch.Tensor) -> dict[str, object]:
    with safe_open(str(source), framework="pt", device="cpu") as file:
        metadata = dict(file.metadata() or {})
        tensors = {name: file.get_tensor(name) for name in file.keys()}
    hidden = tensors.get("hidden_states")
    if hidden is None:
        raise ValueError(f"{source}: missing hidden_states")
    if hidden.shape[-1] != rotation.shape[0]:
        raise ValueError(
            f"{source}: hidden width {hidden.shape[-1]} != rotation width {rotation.shape[0]}"
        )
    tensors["hidden_states"] = torch.matmul(hidden.float(), rotation).to(hidden.dtype)
    metadata.update({"freaksterz_dflash_bridge": "native = rotated @ R", "rotation_shape": json.dumps(list(rotation.shape))})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".safetensors", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source": str(source), "output": str(destination), "shape": list(hidden.shape)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    rotation = load_adapter(args.adapter)
    records = []
    for source in sorted(args.input.rglob("*.safetensors")):
        destination = args.output / source.relative_to(args.input)
        if destination.exists() and not args.overwrite:
            continue
        with safe_open(str(source), framework="pt", device="cpu") as file:
            if "hidden_states" not in file.keys():
                continue
        records.append(convert(source, destination, rotation))
        print(json.dumps(records[-1]), flush=True)
    report = {"converted": len(records), "input": str(args.input), "output": str(args.output)}
    print(json.dumps(report), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
