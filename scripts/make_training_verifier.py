#!/usr/bin/env python3
"""Create frozen native-basis embedding/head weights for DFlash2 training.

Speculators trains DFlash2 with frozen verifier embeddings and LM heads.  This
creates their exact Freaksterz-compatible native-basis equivalents, so training
does not accidentally feed rotated embeddings or score native draft states with
the rotated target head.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def weight_map(directory: Path) -> dict[str, str]:
    index = directory / "model.safetensors.index.json"
    if not index.exists():
        return {}
    return json.loads(index.read_text(encoding="utf-8"))["weight_map"]


def tensor(directory: Path, mapping: dict[str, str], name: str) -> torch.Tensor:
    path = directory / mapping.get(name, "model.safetensors")
    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freaksterz", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = weight_map(args.freaksterz)
    with safe_open(str(args.adapter), framework="pt", device="cpu") as file:
        rotation = file.get_tensor("rotation").float()
        gain_inv = file.get_tensor("gain_inv").float()
    embedding = tensor(args.freaksterz, mapping, "model.language_model.embed_tokens.weight")
    head = tensor(args.freaksterz, mapping, "lm_head.weight")
    if embedding.shape[-1] != rotation.shape[0] or head.shape[-1] != rotation.shape[0]:
        raise ValueError("checkpoint and adapter hidden sizes differ")
    # E_native = E_rot @ R; W_effective = W_rot @ R @ diag(gain_inv).
    native_embedding = (embedding.float() @ rotation).to(torch.bfloat16)
    native_head = ((head.float() @ rotation) * gain_inv).to(torch.bfloat16)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        source = args.freaksterz / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    with NamedTemporaryFile(dir=args.output, suffix=".safetensors", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        save_file({"model.embed_tokens.weight": native_embedding, "lm_head.weight": native_head}, str(temporary), metadata={"freaksterz_dflash_bridge": "native training verifier"})
        os.replace(temporary, args.output / "model.safetensors")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "embedding_shape": list(native_embedding.shape), "head_shape": list(native_head.shape)}))


if __name__ == "__main__":
    main()
