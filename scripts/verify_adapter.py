#!/usr/bin/env python3
"""Validate a built adapter against the pinned base and Freaksterz checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from build_adapter import (BASE_REPO, BASE_REVISION, EMBED, FINAL_NORM, HIDDEN_SIZE,
                           LM_HEAD, MAIN_REPO, MAIN_REVISION, IndexedCheckpoint,
                           metrics, resolve_checkpoint, sampled_rows)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("freaksterz-dflash-adapter.safetensors"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--main", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not args.adapter.is_file():
        raise FileNotFoundError(args.adapter)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable; pass --device cpu")
    base = IndexedCheckpoint(resolve_checkpoint(args.base, BASE_REPO, BASE_REVISION, args.local_files_only))
    main_checkpoint = IndexedCheckpoint(resolve_checkpoint(args.main, MAIN_REPO, MAIN_REVISION, args.local_files_only))
    with safe_open(str(args.adapter), framework="pt", device="cpu") as file:
        if set(file.keys()) != {"gain_inv", "rotation"}:
            raise RuntimeError(f"unexpected adapter tensors: {list(file.keys())}")
        rotation, gain_inv = file.get_tensor("rotation").float(), file.get_tensor("gain_inv").float()
        metadata = file.metadata() or {}
    if tuple(rotation.shape) != (HIDDEN_SIZE, HIDDEN_SIZE) or tuple(gain_inv.shape) != (HIDDEN_SIZE,):
        raise RuntimeError("invalid adapter shapes")
    if metadata.get("target_revision") != MAIN_REVISION or metadata.get("base_revision") != BASE_REVISION:
        raise RuntimeError("adapter revision metadata does not match this runtime")
    expected_gain_inv = (1.0 + base.tensor(FINAL_NORM).float()).reciprocal()
    gain_rel, gain_cos = metrics(gain_inv, expected_gain_inv)
    rotation_device = rotation.to(device)
    base_embed, main_embed = base.tensor(EMBED), main_checkpoint.tensor(EMBED)
    indices, rows = sampled_rows(base_embed, args.samples, 20260829)
    embed_rel, embed_cos = metrics(rows.float().to(device) @ rotation_device.T,
                                   main_embed[indices].float().to(device))
    base_head, main_head = base.tensor(LM_HEAD), main_checkpoint.tensor(LM_HEAD)
    indices, rows = sampled_rows(base_head, args.samples, 20260830)
    head_rel, head_cos = metrics((rows.float().to(device) * (1.0 + base.tensor(FINAL_NORM).float()).to(device)) @ rotation_device.T,
                                 main_head[indices].float().to(device))
    report = {"gain_inv_rel_error": gain_rel, "gain_inv_mean_cosine": gain_cos,
              "embedding_rel_error": embed_rel, "embedding_mean_cosine": embed_cos,
              "lm_head_rel_error": head_rel, "lm_head_mean_cosine": head_cos}
    print(json.dumps(report, indent=2))
    if gain_rel > 1e-6 or embed_cos < 0.9999 or head_cos < 0.9999:
        raise RuntimeError("adapter verification failed")


if __name__ == "__main__":
    main()
