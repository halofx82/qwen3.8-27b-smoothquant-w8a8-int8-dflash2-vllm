#!/usr/bin/env python3
"""Build and validate the Freaksterz-main <-> native-Qwen basis adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


HIDDEN_SIZE = 5120
BASE_REPO = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MAIN_REPO = "Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
MAIN_REVISION = "68746dd1df1b7290aa6c1cb789773a7829fb86fc"
FINAL_NORM = "model.language_model.norm.weight"
EMBED = "model.language_model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path: Path | None, repo_id: str, revision: str,
                       local_files_only: bool) -> Path:
    if path is not None:
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    return Path(snapshot_download(
        repo_id=repo_id, revision=revision, local_files_only=local_files_only,
    ))


class IndexedCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        with (path / "model.safetensors.index.json").open("r", encoding="utf-8") as handle:
            self.weight_map = json.load(handle)["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.path / self.weight_map[name]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def metrics(prediction: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    prediction, reference = prediction.float(), reference.float()
    relative = torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    cosine = torch.nn.functional.cosine_similarity(prediction, reference, dim=-1, eps=1e-12).mean()
    return float(relative), float(cosine)


def sampled_rows(source: torch.Tensor, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(source.shape[0], generator=generator)[:count]
    return indices, source[indices]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, help="local base checkpoint snapshot")
    parser.add_argument("--main", type=Path, help="local Freaksterz main checkpoint snapshot")
    parser.add_argument("--rotation", type=Path, default=Path("rotation-R.pt"))
    parser.add_argument("--output", type=Path, default=Path("freaksterz-dflash-adapter.safetensors"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")
    if not args.rotation.is_file():
        raise FileNotFoundError(args.rotation)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable; pass --device cpu")

    base_path = resolve_checkpoint(args.base, BASE_REPO, BASE_REVISION, args.local_files_only)
    main_path = resolve_checkpoint(args.main, MAIN_REPO, MAIN_REVISION, args.local_files_only)
    print(f"base checkpoint: {base_path}")
    print(f"Freaksterz main checkpoint: {main_path}")
    base, main_checkpoint = IndexedCheckpoint(base_path), IndexedCheckpoint(main_path)

    rotation = torch.load(args.rotation, map_location="cpu", weights_only=True).float().contiguous()
    if tuple(rotation.shape) != (HIDDEN_SIZE, HIDDEN_SIZE) or not torch.isfinite(rotation).all():
        raise RuntimeError("invalid rotation")
    gain = 1.0 + base.tensor(FINAL_NORM).float()
    gain_inv = gain.reciprocal().contiguous()
    if tuple(gain.shape) != (HIDDEN_SIZE,) or not torch.isfinite(gain_inv).all():
        raise RuntimeError("invalid final RMSNorm gain")

    rotation_device = rotation.to(device)
    probe = torch.randn(32, HIDDEN_SIZE, device=device)
    roundtrip_rel, roundtrip_cos = metrics((probe @ rotation_device.T) @ rotation_device, probe)
    base_embed, main_embed = base.tensor(EMBED), main_checkpoint.tensor(EMBED)
    indices, embed_rows = sampled_rows(base_embed, args.samples, 20260829)
    embed_rel, embed_cos = metrics(embed_rows.float().to(device) @ rotation_device.T,
                                   main_embed[indices].float().to(device))
    base_head, main_head = base.tensor(LM_HEAD), main_checkpoint.tensor(LM_HEAD)
    indices, head_rows = sampled_rows(base_head, args.samples, 20260830)
    head_rel, head_cos = metrics((head_rows.float().to(device) * gain.to(device)) @ rotation_device.T,
                                 main_head[indices].float().to(device))
    validation = {
        "roundtrip_rel_error": roundtrip_rel, "roundtrip_mean_cosine": roundtrip_cos,
        "embedding_rel_error": embed_rel, "embedding_mean_cosine": embed_cos,
        "lm_head_rel_error": head_rel, "lm_head_mean_cosine": head_cos,
        "gain_min": float(gain.min()), "gain_max": float(gain.max()), "gain_mean": float(gain.mean()),
    }
    print(json.dumps(validation, indent=2))
    if roundtrip_rel > 2e-5 or embed_cos < 0.9999 or head_cos < 0.9999:
        raise RuntimeError("adapter source validation failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": "1", "hidden_size": str(HIDDEN_SIZE),
        "target_revision": MAIN_REVISION, "base_revision": BASE_REVISION,
        "rotation_sha256": sha256(args.rotation),
        "row_convention": "h_freaksterz = h_native @ rotation.T",
        "head_bridge": "h_main_head = (h_dflash * gain_inv) @ rotation.T",
        "validation": json.dumps(validation, sort_keys=True),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    save_file({"rotation": rotation, "gain_inv": gain_inv}, str(temporary), metadata=metadata)
    os.replace(temporary, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")
    print(f"sha256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
