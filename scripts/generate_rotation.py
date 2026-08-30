#!/usr/bin/env python3
"""Recreate the deterministic residual-stream rotation used by Freaksterz main."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


HIDDEN_SIZE = 5120
SEED = 20260815
EXPECTED_FILE_SHA256 = "8d6dd7bb2278c288f4e74583807bbc6429200839ecf8fe6e9319a23326e6a505"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rotation-R.pt"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")

    # This is the exact rotation-generation block used for the rotated model.
    torch.manual_seed(SEED)
    matrix = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(matrix)
    rotation_fp32 = rotation.float()

    identity = torch.eye(HIDDEN_SIZE, dtype=torch.float64)
    orthogonality_error = float((rotation @ rotation.T - identity).abs().max())
    if orthogonality_error > 1e-10:
        raise RuntimeError(f"rotation is not orthogonal: {orthogonality_error:.3e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rotation_fp32, args.output)
    actual_sha256 = sha256(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")
    print(f"sha256: {actual_sha256}")
    print(f"orthogonality max abs: {orthogonality_error:.3e}")
    if actual_sha256 != EXPECTED_FILE_SHA256:
        raise RuntimeError(
            "rotation checksum mismatch; use the PyTorch build that produced "
            f"{EXPECTED_FILE_SHA256}"
        )


if __name__ == "__main__":
    main()
