#!/usr/bin/env python3
"""Losslessly wrap a native Z-Lab Qwen3 DFlash2 checkpoint for Speculators.

Z-Lab's Qwen3.8 DFlash2 checkpoint already uses the same parameter layout as
``speculators.models.dflash2.DFlash2DraftModel``.  The checkpoint lacks only
the frozen target-owned embedding, output head, and final norm; Speculators
intentionally reconstructs those from ``--verifier`` at load time.

The converter therefore copies the source safetensors byte-for-byte and writes
the Speculators config required by ``DFlash2SpeculatorConfig.from_pretrained``.
It never modifies the source snapshot or initializes trained DFlash2 weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


TRAINED_PREFIXES = ("layers.", "fc.", "hidden_norm.", "norm.", "candidate_selector.")
VERIFIER_FILLED_KEYS = {
    "embed_tokens.weight",
    "lm_head.weight",
    "verifier_lm_head.weight",
    "verifier_norm.weight",
}
NON_TRANSFORMER_KEYS = {
    "architectures",
    "auto_map",
    "block_size",
    "dflash_config",
    "num_target_layers",
}


def json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        json.dump(value, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(source: str, local_files_only: bool) -> Path:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return source_path.resolve()
    return Path(
        snapshot_download(repo_id=source, local_files_only=local_files_only)
    ).resolve()


def source_weight_files(source: Path) -> list[Path]:
    index = source / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        files = [source / name for name in names]
    else:
        files = sorted(source.glob("*.safetensors"))
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"No complete safetensors checkpoint found in {source}")
    return files


def load_source_inventory(files: list[Path]) -> dict[str, tuple[int, ...]]:
    inventory: dict[str, tuple[int, ...]] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            for name in file.keys():
                if name in inventory:
                    raise ValueError(f"Duplicate tensor {name} across source shards")
                inventory[name] = tuple(file.get_slice(name).get_shape())
    return inventory


def build_config(source_config: dict[str, Any], verifier: str) -> DFlash2SpeculatorConfig:
    dflash = source_config.get("dflash_config")
    if not isinstance(dflash, dict):
        raise ValueError("Source config has no dflash_config")
    required = ("block_size", "conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k", "target_layer_ids", "mask_token_id")
    missing = [key for key in required if key not in dflash]
    if missing:
        raise ValueError(f"Source dflash_config misses required fields: {missing}")
    if source_config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError(f"Expected native DFlash2 architecture, got {source_config.get('architectures')}")

    transformer_config = {
        key: value for key, value in source_config.items() if key not in NON_TRANSFORMER_KEYS
    }
    verifier_config, _ = PretrainedConfig.get_config_dict(verifier)
    source_layer_ids = dflash["target_layer_ids"]
    if not isinstance(source_layer_ids, list) or not source_layer_ids:
        raise ValueError("dflash_config.target_layer_ids must be a non-empty list")

    # Z-Lab indexes transformer layers and accesses hidden_states[layer_id + 1],
    # because hidden_states[0] is the embedding output. Speculators records that
    # latter hidden-state-slot convention in aux_hidden_state_layer_ids.
    auxiliary_ids = [int(layer_id) + 1 for layer_id in source_layer_ids]
    block_size = int(dflash["block_size"])
    return DFlash2SpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=int(transformer_config["vocab_size"]),
        block_size=block_size,
        aux_hidden_state_layer_ids=auxiliary_ids,
        mask_token_id=int(dflash["mask_token_id"]),
        sliding_window_non_causal=True,
        sample_from_anchor=False,
        conv_kernel_size=int(dflash["conv_kernel_size"]),
        conv_group_size=int(dflash["conv_group_size"]),
        selector_rank=int(dflash["selector_rank"]),
        selector_top_k=int(dflash["selector_top_k"]),
        speculators_config=SpeculatorsConfig(
            algorithm="dflash2",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block_size - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=verifier,
                architectures=verifier_config.get("architectures", []),
            ),
        ),
    )


def expected_inventory(config: DFlash2SpeculatorConfig) -> dict[str, tuple[int, ...]]:
    # Meta avoids allocating the 3.8 GiB draft merely to inspect its state layout.
    with torch.device("meta"):
        model = DFlash2DraftModel(config)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def verify_tensors(source_files: list[Path], destination: Path, expected: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    source: dict[str, tuple[Path, tuple[int, ...]]] = {}
    for path in source_files:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            for name in file.keys():
                source[name] = (path, tuple(file.get_slice(name).get_shape()))

    destination_files = source_weight_files(destination)
    destination_locations: dict[str, Path] = {}
    destination_shapes: dict[str, tuple[int, ...]] = {}
    for path in destination_files:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            for name in file.keys():
                destination_locations[name] = path
                destination_shapes[name] = tuple(file.get_slice(name).get_shape())

    unexpected_source = sorted(set(source) - set(expected))
    missing_trained = sorted(
        name for name in set(expected) - set(source) if name not in VERIFIER_FILLED_KEYS
    )
    if unexpected_source or missing_trained:
        raise ValueError(
            f"Incompatible layout: unexpected source={unexpected_source}; "
            f"missing trained destination={missing_trained}"
        )

    mapped: list[dict[str, Any]] = []
    for name in sorted(source):
        if destination_shapes.get(name) != source[name][1] or expected.get(name) != source[name][1]:
            raise ValueError(
                f"Shape mismatch for {name}: source={source[name][1]}, "
                f"destination={destination_shapes.get(name)}, expected={expected.get(name)}"
            )
        with safe_open(str(source[name][0]), framework="pt", device="cpu") as src_file, safe_open(
            str(destination_locations[name]), framework="pt", device="cpu"
        ) as dst_file:
            equal = torch.equal(src_file.get_tensor(name), dst_file.get_tensor(name))
        if not equal:
            raise ValueError(f"Value mismatch after conversion: {name}")
        mapped.append({"source": name, "destination": name, "shape": list(source[name][1]), "torch_equal": True})

    conv = [entry["destination"] for entry in mapped if ".attention_conv." in entry["destination"] or ".mlp_conv." in entry["destination"]]
    selector = [entry["destination"] for entry in mapped if entry["destination"].startswith("candidate_selector.")]
    if len(conv) != 20 or len(selector) != 3:
        raise ValueError(f"DFlash2 verification failed: conv={len(conv)}, selector={len(selector)}")
    return {
        "source_tensor_count": len(source),
        "destination_tensor_count": len(destination_shapes),
        "mapped_tensor_count": len(mapped),
        "mapped_tensors": mapped,
        "intentionally_verifier_filled": sorted(VERIFIER_FILLED_KEYS),
        "missing_trained_tensors": missing_trained,
        "unexpected_source_tensors": unexpected_source,
        "dflash2_conv_tensors": conv,
        "dflash2_candidate_selector_tensors": selector,
    }


def copy_checkpoint(source: Path, output: Path, overwrite: bool) -> list[Path]:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in source_weight_files(source):
        destination = output / path.name
        shutil.copyfile(path, destination)
        copied.append(destination)
    index = source / "model.safetensors.index.json"
    if index.exists():
        shutil.copyfile(index, output / index.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Z-Lab repository ID or local snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier", required=True, help="Native-basis Freaksterz training-verifier directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true", help="Skip the full CPU DFlash2 load validation")
    args = parser.parse_args()

    source = resolve_source(args.source, args.local_files_only)
    verifier = str(Path(args.verifier).resolve())
    if not Path(verifier).is_dir():
        raise FileNotFoundError(f"Verifier directory does not exist: {verifier}")
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config = build_config(source_config, verifier)
    expected = expected_inventory(config)
    copy_checkpoint(source, args.output, args.overwrite)
    config.save_pretrained(str(args.output))
    report = verify_tensors(source_weight_files(source), args.output, expected)
    report.update(
        {
            "source": str(source),
            "output": str(args.output.resolve()),
            "source_config_target_layer_ids": source_config["dflash_config"]["target_layer_ids"],
            "speculators_aux_hidden_state_layer_ids": config.aux_hidden_state_layer_ids,
            "config": {
                "block_size": config.block_size,
                "conv_kernel_size": config.conv_kernel_size,
                "conv_group_size": config.conv_group_size,
                "selector_rank": config.selector_rank,
                "selector_top_k": config.selector_top_k,
                "draft_vocab_size": config.draft_vocab_size,
                "hidden_size": config.transformer_layer_config.hidden_size,
            },
            "source_files": [
                {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in source_weight_files(source)
            ],
            "destination_files": [
                {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in source_weight_files(args.output)
            ],
        }
    )
    if report["source_files"] != report["destination_files"]:
        raise ValueError("Safetensors file checksum mismatch after conversion")
    # Config loading is independent from model allocation and is always validated.
    loaded_config = DFlash2SpeculatorConfig.from_pretrained(str(args.output))
    # Transformers adds serialization-only defaults (for example ``auto_map``
    # and ``tie_word_embeddings``) while reading config.json. Validate every
    # architecture-relevant field rather than those transport defaults.
    config_fields = (
        "speculators_model_type",
        "architectures",
        "draft_vocab_size",
        "block_size",
        "aux_hidden_state_layer_ids",
        "mask_token_id",
        "sliding_window_non_causal",
        "sample_from_anchor",
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    )
    config_mismatches = {
        field: {"written": getattr(config, field), "loaded": getattr(loaded_config, field)}
        for field in config_fields
        if getattr(config, field) != getattr(loaded_config, field)
    }
    if config.transformer_layer_config.to_dict() != loaded_config.transformer_layer_config.to_dict():
        config_mismatches["transformer_layer_config"] = "mismatch"
    if config.speculators_config.model_dump() != loaded_config.speculators_config.model_dump():
        config_mismatches["speculators_config"] = "mismatch"
    if config_mismatches:
        raise ValueError(f"Config round-trip mismatch: {config_mismatches}")
    report["config_load"] = {"ok": True, "class": type(loaded_config).__name__}
    if not args.skip_model_load:
        model = DFlash2DraftModel.from_pretrained(str(args.output))
        report["model_load"] = {
            "ok": True,
            "class": type(model).__name__,
            "state_tensor_count": len(model.state_dict()),
        }
        del model
    json_dump(args.output / "conversion-report.json", report)
    print(json.dumps({
        "output": str(args.output),
        "source_tensor_count": report["source_tensor_count"],
        "mapped_tensor_count": report["mapped_tensor_count"],
        "conv_tensors": len(report["dflash2_conv_tensors"]),
        "selector_tensors": len(report["dflash2_candidate_selector_tensors"]),
        "config_load": report["config_load"],
        "model_load": report.get("model_load", {"skipped": True}),
    }, indent=2))


if __name__ == "__main__":
    main()
