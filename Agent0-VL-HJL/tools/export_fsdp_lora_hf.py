#!/usr/bin/env python3
"""Export a sharded verl FSDP LoRA checkpoint as a merged HF model."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import Partial, Replicate, Shard


LORA_SUFFIXES = (".lora_A.default.weight", ".lora_B.default.weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--actor-checkpoint", required=True, type=Path,
                        help="Actor directory containing model_world_size_*_rank_*.pt files")
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory for the merged HF model")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--adapter-only", action="store_true",
                        help="Export the PEFT adapter without merging it into the base")
    return parser.parse_args()


def rank_files(actor_dir: Path) -> list[Path]:
    candidates = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
    parsed: dict[int, Path] = {}
    world_sizes: set[int] = set()
    for path in candidates:
        match = re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", path.name)
        if match is None:
            continue
        world_size, rank = map(int, match.groups())
        world_sizes.add(world_size)
        parsed[rank] = path
    if len(world_sizes) != 1:
        raise RuntimeError(f"Expected one FSDP world size, found {sorted(world_sizes)}")
    world_size = world_sizes.pop()
    if sorted(parsed) != list(range(world_size)):
        raise RuntimeError(f"Expected rank files 0..{world_size - 1}, found {sorted(parsed)}")
    return [parsed[rank] for rank in range(world_size)]


def export_adapter(actor_dir: Path, output: Path, base_model: Path, rank: int, alpha: int) -> Path:
    adapter_name = 'default'
    sidecar = actor_dir / 'role_checkpoint.json'
    if sidecar.exists():
        from agent0_protocol.checkpointed import PROTOCOL, ADAPTER_LAYOUT, ADAPTER_FOR_MODE, ADAPTERS
        info = json.loads(sidecar.read_text())
        if (info.get('protocol_version') != PROTOCOL or
                info.get('adapter_layout_version') != ADAPTER_LAYOUT or
                info.get('mode_to_adapter') != ADAPTER_FOR_MODE or
                set(info.get('versions', {})) != set(ADAPTERS)):
            raise ValueError('Old multi-adapter checkpoint cannot be exported as shared LoRA')
        adapter_name = 'shared'
    lora_suffixes = tuple(suffix.replace('.default.', '.' + adapter_name + '.') for suffix in LORA_SUFFIXES)
    if output.exists():
        if (output / "adapter_model.safetensors").is_file() and (output / "adapter_config.json").is_file():
            print(f"Reusing existing PEFT adapter: {output}", flush=True)
            return output
        raise FileExistsError(f"Refusing to overwrite incomplete adapter directory: {output}")
    files = rank_files(actor_dir)
    pieces: dict[str, list[torch.Tensor]] = {}
    metadata: dict[str, tuple[tuple[int, ...], str, int]] = {}
    module_names: set[str] = set()

    for rank_index, path in enumerate(files):
        state = torch.load(path, map_location="cpu", weights_only=False)
        rank_keys = {key for key in state if any(key.endswith(suffix) for suffix in lora_suffixes)}
        if rank_index == 0:
            adapter_keys = rank_keys
        elif rank_keys != adapter_keys:
            raise RuntimeError(f"LoRA parameter keys differ in {path}")

        for key in sorted(rank_keys):
            tensor = state[key]
            if not hasattr(tensor, "to_local") or len(tensor.placements) != 1:
                raise TypeError(f"Expected a one-dimensional FSDP DTensor for {key}")
            placement = tensor.placements[0]
            if not isinstance(placement, (Shard, Replicate, Partial)):
                raise TypeError(f"Unsupported placement {placement!r} for {key}")
            local = tensor.to_local().detach().cpu().contiguous()
            shape = tuple(tensor.shape)
            descriptor = (shape, repr(placement), tensor.ndim)
            if rank_index == 0:
                metadata[key] = descriptor
            elif metadata[key] != descriptor:
                raise RuntimeError(f"Tensor metadata differs between FSDP ranks for {key}")
            pieces.setdefault(key, []).append(local.clone())
            for suffix in lora_suffixes:
                if key.endswith(suffix):
                    module_names.add(key[:-len(suffix)])

        del state
        gc.collect()
        print(f"Read LoRA tensors from rank {rank_index + 1}/{len(files)}: {path.name}", flush=True)

    if not pieces:
        raise RuntimeError(f"No LoRA tensors found under {actor_dir}")

    adapter_state: dict[str, torch.Tensor] = {}
    for key, local_parts in pieces.items():
        shape, placement_repr, _ = metadata[key]
        if placement_repr.startswith("Shard(dim="):
            match = re.fullmatch(r"Shard\(dim=(\d+)\)", placement_repr)
            if match is None:
                raise RuntimeError(f"Could not parse sharding placement {placement_repr}")
            shard_dim = int(match.group(1))
            full = torch.cat(local_parts, dim=shard_dim)
        elif placement_repr == "Replicate()":
            if any(not torch.equal(local_parts[0], part) for part in local_parts[1:]):
                raise RuntimeError(f"Replicated LoRA tensor differs across ranks: {key}")
            full = local_parts[0]
        elif placement_repr.startswith("Partial("):
            full = torch.stack(local_parts).sum(0)
        else:
            raise RuntimeError(f"Unsupported placement {placement_repr} for {key}")
        if tuple(full.shape) != shape:
            raise RuntimeError(f"Reconstructed {key} has {tuple(full.shape)}, expected {shape}")
        export_key = key.replace("." + adapter_name + ".weight", ".weight")
        adapter_state[export_key] = full.contiguous()

    output.mkdir(parents=True, exist_ok=False)
    save_file(adapter_state, str(output / "adapter_model.safetensors"))
    target_modules = sorted(module.removeprefix("base_model.model.") for module in module_names)
    config = {
        "base_model_name_or_path": str(base_model.resolve()),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(adapter_state)} adapter tensors for {len(target_modules)} target modules", flush=True)
    return output


def merge_adapter(base_model_path: Path, adapter_path: Path, output: Path) -> None:
    from peft import PeftModel
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    from transformers import AutoProcessor

    print(f"Loading base model on CPU: {base_model_path}", flush=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"Loading RL adapter: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    print("Merging RL adapter into the SFT model", flush=True)
    model = model.merge_and_unload(safe_merge=True)

    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(output), safe_serialization=True, max_shard_size="4GB")
    processor = AutoProcessor.from_pretrained(str(base_model_path), trust_remote_code=True)
    processor.save_pretrained(str(output))
    print(f"Merged RL model written to {output}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    adapter_output = args.output.with_name(args.output.name + "_adapter")
    export_adapter(args.actor_checkpoint, adapter_output, args.base_model, args.rank, args.alpha)
    if not args.adapter_only:
        merge_adapter(args.base_model, adapter_output, args.output)
