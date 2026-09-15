#!/usr/bin/env python3
"""Run a small, auditable QLoRA smoke fine-tune for Qwen2.5-VL.

This entry point intentionally keeps the 60 RL task rows out of the SFT loss.
They are snapshotted and recorded as held-out rollout/evaluation inputs.  An
RL row has a prompt and reward metadata, but no assistant target; using it as
ordinary SFT data would silently train on an invented answer.

The script is deliberately independent of ms-swift so it can be run from the
Linux ``agent0vl`` environment with only Transformers, PEFT and bitsandbytes.
It uses the Qwen2.5-VL processor for both the full transcript and the prompt
prefix, which keeps image-token expansion and label masking consistent.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "agent0vl.dataset.v1"
PROTOCOL_VERSION = "agent0vl.protocol.v1"
SCORER_VERSION = "agent0vl.scorer.v1"
TRAINING_SCRIPT_VERSION = "agent0vl.qlora-smoke.v1"
TARGET_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_dump_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def file_inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def inventory_fingerprint(inventory: Iterable[dict[str, Any]]) -> str:
    payload = sorted(
        [
            {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
            for item in inventory
        ],
        key=lambda item: item["path"],
    )
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def count_parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return pq.ParquetFile(path).metadata.num_rows
    except Exception as exc:
        raise RuntimeError(f"Cannot count RL parquet rows ({path}); install pyarrow") from exc


def copy_input_snapshot(sft_path: Path, rl_path: Path, snapshot_dir: Path) -> tuple[Path, Path]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_sft = snapshot_dir / "sft_records.jsonl"
    snapshot_rl = snapshot_dir / "rl_train_multimodal.parquet"
    shutil.copy2(sft_path, snapshot_sft)
    shutil.copy2(rl_path, snapshot_rl)

    source_assets = sft_path.parent / "assets"
    if source_assets.is_dir():
        shutil.copytree(source_assets, snapshot_dir / "assets", dirs_exist_ok=True)
    return snapshot_sft, snapshot_rl


def copy_model_backup(model_dir: Path, backup_dir: Path) -> str:
    """Copy the immutable base model without ever modifying an existing backup."""

    if model_dir.resolve() == backup_dir.resolve():
        raise ValueError("Base model and backup directory must be different")
    if backup_dir.exists():
        return "existing_backup_reused"
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, backup_dir)
    return "new_full_copy"


def resolve_image_path(raw_path: str, data_root: Path) -> str:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = data_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"SFT image does not exist: {candidate}")
    return str(candidate)


def make_user_content(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    """Convert the builder's ``<image>`` marker into Qwen processor content."""

    clean_text = text or ""
    pieces = clean_text.split("<image>")
    content: list[dict[str, Any]] = []
    image_index = 0
    for index, piece in enumerate(pieces):
        if index > 0 and image_index < len(image_paths):
            content.append({"type": "image", "image": image_paths[image_index]})
            image_index += 1
        if piece.strip():
            content.append({"type": "text", "text": piece})
    while image_index < len(image_paths):
        content.insert(image_index, {"type": "image", "image": image_paths[image_index]})
        image_index += 1
    if not content:
        content.append({"type": "text", "text": ""})
    return content


def record_to_qwen_messages(record: dict[str, Any], data_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        raise ValueError(f"Record has no user/assistant pair: {record.get('record_id')}")
    image_paths = [resolve_image_path(str(item), data_root) for item in record.get("images", [])]

    messages: list[dict[str, Any]] = []
    for message in raw_messages:
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if role == "user":
            if isinstance(content, list):
                messages.append({"role": role, "content": content})
            else:
                messages.append(
                    {"role": role, "content": make_user_content(str(content), image_paths)}
                )
        elif role == "assistant":
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": str(content)}],
                }
            )
        else:
            messages.append({"role": role, "content": content})
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"Last message is not assistant: {record.get('record_id')}")
    return messages, image_paths


def move_batch_to_device(batch: dict[str, Any], device: Any, torch: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            moved[key] = value.to(device)
    return moved


def prepare_example(
    record: dict[str, Any],
    processor: Any,
    data_root: Path,
    max_length: int,
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    messages, _ = record_to_qwen_messages(record, data_root)
    prompt_messages = messages[:-1]
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    kwargs: dict[str, Any] = {
        "padding": False,
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs
    full = processor(text=[full_text], **kwargs)

    prompt_kwargs = dict(kwargs)
    prompt_kwargs.pop("return_tensors", None)
    prompt_kwargs["return_tensors"] = "pt"
    prompt = processor(text=[prompt_text], **prompt_kwargs)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    prompt_length = int(prompt["input_ids"].shape[1])
    full_length = int(input_ids.shape[1])
    if prompt_length >= full_length:
        raise ValueError(
            f"Assistant target was truncated for {record.get('record_id')}: "
            f"prompt={prompt_length}, full={full_length}, max_length={max_length}"
        )

    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    labels[attention_mask == 0] = -100

    prepared: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "record_id": record.get("record_id"),
    }
    for key in ("pixel_values", "image_grid_thw", "second_per_grid_ts"):
        if key in full:
            prepared[key] = full[key]
    # Ensure all tensors are batch-shaped; this also makes the training loop
    # independent of processor versions that return a singleton list wrapper.
    for key, value in list(prepared.items()):
        if key != "record_id" and hasattr(value, "dim") and value.dim() == 0:
            prepared[key] = value.unsqueeze(0)
    return prepared


def build_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this smoke run is configured for the RTX 5080")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("high")

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=False,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    frozen_prefixes = ["model.visual", "visual"]
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in frozen_prefixes):
            parameter.requires_grad = False
            frozen_parameters += parameter.numel()

    target_modules = [
        name
        for name, _module in model.named_modules()
        if name.startswith("model.language_model.layers.")
        and name.rsplit(".", 1)[-1] in TARGET_SUFFIXES
    ]
    if not target_modules:
        raise RuntimeError("No language-layer LoRA targets found; refusing to train")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    visual_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("visual" in name or "merger" in name)
    ]
    if visual_trainable:
        raise RuntimeError(f"Visual parameters unexpectedly trainable: {visual_trainable[:5]}")

    freeze_report = {
        "freeze_vit": True,
        "freeze_visual_prefixes": frozen_prefixes,
        "frozen_parameter_count_before_lora": frozen_parameters,
        "lora_target_count": len(target_modules),
        "lora_target_examples": target_modules[:10],
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_examples": [name for name, _ in trainable[:10]],
        "visual_trainable_parameter_names": visual_trainable,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "attention": "sdpa",
    }
    return model, processor, freeze_report


def plot_loss(metrics: list[dict[str, Any]], output_dir: Path) -> str | None:
    if not metrics:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [item["global_step"] for item in metrics]
        losses = [item["loss"] for item in metrics]
        figure, axis = plt.subplots(figsize=(8, 4.8), dpi=160)
        axis.plot(steps, losses, marker="o", linewidth=1.6, markersize=3.5)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Training loss")
        axis.set_title("Qwen2.5-VL-7B 4-bit QLoRA smoke training")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        path = output_dir / "loss_curve.png"
        figure.savefig(path)
        plt.close(figure)
        return str(path)
    except Exception as exc:
        (output_dir / "loss_plot_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--rl-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-backup", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-steps", type=int, default=10)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs and gradient accumulation must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}; use a new run directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    random.seed(args.seed)

    sft_records = load_jsonl(args.sft_data)
    if args.max_train_records:
        sft_records = sft_records[: args.max_train_records]
    rl_rows = count_parquet_rows(args.rl_data)
    repo_root = Path(__file__).resolve().parents[1]
    snapshot_sft, snapshot_rl = copy_input_snapshot(
        args.sft_data.resolve(), args.rl_data.resolve(), args.output_dir / "input_snapshot"
    )
    model_inventory = file_inventory(args.model.resolve())
    model_fingerprint = inventory_fingerprint(model_inventory)
    base_backup_status = copy_model_backup(args.model.resolve(), args.base_model_backup.resolve())

    manifest: dict[str, Any] = {
        "run_status": "starting",
        "started_at": started,
        "training_script_version": TRAINING_SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "scorer_version": SCORER_VERSION,
        "git_revision": run_git_revision(repo_root),
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "base_model": {
            "path": str(args.model.resolve()),
            "backup_path": str(args.base_model_backup.resolve()),
            "backup_status": base_backup_status,
            "content_fingerprint": model_fingerprint,
            "file_count": len(model_inventory),
            "files": model_inventory,
        },
        "data": {
            "sft_source": str(args.sft_data.resolve()),
            "sft_snapshot": str(snapshot_sft),
            "sft_sha256": sha256_file(snapshot_sft),
            "sft_record_count": len(sft_records),
            "rl_source": str(args.rl_data.resolve()),
            "rl_snapshot": str(snapshot_rl),
            "rl_sha256": sha256_file(snapshot_rl),
            "rl_task_row_count": rl_rows,
            "rl_rows_used_for_sft_loss": 0,
            "rl_purpose": "held_out_rollout_evaluation_and_provenance",
        },
        "configuration": {
            "epochs": args.epochs,
            "max_length": args.max_length,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed,
            "device_index": args.device,
            "save_every_steps": args.save_every_steps,
            "resume_from": str(args.resume_from) if args.resume_from else None,
        },
        "outputs": {},
    }
    json_dump_atomic(args.output_dir / "run_manifest.json", manifest)

    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model, processor, freeze_report = build_model_and_processor(args)
        manifest["freeze_report"] = freeze_report
        manifest["library_versions"] = {
            "torch": torch.__version__,
        }
        try:
            import transformers
            import peft
            import bitsandbytes

            manifest["library_versions"].update(
                {
                    "transformers": transformers.__version__,
                    "peft": peft.__version__,
                    "bitsandbytes": bitsandbytes.__version__,
                }
            )
        except Exception:
            pass
        json_dump_atomic(args.output_dir / "run_manifest.json", manifest)

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        if args.resume_from:
            optimizer_state = args.resume_from / "optimizer.pt"
            if optimizer_state.is_file():
                optimizer.load_state_dict(torch.load(optimizer_state, map_location="cpu"))

        data_root = snapshot_sft.parent
        randomizer = random.Random(args.seed)
        metrics: list[dict[str, Any]] = []
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        micro_step = 0
        model.train()

        for epoch in range(args.epochs):
            epoch_records = list(sft_records)
            randomizer.shuffle(epoch_records)
            for record in epoch_records:
                micro_step += 1
                example = prepare_example(record, processor, data_root, args.max_length)
                tensor_batch = {
                    key: value
                    for key, value in example.items()
                    if key not in {"record_id"}
                }
                device = torch.device(f"cuda:{args.device}")
                tensor_batch = move_batch_to_device(tensor_batch, device, torch)
                outputs = model(**tensor_batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch + 1}, record={record.get('record_id')}"
                    )
                (loss / args.gradient_accumulation_steps).backward()
                should_step = (
                    micro_step % args.gradient_accumulation_steps == 0
                    or record is epoch_records[-1]
                )
                if not should_step:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                memory = (
                    torch.cuda.max_memory_allocated(args.device) / (1024 * 1024)
                    if torch.cuda.is_available()
                    else 0.0
                )
                metric = {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "micro_step": micro_step,
                    "record_id": record.get("record_id"),
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": args.learning_rate,
                    "max_memory_allocated_mb": round(memory, 2),
                    "timestamp": utc_now(),
                }
                metrics.append(metric)
                with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, ensure_ascii=False) + "\n")

                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        final_dir = args.output_dir / "adapter_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        torch.save(optimizer.state_dict(), args.output_dir / "optimizer_final.pt")
        plot_path = plot_loss(metrics, args.output_dir)
        json_dump_atomic(args.output_dir / "freeze_report.json", freeze_report)
        manifest["run_status"] = "completed"
        manifest["finished_at"] = utc_now()
        manifest["training"] = {
            "optimizer_steps": global_step,
            "micro_steps": micro_step,
            "metrics_count": len(metrics),
            "final_loss": metrics[-1]["loss"] if metrics else None,
        }
        manifest["outputs"] = {
            "adapter_final": str(final_dir),
            "loss_curve": plot_path,
            "metrics": str(args.output_dir / "metrics.jsonl"),
            "freeze_report": str(args.output_dir / "freeze_report.json"),
        }
        json_dump_atomic(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        manifest["run_status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["traceback"] = traceback.format_exc()
        json_dump_atomic(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
