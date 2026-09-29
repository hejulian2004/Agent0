"""Canonical Responses trajectory dataset for Qwen supervised training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from agent0_protocol.adapters import QwenModelAdapter
from agent0_protocol.schema import CanonicalTrajectory
from agent0_protocol.tools import get_tool_registry
from verl.utils import hf_tokenizer


class Agent0SFTDataset(Dataset):
    """Train on assistant semantic items while masking user and tool observations."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: Any,
        processor: Any | None = None,
        prompt_key: str = "prompt",
        response_key: str = "response",
        image_key: str = "images",
        data_source_key: str = "data_source",
        max_length: int = 8192,
        max_prompt_length: int | None = None,
        truncation: str = "right",
        filter_overlong: bool = False,
        cache_dir: str = "~/.cache/verl/agent0_sft",
        num_workers: int | None = None,
        max_pixels: int = 2048 * 2048,
        min_pixels: int = 512 * 512,
        tensor_only: bool = False,
    ) -> None:
        del prompt_key, response_key, image_key, max_prompt_length, filter_overlong
        del cache_dir, num_workers, max_pixels, min_pixels
        if truncation not in {"error", "left", "right"}:
            raise ValueError("invalid truncation")
        paths = list(parquet_files) if isinstance(parquet_files, (list, ListConfig)) else [parquet_files]
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.adapter = QwenModelAdapter(tokenizer)
        self.processor = processor
        self.max_length = max_length
        self.truncation = truncation
        self.tensor_only = tensor_only
        self.data_source_key = data_source_key
        frames = []
        for value in paths:
            path = Path(value)
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            elif path.suffix == ".jsonl":
                frame = pd.read_json(path, lines=True)
            else:
                raise ValueError(f"unsupported SFT file: {path}")
            if "trajectory" not in frame.columns:
                raise ValueError(f"SFT file requires trajectory column: {path}")
            frames.append(frame)
        self.dataframe = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        value = row["trajectory"]
        if isinstance(value, str):
            value = json.loads(value)
        trajectory = CanonicalTrajectory.from_dict(value)
        if trajectory.tools != get_tool_registry().definitions():
            raise ValueError("SFT trajectory tool definitions differ from registry")
        if self.processor is None and any(
            item.get("type") == "message" and (
                (isinstance(item.get("content"), str) and "<image>" in item["content"])
                or (isinstance(item.get("content"), list) and any(
                    part.get("type") == "input_image" for part in item["content"]
                    if isinstance(part, dict)
                ))
            ) for item in trajectory.items
        ):
            raise ValueError("image SFT requires a VL processor and image tensors")

        token_ids: list[int] = []
        target_mask: list[bool] = []
        for item in trajectory.items:
            text = self.adapter.render([item], trajectory.tools, generate=False)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            trainable = item["type"] in {"reasoning", "function_call"} or (
                item["type"] == "message" and item.get("role") == "assistant"
            )
            token_ids.extend(ids)
            target_mask.extend([trainable] * len(ids))
        if not token_ids or not any(target_mask):
            raise ValueError("SFT trajectory has no trainable assistant tokens")

        if len(token_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError("SFT trajectory exceeds max_length")
            if self.truncation == "left":
                token_ids = token_ids[-self.max_length:]
                target_mask = target_mask[-self.max_length:]
            else:
                token_ids = token_ids[:self.max_length]
                target_mask = target_mask[:self.max_length]
        real_length = len(token_ids)
        padding = self.max_length - real_length
        token_ids.extend([self.tokenizer.pad_token_id or 0] * padding)
        attention_values = [1] * real_length + [0] * padding
        target_mask.extend([False] * padding)
        # The trainer shifts logits by one position before multiplying this mask.
        loss_values = [bool(target_mask[i + 1]) if i + 1 < real_length else False
                       for i in range(self.max_length)]
        attention_mask = torch.tensor(attention_values, dtype=torch.long)
        result: dict[str, Any] = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "position_ids": (attention_mask.cumsum(dim=0) - 1).clamp(min=0),
            "loss_mask": torch.tensor(loss_values, dtype=torch.long),
        }
        if not self.tensor_only:
            result["data_source"] = row.get(self.data_source_key, "unknown")
            result["trajectory_id"] = trajectory.trajectory_id
        return result


class Agent0MultiTurnSFTDataset(Agent0SFTDataset):
    """The same canonical trajectory loader; all assistant items are trained."""

    def __init__(self, *args: Any, train_on_all_assistant: bool = True, **kwargs: Any) -> None:
        if not train_on_all_assistant:
            raise ValueError("canonical multi-turn SFT trains all assistant items")
        super().__init__(*args, **kwargs)
