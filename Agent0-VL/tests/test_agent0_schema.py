"""Dataset records must embed the canonical Responses trajectory."""

from __future__ import annotations

import json

import pytest

from agent0_protocol.schema import CanonicalTrajectory, ProtocolError, SCHEMA_VERSION
from agent0_protocol.tools import get_tool_registry
from tools.data_builder.schema import PROTOCOL_VERSION


def test_schema_version_matches_dataset_protocol():
    assert SCHEMA_VERSION == PROTOCOL_VERSION == "agent0.responses.v1"


def test_tool_snapshot_and_items_roundtrip():
    registry = get_tool_registry()
    trajectory = CanonicalTrajectory("record_1", registry.definitions())
    trajectory.append({"type": "message", "role": "user", "content": "Calculate 1+1"})
    trajectory.append({"type": "function_call", "name": "python_exec", "call_id": "call_1", "arguments": {"code": "print(1+1)"}})
    trajectory.append({"type": "function_call_output", "call_id": "call_1", "output": {"success": True, "stdout": "2"}})
    trajectory.append({"type": "message", "role": "assistant", "content": "2"})
    encoded = json.dumps(trajectory.to_dict())
    decoded = CanonicalTrajectory.from_dict(json.loads(encoded))
    assert decoded.tools == registry.definitions()
    assert decoded.items == trajectory.items


def test_old_record_rejected():
    with pytest.raises(ProtocolError):
        CanonicalTrajectory.from_dict({"schema_version": "agent0vl.dataset.v1", "trajectory_id": "old", "tools": [], "items": []})


def test_sft_dataset_reads_canonical_jsonl(tmp_path):
    from verl.utils.dataset.agent0_sft_dataset import Agent0SFTDataset

    class TinyTokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return list(text.encode("utf-8"))

    trajectory = CanonicalTrajectory("sft_1", get_tool_registry().definitions())
    trajectory.append({"type": "message", "role": "user", "content": "1+1?"})
    trajectory.append({"type": "message", "role": "assistant", "content": "2"})
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"trajectory": trajectory.to_dict()}) + "\n")
    dataset = Agent0SFTDataset(str(path), TinyTokenizer(), max_length=512)
    item = dataset[0]
    assert item["input_ids"].shape[0] == 512
    assert item["loss_mask"].sum() > 0
    assert item["trajectory_id"] == "sft_1"


def test_rl_dataset_reads_canonical_parquet(tmp_path):
    import pandas as pd
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from verl.utils.dataset.rl_dataset import RLHFDataset

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    trajectory = CanonicalTrajectory("rl_1", get_tool_registry().definitions())
    trajectory.append({"type": "message", "role": "user", "content": "hello"})
    path = tmp_path / "rl.parquet"
    pd.DataFrame([{"canonical_trajectory_json": json.dumps(trajectory.to_dict()),
                   "prompt": [{"role": "user", "content": "ignored"}],
                   "reward_model": {"ground_truth": "hello"}}]).to_parquet(path)
    dataset = RLHFDataset(str(path), tokenizer, max_prompt_length=256, num_workers=1)
    item = dataset[0]
    assert item["input_ids"].shape[0] == 256
    assert item["raw_prompt_ids"]
    assert "prompt" not in item
