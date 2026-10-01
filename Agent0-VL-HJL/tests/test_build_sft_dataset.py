"""Unit tests for the canonical SFT dataset builder and audit system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
from agent0_protocol.schema import CanonicalTrajectory, SCHEMA_VERSION
from agent0_protocol.tools import ToolExecutionContext, get_tool_registry
from scripts.build_sft_dataset import (
    AnswerJudge,
    BuildStats,
    FormatCleaner,
    SFTTrajectoryBuilder,
    run_task_rollouts_concurrent,
    verify_sft_dataset_compatibility,
)
from verl.prompts.agent0_templates import render_system_prompt
from verl.utils.dataset.agent0_sft_dataset import Agent0SFTDataset


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 1, "<|endoftext|>": 2}.get(token, 0)


# ==============================================================================
# 1. Answer Judge Tests
# ==============================================================================

def test_answer_judge_extraction():
    assert AnswerJudge.extract_answer("<answer>42</answer>") == "42"
    assert AnswerJudge.extract_answer("Work...\n\\boxed{12.5}\nDone") == "12.5"
    assert AnswerJudge.extract_answer("The answer is: 100") == "100"
    assert AnswerJudge.extract_answer("Therefore, final answer: B.") == "B"
    assert AnswerJudge.extract_answer("Raw conclusion 99") == "Raw conclusion 99"


def test_answer_judge_numerical_and_option_equivalence():
    # Fractions & Floats
    assert AnswerJudge.is_equivalent("0.5", "1/2")
    assert AnswerJudge.is_equivalent("50%", "0.5")
    assert AnswerJudge.is_equivalent("12.500", "12.5")
    assert AnswerJudge.is_equivalent("<answer>100</answer>", "100.0001", tolerance=1e-3)
    assert not AnswerJudge.is_equivalent("50", "40")

    # Options mapping
    q = "What is 2+2? (A) 3 (B) 4 (C) 5"
    assert AnswerJudge.is_equivalent("B", "4", question=q)
    assert AnswerJudge.is_equivalent("4", "B", question=q)
    assert AnswerJudge.is_equivalent("B", "b", question=q)


# ==============================================================================
# 2. Format Cleaner & Audit Tests
# ==============================================================================

def test_format_cleaner_control_characters_and_images(tmp_path):
    raw_text = "Good text\x00\x07with\nnewlines\tand control\x1fchars"
    cleaned = FormatCleaner.strip_control_characters(raw_text)
    assert cleaned == "Good textwith\nnewlines\tand controlchars"

    img_file = tmp_path / "test.png"
    Image.new("RGB", (10, 10), "red").save(img_file, format="PNG")
    assert FormatCleaner.is_image_valid(str(img_file))
    assert not FormatCleaner.is_image_valid("/nonexistent/file.png")
    assert FormatCleaner.is_image_valid("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def test_format_cleaner_audit_and_deduplication():
    builder = SFTTrajectoryBuilder()
    traj = CanonicalTrajectory("audit_1", builder.canonical_tools)
    traj.append({"type": "message", "role": "system", "content": render_system_prompt()})
    traj.append({"type": "message", "role": "user", "content": "Calculate 1+1"})
    traj.append({"type": "function_call", "call_id": "c1", "name": "python_exec", "arguments": {"code": "print(2)"}})
    traj.append({"type": "function_call_output", "call_id": "c1", "output": {"success": True, "stdout": "2"}})
    traj.append({"type": "message", "role": "assistant", "content": "<answer>2</answer>"})

    # Valid audit
    ok, reason = FormatCleaner.audit_trajectory(traj, min_steps=3, ground_truth="2")
    assert ok is True
    assert reason == "ok"

    # Mismatched ground truth rejection
    ok_bad, reason_bad = FormatCleaner.audit_trajectory(traj, min_steps=3, ground_truth="99")
    assert ok_bad is False
    assert "ground_truth_mismatch" in reason_bad

    # Content hashing
    h1 = FormatCleaner.compute_trajectory_hash(traj)
    h2 = FormatCleaner.compute_trajectory_hash(traj)
    assert h1 == h2
    assert len(h1) == 64


# ==============================================================================
# 3. SFT Synthesis & Export Tests
# ==============================================================================

def test_builder_synthesis_and_export(tmp_path):
    builder = SFTTrajectoryBuilder()

    # Synthesize Stage 1 (Vision) and Stage 2 (Math)
    s1_trajs = builder.synthesize_trajectories(4, stage=1, seed=42)
    s2_trajs = builder.synthesize_trajectories(4, stage=2, seed=42)
    all_trajs = s1_trajs + s2_trajs

    assert len(all_trajs) == 8
    for t in all_trajs:
        assert len(t.items) >= 4
        assert t.items[0]["role"] == "system"
        assert t.items[-1]["role"] == "assistant"

    # Export
    out_dir = tmp_path / "sft_out"
    manifest = builder.export(
        all_trajs,
        out_dir,
        train_ratio=0.75,
        export_format="both",
        seed=42,
    )

    assert manifest["total_records"] == 8
    assert manifest["train_records"] == 6
    assert manifest["val_records"] == 2
    assert (out_dir / "train.parquet").is_file()
    assert (out_dir / "val.parquet").is_file()
    assert (out_dir / "train.jsonl").is_file()
    assert (out_dir / "val.jsonl").is_file()
    assert (out_dir / "sft_manifest.json").is_file()

    # Loader verification using Agent0SFTDataset
    check = verify_sft_dataset_compatibility(out_dir / "train.parquet")
    assert check["status"] == "passed"
    assert check["dataset_len"] == 6
    assert check["loss_mask_trainable_tokens"] > 0


# ==============================================================================
# 4. Concurrent Rollout Execution Tests
# ==============================================================================

def test_concurrent_task_rollout_runner(tmp_path):
    builder = SFTTrajectoryBuilder()

    # Mock responses runtime
    class FakeResponsesClient:
        def __init__(self):
            self.count = 0

        def create(self, **kwargs):
            self.count += 1
            inp = kwargs.get("input", [])
            # If function output already received, return final message
            if any(isinstance(x, dict) and x.get("type") == "function_call_output" for x in inp):
                return SimpleNamespace(id="resp_2", output=[{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The calculation gives <answer>42</answer>."}],
                }])
            # First turn: call python_exec
            return SimpleNamespace(id="resp_1", output=[{
                "type": "function_call",
                "call_id": f"call_{self.count}",
                "name": "python_exec",
                "arguments": json.dumps({"code": "print(42)"}),
            }])

    cfg = ResponsesConfig("http://localhost:8000/v1", "test", "test-model")
    runtime = ResponsesRuntime(cfg, builder.registry, client=SimpleNamespace(responses=FakeResponsesClient()), probe_on_init=False)

    tasks = [
        {"id": f"task_{i}", "question": f"Compute task {i}", "ground_truth": "42"}
        for i in range(5)
    ]

    out_jsonl = tmp_path / "stream_out.jsonl"
    trajectories, stats = run_task_rollouts_concurrent(
        tasks,
        runtime,
        builder,
        out_jsonl,
        concurrency=2,
        resume=True,
        min_steps=3,
    )

    assert len(trajectories) == 5
    assert stats.exported == 5
    assert stats.tool_calls == 5
    assert stats.answer_judge_passes == 5
    assert out_jsonl.is_file()
    assert Path(str(out_jsonl) + ".state.json").is_file()

    # Test resume: running again should skip already completed tasks
    trajectories_resumed, stats_resumed = run_task_rollouts_concurrent(
        tasks,
        runtime,
        builder,
        out_jsonl,
        concurrency=2,
        resume=True,
    )
    # Already completed tasks are skipped
    assert stats_resumed.attempted == 0
    assert len(trajectories_resumed) == 5
