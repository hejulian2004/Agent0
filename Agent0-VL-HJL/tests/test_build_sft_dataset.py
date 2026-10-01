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


def test_format_cleaner_normalizes_absolute_to_relative_paths():
    builder = SFTTrajectoryBuilder()
    traj = CanonicalTrajectory("rel_path_1", builder.canonical_tools)
    traj.append({"type": "message", "role": "system", "content": render_system_prompt()})

    # Simulate trajectory containing local absolute path
    from scripts.build_sft_dataset import ROOT
    abs_img = str(ROOT / "outputs" / "intermediate_images" / "test_abs.png")
    traj.append({"type": "message", "role": "user", "content": f"Image: {abs_img}\nInspect image."})
    traj.append({"type": "function_call", "call_id": "c1", "name": "crop_image", "arguments": {"image_path": abs_img, "bbox": [0, 0, 10, 10]}})
    traj.append({"type": "function_call_output", "call_id": "c1", "output": {"success": True, "image_path": abs_img, "output_path": abs_img, "image_size": [10, 10]}})
    traj.append({"type": "message", "role": "assistant", "content": f"The region was saved to {abs_img}."})

    # Clean trajectory text
    FormatCleaner.clean_trajectory_text(traj)

    # All absolute paths must be normalized to relative paths
    user_text = traj.items[1]["content"]
    assert str(ROOT) not in user_text
    assert "outputs/intermediate_images/test_abs.png" in user_text

    call_args = traj.items[2]["arguments"]
    assert str(ROOT) not in call_args["image_path"]
    assert call_args["image_path"] == "outputs/intermediate_images/test_abs.png"

    call_out = traj.items[3]["output"]
    assert str(ROOT) not in call_out["image_path"]
    assert call_out["image_path"] == "outputs/intermediate_images/test_abs.png"


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


def test_builder_export_all_train_records_default_ratio(tmp_path):
    builder = SFTTrajectoryBuilder()
    trajs = builder.synthesize_trajectories(5, stage=1, seed=123)
    out_dir = tmp_path / "sft_full_train"
    manifest = builder.export(
        trajs,
        out_dir,
        train_ratio=1.0,
        export_format="both",
        seed=123,
    )

    assert manifest["total_records"] == 5
    assert manifest["train_records"] == 5
    assert manifest["val_records"] == 0
    assert (out_dir / "train.parquet").is_file()
    assert (out_dir / "train.jsonl").is_file()
    assert not (out_dir / "val.parquet").exists()
    assert not (out_dir / "val.jsonl").exists()

    check = verify_sft_dataset_compatibility(out_dir / "train.parquet")
    assert check["status"] == "passed"
    assert check["dataset_len"] == 5
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


def test_teacher_backend_selection_defaults(monkeypatch):
    monkeypatch.delenv("AGENT0_RESPONSES_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT0_RESPONSES_MODEL", raising=False)
    monkeypatch.delenv("AGENT0_RESPONSES_API_KEY", raising=False)

    from scripts.build_sft_dataset import main
    import sys

    # Dry-run check for local default backend
    test_args = ["scripts.build_sft_dataset", "--synthesize-count", "2", "--output-dir", "/tmp/test_sft_local", "--format", "jsonl"]
    monkeypatch.setattr(sys, "argv", test_args)
    res = main()
    assert res == 0
