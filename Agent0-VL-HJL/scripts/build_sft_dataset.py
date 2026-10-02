"""Build canonical SFT datasets preserving complete tool reasoning trajectories.

Implements format cleaning, deduplication, multi-tier answer judging,
and full trajectory auditing based on the Agent0-VL SERC specification.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd
from PIL import Image, UnidentifiedImageError

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
from agent0_protocol.schema import CanonicalTrajectory, SCHEMA_VERSION, ProtocolError, new_call_id
from agent0_protocol.tools import (
    ToolExecutionContext,
    ToolRegistry,
    execute_call_batch,
    get_tool_registry,
    _get_intermediate_image_dir,
    _to_relative_path,
    _resolve_relative_path,
)
from agent0_protocol.local_prompts import render_system_prompt
from tools.data_builder.sources import SOURCE_STAGES

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Answer Extraction and Equivalence Judging
# ==============================================================================

from tools.data_builder.sft_quality import (
    StrictAnswerJudge as AnswerJudge, solver_text, verify_sft_semantics,
)


# ==============================================================================
# 2. Format Cleaning and Deduplication
# ==============================================================================

class FormatCleaner:
    """Validate format invariants, clean control characters, and deduplicate trajectories."""

    @staticmethod
    def strip_control_characters(text: str) -> str:
        """Remove ASCII control characters except newline, tab, and carriage return."""
        return "".join(c for c in text if ord(c) >= 32 or c in "\n\t\r")

    @staticmethod
    def is_image_valid(image: str) -> bool:
        """Verify image path or base64 data URL."""
        if not image or not isinstance(image, str):
            return False
        if image.startswith("data:"):
            try:
                header, sep, payload = image.partition(",")
                if not sep or ";base64" not in header:
                    return False
                base64.b64decode(payload, validate=True)
                return True
            except (ValueError, binascii.Error):
                return False
        if image.startswith(("http://", "https://")):
            parsed = urlparse(image)
            return bool(parsed.netloc and parsed.path)
        path = _resolve_relative_path(image)
        if not path.is_file():
            return False
        try:
            with Image.open(path) as opened:
                opened.verify()
            return True
        except (OSError, UnidentifiedImageError):
            return False

    @classmethod
    def clean_trajectory_text(cls, trajectory: CanonicalTrajectory) -> None:
        """Clean control characters and normalize all local absolute paths to relative paths."""
        root_str = str(ROOT).rstrip("/") + "/"
        for item in trajectory.items:
            kind = item.get("type")
            if kind == "message":
                content = item.get("content")
                if isinstance(content, str):
                    text = cls.strip_control_characters(content)
                    if root_str in text:
                        text = text.replace(root_str, "")
                    item["content"] = text
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text = cls.strip_control_characters(str(part["text"]))
                            if root_str in text:
                                text = text.replace(root_str, "")
                            part["text"] = text
            elif kind == "reasoning":
                for part in item.get("summary", []):
                    if isinstance(part, dict) and "text" in part:
                        text = cls.strip_control_characters(str(part["text"]))
                        if root_str in text:
                            text = text.replace(root_str, "")
                        part["text"] = text
            elif kind == "function_call":
                args = item.get("arguments", {})
                for k, v in args.items():
                    if isinstance(v, str) and root_str in v:
                        args[k] = v.replace(root_str, "")
            elif kind == "function_call_output":
                out = item.get("output", {})
                if isinstance(out, dict):
                    for k in ("image_path", "output_path"):
                        if isinstance(out.get(k), str) and root_str in out[k]:
                            out[k] = out[k].replace(root_str, "")

    @staticmethod
    def compute_trajectory_hash(trajectory: CanonicalTrajectory) -> str:
        """Compute deterministic SHA-256 fingerprint of trajectory content."""
        hasher = hashlib.sha256()
        for item in trajectory.items:
            hasher.update(str(item.get("type", "")).encode("utf-8"))
            if item.get("type") == "message":
                hasher.update(str(item.get("role", "")).encode("utf-8"))
                hasher.update(str(item.get("content", "")).encode("utf-8"))
            elif item.get("type") == "function_call":
                hasher.update(str(item.get("name", "")).encode("utf-8"))
                hasher.update(json.dumps(item.get("arguments", {}), sort_keys=True).encode("utf-8"))
            elif item.get("type") == "function_call_output":
                hasher.update(json.dumps(item.get("output", {}), sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def audit_trajectory(
        cls,
        trajectory: CanonicalTrajectory,
        *,
        min_steps: int = 3,
        ground_truth: str | None = None,
    ) -> tuple[bool, str]:
        """Audit trajectory compliance with agent0.responses.v1 and SFT trainability."""
        try:
            trajectory.validate()
        except Exception as exc:
            return False, f"protocol_validation_failed: {exc}"

        if len(trajectory.items) < min_steps:
            return False, f"trajectory_too_short ({len(trajectory.items)} < {min_steps})"

        # 1. System prompt check
        if not trajectory.items or trajectory.items[0].get("type") != "message" or trajectory.items[0].get("role") != "system":
            return False, "missing_system_message"

        # 2. User prompt check
        if len(trajectory.items) < 2 or trajectory.items[1].get("type") != "message" or trajectory.items[1].get("role") != "user":
            return False, "missing_initial_user_message"

        # 3. Trainable tokens check
        has_trainable = any(
            item["type"] in {"reasoning", "function_call"} or
            (item["type"] == "message" and item.get("role") == "assistant")
            for item in trajectory.items
        )
        if not has_trainable:
            return False, "no_trainable_assistant_items"

        # 4. Final message must be assistant
        if trajectory.items[-1].get("type") != "message" or trajectory.items[-1].get("role") != "assistant":
            return False, "final_item_not_assistant"

        # 5. Answer correctness check against ground truth if provided
        if ground_truth:
            final_text = solver_text(trajectory)

            question_text = ""
            if len(trajectory.items) > 1 and trajectory.items[1].get("type") == "message":
                q_content = trajectory.items[1].get("content", "")
                if isinstance(q_content, str):
                    question_text = q_content
                elif isinstance(q_content, list):
                    question_text = "".join(str(p.get("text", "")) for p in q_content if isinstance(p, dict))

            if not AnswerJudge.is_equivalent(final_text, ground_truth, question=question_text):
                return False, f"ground_truth_mismatch (prediction={final_text!r}, ground_truth={ground_truth!r})"

        return True, "ok"


# ==============================================================================
# 3. SFT Dataset Builder
# ==============================================================================

def _create_synthetic_image(color: str = "blue", width: int = 120, height: int = 80) -> Path:
    """Create a persistent visual image in outputs/intermediate_images."""
    target_dir = _get_intermediate_image_dir()
    filename = f"sft_base_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.png"
    filepath = target_dir / filename
    img = Image.new("RGB", (width, height), color=color)
    for x in range(width // 4, width // 2):
        for y in range(height // 4, height // 2):
            img.putpixel((x, y), (240, 240, 240))
    img.save(filepath, format="PNG")
    return filepath


class SFTTrajectoryBuilder:
    """Ingest, synthesize, clean, audit, and export complete SFT trajectories."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_tool_registry()
        self.canonical_tools = self.registry.definitions()
        self.answer_judge_hashes = set()

    def normalize_trajectory(self, trajectory: CanonicalTrajectory) -> CanonicalTrajectory:
        """Ensure tool snapshot strictly matches current registry definitions and clean text."""
        if "serc" in trajectory.metadata:
            # Generated role evidence must retain the exact observed bytes.
            trajectory.validate()
            return trajectory
        if trajectory.items and trajectory.items[0].get("role") == "system" and trajectory.items[0].get("content") != render_system_prompt():
            raise ProtocolError("conflicting_system_prompt")
        trajectory.tools = copy.deepcopy(self.canonical_tools)
        if not trajectory.items or trajectory.items[0].get("type") != "message" or trajectory.items[0].get("role") != "system":
            trajectory.items.insert(0, {
                "type": "message",
                "role": "system",
                "content": render_system_prompt(),
            })
        FormatCleaner.clean_trajectory_text(trajectory)
        trajectory.validate()
        return trajectory

    def load_from_jsonl(self, jsonl_path: Path) -> list[CanonicalTrajectory]:
        """Load and clean trajectories from a JSONL file."""
        trajectories: list[CanonicalTrajectory] = []
        if not jsonl_path.is_file():
            logger.warning(f"File not found: {jsonl_path}")
            return []

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    traj_data = data.get("trajectory", data)
                    if isinstance(traj_data, dict) and "items" in traj_data:
                        if "serc" not in traj_data.get("metadata", {}):
                            traj_data["tools"] = copy.deepcopy(self.canonical_tools)
                        if "trajectory_id" not in traj_data:
                            traj_data["trajectory_id"] = f"traj_jsonl_{line_no}_{uuid.uuid4().hex[:6]}"
                        traj = CanonicalTrajectory.from_dict(traj_data)
                        trajectories.append(self.normalize_trajectory(traj))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ProtocolError(f"Invalid input at {jsonl_path}:{line_no}") from exc
                except Exception as exc:
                    logger.warning("Rejected input row %s:%s: %s", jsonl_path, line_no, type(exc).__name__)

        return trajectories

    def load_from_json_dir(self, dir_path: Path) -> list[CanonicalTrajectory]:
        """Load and clean all canonical JSON trajectory files from a directory."""
        trajectories: list[CanonicalTrajectory] = []
        if not dir_path.is_dir():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        for p in sorted(dir_path.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "items" in data:
                    if "serc" not in data.get("metadata", {}):
                        data["tools"] = copy.deepcopy(self.canonical_tools)
                    traj = CanonicalTrajectory.from_dict(data)
                    trajectories.append(self.normalize_trajectory(traj))
            except Exception as exc:
                logger.debug(f"Skipping {p}: {exc}")

        return trajectories

    def synthesize_trajectories(
        self,
        count: int,
        *,
        stage: int = 1,
        seed: int = 42,
    ) -> list[CanonicalTrajectory]:
        """Synthesize verified, full-trajectory tool interactions for Stage 1 or Stage 2."""
        rng = random.Random(seed)
        synthesized: list[CanonicalTrajectory] = []

        # Stage 1: Visual Perception & Image Manipulation
        stage1_templates = [
            {
                "topic": "vision_crop_inspect",
                "question": "Crop the upper-left quadrant [0, 0, 40, 30] of this image and analyze its dimensions and color properties.",
                "color": "blue",
                "crop_bbox": [0, 0, 40, 30],
                "thought": "I will crop the specified upper-left region from the image using crop_image, then analyze the resulting image with visual_analyzer.",
                "answer": "The upper-left quadrant was cropped and analyzed. The resulting region has size [40, 30] with distinct color contrast.",
            },
            {
                "topic": "vision_zoom_inspect",
                "question": "Magnify the image by a scale factor of 2.0 and verify its updated dimensions.",
                "color": "green",
                "scale": 2.0,
                "thought": "I will apply zoom_image with a 2.0x scale factor to examine higher resolution details.",
                "answer": "The image has been zoomed by a factor of 2.0, successfully magnifying all fine surface features.",
            },
            {
                "topic": "vision_rotate_align",
                "question": "Rotate the image by 90 degrees clockwise to align with the visual reference frame.",
                "color": "purple",
                "angle": 90.0,
                "thought": "I will rotate the image by 90 degrees using rotate_image and inspect the resulting visual geometry.",
                "answer": "The image was rotated 90 degrees clockwise, successfully aligning the orientation.",
            },
            {
                "topic": "vision_multistep_crop_zoom",
                "question": "Extract the center region [10, 10, 50, 40] of the image and then zoom in by 1.5x for detailed inspection.",
                "color": "navy",
                "crop_bbox": [10, 10, 50, 40],
                "scale": 1.5,
                "thought": "First, I will crop the center region. Then I will pass the cropped image path to zoom_image for 1.5x magnification.",
                "answer": "The center region was cropped and subsequently magnified by 1.5x, confirming sharp and detailed visual structures.",
            },
        ]

        # Stage 2: Complex Math & Symbolic Reasoning with Python & Retrieval
        stage2_templates = [
            {
                "topic": "math_python_perimeter",
                "question": "A rectangle has a perimeter of 56 cm and its length is 3 times its width. Find the width, length, and area of the rectangle in square centimeters.",
                "code": "w = 56 / (2 * (3 + 1))\nl = 3 * w\narea = l * w\nprint(f'w={w}, l={l}, area={area}')",
                "thought": "Let width be w. Perimeter P = 2*(l + w) = 2*(3w + w) = 8w = 56, so w = 7 cm, l = 21 cm. Area = 21 * 7 = 147 cm^2.",
                "answer": "The width is 7 cm, the length is 21 cm, and the area is 147 square centimeters.",
            },
            {
                "topic": "math_python_quadratic",
                "question": "Find the roots of the quadratic equation 2*x^2 - 14*x + 20 = 0 and calculate their product and sum.",
                "code": "import math\na, b, c = 2, -14, 20\nd = b**2 - 4*a*c\nr1 = (-b + math.sqrt(d)) / (2*a)\nr2 = (-b - math.sqrt(d)) / (2*a)\nprint(f'roots=({r1}, {r2}), sum={r1+r2}, prod={r1*r2}')",
                "thought": "Using the quadratic formula for 2x^2 - 14x + 20 = 0: discriminant d = 196 - 160 = 36. Roots are (14 +/- 6)/4, which gives 5 and 2.",
                "answer": "The roots of the equation are x = 5 and x = 2. Their sum is 7 and their product is 10.",
            },
            {
                "topic": "knowledge_retrieval_agent",
                "question": "What are the core stages of the Self-Evolving Reasoning Cycle (SERC) in Agent0?",
                "query": "SERC Self-Evolving Reasoning Cycle",
                "thought": "I will query local knowledge documents for the definition and stages of SERC.",
                "answer": "The Self-Evolving Reasoning Cycle (SERC) comprises multi-step reasoning with sandboxed tool execution, step-by-step verification, and confidence-gated self-repair.",
            },
            {
                "topic": "math_python_statistics",
                "question": "Calculate the mean, variance, and standard deviation of the dataset: [12, 15, 23, 29, 31, 38, 45].",
                "code": "import numpy as np\ndata = [12, 15, 23, 29, 31, 38, 45]\nmean = np.mean(data)\nvar = np.var(data)\nstd = np.std(data)\nprint(f'mean={mean:.2f}, var={var:.2f}, std={std:.2f}')",
                "thought": "I will calculate the statistical summary metrics using Python.",
                "answer": "The mean of the dataset is 27.57, the population variance is 126.82, and the standard deviation is approximately 11.26.",
            },
        ]

        active_templates = stage1_templates if stage == 1 else stage2_templates

        for i in range(count):
            traj_id = f"sft_stage{stage}_{i+1:04d}_{uuid.uuid4().hex[:6]}"
            traj = CanonicalTrajectory(
                trajectory_id=traj_id,
                tools=copy.deepcopy(self.canonical_tools),
                metadata={
                    "source": f"synthetic_sft_stage{stage}",
                    "stage": stage,
                    "index": i + 1,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # System message
            traj.append({
                "type": "message",
                "role": "system",
                "content": render_system_prompt(),
            })

            if stage == 1:
                # Stage 1: Visual Perception & Image Operations
                pattern = i % 4
                colors = ["blue", "green", "navy", "purple", "teal", "maroon", "olive", "coral"]
                col = colors[i % len(colors)]
                base_img = _to_relative_path(_create_synthetic_image(color=col, width=120 + (i % 5) * 10, height=80 + (i % 5) * 8))
                context = ToolExecutionContext(image=_resolve_relative_path(base_img))
                try:
                    if pattern == 0:
                        # Crop + Analyze
                        bbox = [i % 5, i % 5, 40 + (i % 10), 30 + (i % 10)]
                        traj.append({"type": "message", "role": "user", "content": f"Image: {base_img}\nCrop the region {bbox} and analyze color properties."})
                        traj.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": f"I will crop region {bbox} from {base_img} and analyze the resulting image."}]})
                        c1 = new_call_id()
                        call1 = {"type": "function_call", "call_id": c1, "name": "crop_image", "arguments": {"image_path": str(base_img), "bbox": bbox}}
                        traj.append(call1)
                        out1 = self.registry.execute(call1, context)
                        traj.append({"type": "function_call_output", "call_id": c1, "output": out1})

                        c2 = new_call_id()
                        crop_p = out1.get("image_path", str(base_img))
                        call2 = {"type": "function_call", "call_id": c2, "name": "visual_analyzer", "arguments": {"image_path": str(crop_p)}}
                        traj.append(call2)
                        out2 = self.registry.execute(call2, context)
                        traj.append({"type": "function_call_output", "call_id": c2, "output": out2})

                        traj.append({"type": "message", "role": "assistant", "content": f"<answer>The region {bbox} was cropped and analyzed. Dominant color matches {col} background.</answer>"})

                    elif pattern == 1:
                        # Zoom
                        scale = round(1.5 + (i % 4) * 0.5, 1)
                        traj.append({"type": "message", "role": "user", "content": f"Image: {base_img}\nMagnify this image by a factor of {scale}."})
                        c = new_call_id()
                        call = {"type": "function_call", "call_id": c, "name": "zoom_image", "arguments": {"image_path": str(base_img), "scale": scale}}
                        traj.append(call)
                        out = self.registry.execute(call, context)
                        traj.append({"type": "function_call_output", "call_id": c, "output": out})
                        traj.append({"type": "message", "role": "assistant", "content": f"<answer>The image was magnified by {scale}x, resulting in dimensions {out.get('image_size')}.</answer>"})

                    elif pattern == 2:
                        # Rotate
                        angles = [90.0, 180.0, 270.0, -90.0]
                        ang = angles[i % len(angles)]
                        traj.append({"type": "message", "role": "user", "content": f"Image: {base_img}\nRotate this image by {ang} degrees."})
                        c = new_call_id()
                        call = {"type": "function_call", "call_id": c, "name": "rotate_image", "arguments": {"image_path": str(base_img), "angle": ang}}
                        traj.append(call)
                        out = self.registry.execute(call, context)
                        traj.append({"type": "function_call_output", "call_id": c, "output": out})
                        traj.append({"type": "message", "role": "assistant", "content": f"<answer>The image was rotated by {ang} degrees successfully.</answer>"})

                    else:
                        # Multi-step: Crop + Zoom
                        bbox = [10 + (i % 5), 10 + (i % 5), 50 + (i % 5), 40 + (i % 5)]
                        scale = 1.5
                        traj.append({"type": "message", "role": "user", "content": f"Image: {base_img}\nExtract box {bbox} and magnify it by {scale}x."})
                        c1 = new_call_id()
                        call1 = {"type": "function_call", "call_id": c1, "name": "crop_image", "arguments": {"image_path": str(base_img), "bbox": bbox}}
                        traj.append(call1)
                        out1 = self.registry.execute(call1, context)
                        traj.append({"type": "function_call_output", "call_id": c1, "output": out1})

                        c2 = new_call_id()
                        crop_p = out1.get("image_path", str(base_img))
                        call2 = {"type": "function_call", "call_id": c2, "name": "zoom_image", "arguments": {"image_path": str(crop_p), "scale": scale}}
                        traj.append(call2)
                        out2 = self.registry.execute(call2, context)
                        traj.append({"type": "function_call_output", "call_id": c2, "output": out2})
                        traj.append({"type": "message", "role": "assistant", "content": f"<answer>Region {bbox} was cropped and magnified {scale}x to size {out2.get('image_size')}.</answer>"})

                finally:
                    context.close()

            else:
                # Stage 2: Complex Math & Symbolic Reasoning with Python & Retrieval
                pattern = i % 4
                if pattern == 0:
                    # Geometry problem
                    w = 4 + (i % 7)
                    k = 2 + (i % 3)
                    p = 2 * (k * w + w)
                    l = k * w
                    area = l * w
                    q = f"A rectangle has a perimeter of {p} cm and its length is {k} times its width. Find the area in square centimeters."
                    code = f"w = {p} / (2 * ({k} + 1))\nl = {k} * w\narea = l * w\nprint(f'area={{area}}')"
                    thought = f"Let width be w. Perimeter 2*({k}w + w) = {2*(k+1)}w = {p}, so w = {w}, l = {l}, area = {area}."
                    ans = f"The width is {w} cm, length is {l} cm. <answer>{area}</answer> square centimeters."

                    traj.append({"type": "message", "role": "user", "content": q})
                    traj.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": thought}]})
                    cid = new_call_id()
                    call = {"type": "function_call", "call_id": cid, "name": "python_exec", "arguments": {"code": code}}
                    traj.append(call)
                    out = self.registry.execute(call, {})
                    traj.append({"type": "function_call_output", "call_id": cid, "output": out})
                    traj.append({"type": "message", "role": "assistant", "content": ans})

                elif pattern == 1:
                    # Algebra: roots
                    r1 = 1 + (i % 5)
                    r2 = 3 + (i % 6)
                    b_coef = -(r1 + r2)
                    c_coef = r1 * r2
                    q = f"Find the sum and product of the roots of the equation x^2 + ({b_coef})*x + ({c_coef}) = 0."
                    code = f"import sympy as sp\nx = sp.symbols('x')\neq = x**2 + ({b_coef})*x + ({c_coef})\nroots = sp.solve(eq, x)\nprint(f'roots={{roots}}, sum={{sum(roots)}}, prod={{roots[0]*roots[1]}}')"
                    thought = f"For equation x^2 + ({b_coef})x + {c_coef} = 0, roots are {r1} and {r2}. Sum is {r1+r2}, product is {c_coef}."
                    ans = f"The roots are {r1} and {r2}. Sum is {r1+r2} and product is <answer>{c_coef}</answer>."

                    traj.append({"type": "message", "role": "user", "content": q})
                    traj.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": thought}]})
                    cid = new_call_id()
                    call = {"type": "function_call", "call_id": cid, "name": "python_exec", "arguments": {"code": code}}
                    traj.append(call)
                    out = self.registry.execute(call, {})
                    traj.append({"type": "function_call_output", "call_id": cid, "output": out})
                    traj.append({"type": "message", "role": "assistant", "content": ans})

                elif pattern == 2:
                    # Knowledge retrieval
                    queries = ["Agent0 architecture", "SERC self-evolving reasoning", "responses API runtime", "trajectory schema v1"]
                    query = queries[i % len(queries)]
                    q = f"Search documents and summarize key details about: {query}."
                    traj.append({"type": "message", "role": "user", "content": q})
                    cid = new_call_id()
                    call = {"type": "function_call", "call_id": cid, "name": "retrieve", "arguments": {"query": query}}
                    traj.append(call)
                    out = self.registry.execute(call, {})
                    traj.append({"type": "function_call_output", "call_id": cid, "output": out})
                    traj.append({"type": "message", "role": "assistant", "content": f"<answer>Based on retrieved records, {query} is documented in data/knowledge with full specification.</answer>"})

                else:
                    # Compound Interest / Financial calculation
                    p_val = 1000 * (1 + (i % 10))
                    r_val = 0.05 + (i % 5) * 0.01
                    years = 2 + (i % 4)
                    q = f"Calculate the total compound amount for principal ${p_val} at annual rate {int(r_val*100)}% compounded annually for {years} years."
                    code = f"p = {p_val}\nr = {r_val}\nt = {years}\na = p * (1 + r)**t\nprint(f'total={{a:.2f}}')"
                    total = round(p_val * ((1 + r_val) ** years), 2)
                    traj.append({"type": "message", "role": "user", "content": q})
                    cid = new_call_id()
                    call = {"type": "function_call", "call_id": cid, "name": "python_exec", "arguments": {"code": code}}
                    traj.append(call)
                    out = self.registry.execute(call, {})
                    traj.append({"type": "function_call_output", "call_id": cid, "output": out})
                    traj.append({"type": "message", "role": "assistant", "content": f"The accumulated compound amount is <answer>${total:.2f}</answer>."})

            traj.validate()
            synthesized.append(traj)

        return synthesized

    def filter_and_audit(
        self,
        trajectories: Sequence[CanonicalTrajectory],
        *,
        min_steps: int = 3,
        verify_semantics: bool = True,
        ground_truth_map: Mapping[str, str] | None = None,
    ) -> tuple[list[CanonicalTrajectory], dict[str, int]]:
        """Filter trajectories with audit checks, deduplication, and answer verification."""
        valid: list[CanonicalTrajectory] = []
        stats: dict[str, int] = {}
        seen_hashes: set[str] = set()
        seen_samples: set[str] = set()

        for traj in trajectories:
            gt = (ground_truth_map or {}).get(traj.trajectory_id)
            ok, reason = FormatCleaner.audit_trajectory(
                traj,
                min_steps=min_steps,
                ground_truth=gt,
            )
            if not ok:
                stats[reason] = stats.get(reason, 0) + 1
                continue

            # Deduplication check
            h = FormatCleaner.compute_trajectory_hash(traj)
            if h in seen_hashes:
                stats["duplicate_trajectory"] = stats.get("duplicate_trajectory", 0) + 1
                continue
            sample_hash = traj.metadata.get("sample_hash")
            if sample_hash and sample_hash in seen_samples:
                stats["duplicate_sample"] = stats.get("duplicate_sample", 0) + 1
                continue

            # Semantic verification check
            if verify_semantics or "serc" in traj.metadata:
                v = verify_sft_semantics(traj, self.registry, self.answer_judge_hashes)
                if not v.valid:
                    stats["semantic_verifier_failed"] = stats.get("semantic_verifier_failed", 0) + 1
                    continue

            seen_hashes.add(h)
            if sample_hash:
                seen_samples.add(sample_hash)
            valid.append(traj)

        return valid, stats

    def export(
        self,
        trajectories: Sequence[CanonicalTrajectory],
        output_dir: Path,
        *,
        train_ratio: float = 0.9,
        export_format: str = "both",
        data_source: str = "agent0_sft",
        seed: int = 42,
    ) -> dict[str, Any]:
        """Export complete trajectories into train/val .parquet and .jsonl files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        shuffled = list(trajectories)
        rng.shuffle(shuffled)

        records: list[dict[str, Any]] = []
        for traj in shuffled:
            if "serc" in traj.metadata:
                result = verify_sft_semantics(traj, self.registry, self.answer_judge_hashes)
                if not result.valid:
                    raise ProtocolError("Export audit failed: " + "; ".join(result.issues))
            num_calls = sum(1 for item in traj.items if item.get("type") == "function_call")
            tool_names = [item["name"] for item in traj.items if item.get("type") == "function_call"]
            records.append({
                "trajectory_id": traj.trajectory_id,
                "data_source": traj.metadata.get("source", data_source),
                "trajectory": traj.to_dict(),
                "num_items": len(traj.items),
                "num_tool_calls": num_calls,
                "tools_used": list(set(tool_names)),
            })

        split_idx = int(len(records) * max(0.0, min(1.0, train_ratio)))
        if split_idx == 0 and len(records) > 0:
            split_idx = 1
        train_records = records[:split_idx]
        val_records = records[split_idx:] if split_idx < len(records) else []

        exported_files: list[str] = []

        # 1. Export Parquet (storing trajectory as valid JSON string for robust PyArrow loading)
        if export_format in {"both", "parquet"}:
            train_pq_records = [
                {**r, "trajectory": json.dumps(r["trajectory"], ensure_ascii=False), "tools_used": json.dumps(r["tools_used"])}
                for r in train_records
            ]
            train_df = pd.DataFrame(train_pq_records)
            train_pq_path = output_dir / "train.parquet"
            train_df.to_parquet(train_pq_path, index=False)
            exported_files.append(str(train_pq_path))

            if val_records:
                val_pq_records = [
                    {**r, "trajectory": json.dumps(r["trajectory"], ensure_ascii=False), "tools_used": json.dumps(r["tools_used"])}
                    for r in val_records
                ]
                val_df = pd.DataFrame(val_pq_records)
                val_pq_path = output_dir / "val.parquet"
                val_df.to_parquet(val_pq_path, index=False)
                exported_files.append(str(val_pq_path))

        # 2. Export JSONL
        if export_format in {"both", "jsonl"}:
            train_jsonl_path = output_dir / "train.jsonl"
            with train_jsonl_path.open("w", encoding="utf-8") as f:
                for rec in train_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            exported_files.append(str(train_jsonl_path))

            if val_records:
                val_jsonl_path = output_dir / "val.jsonl"
                with val_jsonl_path.open("w", encoding="utf-8") as f:
                    for rec in val_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                exported_files.append(str(val_jsonl_path))

        # 3. Export Manifest
        all_tools = [tool for rec in records for tool in rec.get("tools_used", [])]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "prompt_sha256": hashlib.sha256(render_system_prompt().encode()).hexdigest(),
            "answer_judge_accepted_hashes": sorted(self.answer_judge_hashes),
            "total_records": len(records),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "train_ratio": train_ratio,
            "exported_files": exported_files,
            "tool_call_frequencies": {tool: all_tools.count(tool) for tool in set(all_tools)},
            "average_trajectory_items": (
                round(sum(r["num_items"] for r in records) / len(records), 2)
                if records else 0.0
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        manifest_path = output_dir / "sft_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        exported_files.append(str(manifest_path))

        return manifest


@dataclass
class BuildStats:
    """Cumulative counters and metrics for concurrent SFT trajectory generation."""
    attempted: int = 0
    exported: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    answer_judge_calls: int = 0
    answer_judge_passes: int = 0
    answer_judge_errors: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "exported": self.exported,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "answer_judge_calls": self.answer_judge_calls,
            "answer_judge_passes": self.answer_judge_passes,
            "answer_judge_errors": self.answer_judge_errors,
            "tool_success_rate": round(self.successful_tool_calls / max(1, self.tool_calls), 4),
            "export_rate": round(self.exported / max(1, self.attempted), 4),
            "rejected_breakdown": dict(self.rejected),
        }


def run_task_rollouts_concurrent(
    tasks: list[dict[str, Any]],
    runtime: ResponsesRuntime,
    builder: SFTTrajectoryBuilder,
    output_jsonl: Path,
    *,
    concurrency: int = 8,
    resume: bool = True,
    verify_semantics: bool = True,
    min_steps: int = 3,
) -> tuple[list[CanonicalTrajectory], BuildStats]:
    """Execute multi-turn tool rollouts with strict concurrency control against local or remote model."""
    from tools.data_builder.sft_stream import run_stream
    return run_stream(tasks, runtime, builder, output_jsonl, concurrency=concurrency,
                      resume=resume, verify_semantics=verify_semantics, min_steps=min_steps)


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 1, "<|endoftext|>": 2}.get(token, 0)


def verify_sft_dataset_compatibility(dataset_path: Path) -> dict[str, Any]:
    """Validate every canonical row; tensor checks require a real VL processor."""
    from agent0_protocol.adapters import QwenModelAdapter
    from verl.utils.dataset.agent0_sft_dataset import Agent0SFTDataset

    frame = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_json(dataset_path, lines=True)
    if frame.empty or "trajectory" not in frame:
        raise ProtocolError("SFT export is empty or missing trajectory column")
    image_rows = 0
    registry = get_tool_registry()
    adapter = QwenModelAdapter(DummyTokenizer())
    max_text_length = 1
    for value in frame["trajectory"]:
        trajectory = CanonicalTrajectory.from_dict(json.loads(value) if isinstance(value, str) else value)
        if trajectory.tools != registry.definitions():
            raise ProtocolError("SFT tool registry mismatch")
        if trajectory.items[0].get("content") != render_system_prompt():
            raise ProtocolError("SFT system prompt mismatch")
        ok, reason = FormatCleaner.audit_trajectory(trajectory)
        if not ok:
            raise ProtocolError(reason)
        has_images = any(
            (item.get("type") == "message" and isinstance(item.get("content"), list)
             and any(part.get("type") == "input_image" for part in item["content"] if isinstance(part, dict)))
            or (item.get("type") == "function_call_output" and
                (item.get("output", {}).get("image_url") or item.get("output", {}).get("image_urls") or item.get("output", {}).get("images")))
            for item in trajectory.items)
        if has_images:
            image_rows += 1
        else:
            max_text_length = max(max_text_length, sum(len(adapter.render([item], trajectory.tools, generate=False).encode()) for item in trajectory.items))
    if image_rows:
        return {"status": "schema_passed", "dataset_len": len(frame), "image_rows": image_rows,
                "tensor_check": "requires_real_vl_processor"}
    # Dummy byte tokens verify the local container/mask path without truncation;
    # this is not a real model's context-length or GPU feasibility check.
    ds = Agent0SFTDataset(str(dataset_path), DummyTokenizer(), max_length=max_text_length, truncation="error")
    for index in range(len(ds)):
        if int(ds[index]["loss_mask"].sum().item()) <= 0:
            raise ProtocolError("SFT row has no trainable tokens")
    return {"status": "passed", "dataset_len": len(ds), "input_ids_len": max_text_length,
            "loss_mask_trainable_tokens": int(ds[0]["loss_mask"].sum().item())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input JSONL file or directory of canonical JSON trajectories.",
    )
    parser.add_argument("--judge-state", type=Path, nargs="*", default=[], help="Committed source states authorizing imported LLM-accepted trajectories.")
    parser.add_argument("--source", choices=sorted(SOURCE_STAGES))
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--source-split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-retries", type=int, default=3)
    parser.add_argument("--max-reasoning-steps", type=int, default=8)
    parser.add_argument("--sandbox-timeout", type=float, default=float(os.environ.get("SANDBOX_RUN_TIMEOUT", "10")))
    parser.add_argument("--teacher-max-output-tokens", type=int, default=0,
                        help="Optional per-request limit; 0 omits the API output limit.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Tasks JSONL with question, required reference, and optional image/images; runs audited SERC rollouts.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=ROOT / "data" / "sft",
        help="Target output directory (default: data/sft).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=1,
        help="SFT Stage (1: Visual Perception & Tool Use; 2: Math & Complex Reasoning).",
    )
    parser.add_argument(
        "--format",
        choices=["both", "parquet", "jsonl"],
        default="both",
        help="Dataset serialization format (default: both).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Ratio of samples allocated to training set (default: 1.0, all samples for training; dedicated datasets used for evaluation).",
    )
    parser.add_argument(
        "--teacher-backend",
        choices=["local", "remote"],
        default="local",
        help="Teacher model backend: 'local' (default, local Qwen 27B) or 'remote' (OpenAI/cloud API).",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=1,
        help="Number of speculative tokens to predict per step (default: 1).",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=8,
        help="Max concurrent rollout requests to local/remote model (default: 8).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume task rollouts skipping completed samples.",
    )
    parser.add_argument(
        "--teacher-base-url",
        type=str,
        default=None,
        help="Base URL for teacher model endpoint (default from AGENT0_RESPONSES_BASE_URL).",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Model identifier (e.g. Qwen/Qwen2.5-VL-7B-Instruct or qwen2.5-vl-27b).",
    )
    parser.add_argument(
        "--teacher-api-key",
        type=str,
        default=None,
        help="API Key for teacher endpoint (default from AGENT0_RESPONSES_API_KEY).",
    )
    parser.add_argument(
        "--teacher-timeout",
        type=float,
        default=180.0,
        help="Teacher request timeout in seconds (default: 180.0).",
    )
    parser.add_argument(
        "--synthesize-count",
        type=int,
        default=0,
        help="Number of multi-turn tool reasoning trajectories to synthesize.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=3,
        help="Minimum trajectory item count (default: 3).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip static checks for legacy input only; generated SERC evidence is always audited.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="agent0_sft",
        help="Data source identifier string.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and synthesis.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    builder = SFTTrajectoryBuilder()
    for state_path in args.judge_state:
        state = json.loads(state_path.read_text())
        if not state.get("fingerprint"):
            parser.error("Judge state has no generation fingerprint")
        builder.answer_judge_hashes.update(state.get("answer_judge_accepted_hashes", []))
    raw_trajectories: list[CanonicalTrajectory] = []
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. If --tasks is provided: run concurrent rollouts via ResponsesRuntime against local model (e.g. Qwen 27B)
    if args.tasks or args.source:
        if args.tasks and args.source:
            parser.error("Use --tasks or --source, not both")
        if args.source:
            if not args.source_path:
                parser.error("--source requires --source-path")
            from tools.data_builder.sources import load_source
            tasks = load_source(args.source, args.source_path, stage=args.stage, source_split=args.source_split)
        else:
            tasks_path = Path(args.tasks)
            if not tasks_path.is_absolute():
                tasks_path = ROOT / tasks_path
            tasks = []
            with tasks_path.open("r", encoding="utf-8") as f:
                for number, line in enumerate(f, 1):
                    if line.strip():
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            parser.error(f"Invalid task at line {number}")
                        tasks.append(value)
        logging.info("Loaded %d tasks", len(tasks))

        if args.teacher_backend == "remote":
            default_url = "https://api.openai.com/v1"
            default_model = "gpt-4o"
            default_key = os.environ.get("AGENT0_RESPONSES_API_KEY", "")
            default_timeout = 180.0
            default_concurrency = int(os.environ.get("AGENT0_CONCURRENCY", "8"))
        else:
            # Default: Local Qwen 27B teacher model
            default_url = "http://127.0.0.1:8000/v1"
            default_model = "qwen3.8-27b"
            default_key = "EMPTY"
            default_timeout = 300.0
            default_concurrency = int(os.environ.get("AGENT0_CONCURRENCY", "8"))

        teacher_url = args.teacher_base_url or os.environ.get("AGENT0_RESPONSES_BASE_URL") or default_url
        teacher_model = args.teacher_model or os.environ.get("AGENT0_RESPONSES_MODEL") or default_model
        teacher_key = args.teacher_api_key or os.environ.get("AGENT0_RESPONSES_API_KEY") or default_key
        teacher_timeout = args.teacher_timeout if args.teacher_timeout != 180.0 else float(os.environ.get("AGENT0_RESPONSES_TIMEOUT_SECONDS", str(default_timeout)))
        concurrency = args.concurrency if args.concurrency != 8 else default_concurrency

        cfg = ResponsesConfig(
            base_url=teacher_url,
            api_key=teacher_key,
            model=teacher_model,
            timeout_seconds=teacher_timeout,
            max_retries=args.teacher_retries,
            max_tool_rounds=args.max_reasoning_steps,
            temperature=args.teacher_temperature,
            top_p=args.teacher_top_p,
            sandbox_timeout_seconds=args.sandbox_timeout,
            max_output_tokens=args.teacher_max_output_tokens,
        )
        runtime = ResponsesRuntime(cfg, builder.registry, probe_on_init=False)
        stream_jsonl = out_dir / "tasks_stream_trajectories.jsonl"
        task_trajs, roll_stats = run_task_rollouts_concurrent(
            tasks,
            runtime,
            builder,
            stream_jsonl,
            concurrency=concurrency,
            resume=args.resume,
            verify_semantics=not args.no_verify,
            min_steps=args.min_steps,
        )
        logging.info(f"Task rollouts complete: {roll_stats.to_dict()}")
        raw_trajectories.extend(task_trajs)

    # 2. Ingest from input path if provided
    if args.input:
        in_path = Path(args.input)
        if not in_path.is_absolute():
            in_path = ROOT / in_path

        if in_path.is_file():
            logging.info(f"Loading trajectories from file: {in_path}")
            raw_trajectories.extend(builder.load_from_jsonl(in_path))
        elif in_path.is_dir():
            logging.info(f"Loading canonical trajectories from directory: {in_path}")
            raw_trajectories.extend(builder.load_from_json_dir(in_path))

    # 3. Synthesize if requested or if no raw trajectories were available
    synth_target = args.synthesize_count
    if (args.tasks or args.source or args.input) and not raw_trajectories:
        logging.error("No supplied/generated trajectories accepted; refusing synthetic fallback")
        return 1
    if not raw_trajectories and synth_target == 0:
        logging.info(f"No input trajectories found; synthesizing 10 Stage-{args.stage} verified multi-turn tool trajectories...")
        synth_target = 10

    if synth_target > 0:
        logging.info(f"Synthesizing {synth_target} Stage-{args.stage} verified canonical multi-turn trajectories...")
        synth_trajs = builder.synthesize_trajectories(synth_target, stage=args.stage, seed=args.seed)
        raw_trajectories.extend(synth_trajs)

    logging.info(f"Collected {len(raw_trajectories)} raw candidate trajectories.")

    # 4. Filter, clean, and audit
    valid_trajectories, reject_stats = builder.filter_and_audit(
        raw_trajectories,
        min_steps=args.min_steps,
        verify_semantics=not args.no_verify,
    )
    logging.info(f"Retained {len(valid_trajectories)} valid full trajectories (Rejected: {reject_stats}).")

    if not valid_trajectories:
        logging.error("No valid trajectories remaining after audit. Dataset build aborted.")
        return 1

    # 5. Export
    manifest = builder.export(
        valid_trajectories,
        out_dir,
        train_ratio=args.train_ratio,
        export_format=args.format,
        data_source=args.data_source,
        seed=args.seed,
    )

    logging.info(f"Exported SFT dataset to {out_dir}:")
    logging.info(f"  Total records: {manifest['total_records']}")
    logging.info(f"  Train records: {manifest['train_records']}")
    logging.info(f"  Val records:   {manifest['val_records']}")
    logging.info(f"  Tool calls:    {manifest['tool_call_frequencies']}")

    # 6. Verify loader compatibility
    test_target = out_dir / "train.parquet"
    if not test_target.is_file():
        test_target = out_dir / "train.jsonl"

    if test_target.is_file():
        loader_check = verify_sft_dataset_compatibility(test_target)
        logging.info(f"Agent0SFTDataset compatibility check: {loader_check}")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
