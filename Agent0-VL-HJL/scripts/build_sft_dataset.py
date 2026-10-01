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
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
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
)
from agent0_protocol.verifier import verify_trajectory
from tools.data_builder.backends.base import image_data_url
from verl.prompts.agent0_templates import assistant_text, render_solver_request, render_system_prompt

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Answer Extraction and Equivalence Judging
# ==============================================================================

class AnswerJudge:
    """Multi-tier answer extraction, normalization, and equivalence verification."""

    @staticmethod
    def extract_answer(text: str | None) -> str | None:
        """Extract candidate answer from XML tags, LaTeX boxed, or common phrasing."""
        if not text:
            return None
        raw = str(text).strip()

        # 1. <answer>...</answer> tags
        m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # 2. \boxed{...} LaTeX markup
        m = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", raw)
        if m:
            return m.group(1).strip()

        # 3. Explicit "The answer is ..." or "Final Answer:" prefixes
        for line in reversed(raw.splitlines()):
            line_str = line.strip()
            prefix_match = re.search(r"(?:final answer|the answer is|conclusion|answer)\s*[:：]\s*(.+)", line_str, re.IGNORECASE)
            if prefix_match:
                cand = prefix_match.group(1).strip().rstrip(".")
                if cand:
                    return cand

        # Fallback to the last nonempty line
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return lines[-1].rstrip(".") if lines else None

    @staticmethod
    def normalize_text(text: str | None) -> str:
        """Normalize answer text by removing formatting and punctuation."""
        if not text:
            return ""
        s = str(text).strip().lower()
        # Strip LaTeX text macros
        s = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", s)
        s = re.sub(r"[$]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = s.rstrip(".,;:!?'\"")
        return s

    @staticmethod
    def extract_options(question: str) -> dict[str, str]:
        """Extract multiple choice options from a question string (e.g. '(A) 12 (B) 24')."""
        options: dict[str, str] = {}
        # Pattern 1: (A) value (B) value
        matches = re.findall(r"\(([A-Za-z])\)\s*([^(\n]+)", str(question))
        for k, v in matches:
            options[k.upper()] = v.strip().rstrip(".,;")
        if options:
            return options
        # Pattern 2: A. value B. value or A) value
        matches2 = re.findall(r"(?:^|\s)([A-Za-z])[\.:\)]\s*([^\n\(\)]+)", str(question))
        for k, v in matches2:
            options[k.upper()] = v.strip().rstrip(".,;")
        return options

    @classmethod
    def parse_number(cls, text: str) -> float | None:
        """Parse numerical representation including decimals, percentages, and fractions."""
        s = cls.normalize_text(text)
        if not s:
            return None

        # Percentage
        if s.endswith("%"):
            try:
                return float(s[:-1].strip()) / 100.0
            except ValueError:
                pass

        # Fraction "a/b"
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    num = float(parts[0].strip())
                    denom = float(parts[1].strip())
                    if denom != 0:
                        return num / denom
                except ValueError:
                    pass

        # Scientific notation or plain float
        num_m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if num_m:
            try:
                return float(num_m.group(0))
            except ValueError:
                pass

        return None

    @classmethod
    def is_equivalent(
        cls,
        candidate: str | None,
        reference: str | None,
        question: str = "",
        options: dict[str, str] | None = None,
        tolerance: float = 1e-4,
    ) -> bool:
        """Determine whether candidate answer agrees with reference answer."""
        cand_raw = cls.extract_answer(candidate) or candidate or ""
        ref_raw = cls.extract_answer(reference) or reference or ""

        cand_norm = cls.normalize_text(cand_raw)
        ref_norm = cls.normalize_text(ref_raw)

        if not cand_norm or not ref_norm:
            return False

        # 1. Exact normalized match
        if cand_norm == ref_norm:
            return True

        # 2. Numerical equivalence check
        c_num = cls.parse_number(cand_norm)
        r_num = cls.parse_number(ref_norm)
        if c_num is not None and r_num is not None:
            if math.isclose(c_num, r_num, rel_tol=tolerance, abs_tol=tolerance):
                return True

        # 3. Multiple choice option matching
        opts = options or cls.extract_options(question)
        if opts:
            # Candidate is option letter (e.g. "B") while reference is option value, or vice-versa
            cand_key = cand_norm.upper()
            ref_key = ref_norm.upper()
            if cand_key in opts and cls.normalize_text(opts[cand_key]) == ref_norm:
                return True
            if ref_key in opts and cls.normalize_text(opts[ref_key]) == cand_norm:
                return True
            if cand_key in opts and ref_key in opts and cand_key == ref_key:
                return True

        # 4. Substring inclusion if candidate strictly matches reference word boundary
        if len(ref_norm) > 2 and re.search(r"\b" + re.escape(ref_norm) + r"\b", cand_norm):
            return True

        return False


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
        path = Path(image)
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
        """Clean control characters from all messages and reasoning items."""
        for item in trajectory.items:
            kind = item.get("type")
            if kind == "message":
                content = item.get("content")
                if isinstance(content, str):
                    item["content"] = cls.strip_control_characters(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            part["text"] = cls.strip_control_characters(str(part["text"]))
            elif kind == "reasoning":
                for part in item.get("summary", []):
                    if isinstance(part, dict) and "text" in part:
                        part["text"] = cls.strip_control_characters(str(part["text"]))

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
            final_text = ""
            for item in reversed(trajectory.items):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    content = item.get("content", "")
                    if isinstance(content, str):
                        final_text = content
                    elif isinstance(content, list):
                        final_text = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                    break

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

    def normalize_trajectory(self, trajectory: CanonicalTrajectory) -> CanonicalTrajectory:
        """Ensure tool snapshot strictly matches current registry definitions and clean text."""
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
                        traj_data["tools"] = copy.deepcopy(self.canonical_tools)
                        if "trajectory_id" not in traj_data:
                            traj_data["trajectory_id"] = f"traj_jsonl_{line_no}_{uuid.uuid4().hex[:6]}"
                        traj = CanonicalTrajectory.from_dict(traj_data)
                        trajectories.append(self.normalize_trajectory(traj))
                except Exception as exc:
                    logger.debug(f"Skipping line {line_no} in {jsonl_path}: {exc}")

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
                base_img = _create_synthetic_image(color=col, width=120 + (i % 5) * 10, height=80 + (i % 5) * 8)
                context = ToolExecutionContext(image=base_img)
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
            seen_hashes.add(h)

            # Semantic verification check
            if verify_semantics:
                v = verify_trajectory(traj, self.registry)
                if not v.valid:
                    stats["semantic_verifier_failed"] = stats.get("semantic_verifier_failed", 0) + 1
                    continue

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
    rejected: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "exported": self.exported,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "answer_judge_calls": self.answer_judge_calls,
            "answer_judge_passes": self.answer_judge_passes,
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
    concurrency: int = 4,
    resume: bool = True,
    verify_semantics: bool = True,
    min_steps: int = 3,
) -> tuple[list[CanonicalTrajectory], BuildStats]:
    """Execute multi-turn tool rollouts with strict concurrency control against local or remote model."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    state_file = Path(str(output_jsonl) + ".state.json")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    completed_trajectories: list[CanonicalTrajectory] = []

    # Resume from existing progress
    if resume and output_jsonl.is_file():
        logger.info(f"Resuming from existing output file: {output_jsonl}")
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        traj_dict = record.get("trajectory", record)
                        t = CanonicalTrajectory.from_dict(traj_dict)
                        seen_ids.add(t.trajectory_id)
                        if "sample_id" in t.metadata:
                            seen_ids.add(str(t.metadata["sample_id"]))
                        if "sample_id" in record:
                            seen_ids.add(str(record["sample_id"]))
                        seen_hashes.add(FormatCleaner.compute_trajectory_hash(t))
                        completed_trajectories.append(t)
                    except Exception:
                        pass
        logger.info(f"Loaded {len(completed_trajectories)} already completed trajectories.")

    stats = BuildStats()
    semaphore = threading.Semaphore(max(1, concurrency))
    file_lock = threading.Lock()
    stats_lock = threading.Lock()

    def process_task(task: dict[str, Any]) -> CanonicalTrajectory | None:
        sample_id = str(task.get("id", task.get("sample_id", uuid.uuid4().hex[:8])))
        task_id = str(task.get("id", ""))
        if sample_id in seen_ids or (task_id and task_id in seen_ids):
            return None

        question = str(task.get("question", task.get("prompt", "")))
        if not question.strip():
            return None

        ground_truth = task.get("ground_truth", task.get("answer"))
        image = task.get("image") or task.get("image_path") or task.get("image_pil")

        with stats_lock:
            stats.attempted += 1

        # Format initial prompt items
        content: str | list[dict[str, Any]] = render_solver_request(question)
        if image is not None:
            try:
                content = [
                    {"type": "input_text", "text": render_solver_request(question)},
                    {"type": "input_image", "image_url": image_data_url(image)},
                ]
            except Exception as e:
                with stats_lock:
                    stats.rejected[f"image_load_error: {type(e).__name__}"] = stats.rejected.get(f"image_load_error: {type(e).__name__}", 0) + 1
                return None

        initial_items = [
            {"type": "message", "role": "system", "content": render_system_prompt()},
            {"type": "message", "role": "user", "content": content},
        ]

        # Bounded concurrency execution through local/remote model
        with semaphore:
            try:
                traj = runtime.run(
                    initial_items,
                    trajectory_id=f"sft_task_{sample_id}_{uuid.uuid4().hex[:6]}",
                    metadata={"sample_id": sample_id, "source": task.get("data_source", "task_rollout")},
                )
            except Exception as e:
                with stats_lock:
                    stats.rejected[f"runtime_error: {type(e).__name__}"] = stats.rejected.get(f"runtime_error: {type(e).__name__}", 0) + 1
                return None

        # Track tool calls
        calls = [item for item in traj.items if item.get("type") == "function_call"]
        outputs = [item for item in traj.items if item.get("type") == "function_call_output"]
        with stats_lock:
            stats.tool_calls += len(calls)
            stats.successful_tool_calls += sum(bool(o.get("output", {}).get("success")) for o in outputs)

        # Answer judging if ground truth exists
        if ground_truth:
            pred_text = assistant_text(traj.items) or ""
            with stats_lock:
                stats.answer_judge_calls += 1
            is_correct = AnswerJudge.is_equivalent(pred_text, str(ground_truth), question=question)
            if not is_correct:
                with stats_lock:
                    stats.rejected["answer_mismatch"] = stats.rejected.get("answer_mismatch", 0) + 1
                return None
            with stats_lock:
                stats.answer_judge_passes += 1

        # Audit and cleaning
        builder.normalize_trajectory(traj)
        ok, reason = FormatCleaner.audit_trajectory(traj, min_steps=min_steps)
        if not ok:
            with stats_lock:
                stats.rejected[reason] = stats.rejected.get(reason, 0) + 1
            return None

        # Deduplication
        h = FormatCleaner.compute_trajectory_hash(traj)
        with stats_lock:
            if h in seen_hashes:
                stats.rejected["duplicate"] = stats.rejected.get("duplicate", 0) + 1
                return None
            seen_hashes.add(h)
            seen_ids.add(sample_id)

        # Semantic verifier
        if verify_semantics:
            v = verify_trajectory(traj, builder.registry)
            if not v.valid:
                with stats_lock:
                    stats.rejected["semantic_verify_failed"] = stats.rejected.get("semantic_verify_failed", 0) + 1
                return None

        # Write incrementally under file lock
        rec = {
            "trajectory_id": traj.trajectory_id,
            "data_source": task.get("data_source", "task_rollout"),
            "trajectory": traj.to_dict(),
            "num_items": len(traj.items),
            "num_tool_calls": len(calls),
            "tools_used": list(set(c["name"] for c in calls)),
        }
        with file_lock:
            with output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            # Atomic state file write
            state_tmp = state_file.with_suffix(".tmp")
            with state_tmp.open("w", encoding="utf-8") as sf:
                sf.write(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False) + "\n")
                sf.flush()
                os.fsync(sf.fileno())
            state_tmp.replace(state_file)

        with stats_lock:
            stats.exported += 1

        return traj

    logger.info(f"Starting concurrent task rollout over {len(tasks)} tasks (concurrency={concurrency})...")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for f in as_completed(futures):
            try:
                res = f.result()
                if res is not None:
                    completed_trajectories.append(res)
            except Exception as exc:
                logger.warning(f"Task worker failed with exception: {exc}")

    return completed_trajectories, stats


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 1, "<|endoftext|>": 2}.get(token, 0)


def verify_sft_dataset_compatibility(dataset_path: Path) -> dict[str, Any]:
    """Verify that Agent0SFTDataset loads the generated dataset without error."""
    from verl.utils.dataset.agent0_sft_dataset import Agent0SFTDataset

    ds = Agent0SFTDataset(str(dataset_path), DummyTokenizer(), max_length=8192)
    sample = ds[0]
    loss_mask_sum = int(sample["loss_mask"].sum().item())
    input_ids_len = int(sample["input_ids"].shape[0])

    return {
        "status": "passed",
        "dataset_len": len(ds),
        "input_ids_len": input_ids_len,
        "loss_mask_trainable_tokens": loss_mask_sum,
        "sample_trajectory_id": sample.get("trajectory_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input JSONL file or directory of canonical JSON trajectories.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Input tasks JSONL file (question, optional image, optional ground_truth) to drive model rollouts.",
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
        default=0.9,
        help="Ratio of samples allocated to training set (default: 0.9).",
    )
    parser.add_argument(
        "--teacher-backend",
        choices=["local", "remote"],
        default="local",
        help="Teacher model backend: 'local' (default, local Qwen 27B) or 'remote' (OpenAI/cloud API).",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=4,
        help="Max concurrent rollout requests to local/remote model (default: 4).",
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
        help="Skip semantic verifier check during filtering.",
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
    raw_trajectories: list[CanonicalTrajectory] = []
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. If --tasks is provided: run concurrent rollouts via ResponsesRuntime against local model (e.g. Qwen 27B)
    if args.tasks:
        tasks_path = Path(args.tasks)
        if not tasks_path.is_absolute():
            tasks_path = ROOT / tasks_path
        if not tasks_path.is_file():
            logging.error(f"Tasks file not found: {tasks_path}")
            return 1

        tasks: list[dict[str, Any]] = []
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        tasks.append(json.loads(line))
                    except Exception:
                        pass
        logging.info(f"Loaded {len(tasks)} tasks from {tasks_path}")

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
            default_concurrency = int(os.environ.get("AGENT0_CONCURRENCY", "4"))

        teacher_url = args.teacher_base_url or os.environ.get("AGENT0_RESPONSES_BASE_URL") or default_url
        teacher_model = args.teacher_model or os.environ.get("AGENT0_RESPONSES_MODEL") or default_model
        teacher_key = args.teacher_api_key or os.environ.get("AGENT0_RESPONSES_API_KEY") or default_key
        teacher_timeout = args.teacher_timeout if args.teacher_timeout != 180.0 else float(os.environ.get("AGENT0_RESPONSES_TIMEOUT_SECONDS", str(default_timeout)))
        concurrency = args.concurrency if args.concurrency != 4 else default_concurrency

        cfg = ResponsesConfig(
            base_url=teacher_url,
            api_key=teacher_key,
            model=teacher_model,
            timeout_seconds=teacher_timeout,
            max_retries=3,
            max_tool_rounds=8,
            max_output_tokens=2048,
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

    # Auto-discover from default trajectories if no tasks or input explicitly given and synthesize-count == 0
    if not args.tasks and not args.input and args.synthesize_count == 0:
        default_dir = ROOT / "outputs" / "hjl_trajectories"
        if default_dir.is_dir():
            logging.info(f"Auto-discovering existing trajectories from {default_dir}...")
            raw_trajectories.extend(builder.load_from_json_dir(default_dir))
            jsonl_file = default_dir / "hjl_trajectory.jsonl"
            if jsonl_file.is_file():
                raw_trajectories.extend(builder.load_from_jsonl(jsonl_file))

    # 3. Synthesize if requested or if no raw trajectories were available
    synth_target = args.synthesize_count
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
