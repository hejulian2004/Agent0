"""Standard ToolResult representation and visual tool adapters for HJL."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from agent0_protocol.schema import new_call_id
from agent0_protocol.tools import (
    ToolExecutionContext,
    ToolRegistry,
    _current_image_path,
    get_tool_registry,
)


@dataclass
class ToolResult:
    """Standardized result returned by all HJL visual tools."""

    success: bool
    output_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "output_path": self.output_path,
            "metadata": copy.deepcopy(self.metadata),
            "error": self.error,
        }


def validate_reference_metadata(ref_metadata: Mapping[str, Any]) -> None:
    """Runtime enforcement ensuring retrieved reference artifacts strictly prevent dataset leakage."""
    split = ref_metadata.get("split")
    if split != "train":
        raise ValueError(
            f"Reference leakage detected: only train-split normal references are allowed, got {split!r}."
        )
    if not ref_metadata.get("is_normal"):
        raise ValueError("Reference must be a confirmed normal training sample, got is_normal=False.")


def _compare_images_simple(image_a_path: Path, image_b_path: Path) -> dict[str, Any]:
    """Compute difference statistics between two distinct images."""
    try:
        with Image.open(image_a_path) as img_a, Image.open(image_b_path) as img_b:
            img_a_rgb = img_a.convert("RGB")
            img_b_resized = img_b.convert("RGB").resize(img_a_rgb.size)

            diff_count = 0
            total_diff = 0.0
            width, height = img_a_rgb.size
            total_pixels = width * height

            step = max(1, int(math.sqrt(total_pixels / 10000)))
            samples = 0

            for y in range(0, height, step):
                for x in range(0, width, step):
                    pa = img_a_rgb.getpixel((x, y))
                    pb = img_b_resized.getpixel((x, y))
                    delta = (abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) + abs(pa[2] - pb[2])) / 3.0
                    total_diff += delta
                    if delta > 30:
                        diff_count += 1
                    samples += 1

            mean_delta = total_diff / max(1, samples)
            anomaly_pixel_ratio = diff_count / max(1, samples)
            similarity = max(0.0, min(1.0, 1.0 - (mean_delta / 255.0)))

            return {
                "similarity": round(similarity, 4),
                "mean_color_delta": round(mean_delta, 2),
                "anomaly_pixel_ratio": round(anomaly_pixel_ratio, 4),
                "reference_compared": image_b_path.name,
                "reference_path": str(image_b_path),
            }
    except Exception as exc:
        return {"error": str(exc), "similarity": 0.5}


def _locate_or_create_reference_image(
    category: str = "industrial_component",
    corpus_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Retrieve train-split normal reference image or synthesize a pristine normal template."""
    if corpus_dir:
        c_path = Path(corpus_dir) / category / "train" / "good"
        if c_path.is_dir():
            files = sorted(c_path.glob("*.png")) + sorted(c_path.glob("*.jpg"))
            if files:
                meta = {
                    "dataset": "industrial_corpus",
                    "category": category,
                    "split": "train",
                    "is_normal": True,
                    "sample_id": files[0].stem,
                    "reference_path": str(files[0]),
                }
                validate_reference_metadata(meta)
                return files[0], meta

    # Check local benchmark data path if present
    bench_path = Path(f"data/mvtec/{category}/train/good")
    if bench_path.is_dir():
        files = sorted(bench_path.glob("*.png")) + sorted(bench_path.glob("*.jpg"))
        if files:
            meta = {
                "dataset": "mvtec",
                "category": category,
                "split": "train",
                "is_normal": True,
                "sample_id": files[0].stem,
                "reference_path": str(files[0]),
            }
            validate_reference_metadata(meta)
            return files[0], meta

    # Create / cache a pristine normal reference template for this category
    cache_dir = Path(tempfile.gettempdir()) / "hjl_reference_corpus" / category / "train" / "good"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_file = cache_dir / "normal_ref_000.png"
    if not ref_file.is_file():
        img = Image.new("RGB", (100, 100), color=(180, 180, 180))
        img.save(ref_file, format="PNG")

    meta = {
        "dataset": "standard_templates",
        "category": category,
        "split": "train",
        "is_normal": True,
        "sample_id": "normal_ref_000",
        "reference_path": str(ref_file),
    }
    validate_reference_metadata(meta)
    return ref_file, meta


def execute_adapted_tool(
    name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
    registry: ToolRegistry | None = None,
) -> ToolResult:
    """Execute an adapted HJL visual tool. Pure function returning ToolResult without mutating HJLState."""
    reg = registry or get_tool_registry()

    try:
        if name == "crop_region":
            bbox = arguments.get("bbox", [0, 0, 10, 10])
            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "crop_image",
                "arguments": {"bbox": bbox},
            }
            output = reg.execute(call, context)
            if output.get("success"):
                path = str(_current_image_path(context))
                return ToolResult(
                    success=True,
                    output_path=path,
                    metadata={"image_size": output.get("image_size"), "bbox": bbox},
                )
            return ToolResult(success=False, error=output.get("error", "crop_image failed"))

        elif name == "zoom_region":
            scale = float(arguments.get("scale", 2.0))
            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "zoom_image",
                "arguments": {"scale": scale},
            }
            output = reg.execute(call, context)
            if output.get("success"):
                path = str(_current_image_path(context))
                return ToolResult(
                    success=True,
                    output_path=path,
                    metadata={"image_size": output.get("image_size"), "scale": scale},
                )
            return ToolResult(success=False, error=output.get("error", "zoom_image failed"))

        elif name == "rotate_image":
            angle = float(arguments.get("angle", 90.0))
            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "rotate_image",
                "arguments": {"angle": angle},
            }
            output = reg.execute(call, context)
            if output.get("success"):
                path = str(_current_image_path(context))
                return ToolResult(
                    success=True,
                    output_path=path,
                    metadata={"image_size": output.get("image_size"), "angle": angle},
                )
            return ToolResult(success=False, error=output.get("error", "rotate_image failed"))

        elif name == "retrieve_normal_reference":
            category = str(arguments.get("category", "industrial_component"))
            corpus_dir = arguments.get("corpus_dir")
            ref_path, meta = _locate_or_create_reference_image(category, corpus_dir)
            return ToolResult(
                success=True,
                output_path=str(ref_path),
                metadata=meta,
            )

        elif name == "compare_with_reference":
            ref_path_str = arguments.get("reference_path")
            current_path = _current_image_path(context)

            if not ref_path_str:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'reference_path' for compare_with_reference.",
                )

            ref_path = Path(ref_path_str)
            if not ref_path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Reference image not found at: {ref_path_str}",
                )

            # Strictly reject comparing active image against itself
            if ref_path.resolve() == current_path.resolve():
                return ToolResult(
                    success=False,
                    error="Self-comparison rejected: reference image cannot be the current active inspection image.",
                )

            diff_meta = _compare_images_simple(current_path, ref_path)
            if "error" in diff_meta:
                return ToolResult(success=False, error=diff_meta["error"])

            return ToolResult(
                success=True,
                output_path=str(current_path),
                metadata=diff_meta,
            )

        elif name == "localize_candidate":
            current_path = _current_image_path(context)
            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "visual_analyzer",
                "arguments": {},
            }
            output = reg.execute(call, context)
            candidates = []
            if output.get("success"):
                dark_box = output.get("analysis", {}).get("dark_bbox")
                if dark_box:
                    candidates.append({"bbox": dark_box, "confidence": 0.85, "label": "salient_region"})

            if not candidates:
                with Image.open(current_path) as img:
                    w, h = img.size
                    candidates.append({
                        "bbox": [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)],
                        "confidence": 0.5,
                        "label": "center_region",
                    })

            return ToolResult(
                success=True,
                output_path=str(current_path),
                metadata={"candidate_regions": candidates},
            )

        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

    except Exception as exc:
        return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
