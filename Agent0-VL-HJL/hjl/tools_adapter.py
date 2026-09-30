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
    retriable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "output_path": self.output_path,
            "metadata": copy.deepcopy(self.metadata),
            "error": self.error,
            "retriable": bool(self.retriable),
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


def _compare_images_simple(
    image_a_path: Path,
    image_b_path: Path,
    normalized_bbox: list[float] | None = None,
    rotation_deg: float = 0.0,
) -> dict[str, Any]:
    """Compute difference statistics between two distinct images with spatial & rotational alignment."""
    try:
        with Image.open(image_a_path) as img_a, Image.open(image_b_path) as img_b:
            img_a_rgb = img_a.convert("RGB")
            w_ref, h_ref = img_b.size

            if normalized_bbox and len(normalized_bbox) == 4:
                nx1, ny1, nx2, ny2 = normalized_bbox
                rx1 = max(0, min(w_ref - 1, int(round(nx1 * w_ref))))
                ry1 = max(0, min(h_ref - 1, int(round(ny1 * h_ref))))
                rx2 = max(rx1 + 1, min(w_ref, int(round(nx2 * w_ref))))
                ry2 = max(ry1 + 1, min(h_ref, int(round(ny2 * h_ref))))
                ref_patch = img_b.convert("RGB").crop((rx1, ry1, rx2, ry2))
            else:
                ref_patch = img_b.convert("RGB")

            rot = float(rotation_deg) % 360.0
            if rot != 0.0:
                ref_patch = ref_patch.rotate(rot, expand=True)

            img_b_resized = ref_patch.resize(img_a_rgb.size, Image.Resampling.BILINEAR)

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
    allow_synthetic: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Retrieve train-split normal reference image or synthesize if explicitly allowed."""
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

    if not allow_synthetic:
        raise FileNotFoundError(
            f"No train-normal reference available for category '{category}' in reference corpus. "
            "Synthetic references are strictly rejected in live/production mode."
        )

    # In mock/offline testing mode only: create / cache a synthetic reference template
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
            if "bbox" not in arguments:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'bbox' for crop_region.",
                    retriable=False,
                )
            bbox = arguments["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                return ToolResult(
                    success=False,
                    error=f"Invalid bbox dimensions: {bbox}",
                    retriable=False,
                )

            use_original = bool(arguments.get("use_original", False))
            work_context = context.fork()
            if use_original and "original_image_path" in context:
                work_context["current_image_path"] = context["original_image_path"]

            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "crop_image",
                "arguments": {"bbox": bbox},
            }
            output = reg.execute(call, work_context)
            if output.get("success"):
                context.adopt_image_from(work_context)
                path = str(_current_image_path(context))
                return ToolResult(
                    success=True,
                    output_path=path,
                    metadata={"image_size": output.get("image_size"), "bbox": bbox},
                )
            return ToolResult(success=False, error=output.get("error", "crop_image failed"), retriable=False)

        elif name == "zoom_region":
            if "scale" not in arguments:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'scale' for zoom_region.",
                    retriable=False,
                )
            try:
                scale = float(arguments["scale"])
                if scale <= 0:
                    raise ValueError("scale must be positive")
            except (ValueError, TypeError) as exc:
                return ToolResult(
                    success=False,
                    error=f"Invalid scale for zoom_region: {exc}",
                    retriable=False,
                )

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
            return ToolResult(success=False, error=output.get("error", "zoom_image failed"), retriable=False)

        elif name == "rotate_image":
            if "angle" not in arguments:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'angle' for rotate_image.",
                    retriable=False,
                )
            try:
                angle = float(arguments["angle"])
            except (ValueError, TypeError) as exc:
                return ToolResult(
                    success=False,
                    error=f"Invalid angle for rotate_image: {exc}",
                    retriable=False,
                )

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
            return ToolResult(success=False, error=output.get("error", "rotate_image failed"), retriable=False)

        elif name == "retrieve_normal_reference":
            category = str(arguments.get("category", "industrial_component"))
            corpus_dir = arguments.get("corpus_dir")
            allow_synthetic = bool(arguments.get("allow_synthetic", False))
            try:
                ref_path, meta = _locate_or_create_reference_image(
                    category=category,
                    corpus_dir=corpus_dir,
                    allow_synthetic=allow_synthetic,
                )
                return ToolResult(
                    success=True,
                    output_path=str(ref_path),
                    metadata=meta,
                )
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"No train-normal reference available in reference corpus: {exc}",
                    retriable=False,
                )

        elif name == "compare_with_reference":
            ref_path_str = arguments.get("reference_path")
            current_path = _current_image_path(context)

            if not ref_path_str:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'reference_path' for compare_with_reference.",
                    retriable=False,
                )

            ref_path = Path(ref_path_str)
            if not ref_path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Reference image not found at: {ref_path_str}",
                    retriable=False,
                )

            # Strictly reject comparing active image against itself
            if ref_path.resolve() == current_path.resolve():
                return ToolResult(
                    success=False,
                    error="Self-comparison rejected: reference image cannot be the current active inspection image.",
                    retriable=False,
                )

            normalized_bbox = arguments.get("normalized_bbox")
            rotation_deg = float(arguments.get("rotation_deg", 0.0))

            diff_meta = _compare_images_simple(
                current_path,
                ref_path,
                normalized_bbox=normalized_bbox,
                rotation_deg=rotation_deg,
            )
            if "error" in diff_meta:
                return ToolResult(success=False, error=diff_meta["error"], retriable=False)

            return ToolResult(
                success=True,
                output_path=str(current_path),
                metadata=diff_meta,
            )

        elif name == "localize_candidate":
            analysis_context = context.fork()
            if arguments.get("use_original", True) and "original_image_path" in context:
                analysis_context["current_image_path"] = context["original_image_path"]

            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "visual_analyzer",
                "arguments": {},
            }
            output = reg.execute(call, analysis_context)
            candidates = []
            if output.get("success"):
                dark_box = output.get("analysis", {}).get("dark_bbox")
                if dark_box:
                    candidates.append({"bbox": dark_box, "confidence": 0.85, "label": "salient_region"})

            current_path = _current_image_path(context)
            return ToolResult(
                success=True,
                output_path=str(current_path),
                metadata={"candidate_regions": candidates},
            )

        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}", retriable=False)

    except Exception as exc:
        return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}", retriable=False)
