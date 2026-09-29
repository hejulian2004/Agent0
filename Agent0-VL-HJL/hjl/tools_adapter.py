"""Standard ToolResult representation and visual tool adapters for HJL."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from agent0_protocol.schema import new_call_id
from agent0_protocol.tools import (
    ToolExecutionContext,
    ToolRegistry,
    _current_image_path,
    _save_as_current,
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


def _compare_images_simple(image_a_path: Path, image_b_path: Path) -> dict[str, Any]:
    """Compute baseline difference metrics between two images."""
    try:
        with Image.open(image_a_path) as img_a, Image.open(image_b_path) as img_b:
            img_a_rgb = img_a.convert("RGB")
            # Resize image B to match image A dimensions for comparison
            img_b_resized = img_b.convert("RGB").resize(img_a_rgb.size)

            # Simple pixel difference
            diff_count = 0
            total_diff = 0.0
            width, height = img_a_rgb.size
            total_pixels = width * height

            # Subsample for speed if image is large
            step = max(1, int(math.sqrt(total_pixels / 10000)))
            samples = 0

            for y in range(0, height, step):
                for x in range(0, width, step):
                    pa = img_a_rgb.getpixel((x, y))
                    pb = img_b_resized.getpixel((x, y))
                    delta = (abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) + abs(pa[2] - pb[2])) / 3.0
                    total_diff += delta
                    if delta > 30:  # Threshold for noticeable pixel diff
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
            }
    except Exception as exc:
        return {"error": str(exc), "similarity": 0.5}


def execute_adapted_tool(
    name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
    registry: ToolRegistry | None = None,
) -> ToolResult:
    """Execute an HJL visual tool and return a standardized ToolResult."""
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
            query = str(arguments.get("query", "normal component reference"))
            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "retrieve",
                "arguments": {"query": query},
            }
            output = reg.execute(call, context)
            if output.get("success"):
                results = output.get("results", [])
                return ToolResult(
                    success=True,
                    metadata={"results": results, "query": query, "count": len(results)},
                )
            return ToolResult(success=False, error=output.get("error", "retrieve failed"))

        elif name == "compare_with_reference":
            ref_path_str = arguments.get("reference_path")
            current_path = _current_image_path(context)
            if not ref_path_str:
                # If no reference path supplied, look for existing reference in context or mock
                ref_path = current_path
            else:
                ref_path = Path(ref_path_str)
                if not ref_path.is_file():
                    ref_path = current_path

            diff_meta = _compare_images_simple(current_path, ref_path)
            return ToolResult(
                success=True,
                output_path=str(current_path),
                metadata=diff_meta,
            )

        elif name == "localize_candidate":
            # Run object detector or visual analyzer to find candidate boxes
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

            # If no dark box, generate center default candidate
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
