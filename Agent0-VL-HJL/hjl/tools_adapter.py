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

# --- HJL Tool Definitions (Currently deregistered/commented out for general Agent0-VL) ---
HJL_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # {
    #     "type": "function",
    #     "name": "crop_region",
    #     "description": "Crop a specific region of interest using [x1, y1, x2, y2] bounding box coordinates.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "bbox": {
    #                 "type": "array",
    #                 "items": {"type": "integer"},
    #                 "description": "Bounding box coordinates [x1, y1, x2, y2]",
    #             },
    #             "use_original": {
    #                 "type": "boolean",
    #                 "description": "Whether to crop from the original full image rather than current crop",
    #             },
    #         },
    #         "required": ["bbox"],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # },
    # {
    #     "type": "function",
    #     "name": "zoom_region",
    #     "description": "Resize and magnify the current inspection region by a positive scale factor.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "scale": {
    #                 "type": "number",
    #                 "description": "Scale magnification factor (> 0.0)",
    #             },
    #             "bbox": {
    #                 "type": ["array", "null"],
    #                 "items": {"type": "integer"},
    #                 "description": "Associated original region bounding box",
    #             },
    #         },
    #         "required": ["scale"],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # },
    # {
    #     "type": "function",
    #     "name": "rotate_image",
    #     "description": "Rotate the current visual inspection region by a specified angle in degrees.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "angle": {
    #                 "type": "number",
    #                 "description": "Rotation angle in degrees",
    #             },
    #         },
    #         "required": ["angle"],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # },
    # {
    #     "type": "function",
    #     "name": "retrieve_normal_reference",
    #     "description": "Retrieve a verified defect-free normal training reference image for the product category.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "category": {
    #                 "type": "string",
    #                 "description": "Product category name",
    #             },
    #             "corpus_dir": {
    #                 "type": ["string", "null"],
    #                 "description": "Optional path to reference image corpus",
    #             },
    #             "allow_synthetic": {
    #                 "type": "boolean",
    #                 "description": "Whether synthetic template references are permitted (offline/mock only)",
    #             },
    #         },
    #         "required": ["category"],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # },
    # {
    #     "type": "function",
    #     "name": "compare_with_reference",
    #     "description": "Compare the current local inspection region against a normal reference template.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "reference_path": {
    #                 "type": "string",
    #                 "description": "Filesystem path to the normal reference image",
    #             },
    #             "normalized_bbox": {
    #                 "type": ["array", "null"],
    #                 "items": {"type": "number"},
    #                 "description": "Normalized bounding box [nx1, ny1, nx2, ny2] in [0, 1] range",
    #             },
    #             "rotation_deg": {
    #                 "type": "number",
    #                 "description": "Rotation angle applied to the test ROI in degrees",
    #             },
    #             "bbox": {
    #                 "type": ["array", "null"],
    #                 "items": {"type": "integer"},
    #                 "description": "Associated original region bounding box",
    #             },
    #         },
    #         "required": ["reference_path"],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # },
    # {
    #     "type": "function",
    #     "name": "localize_candidate",
    #     "description": "Scan and propose suspicious defect candidate bounding box regions on the full image.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "use_original": {
    #                 "type": "boolean",
    #                 "description": "Whether to perform localization against original uncropped image",
    #             },
    #         },
    #         "required": [],
    #         "additionalProperties": False,
    #     },
    #     "strict": True,
    # },
]


def get_hjl_tool_definitions(*, agent_visible: bool = False) -> list[dict[str, Any]]:
    """Return JSON schemas for all canonical HJL visual tools.

    If agent_visible is True, strips runtime-injected arguments (corpus_dir, allow_synthetic, category)
    so baseline agent models and canonical training trajectories reason only over semantic action parameters.
    """
    defs = copy.deepcopy(HJL_TOOL_DEFINITIONS)
    if agent_visible:
        for t in defs:
            if t["name"] == "retrieve_normal_reference":
                props = t["parameters"]["properties"]
                props.pop("corpus_dir", None)
                props.pop("allow_synthetic", None)
                props.pop("category", None)
                t["parameters"]["required"] = []
    return defs


def _is_retriable_error(error: str | None) -> bool:
    """Classify whether a tool execution failure is transient and eligible for RETRY_TOOL."""
    if not error:
        return False
    text = str(error).lower()
    return any(
        x in text
        for x in (
            "timeouterror",
            "timeout",
            "timed out",
            "connectionerror",
            "connection reset",
            "temporarily unavailable",
            "temporary failure",
        )
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
    category: str = "visual_object",
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
                    "dataset": "reference_corpus",
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

    if reg.contains(name):
        try:
            out = reg.execute(
                {"type": "function_call", "name": name, "call_id": new_call_id(), "arguments": arguments},
                context,
            )
            success = bool(out.get("success", False))
            error = out.get("error") if not success else None
            return ToolResult(
                success=success,
                output_path=out.get("image_path") or out.get("output_path"),
                metadata=copy.deepcopy(out),
                error=error,
                retriable=_is_retriable_error(error),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                retriable=_is_retriable_error(str(exc)),
            )

    # 1. Unified JSON Schema validation against HJL_TOOL_DEFINITIONS
    tool_defs = {t["name"]: t for t in HJL_TOOL_DEFINITIONS}
    if name not in tool_defs:
        return ToolResult(
            success=False,
            error=f"HJL tool '{name}' is currently deregistered. Use canonical Agent0-VL tools instead.",
            retriable=False,
        )

    try:
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(tool_defs[name]["parameters"])
        errors = sorted(validator.iter_errors(arguments), key=lambda e: str(e.path))
        if errors:
            return ToolResult(
                success=False,
                error=f"Schema validation error for '{name}': {errors[0].message}",
                retriable=False,
            )
    except Exception as exc:
        return ToolResult(success=False, error=f"Schema validator error: {exc}", retriable=False)

    try:
        if name == "crop_region":
            if "bbox" not in arguments:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'bbox' for crop_region.",
                    retriable=False,
                )
            bbox = arguments["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(type(v) is int for v in bbox):
                return ToolResult(
                    success=False,
                    error=f"bbox must be a list of 4 integers, got {bbox}",
                    retriable=False,
                )
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or bbox[0] < 0 or bbox[1] < 0:
                return ToolResult(
                    success=False,
                    error=f"Invalid bbox dimensions [x1,y1,x2,y2]: {bbox}",
                    retriable=False,
                )

            if "use_original" in arguments and type(arguments["use_original"]) is not bool:
                return ToolResult(
                    success=False,
                    error="use_original must be a boolean",
                    retriable=False,
                )
            use_original = bool(arguments.get("use_original", False))

            work_context = context.fork()
            if "image_path" in arguments and arguments["image_path"]:
                work_context["current_image_path"] = arguments["image_path"]
            elif use_original and "original_image_path" in context:
                work_context["current_image_path"] = context["original_image_path"]

            crop_args: dict[str, Any] = {"bbox": bbox}
            if "image_path" in arguments:
                crop_args["image_path"] = arguments["image_path"]

            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "crop_image",
                "arguments": crop_args,
            }
            output = reg.execute(call, work_context)
            if output.get("success"):
                context.adopt_image_from(work_context)
                path = str(_current_image_path(context))
                return ToolResult(
                    success=True,
                    output_path=path,
                    metadata={"image_size": output.get("image_size"), "bbox": bbox, "image_path": path},
                )
            err = output.get("error", "crop_image failed")
            return ToolResult(success=False, error=err, retriable=_is_retriable_error(err))

        elif name == "zoom_region":
            if "scale" not in arguments:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'scale' for zoom_region.",
                    retriable=False,
                )
            scale = arguments["scale"]
            if type(scale) is bool or not isinstance(scale, (int, float)) or scale <= 0:
                return ToolResult(
                    success=False,
                    error=f"scale must be a positive number, got {scale!r}",
                    retriable=False,
                )
            scale = float(scale)

            zoom_args: dict[str, Any] = {"scale": scale}
            if "image_path" in arguments:
                zoom_args["image_path"] = arguments["image_path"]

            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "zoom_image",
                "arguments": zoom_args,
            }
            output = reg.execute(call, context)
            if output.get("success"):
                path = str(_current_image_path(context))
                meta: dict[str, Any] = {"image_size": output.get("image_size"), "scale": scale, "image_path": path}
                if "bbox" in arguments and isinstance(arguments["bbox"], list):
                    meta["bbox"] = list(arguments["bbox"])
                return ToolResult(
                    success=True,
                    output_path=path,
                    metadata=meta,
                )
            err = output.get("error", "zoom_image failed")
            return ToolResult(success=False, error=err, retriable=_is_retriable_error(err))

        elif name == "rotate_image":
            if "angle" not in arguments:
                return ToolResult(
                    success=False,
                    error="Missing required argument 'angle' for rotate_image.",
                    retriable=False,
                )
            angle = arguments["angle"]
            if type(angle) is bool or not isinstance(angle, (int, float)):
                return ToolResult(
                    success=False,
                    error=f"angle must be a number, got {angle!r}",
                    retriable=False,
                )
            angle = float(angle)

            rot_args: dict[str, Any] = {"angle": angle}
            if "image_path" in arguments:
                rot_args["image_path"] = arguments["image_path"]

            call = {
                "type": "function_call",
                "call_id": new_call_id(),
                "name": "rotate_image",
                "arguments": rot_args,
            }
            output = reg.execute(call, context)
            if output.get("success"):
                path = str(_current_image_path(context))
                return ToolResult(
                    success=True,
                    output_path=path,
                    metadata={"image_size": output.get("image_size"), "angle": angle, "image_path": path},
                )
            err = output.get("error", "rotate_image failed")
            return ToolResult(success=False, error=err, retriable=_is_retriable_error(err))

        # --- Industrial Anomaly Detection Tools (Commented out for general Agent0-VL) ---
        # elif name == "retrieve_normal_reference":
        #     if "category" not in arguments:
        #         return ToolResult(
        #             success=False,
        #             error="Missing required argument 'category' for retrieve_normal_reference.",
        #             retriable=False,
        #         )
        #     category = str(arguments["category"])
        #     corpus_dir = arguments.get("corpus_dir")
        #
        #     if "allow_synthetic" in arguments and type(arguments["allow_synthetic"]) is not bool:
        #         return ToolResult(
        #             success=False,
        #             error="allow_synthetic must be a boolean",
        #             retriable=False,
        #         )
        #     allow_synthetic = bool(arguments.get("allow_synthetic", False))
        #
        #     try:
        #         ref_path, meta = _locate_or_create_reference_image(
        #             category=category,
        #             corpus_dir=corpus_dir,
        #             allow_synthetic=allow_synthetic,
        #         )
        #         return ToolResult(
        #             success=True,
        #             output_path=str(ref_path),
        #             metadata=meta,
        #         )
        #     except Exception as exc:
        #         return ToolResult(
        #             success=False,
        #             error=f"No train-normal reference available in reference corpus: {exc}",
        #             retriable=False,
        #         )
        #
        # elif name == "compare_with_reference":
        #     ref_path_str = arguments.get("reference_path")
        #     current_path = _current_image_path(context)
        #
        #     if not ref_path_str or not isinstance(ref_path_str, str):
        #         return ToolResult(
        #             success=False,
        #             error="Missing required argument 'reference_path' for compare_with_reference.",
        #             retriable=False,
        #         )
        #
        #     ref_path = Path(ref_path_str)
        #     if not ref_path.is_file():
        #         return ToolResult(
        #             success=False,
        #             error=f"Reference image not found at: {ref_path_str}",
        #             retriable=False,
        #         )
        #
        #     # Strictly reject comparing active image against itself
        #     if ref_path.resolve() == current_path.resolve():
        #         return ToolResult(
        #             success=False,
        #             error="Self-comparison rejected: reference image cannot be the current active inspection image.",
        #             retriable=False,
        #         )
        #
        #     normalized_bbox = arguments.get("normalized_bbox")
        #     if normalized_bbox is not None:
        #         if (
        #             not isinstance(normalized_bbox, list)
        #             or len(normalized_bbox) != 4
        #             or not all(isinstance(v, (int, float)) and type(v) is not bool for v in normalized_bbox)
        #         ):
        #             return ToolResult(
        #                 success=False,
        #                 error="normalized_bbox must be a list of 4 numbers",
        #                 retriable=False,
        #             )
        #
        #     rotation_deg = float(arguments.get("rotation_deg", 0.0))
        #
        #     diff_meta = _compare_images_simple(
        #         current_path,
        #         ref_path,
        #         normalized_bbox=normalized_bbox,
        #         rotation_deg=rotation_deg,
        #     )
        #     if "error" in diff_meta:
        #         return ToolResult(success=False, error=diff_meta["error"], retriable=False)
        #
        #     if "bbox" in arguments and isinstance(arguments["bbox"], list):
        #         diff_meta["bbox"] = list(arguments["bbox"])
        #
        #     return ToolResult(
        #         success=True,
        #         output_path=str(current_path),
        #         metadata=diff_meta,
        #     )

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
            if not output.get("success"):
                err = output.get("error", "visual_analyzer failed")
                return ToolResult(
                    success=False,
                    error=err,
                    retriable=_is_retriable_error(err),
                )

            candidates = []
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
        return ToolResult(
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            retriable=_is_retriable_error(str(exc)),
        )
