"""Single tool registry used by API, model adapters, datasets and verifier."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import re
import tempfile
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator

from .schema import ProtocolError


Handler = Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]


class _ImageFileOwner:
    """Track temporary input and transformed images for one trajectory."""

    def __init__(self) -> None:
        self.paths: set[str] = set()

    def add(self, path: str | Path) -> None:
        self.paths.add(str(path))

    def cleanup(self) -> None:
        for value in self.paths:
            try:
                Path(value).unlink(missing_ok=True)
            except OSError:
                pass
        self.paths.clear()

    def __del__(self) -> None:  # pragma: no cover - safety cleanup on exceptions
        self.cleanup()


class ToolExecutionContext(dict[str, Any]):
    """Per-trajectory context carrying the original and active image.

    Image transforms replace ``current_image_path`` for the next tool round.
    Sibling calls in one Responses output can use forked snapshots so they do
    not accidentally consume each other's intermediate images.
    """

    def __init__(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        image: Any | None = None,
        _owner: _ImageFileOwner | None = None,
    ) -> None:
        super().__init__(values or {})
        self._image_files = _owner or _ImageFileOwner()
        self._image_changed = False
        self._checkpoints: list[dict[str, Any]] = []
        if image is not None:
            self.set_current_image(image)

    def set_current_image(self, image: Any) -> None:
        path, owned = _materialize_image(image)
        if owned:
            self._image_files.add(path)
        if "original_image_path" not in self:
            self["original_image_path"] = str(path)
        self["current_image_path"] = str(path)
        self._image_changed = True

    def set_generated_image(self, path: str | Path) -> None:
        self._image_files.add(path)
        self["current_image_path"] = str(path)
        self._image_changed = True

    def checkpoint(self) -> dict[str, Any]:
        """Save a snapshot of the current image state for potential rollback."""
        return {
            "current_image_path": self.get("current_image_path"),
            "image_changed": self._image_changed,
            "current_image_error": self.get("current_image_error"),
            "owned_paths": set(self._image_files.paths),
        }

    def save_checkpoint(self) -> dict[str, Any]:
        """Push a snapshot onto the internal checkpoint stack and return it."""
        cp = self.checkpoint()
        self._checkpoints.append(cp)
        return cp

    def rollback(self, checkpoint: dict[str, Any] | None = None) -> None:
        """Roll back the image state to a saved checkpoint.

        If checkpoint is not provided, pops and restores the most recent checkpoint
        on the internal stack. Cleans up intermediate temporary files that
        were created after the checkpoint.
        """
        if checkpoint is None:
            if not self._checkpoints:
                return
            checkpoint = self._checkpoints.pop()

        target_path = checkpoint.get("current_image_path")
        if target_path is not None:
            self["current_image_path"] = target_path
        else:
            self.pop("current_image_path", None)

        self._image_changed = checkpoint.get("image_changed", False)

        error = checkpoint.get("current_image_error")
        if error is not None:
            self["current_image_error"] = error
        else:
            self.pop("current_image_error", None)

        owned_at_checkpoint = checkpoint.get("owned_paths")
        if owned_at_checkpoint is not None:
            discarded_paths = self._image_files.paths - owned_at_checkpoint
            for p in discarded_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
            self._image_files.paths &= owned_at_checkpoint

    def restore_checkpoint(self) -> None:
        """Convenience alias to rollback to the last saved checkpoint."""
        self.rollback(None)

    def fork(self) -> "ToolExecutionContext":
        forked = ToolExecutionContext(self, _owner=self._image_files)
        forked._image_changed = self._image_changed
        forked._checkpoints = [dict(cp) for cp in self._checkpoints]
        return forked

    def adopt_image_from(self, other: "ToolExecutionContext") -> None:
        if other._image_changed:
            self["current_image_path"] = other["current_image_path"]
            self._image_changed = True

    def close(self) -> None:
        self._checkpoints.clear()
        self._image_files.cleanup()


def _image_suffix(mime: str | None = None, data: bytes = b"") -> str:
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get((mime or "").lower())
    if suffix:
        return suffix
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def _write_temporary_image(data: bytes, mime: str | None = None) -> tuple[Path, bool]:
    if not data:
        raise ValueError("image input is empty")
    with tempfile.NamedTemporaryFile(prefix="agent0-image-", suffix=_image_suffix(mime, data), delete=False) as output:
        output.write(data)
        return Path(output.name), True


def _materialize_image(value: Any) -> tuple[Path, bool]:
    """Return a local image path; only generated temporary files are owned."""
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            value = value["path"]
        elif value.get("image_url"):
            value = value["image_url"]

    if isinstance(value, (str, os.PathLike)):
        raw = os.fspath(value)
        if raw.startswith("data:"):
            header, separator, payload = raw.partition(",")
            if not separator:
                raise ValueError("input_image data URL is malformed")
            mime = header[5:].split(";", 1)[0]
            try:
                if header.endswith(";base64"):
                    data = base64.b64decode(payload, validate=True)
                else:
                    from urllib.parse import unquote_to_bytes

                    data = unquote_to_bytes(payload)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("input_image data URL cannot be decoded") from exc
            return _write_temporary_image(data, mime)
        path = Path(raw).expanduser()
        if not path.is_file():
            raise ValueError("image tools require a local current input image")
        return path.resolve(), False

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _write_temporary_image(bytes(value))

    if hasattr(value, "save"):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return _write_temporary_image(buffer.getvalue(), "image/png")

    raise TypeError(f"unsupported current image value: {type(value).__name__}")


def execute_call_batch(
    registry: "ToolRegistry",
    calls: list[Mapping[str, Any]],
    context: ToolExecutionContext,
) -> list[dict[str, Any]]:
    """Execute one model turn's calls against one image snapshot.

    If several calls transform the image at once, the last successful result
    becomes active for the following model turn.
    """
    snapshot = context.fork()
    changed_contexts: list[ToolExecutionContext] = []
    results: list[dict[str, Any]] = []
    for call in calls:
        call_context = snapshot.fork()
        try:
            output = registry.execute(call, call_context)
        except Exception as exc:
            output = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        if output.get("success") is True and call_context._image_changed:
            changed_contexts.append(call_context)
        results.append({
            "type": "function_call_output",
            "call_id": call.get("call_id"),
            "output": output,
        })
    if changed_contexts:
        context.adopt_image_from(changed_contexts[-1])
    return results


def input_image_from_items(items: Any) -> Any | None:
    """Find the first Responses ``input_image`` in canonical input items."""
    for item in items:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "input_image":
                image = part.get("image_url")
                if image is not None:
                    return image
    return None


@dataclass(frozen=True)
class RegisteredTool:
    schema: dict[str, Any]
    handler: Handler


class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegisteredTool] = {}

    def register(self, schema: Mapping[str, Any], handler: Handler) -> None:
        definition = copy.deepcopy(dict(schema))
        name = definition.get("name")
        if definition.get("type") != "function" or not isinstance(name, str) or not name:
            raise ProtocolError("tool registration requires type=function and a name")
        if name in self._entries:
            raise ProtocolError(f"duplicate tool registration: {name}")
        if definition.get("strict") is not True:
            raise ProtocolError(f"tool {name} must set strict=true")
        parameters = definition.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ProtocolError(f"tool {name} needs an object parameters schema")
        Draft202012Validator.check_schema(parameters)
        self._entries[name] = RegisteredTool(definition, handler)

    def definitions(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(entry.schema) for entry in self._entries.values()]

    def contains(self, name: str) -> bool:
        return name in self._entries

    def execute(self, call: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if call.get("type") != "function_call":
            raise ProtocolError("expected a function_call item")
        name = call.get("name")
        if name not in self._entries:
            raise ProtocolError(f"unregistered function: {name!r}")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            raise ProtocolError("function arguments must be an object")
        entry = self._entries[str(name)]
        Draft202012Validator(entry.schema["parameters"]).validate(arguments)
        try:
            result = entry.handler(copy.deepcopy(arguments), context if context is not None else {})
            if not isinstance(result, dict):
                raise TypeError("handler must return an object")
            return result
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def _current_image_path(context: Mapping[str, Any]) -> Path:
    value = context.get("current_image_path")
    if not value:
        detail = context.get("current_image_error")
        raise ValueError(str(detail) if detail else "the current input image is unavailable")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError("the current image is no longer available")
    return path


def _save_as_current(image: Any, context: Mapping[str, Any]) -> None:
    try:
        with tempfile.NamedTemporaryFile(prefix="agent0-image-", suffix=".png", delete=False) as output:
            path = Path(output.name)
        image.save(path, format="PNG")
    except Exception:
        if "path" in locals():
            path.unlink(missing_ok=True)
        raise
    if isinstance(context, ToolExecutionContext):
        context.set_generated_image(path)
    elif isinstance(context, MutableMapping):
        context["current_image_path"] = str(path)


def _python_exec(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    from sandbox import get_parallel_sandbox

    runner = get_parallel_sandbox()
    timeout = int(context.get("sandbox_timeout", os.getenv("SANDBOX_RUN_TIMEOUT", "10")))
    success, stdout, stderr = asyncio.run(
        runner([arguments["code"]], num_processes=1, run_timeout=timeout)
    )
    return {
        "success": bool(success[0]),
        "stdout": str(stdout[0])[:512],
        "stderr": str(stderr[0])[:512],
    }


def _crop_image(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    from PIL import Image

    source = _current_image_path(context)
    bbox = arguments["bbox"]
    with Image.open(source) as image:
        if not (0 <= bbox[0] < bbox[2] <= image.width and 0 <= bbox[1] < bbox[3] <= image.height):
            raise ValueError("bbox is outside the image")
        cropped = image.crop(tuple(bbox))
        _save_as_current(cropped, context)
        size = cropped.size
    return {"success": True, "image_size": [size[0], size[1]]}


_OCR_ENGINE: Any = None
_DETECTOR: Any = None


def _ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr import RapidOCR

        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _run_ocr(image_path: str) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        image.load()
        image_array = np.asarray(image.convert("RGB"))
    result = _ocr_engine()(image_array)
    lines = []
    if result is not None and result.txts:
        for box, text, confidence in zip(result.boxes, result.txts, result.scores):
            coordinates = [[int(round(float(x))), int(round(float(y)))] for x, y in box]
            lines.append({"text": str(text), "confidence": float(confidence), "polygon": coordinates})
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines}


def _ocr(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    result = _run_ocr(str(_current_image_path(context)))
    return {"success": True, **result}


def _zoom_image(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    from PIL import Image

    source = _current_image_path(context)
    scale = arguments["scale"]
    if not (0.1 <= scale <= 8.0):
        raise ValueError("scale must be between 0.1 and 8")
    with Image.open(source) as image:
        output_image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
        _save_as_current(output_image, context)
        size = output_image.size
        output_image.close()
    return {"success": True, "image_size": [size[0], size[1]]}


def _rotate_image(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    from PIL import Image

    angle = arguments["angle"]
    if not (-360.0 <= angle <= 360.0):
        raise ValueError("angle must be between -360 and 360 degrees")
    with Image.open(_current_image_path(context)) as image:
        output_image = image.rotate(angle, expand=True)
        _save_as_current(output_image, context)
        width, height = output_image.size
        output_image.close()
    return {"success": True, "image_size": [width, height]}


def _visual_analyzer(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    with Image.open(_current_image_path(context)) as image:
        width, height = image.size
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    result: dict[str, Any] = {
        "size": [width, height],
        "mean_rgb": [round(float(value), 1) for value in array.mean(axis=(0, 1))],
    }
    quantized = (array.astype(np.uint8) // 32) * 32
    colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
    result["dominant_rgb_approx"] = [int(value) for value in colors[int(counts.argmax())]]
    gray = array.mean(axis=2)
    ys, xs = np.where(gray < float(gray.mean()))
    result["dark_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if len(xs) else None
    return {"success": True, "analysis": result}


def _plot_parser(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Return chart text/locations; it deliberately does not invent plotted values."""
    from PIL import Image

    image_path = _current_image_path(context)
    with Image.open(image_path) as image:
        width, height = image.size
    result = _run_ocr(str(image_path))
    return {
        "success": True,
        "image_size": {"width": width, "height": height},
        "labels": result["lines"],
    }


def _object_detector(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    global _DETECTOR
    from ultralytics import YOLO

    model_path = str(context.get("detector_model") or os.getenv("AGENT0_DETECTOR_MODEL", ".venv/models/yolo26n.pt"))
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = Path(__file__).resolve().parents[1] / model_file
    if not model_file.is_file():
        alt = Path(__file__).resolve().parents[1] / "models" / model_file.name
        if alt.is_file():
            model_file = alt
        else:
            raise FileNotFoundError(f"detector model weights not found: {model_file}")
    if _DETECTOR is None:
        _DETECTOR = YOLO(str(model_file))
    confidence = float(context.get("detector_confidence", os.getenv("AGENT0_DETECTOR_CONFIDENCE", "0.25")))
    max_detections = int(context.get("detector_max_detections", os.getenv("AGENT0_DETECTOR_MAX_DETECTIONS", "100")))
    device = str(context.get("detector_device", os.getenv("AGENT0_DETECTOR_DEVICE", "cpu")))
    results = _DETECTOR.predict(
        source=str(_current_image_path(context)), conf=confidence, max_det=max_detections,
        device=device, verbose=False,
    )
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        label = str(results[0].names[class_id])
        detections.append({
            "label": label,
            "confidence": round(float(box.conf.item()), 5),
            "bbox_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
        })
    return {"success": True, "model": model_file.name, "detections": detections}


def _retrieve(arguments: dict[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    corpus_dir = Path(context.get("retrieval_corpus_dir") or os.getenv("AGENT0_RETRIEVAL_CORPUS_DIR", "data/knowledge"))
    if not corpus_dir.is_absolute():
        corpus_dir = Path(__file__).resolve().parents[1] / corpus_dir
    corpus_dir = corpus_dir.resolve()
    terms = [term.casefold() for term in re.findall(r"[\w.-]+", arguments["query"]) if len(term) > 1]
    if not terms:
        raise ValueError("query must contain at least one searchable term")
    allowed_suffixes = {".txt", ".md", ".rst", ".json", ".jsonl", ".csv"}
    candidates = [path for path in corpus_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed_suffixes] if corpus_dir.is_dir() else []
    max_results = int(os.getenv("AGENT0_RETRIEVAL_MAX_RESULTS", "5"))
    snippet_chars = int(os.getenv("AGENT0_RETRIEVAL_SNIPPET_CHARS", "800"))
    matches = []
    for path in candidates:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = sum(content.casefold().count(term) for term in terms)
        if score:
            matches.append((score, path, content))
    matches.sort(key=lambda item: (-item[0], str(item[1])))
    return {
        "success": True,
        "corpus_dir": str(corpus_dir),
        "results": [
            {"source": str(path.relative_to(corpus_dir)), "score": score, "snippet": content[:snippet_chars]}
            for score, path, content in matches[:max_results]
        ],
    }


def _default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        {
            "type": "function",
            "name": "python_exec",
            "description": "Run Python; preloaded: math, np, Image, cv2, sp, RapidOCR.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _python_exec,
    )
    registry.register(
        {
            "type": "function",
            "name": "crop_image",
            "description": "Crop current image to [x1,y1,x2,y2]; result becomes current.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["bbox"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _crop_image,
    )
    registry.register(
        {
            "type": "function",
            "name": "zoom_image",
            "description": "Resize current image by scale; result becomes current.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scale": {"type": "number"},
                },
                "required": ["scale"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _zoom_image,
    )
    registry.register(
        {
            "type": "function",
            "name": "rotate_image",
            "description": "Rotate current image by degrees; result becomes current.",
            "parameters": {
                "type": "object",
                "properties": {"angle": {"type": "number"}},
                "required": ["angle"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _rotate_image,
    )
    registry.register(
        {
            "type": "function",
            "name": "ocr",
            "description": "OCR current image text.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _ocr,
    )
    registry.register(
        {
            "type": "function",
            "name": "plot_parser",
            "description": "OCR current chart labels and locations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _plot_parser,
    )
    registry.register(
        {
            "type": "function",
            "name": "visual_analyzer",
            "description": "Summarize current image size, colors, and dark bounds.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _visual_analyzer,
    )
    registry.register(
        {
            "type": "function",
            "name": "object_detector",
            "description": "Detect COCO objects in current image.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _object_detector,
    )
    registry.register(
        {
            "type": "function",
            "name": "retrieve",
            "description": "Keyword-search local documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        _retrieve,
    )
    return registry


_REGISTRY: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _default_registry()
    return _REGISTRY
