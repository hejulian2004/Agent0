"""One semantic rendering and image order for local SFT and RL."""
import base64
import io
from pathlib import Path

from PIL import Image
from agent0_protocol.adapters import QwenModelAdapter, ResponsesAdapter
from agent0_protocol.schema import ProtocolError


def load_image(value):
    if isinstance(value, Image.Image):
        return value.copy().convert("RGB")
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, bytes):
        stream = io.BytesIO(value)
    elif str(value).startswith("data:image/"):
        stream = io.BytesIO(base64.b64decode(str(value).split(",", 1)[1], validate=True))
    else:
        stream = Path(value)
    with Image.open(stream) as image:
        image.load()
        return image.convert("RGB")


def item_images(item):
    if item["type"] == "message":
        content = item["content"]
        return [p["image_url"] for p in content if p.get("type") == "input_image"] if isinstance(content, list) else []
    if item["type"] == "function_call_output":
        wire = ResponsesAdapter.function_result(item)["output"]
        return [p["image_url"] for p in wire if p.get("type") == "input_image"] if isinstance(wire, list) else []
    return []


def render_with_images(trajectory, tokenizer, *, generate=False):
    images = [load_image(value) for item in trajectory.items for value in item_images(item)]
    text = QwenModelAdapter(tokenizer).render(trajectory.items, trajectory.tools, generate=generate)
    if text.count("<image>") != len(images):
        raise ProtocolError("Canonical image markers and image objects differ")
    return text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>"), images


def assistant_labels(ids, tokenizer):
    """Supervise assistant spans, including structured calls and reasoning."""
    start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    labels = [-100] * len(ids)
    cursor = 0
    while cursor < len(ids):
        if ids[cursor] != start:
            cursor += 1
            continue
        begin = cursor
        cursor += 1
        while cursor < len(ids) and ids[cursor] != end:
            cursor += 1
        stop = min(cursor + 1, len(ids))
        header = tokenizer.decode(ids[begin + 1:min(begin + 8, stop)], skip_special_tokens=False)
        if header.startswith("assistant\n"):
            prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            for index in range(begin + len(prefix), stop):
                if ids[index] != image_pad:
                    labels[index] = ids[index]
        cursor = stop
    return labels
