from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
from agent0_protocol.schema import CanonicalTrajectory
from agent0_protocol.tools import ToolExecutionContext, execute_call_batch, get_tool_registry
from agent0_protocol.verifier import repair_function, retry_function


class ChainedImageResponses:
    """Return crop, then inspect, then finish a Responses tool loop."""

    def __init__(self) -> None:
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        step = len(self.requests)
        if step == 1:
            output = [{
                "type": "function_call",
                "call_id": "crop_1",
                "name": "crop_image",
                "arguments": json.dumps({"bbox": [0, 0, 3, 2]}),
            }]
        elif step == 2:
            output = [{
                "type": "function_call",
                "call_id": "visual_1",
                "name": "visual_analyzer",
                "arguments": "{}",
            }]
        else:
            output = [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The crop is 3 by 2."}],
            }]
        return SimpleNamespace(id=f"resp_{step}", output=output)


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def test_image_tools_chain_intermediate_images_and_hide_paths():
    registry = get_tool_registry()
    definitions = {tool["name"]: tool for tool in registry.definitions()}
    for name in ("crop_image", "zoom_image", "rotate_image", "ocr", "plot_parser", "visual_analyzer", "object_detector"):
        assert "image_path" not in definitions[name]["parameters"]["properties"]

    context = ToolExecutionContext(image=Image.new("RGB", (8, 6), "red"))
    original_path = context["current_image_path"]
    try:
        crop_result = execute_call_batch(
            registry,
            [_tool_call("crop_1", "crop_image", {"bbox": [0, 0, 3, 2]})],
            context,
        )
        assert crop_result[0]["output"] == {"success": True, "image_size": [3, 2]}
        assert context["current_image_path"] != original_path

        inspect_result = execute_call_batch(
            registry,
            [_tool_call("visual_1", "visual_analyzer", {})],
            context,
        )
        assert inspect_result[0]["output"]["analysis"]["size"] == [3, 2]
        assert "image_path" not in inspect_result[0]["output"]
    finally:
        context.close()


def test_responses_runtime_injects_input_image_and_carries_crop_forward():
    image_buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "blue").save(image_buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")
    fake = ChainedImageResponses()
    runtime = ResponsesRuntime(
        ResponsesConfig("http://localhost:8000/v1", "test", "model", max_tool_rounds=3),
        client=SimpleNamespace(responses=fake),
        probe_on_init=False,
    )

    trajectory = runtime.run([{
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Crop and inspect."},
            {"type": "input_image", "image_url": image_url},
        ],
    }])

    assert [item["name"] for item in trajectory.items if item["type"] == "function_call"] == [
        "crop_image", "visual_analyzer"
    ]
    # Check that crop_1 returned the newly transformed image to the model via input_image
    crop_call_output = next(
        item for item in fake.requests[1]["input"]
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "crop_1"
    )
    assert isinstance(crop_call_output["output"], list)
    types = [part["type"] for part in crop_call_output["output"]]
    assert "input_text" in types
    assert "input_image" in types
    img_part = next(part for part in crop_call_output["output"] if part["type"] == "input_image")
    assert img_part["image_url"].startswith("data:image/png;base64,")

    visual_call_output = next(
        item for item in fake.requests[2]["input"]
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "visual_1"
    )
    visual_output = json.loads(visual_call_output["output"])
    assert visual_output["analysis"]["size"] == [3, 2]


def test_tool_execution_context_checkpoint_and_rollback():
    registry = get_tool_registry()
    context = ToolExecutionContext(image=Image.new("RGB", (10, 8), "red"))
    original_path = context["current_image_path"]
    try:
        cp = context.checkpoint()
        assert cp["current_image_path"] == original_path

        # Step modifies the image
        res1 = execute_call_batch(
            registry,
            [_tool_call("crop_1", "crop_image", {"bbox": [0, 0, 4, 3]})],
            context,
        )
        assert res1[0]["output"]["success"] is True
        cropped_path = context["current_image_path"]
        assert cropped_path != original_path
        assert Path(cropped_path).is_file()

        # Rollback should restore original path and delete intermediate file
        context.rollback(cp)
        assert context["current_image_path"] == original_path
        assert not Path(cropped_path).exists()
        assert Path(original_path).is_file()

        # Subsequent inspection sees the original image dimensions
        res2 = execute_call_batch(
            registry,
            [_tool_call("visual_1", "visual_analyzer", {})],
            context,
        )
        assert res2[0]["output"]["analysis"]["size"] == [10, 8]
    finally:
        context.close()


def test_tool_execution_context_stack_checkpoints():
    registry = get_tool_registry()
    context = ToolExecutionContext(image=Image.new("RGB", (10, 8), "blue"))
    orig_path = context["current_image_path"]
    try:
        context.save_checkpoint()
        execute_call_batch(registry, [_tool_call("c1", "crop_image", {"bbox": [0, 0, 6, 6]})], context)
        step1_path = context["current_image_path"]
        assert step1_path != orig_path

        context.save_checkpoint()
        execute_call_batch(registry, [_tool_call("c2", "crop_image", {"bbox": [0, 0, 2, 2]})], context)
        step2_path = context["current_image_path"]
        assert step2_path != step1_path

        # Roll back to step 1
        context.restore_checkpoint()
        assert context["current_image_path"] == step1_path
        assert not Path(step2_path).exists()
        assert Path(step1_path).is_file()

        # Roll back to original
        context.restore_checkpoint()
        assert context["current_image_path"] == orig_path
        assert not Path(step1_path).exists()
        assert Path(orig_path).is_file()
    finally:
        context.close()


def test_verifier_repair_function_with_image_rollback():
    registry = get_tool_registry()
    context = ToolExecutionContext(image=Image.new("RGB", (10, 8), "green"))
    orig_path = context["current_image_path"]
    try:
        cp = context.checkpoint()
        trajectory = CanonicalTrajectory("traj_repair", registry.definitions())
        trajectory.append({"type": "function_call", "call_id": "bad_crop", "name": "crop_image", "arguments": {"bbox": [0, 0, 1, 1]}})
        out = execute_call_batch(registry, [trajectory.items[0]], context)
        trajectory.append(out[0])
        bad_path = context["current_image_path"]
        assert bad_path != orig_path

        # Repair with rollback to pre-tool checkpoint and valid arguments
        repair_id = repair_function(
            trajectory,
            "bad_crop",
            registry,
            new_arguments={"bbox": [0, 0, 5, 4]},
            context=context,
            rollback_checkpoint=cp,
        )
        assert repair_id != "bad_crop"
        assert context["current_image_path"] != orig_path
        assert not Path(bad_path).exists()

        # Verify output of the repaired call
        repaired_output = trajectory.items[-1]["output"]
        assert repaired_output["success"] is True
        assert repaired_output["image_size"] == [5, 4]
    finally:
        context.close()
