from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
from agent0_protocol.schema import ProtocolError
from agent0_protocol.tools import ToolExecutionContext, _python_exec
from scripts.build_sft_dataset import SFTTrajectoryBuilder, run_task_rollouts_concurrent
from tools.data_builder import sft_quality as quality
from tools.data_builder.sources import normalize_source_question, load_source, _extract_question, _raw_ground_truth
from verl.prompts.agent0_templates import render_system_prompt


def message(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def verifier(confidence=0.9):
    return message(json.dumps({"score": 1, "confidence": confidence, "critique": "check", "tool_check": True}))


class ScriptedClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return SimpleNamespace(id="response", output=next(self.outputs))


def runtime_for(outputs):
    client = ScriptedClient(outputs)
    runtime = ResponsesRuntime(ResponsesConfig("http://localhost:8000/v1", "test", "model"),
                               client=SimpleNamespace(responses=client), probe_on_init=False)
    return runtime, client


def initial():
    return [{"type": "message", "role": "system", "content": render_system_prompt()},
            {"type": "message", "role": "user", "content": "Compute 40 + 2"}]


def run(outputs):
    runtime, client = runtime_for(outputs)
    trajectory = quality.run_serc(runtime, initial(), trajectory_id="test")
    return trajectory, runtime, client


def test_iterative_patch_and_no_change_preserve_all_turns():
    outputs = [[message("<answer>40</answer>")], [verifier(0.1)],
               [message('{"action":"NO_CHANGE"}')], [message("<answer>41</answer>")], [verifier(0.2)],
               [message('{"action":"PATCH","patch":"recalculate"}')], [message("<answer>42</answer>")], [verifier()]]
    trajectory, runtime, client = run(outputs)
    rounds = trajectory.metadata["serc"]["steps"][0]
    assert len(rounds) == 3
    assert quality.solver_text(trajectory) == "<answer>42</answer>"
    assert trajectory.items[0]["content"] == render_system_prompt()
    quality.audit_serc(trajectory)
    assert quality.judge_answer(runtime, trajectory, {"ground_truth": "42"}) is False
    assert quality.verify_sft_semantics(trajectory, runtime.registry).valid
    assert all("max_output_tokens" not in request for request in client.requests)
    assert all(0 < request["timeout"] <= 300 for request in client.requests)


def test_failed_tool_triggers_repair_even_with_final_answer(monkeypatch):
    def execute(registry, calls, context):
        return [{"type": "function_call_output", "call_id": call["call_id"],
                 "output": {"success": False, "error": "real failure"}} for call in calls]
    monkeypatch.setattr(quality, "execute_call_batch", execute)
    call = {"type": "function_call", "call_id": "failed", "name": "python_exec", "arguments": '{"code":"raise ValueError()"}'}
    trajectory, runtime, client = run([[message("<answer>42</answer>"), call], [verifier()],
                                     [message('{"action":"PATCH","patch":"compute without failed tool"}')],
                                     [message("<answer>42</answer>")], [verifier()]])
    assert len(trajectory.metadata["serc"]["steps"][0]) == 2
    assert any(item.get("output", {}).get("success") is False for item in trajectory.items)
    quality.judge_answer(runtime, trajectory, {"ground_truth": "42"})
    assert quality.verify_sft_semantics(trajectory, runtime.registry).valid
    trajectory.metadata["serc"]["steps"][0].pop()
    assert not quality.verify_sft_semantics(trajectory, runtime.registry).valid


def test_six_repairs_exhaust_before_answer_judge():
    outputs = []
    for number in range(7):
        outputs.extend([[message("<answer>42</answer>")], [verifier(0.1)]])
        if number < 6:
            outputs.append([message('{"action":"NO_CHANGE"}')])
    runtime, client = runtime_for(outputs)
    with pytest.raises(ProtocolError, match="repair_exhausted"):
        quality.run_serc(runtime, initial(), trajectory_id="exhausted")
    assert len(client.requests) == 20


@pytest.mark.parametrize("text", ['{"confidence":0.9}', '{"score":1,"confidence":NaN,"critique":"x","tool_check":true}',
                                   '{"score":1,"confidence":1.1,"critique":"x","tool_check":true}'])
def test_invalid_verifier_fails_closed(text):
    with pytest.raises(ProtocolError, match="invalid_verifier"):
        run([[message("<answer>42</answer>")], [message(text)]])


def test_missing_marker_judge_is_independent_and_proof_required():
    trajectory, runtime, client = run([[message("forty two")], [verifier()],
                                      [message('{"equivalent":true,"reason":"same value"}')]])
    task = {"question": "PRIVATE QUESTION", "ground_truth": "42", "options": {"A": "42", "B": "41"}}
    assert quality.judge_answer(runtime, trajectory, task)
    request = client.requests[-1]
    assert "tools" not in request
    assert request["temperature"] == 0
    assert request["input"][0]["content"] == quality.JUDGE_PROMPT
    payload = json.loads(request["input"][1]["content"])
    assert set(payload) == {"candidate_answer", "reference_answer", "options", "answer_extracted"}
    assert payload["candidate_answer"] == "forty two"
    assert payload["answer_extracted"] is False
    assert "PRIVATE QUESTION" not in json.dumps(request)
    digest = quality.content_hash(trajectory)
    assert not quality.verify_sft_semantics(trajectory, runtime.registry).valid
    assert quality.verify_sft_semantics(trajectory, runtime.registry, {digest}).valid
    trajectory.items[trajectory.metadata["final_solver_index"]]["content"] = "changed"
    assert not quality.verify_sft_semantics(trajectory, runtime.registry, {digest}).valid


@pytest.mark.parametrize("decision", ['{"equivalent":"true","reason":"same"}', '{"equivalent":true}',
                                        '{"equivalent":false,"reason":"unclear"}', 'not json'])
def test_invalid_or_negative_judge_rejected(decision):
    trajectory, runtime, client = run([[message("raw answer")], [verifier()], [message(decision)]])
    with pytest.raises(ProtocolError):
        quality.judge_answer(runtime, trajectory, {"ground_truth": "42"})


def test_strict_matching_does_not_accept_numeric_or_word_substrings():
    judge = quality.StrictAnswerJudge
    assert not judge.is_equivalent("there are 42 cats", "42")
    assert not judge.is_equivalent("not gold", "gold")
    assert judge.is_equivalent("50%", "1/2")
    assert judge.is_equivalent("B", "42", options={"A": "41", "B": "42"})
    assert judge.extract_answer("raw final line") is None


def stream(tmp_path, outputs, tasks=None):
    runtime, client = runtime_for(outputs)
    builder = SFTTrajectoryBuilder()
    tasks = tasks or [{"id": "one", "question": "40+2?", "ground_truth": "42"}]
    output = tmp_path / "stream.jsonl"
    result, stats = run_task_rollouts_concurrent(tasks, runtime, builder, output, concurrency=1, resume=True)
    return result, stats, output, runtime, builder, client, tasks


def test_resume_rolls_back_only_uncommitted_suffix_and_skips_rejected(tmp_path):
    tasks = [{"id": "one", "question": "one", "ground_truth": "42"},
             {"id": "two", "question": "two", "ground_truth": "42"}]
    rows, stats, output, runtime, builder, client, tasks = stream(tmp_path, [[message("<answer>42</answer>")], [verifier()],
                           [message("<answer>41</answer>")], [verifier()],
                           [message('{"equivalent":false,"reason":"wrong"}')]], tasks)
    assert len(rows) == 1
    assert stats.attempted == 2
    committed = output.read_bytes()
    with output.open("ab") as handle:
        handle.write(b'{"uncommitted":')
    requests = len(client.requests)
    result, stats = run_task_rollouts_concurrent(tasks, runtime, builder, output, concurrency=1, resume=True)
    assert len(result) == 1 and stats.attempted == 0
    assert len(client.requests) == requests
    assert output.read_bytes() == committed
    state = json.loads(Path(str(output) + ".state.json").read_text())
    assert state["completed_indices"] == [0, 1]
    assert state["stats"]["attempted"] == 2


def test_resume_refuses_changed_tasks_legacy_state_and_committed_loss(tmp_path):
    rows, stats, output, runtime, builder, client, tasks = stream(tmp_path, [[message("<answer>42</answer>")], [verifier()]])
    changed = [{**tasks[0], "ground_truth": "41"}]
    with pytest.raises(ProtocolError, match="fingerprint"):
        run_task_rollouts_concurrent(changed, runtime, builder, output, resume=True)
    with output.open("r+b") as handle:
        handle.truncate(1)
    with pytest.raises(ProtocolError, match="committed bytes"):
        run_task_rollouts_concurrent(tasks, runtime, builder, output, resume=True)
    Path(str(output) + ".state.json").unlink()
    with pytest.raises(ProtocolError, match="legacy"):
        run_task_rollouts_concurrent(tasks, runtime, builder, output, resume=True)


def test_llm_proof_rechecked_at_resume_and_export(tmp_path):
    rows, stats, output, runtime, builder, client, tasks = stream(tmp_path, [[message("forty two")], [verifier()],
                                             [message('{"equivalent":true,"reason":"same"}')]])
    assert len(rows) == 1
    builder.export(rows, tmp_path / "export", export_format="jsonl")
    builder.answer_judge_hashes.clear()
    with pytest.raises(ProtocolError, match="Export audit"):
        builder.export(rows, tmp_path / "bad", export_format="jsonl")
    state_path = Path(str(output) + ".state.json")
    state = json.loads(state_path.read_text())
    state["answer_judge_accepted_hashes"] = []
    state_path.write_text(json.dumps(state))
    with pytest.raises(ProtocolError, match="resume audit"):
        run_task_rollouts_concurrent(tasks, runtime, builder, output, resume=True)


def test_python_receives_real_paths_and_exports_all_valid_images(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_PRELOAD_PACKAGES", '["pillow"]')
    image_path = tmp_path / "input.png"
    Image.new("RGB", (6, 4), "red").save(image_path)
    root = tmp_path / "outputs"
    context = ToolExecutionContext({"sft_output_root": str(root)}, image=str(image_path))
    result = _python_exec({"code": "assert Image.open(image_path).size == (6,4)\nImage.new('RGB',(2,2)).save(os.path.join(output_dir,'a.png'))\nImage.new('RGB',(3,3)).save(os.path.join(output_dir,'b.png'))"}, context)
    assert result["success"] is True, result
    assert len(result["images"]) == 2
    from agent0_protocol.adapters import ResponsesAdapter
    wire = ResponsesAdapter.function_result({"type": "function_call_output", "call_id": "images", "output": result})
    assert len([p for p in wire["output"] if p["type"] == "input_image"]) == 2
    context.close()
    assert all(Path(path).is_file() for path in result["images"])


def test_sources_preserve_question_reference_pair_and_read_time_cleanup(tmp_path):
    row = {"conversations": [{"from": "human", "value": "first question"},
                              {"from": "gpt", "value": "first answer"},
                              {"from": "human", "value": "second question"},
                              {"from": "gpt", "value": "second answer"}]}
    assert _extract_question(row) == "first question"
    assert _raw_ground_truth(row, "llava_ov_image") == "first answer"
    raw = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.\n**user question:**\nFind x. Show work."
    assert normalize_source_question(raw, "retool") == "Find x. Show work."
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps({"task_id": "x", "question": raw, "images": [], "source_dataset": "retool", "ground_truth": "42"}) + "\n")
    original = source.read_bytes()
    sample = load_source("normalized", source)[0]
    assert sample["question"] == "Find x. Show work."
    assert sample["ground_truth"] == "42"
    assert source.read_bytes() == original


def test_request_timeout_is_not_retried():
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        raise TimeoutError("timeout")
    runtime = ResponsesRuntime(ResponsesConfig("http://localhost:8000/v1", "test", "model", max_retries=3),
                               client=SimpleNamespace(responses=SimpleNamespace(create=create)), probe_on_init=False)
    with pytest.raises(TimeoutError):
        runtime._create(input="test", max_output_tokens=0)
    assert len(requests) == 1
    assert "max_output_tokens" not in requests[0]


def test_actual_process_workers_with_fake_local_responses_server(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    # Exercises spawn, SDK serialization and committed output without a model.
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            last = payload["input"][-1]["content"]
            question = payload["input"][1].get("content", "")
            if isinstance(question, str) and "intentional failure" in question:
                body = b'{"error":{"message":"test rejection","type":"invalid_request_error"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            text = json.dumps({"score": 1, "confidence": 0.9, "critique": "ok", "tool_check": True}) if isinstance(last, str) and last.startswith("Verify semantic trajectory") else "<answer>42</answer>"
            body = json.dumps({"id": "resp_test", "object": "response", "created_at": 0, "model": "test", "status": "completed",
                               "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime = ResponsesRuntime(ResponsesConfig(f"http://127.0.0.1:{server.server_port}/v1", "test", "model"), probe_on_init=False)
    try:
        tasks = [{"question": f"question {i}", "ground_truth": "42"} for i in range(3)]
        tasks.append({"question": "intentional failure", "ground_truth": "42"})
        rows, stats = run_task_rollouts_concurrent(tasks, runtime, SFTTrajectoryBuilder(), tmp_path / "process.jsonl", concurrency=2)
        assert len(rows) == 3, stats.to_dict()
        assert stats.rejected == {"teacher_or_runtime_error: BadRequestError": 1}
    finally:
        runtime.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_image_export_checks_schema_without_claiming_tensor_validation(tmp_path):
    from scripts.build_sft_dataset import verify_sft_dataset_compatibility
    from tools.data_builder.backends.base import image_data_url
    trajectory, runtime, client = run([[message("<answer>42</answer>")], [verifier()]])
    quality.judge_answer(runtime, trajectory, {"ground_truth": "42"})
    trajectory.items[1]["content"] = [{"type": "input_text", "text": "Question"},
                                     {"type": "input_image", "image_url": image_data_url(Image.new("RGB", (2, 2)))}]
    builder = SFTTrajectoryBuilder()
    builder.export([trajectory], tmp_path / "vl", export_format="jsonl")
    check = verify_sft_dataset_compatibility(tmp_path / "vl/train.jsonl")
    assert check["status"] == "schema_passed"
    assert check["image_rows"] == 1
    assert check["tensor_check"] == "requires_real_vl_processor"


def test_verifier_prompt_does_not_recursively_embed_prior_roles(monkeypatch):
    def execute(registry, calls, context):
        return [{"type": "function_call_output", "call_id": call["call_id"],
                 "output": {"success": True, "stdout": "42"}} for call in calls]
    monkeypatch.setattr(quality, "execute_call_batch", execute)
    outputs = []
    for step in range(3):
        outputs += [[{"type": "function_call", "call_id": f"call_{step}", "name": "python_exec", "arguments": '{"code":"print(42)"}'}], [verifier()]]
    outputs += [[message("<answer>42</answer>")], [verifier()]]
    trajectory, runtime, client = run(outputs)
    prompts = [request["input"][-1]["content"] for request in client.requests
               if request["input"][-1].get("role") == "user"
               and isinstance(request["input"][-1].get("content"), str)
               and request["input"][-1]["content"].startswith("Verify semantic trajectory")]
    assert len(prompts) == 4
    assert all(prompt.count("Verify semantic trajectory") == 1 for prompt in prompts)
    assert all("data:image" not in prompt for prompt in prompts)
    assert max(map(len, prompts)) < 3 * len(prompts[0])
    assert len(trajectory.metadata["serc"]["steps"]) == 4
    quality.audit_serc(trajectory)
