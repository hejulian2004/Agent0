"""Continuous SFT scheduling with fingerprinted, committed-output recovery."""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
import multiprocessing
from dataclasses import asdict
from pathlib import Path

from agent0_protocol.schema import CanonicalTrajectory, ProtocolError
from tools.data_builder.backends.base import image_data_url
from tools.data_builder.sources import normalize_source_question, normalize_prepared_reference
from tools.data_builder.sft_quality import (
    FLOW_VERSION, content_hash, run_serc, judge_answer, verify_sft_semantics,
)
from agent0_protocol.local_prompts import render_solver_request, render_system_prompt


def _serializable(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tobytes"):
        return {"pixels": hashlib.sha256(value.tobytes()).hexdigest(), "size": value.size, "mode": value.mode}
    raise TypeError(f"Unsupported task value: {type(value).__name__}")


def sample_hash(task):
    task = _prepare(task)
    images = task.get("images")
    if images is None:
        image = task.get("image") or task.get("image_path") or task.get("image_pil")
        images = [] if image is None else [image]
    payload = {"question": " ".join(task["question"].casefold().split()),
               "images": [hashlib.sha256(image_data_url(image).encode()).hexdigest() for image in images]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _run_task_in_process(task, index, config, min_steps):
    from agent0_protocol.responses_runtime import ResponsesRuntime
    from scripts.build_sft_dataset import SFTTrajectoryBuilder
    runtime = ResponsesRuntime(config, probe_on_init=False)
    try:
        return _run_task(task, index, runtime, SFTTrajectoryBuilder(), min_steps)
    except ProtocolError:
        raise
    except Exception as exc:
        # SDK exceptions carry HTTP objects and cannot safely cross a process
        # boundary. Return a serializable, credential-free failure reason.
        error = ProtocolError("teacher_or_runtime_error: " + type(exc).__name__)
        error.answer_judge_called = getattr(exc, "answer_judge_called", False)
        raise error from None
    finally:
        runtime.client.close()


@contextmanager
def _executor(runtime, concurrency):
    if getattr(runtime, "_injected_client", False):
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            yield executor, False
        return
    executor = ProcessPoolExecutor(max_workers=concurrency, mp_context=multiprocessing.get_context("spawn"))
    try:
        yield executor, True
    except BaseException:
        # Only these worker processes belong to this builder. Close their HTTP
        # connections without touching any teacher service.
        workers = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1)
        raise
    else:
        executor.shutdown(wait=True)


def fingerprint(tasks, runtime, min_steps, verify_semantics):
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "agent0_protocol").glob("*.py")),
             *sorted((root / "sandbox").glob("*.py")),
             *sorted((root / "tools/data_builder").glob("sft_*.py")),
             root / "tools/data_builder/sources.py", root / "scripts/build_sft_dataset.py",
             root / "verl/prompts/agent0_templates.py"]
    if runtime.config.checkpointed_settings:
        paths.extend([root / 'tools/data_builder/checkpointed_quality.py',
                      root / 'tools/checkpointed_sft.py',
                      root / 'tools/canonical_multimodal.py'])
    code = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    config = asdict(runtime.config)
    config.pop("api_key")
    images = {}
    for task in tasks:
        values = task.get("images") or [task.get("image") or task.get("image_path") or task.get("image_pil")]
        for image in values:
            if isinstance(image, (str, Path)) and not str(image).startswith(("data:", "http://", "https://")):
                path = Path(image)
                if path.is_file():
                    images[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    environment = {name: os.environ.get(name) for name in (
        "SANDBOX_BACKEND", "SANDBOX_ENDPOINT", "SANDBOX_RUN_TIMEOUT", "SANDBOX_CPU_TIMEOUT",
        "SANDBOX_MEM_LIMIT_MB", "SANDBOX_MAX_OUTPUT_BYTES", "SANDBOX_PRELOAD_PACKAGES",
        "AGENT0_INTERMEDIATE_IMAGE_DIR", "AGENT0_DETECTOR_WEIGHTS", "AGENT0_RETRIEVAL_DIR")}
    payload = {"environment": environment, "tasks": tasks, "images": images, "config": config, "tools": runtime.registry.definitions(),
               "prompt": render_system_prompt(), "flow": FLOW_VERSION, "code": code,
               "min_steps": min_steps, "verify_semantics": verify_semantics}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=_serializable).encode()).hexdigest()


def _write_state(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare(task):
    task = dict(task)
    task["question"] = normalize_source_question(str(task.get("question", task.get("prompt", ""))),
                                                  task.get("source_dataset", task.get("data_source", "")))
    task = normalize_prepared_reference(task, task.get("source_dataset", task.get("data_source", "")))
    return task


def _run_task(task, index, runtime, builder, min_steps):
    task = _prepare(task)
    reference = task.get("ground_truth", task.get("answer"))
    if reference is None or not str(reference).strip():
        raise ProtocolError("missing_reference")
    question = task["question"]
    images = task.get("images")
    if images is None:
        image = task.get("image") or task.get("image_path") or task.get("image_pil")
        images = [] if image is None else [image]
    text = render_solver_request(question)
    if images:
        text += "\nInput image paths: " + json.dumps([str(image) for image in images if isinstance(image, (str, Path))], ensure_ascii=False)
    content = [{"type": "input_text", "text": text},
               *[{"type": "input_image", "image_url": image_data_url(image)} for image in images]] if images else text
    initial = [
        {"type": "message", "role": "system", "content": render_system_prompt()},
        {"type": "message", "role": "user", "content": content}]
    metadata = {"sample_id": str(task.get("id", task.get("task_id", task.get("sample_id", index)))),
                "source": task.get("source_dataset", task.get("data_source", "task_rollout")),
                "sample_hash": sample_hash(task), "candidate_index": index, "question": question}
    if runtime.config.checkpointed_settings:
        from tools.data_builder.checkpointed_quality import construct
        trajectory = construct(runtime, initial, task, metadata)
    else:
        trajectory = run_serc(runtime, initial,
        trajectory_id=f"sft_{task.get('source_dataset', 'task')}_{index}", metadata={"sample_id": str(task.get("id", task.get("task_id", task.get("sample_id", index)))),
                                               "source": task.get("source_dataset", task.get("data_source", "task_rollout")),
                                               "sample_hash": sample_hash(task), "candidate_index": index, "question": question})
    ok, reason = builder_audit(builder, trajectory, min_steps)
    if not ok:
        raise ProtocolError(reason)
    try:
        llm_used = judge_answer(runtime, trajectory, task)
    except Exception as exc:
        exc.answer_judge_called = trajectory.metadata.get("answer_judge_called", False)
        raise
    result = verify_sft_semantics(trajectory, builder.registry, {content_hash(trajectory)} if llm_used else set())
    if not result.valid:
        raise ProtocolError("; ".join(result.issues))
    return trajectory, llm_used


def builder_audit(builder, trajectory, min_steps):
    if 'checkpointed_flow' in trajectory.metadata:
        from tools.data_builder.checkpointed_quality import audit_flow
        try:
            trajectory.validate()
            audit_flow(trajectory.metadata['checkpointed_flow'])
            return True, ''
        except (ValueError, KeyError, TypeError) as exc:
            return False, str(exc)
    from scripts.build_sft_dataset import FormatCleaner
    return FormatCleaner.audit_trajectory(trajectory, min_steps=min_steps)


class _QuotaReached(BaseException):
    pass


def run_stream(tasks, runtime, builder, output_jsonl, *, concurrency=8, resume=True, verify_semantics=True, min_steps=3, target_exported=None):
    from scripts.build_sft_dataset import BuildStats
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    # Lock the whole session: two builders must not append to the same stream.
    import fcntl
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output_jsonl) + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("SFT output is owned by another active builder") from exc
        return _run_locked(tasks, runtime, builder, output_jsonl, concurrency, resume, verify_semantics, min_steps, BuildStats(), target_exported)


def _run_locked(tasks, runtime, builder, output, concurrency, resume, verify_semantics, min_steps, stats, target_exported=None):
    state_path = Path(str(output) + ".state.json")
    expected = hashlib.sha256((fingerprint(tasks, runtime, min_steps, verify_semantics) + str(target_exported)).encode()).hexdigest()
    if target_exported is not None and target_exported < 1:
        raise ValueError("target_exported must be positive")
    trajectories = []
    seen = set()
    seen_samples = set()
    if state_path.exists() or (output.exists() and output.stat().st_size):
        if not resume:
            raise ProtocolError("Existing SFT output: choose a new output directory or --resume")
        if not state_path.exists():
            raise ProtocolError("Missing fingerprinted state; refusing unsafe legacy resume")
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != expected:
            raise ProtocolError("SFT generation fingerprint mismatch; use a new output directory")
        committed = state["committed_output_bytes"]
        if not output.exists() or output.stat().st_size < committed:
            raise ProtocolError("SFT output is missing committed bytes")
        # Validate committed data before rolling back an uncommitted suffix.
        hashes = set(state.get("accepted_hashes", []))
        proofs = set(state.get("answer_judge_accepted_hashes", []))
        for line in output.read_bytes()[:committed].decode().splitlines():
            record = json.loads(line)
            trajectory = CanonicalTrajectory.from_dict(record["trajectory"])
            result = verify_sft_semantics(trajectory, builder.registry, proofs)
            digest = content_hash(trajectory)
            if not result.valid or digest not in hashes or digest in seen:
                raise ProtocolError("Committed trajectory failed resume audit")
            if trajectory.metadata["answer_validation"]["method"] == "llm" and digest not in proofs:
                raise ProtocolError("Missing committed answer-judge proof")
            ok, reason = builder_audit(builder, trajectory, min_steps)
            if not ok:
                raise ProtocolError(reason)
            trajectories.append(trajectory)
            seen.add(digest)
            sample_digest = trajectory.metadata["sample_hash"]
            if sample_digest in seen_samples:
                raise ProtocolError("Duplicate committed sample")
            seen_samples.add(sample_digest)
        if seen != hashes or len(trajectories) != state["exported_rows"]:
            raise ProtocolError("SFT committed row count/hash mismatch")
        if output.stat().st_size > committed:
            with output.open("r+b") as handle:
                handle.truncate(committed)
                handle.flush()
                os.fsync(handle.fileno())
    else:
        output.touch()
        state = {"fingerprint": expected, "committed_output_bytes": 0, "completed_indices": [],
                 "exported_rows": 0, "accepted_hashes": [], "answer_judge_accepted_hashes": [],
                 "rejected": {}, "stats": {}}
        _write_state(state_path, state)
    previous_stats = dict(state.get("stats", {}))
    completed = set(state["completed_indices"])
    if any(type(index) is not int or not 0 <= index < len(tasks) for index in completed):
        raise ProtocolError("Invalid completed candidate indices")
    pending = iter((index, task) for index, task in enumerate(tasks) if index not in completed)
    # Only submit a bounded number of futures; flush and refill immediately.
    try:
        if target_exported is not None and len(trajectories) >= target_exported:
            raise _QuotaReached()
        with output.open("a", encoding="utf-8") as handle, _executor(runtime, concurrency) as (executor, use_processes):
            futures = {}
            def refill():
                while len(futures) < concurrency:
                    try:
                        index, task = next(pending)
                    except StopIteration:
                        break
                    stats.attempted += 1
                    if use_processes:
                        future = executor.submit(_run_task_in_process, task, index, runtime.config, min_steps)
                    else:
                        future = executor.submit(_run_task, task, index, runtime, builder, min_steps)
                    futures[future] = index
            refill()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    trajectory = None
                    try:
                        trajectory, llm_used = future.result()
                        digest = content_hash(trajectory)
                        sample_digest = trajectory.metadata["sample_hash"]
                        if digest in seen or sample_digest in seen_samples:
                            raise ProtocolError("duplicate_trajectory")
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:
                        reason = str(exc) if isinstance(exc, ProtocolError) else type(exc).__name__
                        if getattr(exc, "answer_judge_called", False):
                            stats.answer_judge_calls += 1
                            stats.answer_judge_errors += int(reason != "answer_mismatch")
                        stats.rejected[reason] = stats.rejected.get(reason, 0) + 1
                        state["rejected"][str(index)] = reason
                        trajectory = None
                    if trajectory is not None:
                        calls = [item for item in trajectory.items if item["type"] == "function_call"]
                        results = [item for item in trajectory.items if item["type"] == "function_call_output"]
                        record = {"trajectory_id": trajectory.trajectory_id, "data_source": trajectory.metadata["source"],
                                  "trajectory": trajectory.to_dict(), "num_items": len(trajectory.items),
                                  "num_tool_calls": len(calls), "tools_used": sorted({call["name"] for call in calls})}
                        # I/O failures must abort; never mark a partial write rejected
                        # or advance state past bytes that were not durably written.
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        stats.exported += 1
                        stats.tool_calls += len(calls)
                        stats.successful_tool_calls += sum(result["output"].get("success") is True for result in results)
                        stats.answer_judge_calls += int(llm_used)
                        stats.answer_judge_passes += 1
                        trajectories.append(trajectory)
                        seen.add(digest)
                        seen_samples.add(sample_digest)
                        state["accepted_hashes"].append(digest)
                        if llm_used:
                            state["answer_judge_accepted_hashes"].append(digest)
                    completed.add(index)
                    state.update(completed_indices=sorted(completed), exported_rows=len(trajectories),
                                 committed_output_bytes=output.stat().st_size)
                    current_stats = stats.to_dict()
                    cumulative = dict(current_stats)
                    for key in ("attempted", "exported", "tool_calls", "successful_tool_calls", "answer_judge_calls", "answer_judge_passes", "answer_judge_errors"):
                        cumulative[key] += previous_stats.get(key, 0)
                    cumulative["rejected_breakdown"] = dict(previous_stats.get("rejected_breakdown", {}))
                    for reason, count in stats.rejected.items():
                        cumulative["rejected_breakdown"][reason] = cumulative["rejected_breakdown"].get(reason, 0) + count
                    cumulative["export_rate"] = round(cumulative["exported"] / max(1, cumulative["attempted"]), 4)
                    cumulative["tool_success_rate"] = round(cumulative["successful_tool_calls"] / max(1, cumulative["tool_calls"]), 4)
                    state["stats"] = cumulative
                    state["target_exported"] = target_exported
                    state["target_reached"] = target_exported is not None and len(trajectories) >= target_exported
                    _write_state(state_path, state)
                    if state["target_reached"]:
                        raise _QuotaReached()
                refill()
    except _QuotaReached:
        pass
    builder.answer_judge_hashes = set(state["answer_judge_accepted_hashes"])
    return trajectories, stats
