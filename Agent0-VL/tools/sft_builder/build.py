"""Build a small, trainable Agent0-VL SFT subset.

The implementation follows the released Agent0-VL runtime at the level needed
for the paper's SFT bridge:

``source -> Solver -> sandbox -> observation -> Solver -> final -> Verifier -> Repair``

Low-confidence steps enter the same-message Repair -> Solver regeneration ->
re-execution -> Verifier flow used by the paper.  A trajectory is exported only
after the repaired step passes the post-repair verifier.  The output contains
only ``messages`` and ``images`` so the official SFT scripts can load it.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dedup import exact_deduplicate
from .prompts import (
    continue_solver_prompt,
    load_stage_solver_prompt,
    regenerate_prompt,
    repair_prompt,
    verifier_prompt,
)
from .sources import SOURCE_STAGES, SourceFormatError, format_user_question, iter_source_samples
from .teacher import OpenAICompatibleTeacher, TeacherError


# Image-analysis snippets must never try to open a GUI from the sandbox.
os.environ.setdefault("MPLBACKEND", "Agg")


Message = Dict[str, str]
CODE_BLOCK_PATTERN = r"```(?:python|py)?[ \t]*\r?\n?(.*?)```"
BOXED_PATTERN = r"\\boxed\{([^}]+)\}"
FINAL_PATTERN = r"FINAL_ANSWER:\s*(.+?)(?:\n|$)"
VERIFICATION_JSON_PATTERN = r'\{[^{}]*"step_index"[^{}]*\}'
REPAIR_JSON_PATTERN = r'\{[^{}]*"action"[^{}]*\}'
REPAIR_THRESHOLD = 0.7
DEFAULT_MAX_REASONING_STEPS = 8
DEFAULT_MAX_OBSERVATION_LENGTH = 512
IMAGE_SANDBOX_LOCK = threading.Lock()


@dataclass
class SolverStep:
    """One Solver assistant response and its immediate sandbox observation."""

    step_index: int
    content: str
    tool_outputs: Optional[str] = None


@dataclass
class Trajectory:
    """Mutable in-memory trajectory before export filtering."""

    sample: Dict[str, Any]
    messages: List[Message]
    solver_steps: List[SolverStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    initial_verification: Optional[Dict[str, Any]] = None
    repair: Optional[Dict[str, Any]] = None
    repair_verification: Optional[Dict[str, Any]] = None
    tool_call_count: int = 0
    successful_tool_call_count: int = 0
    success: bool = False
    failure_reason: Optional[str] = None


@dataclass
class BuildStats:
    """Small CLI summary; no manifest is written."""

    attempted: int = 0
    exported: int = 0
    solver_failures: int = 0
    verifier_failures: int = 0
    reference_failures: int = 0
    unverified_skips: int = 0
    duplicate_skips: int = 0
    quality_failures: int = 0
    failure_reasons: Dict[str, int] = field(default_factory=dict)

    def record_failure(self, reason: str) -> None:
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1


def truncate_content(content: str, max_length: int = 2000) -> str:
    """Match the released rollout's head/tail truncation helper."""

    if len(content) <= max_length:
        return content
    half = max_length // 2
    return content[:half] + f"\n..._Truncated to {max_length} chars_...\n" + content[-half:]


def initial_messages(sample: Mapping[str, Any]) -> List[Message]:
    """Create the only initial message; source answers are not copied."""

    question = str(sample.get("question", ""))
    images = list(sample.get("images", []))
    if not question.strip():
        raise ValueError("Source sample has an empty question")
    content = format_user_question(question, images)
    if content.lower().count("<image>") != len(images):
        raise ValueError("The initial user content and images have different marker counts")
    return [{"role": "user", "content": content}]


def extract_python_blocks(text: str) -> List[str]:
    """Use the exact fenced-Python extraction pattern from the runtime."""

    return [match.strip() for match in re.findall(CODE_BLOCK_PATTERN, text, re.DOTALL) if match.strip()]


def extract_final_answer(text: str) -> Optional[str]:
    """Use the released runtime's ``\\boxed`` / ``FINAL_ANSWER`` parsing."""

    boxed = re.findall(BOXED_PATTERN, text)
    if boxed:
        return boxed[-1].strip()
    final = re.findall(FINAL_PATTERN, text, re.IGNORECASE)
    if final:
        return final[-1].strip()
    return None


def _sandbox_backend():
    """Select the same local/HTTP sandbox backend as the released runtime."""

    if os.getenv("SANDBOX_ENDPOINT"):
        from sandbox.local_sandbox import parallel_sandbox
    else:
        from sandbox.internal_sandbox import parallel_sandbox
    return parallel_sandbox


def execute_python(
    code_blocks: Sequence[str],
    sandbox_timeout: float = 10.0,
    images: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Execute code through the project sandbox, injecting the first local image.

    The released rollout uses a plain subprocess sandbox.  For local image
    samples the builder uses the repository's image-aware sandbox so generated
    code can read the supplied image through the ``image_path`` variable.
    Text-only samples retain the released parallel subprocess path.
    """

    if not code_blocks:
        return []
    first_image = str(images[0]) if images else ""
    if first_image and not first_image.startswith(("data:", "http://", "https://")) and Path(first_image).is_file():
        try:
            from verl.utils.sandbox import execute_code_in_sandbox

            temp_dir = Path(tempfile.mkdtemp(prefix="agent0_sandbox_"))
            results: List[Dict[str, Any]] = []
            execution_context = None
            try:
                # execute_code_in_sandbox uses timeout_decorator's
                # multiprocessing fallback. Serializing this small critical
                # section avoids unsafe fork calls from concurrent builder
                # threads while leaving teacher requests fully concurrent.
                with IMAGE_SANDBOX_LOCK:
                    for index, code in enumerate(code_blocks, 1):
                        processed_paths, stdout, error, execution_context = execute_code_in_sandbox(
                            code_to_execute=code,
                            input_image_path=first_image,
                            item_id=f"sft-image-{index}",
                            temp_output_dir=str(temp_dir),
                            previous_execution_context=execution_context,
                            timeout=max(1, int(sandbox_timeout)),
                        )
                        del processed_paths
                        results.append({
                            "success": error is None,
                            "stdout": truncate_content(str(stdout or ""), DEFAULT_MAX_OBSERVATION_LENGTH),
                            "stderr": truncate_content(str(error or ""), DEFAULT_MAX_OBSERVATION_LENGTH) if error else "",
                        })
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return results
        except Exception as exc:
            return [{"success": False, "stdout": "", "stderr": str(exc)}] * len(code_blocks)

    try:
        parallel_sandbox = _sandbox_backend()
        success_list, stdout_list, stderr_list = asyncio.run(
            parallel_sandbox(
                list(code_blocks),
                num_processes=min(256, len(code_blocks)),
                run_timeout=sandbox_timeout,
            )
        )
    except Exception as exc:
        return [{"success": False, "stdout": "", "stderr": str(exc)}] * len(code_blocks)

    results: List[Dict[str, Any]] = []
    for success, stdout, stderr in zip(success_list, stdout_list, stderr_list):
        results.append({
            "success": bool(success),
            "stdout": truncate_content(str(stdout), DEFAULT_MAX_OBSERVATION_LENGTH),
            "stderr": truncate_content(str(stderr), DEFAULT_MAX_OBSERVATION_LENGTH) if stderr else "",
        })
    while len(results) < len(code_blocks):
        results.append({"success": False, "stdout": "", "stderr": "Sandbox returned no result"})
    return results


def format_observation(results: Sequence[Mapping[str, Any]]) -> str:
    """Format observations exactly like the released rollout."""

    text = "\n[Code Execution Result]\n"
    for result in results:
        stderr = str(result.get("stderr", ""))
        stdout = str(result.get("stdout", ""))
        if stderr:
            text += f"Error: {stderr}\n"
        elif stdout:
            text += f"Output: {stdout}\n"
        else:
            text += "No output\n"
    return text


def parse_verification_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse and clamp the same fields as the released Verifier parser."""

    matches = re.findall(VERIFICATION_JSON_PATTERN, text, re.DOTALL)
    if not matches:
        return None
    try:
        verification = json.loads(matches[-1])
        if not isinstance(verification, dict):
            return None
        if not all(key in verification for key in ("step_index", "score", "confidence")):
            return None
        verification["score"] = max(-1.0, min(1.0, float(verification.get("score", 0))))
        verification["confidence"] = max(0.0, min(1.0, float(verification.get("confidence", 0.5))))
        return verification
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def parse_repair_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse the released ``PATCH`` / ``NO_CHANGE`` repair object."""

    matches = re.findall(REPAIR_JSON_PATTERN, text, re.DOTALL)
    if not matches:
        return None
    try:
        repair = json.loads(matches[-1])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(repair, dict):
        return None
    return repair if repair.get("action") in {"PATCH", "NO_CHANGE"} else None


def _generate_solver_step(
    trajectory: Trajectory,
    teacher: Any,
    solver_system_prompt: str,
    *,
    step_index: int,
    sandbox_timeout: float,
) -> str:
    """Generate one Solver segment and execute its fenced Python, if any.

    Returns ``finished`` when a final answer was emitted, ``continue`` when the
    tool observation needs another Solver turn, and ``failed`` for malformed
    Solver output.  The helper is also used for a corrected segment after a
    PATCH so repaired trajectories retain the real tool evidence.
    """

    images = list(trajectory.sample.get("images", []))
    response = str(teacher.generate(trajectory.messages, images, solver_system_prompt))
    trajectory.messages.append({"role": "assistant", "content": response})
    step = SolverStep(step_index=step_index, content=response)
    trajectory.solver_steps.append(step)

    # Match the released runtime: a final answer takes precedence over tool
    # execution when both occur in one response.
    final_answer = extract_final_answer(response)
    if final_answer is not None:
        trajectory.final_answer = final_answer
        trajectory.success = True
        return "finished"

    code_blocks = extract_python_blocks(response)
    if not code_blocks:
        trajectory.success = False
        trajectory.failure_reason = "no_final_answer_or_python"
        return "failed"

    results = execute_python(code_blocks, sandbox_timeout=sandbox_timeout, images=images)
    trajectory.tool_call_count += len(results)
    trajectory.successful_tool_call_count += sum(1 for result in results if result.get("success"))
    observation = format_observation(results)
    step.tool_outputs = observation
    trajectory.messages.append({"role": "user", "content": observation})

    final_answer = extract_final_answer(observation)
    if final_answer is not None:
        trajectory.final_answer = final_answer
        trajectory.success = True
        return "finished"
    return "continue"


def _append_verifier(
    trajectory: Trajectory,
    teacher: Any,
    solver_system_prompt: str,
) -> Optional[Dict[str, Any]]:
    """Append one verifier turn for the latest Solver segment."""

    if not trajectory.solver_steps:
        trajectory.success = False
        trajectory.failure_reason = "no_solver_step"
        return None
    last_step = trajectory.solver_steps[-1]
    verify_user = verifier_prompt(
        step_content=truncate_content(last_step.content, 2000),
        tool_outputs=truncate_content(last_step.tool_outputs or "None", 1000),
        step_index=last_step.step_index,
    )
    trajectory.messages.append({"role": "user", "content": verify_user})
    verify_text = str(
        teacher.generate(trajectory.messages, list(trajectory.sample.get("images", [])), solver_system_prompt)
    )
    trajectory.messages.append({"role": "assistant", "content": verify_text})
    verification = parse_verification_output(verify_text)
    if verification is None:
        trajectory.success = False
        trajectory.failure_reason = "invalid_verifier_json"
        return None
    trajectory.verification = verification
    return verification


def _repair_and_reverify(
    trajectory: Trajectory,
    teacher: Any,
    solver_system_prompt: str,
    *,
    step_index: int,
    sandbox_timeout: float,
) -> str:
    """Repair a low-confidence step, regenerate it, and verify it again."""

    verification = trajectory.verification or {}
    last_step = trajectory.solver_steps[-1]
    repair_user = repair_prompt(
        original_step=truncate_content(last_step.content, 1000),
        critique=str(verification.get("critique", "Low confidence")),
        score=float(verification.get("score", 0.0)),
        confidence=float(verification.get("confidence", 0.5)),
        step_index=step_index,
    )
    trajectory.messages.append({"role": "user", "content": repair_user})
    repair_text = str(
        teacher.generate(trajectory.messages, list(trajectory.sample.get("images", [])), solver_system_prompt)
    )
    trajectory.messages.append({"role": "assistant", "content": repair_text})
    repair = parse_repair_output(repair_text)
    if repair is None:
        trajectory.success = False
        trajectory.failure_reason = "invalid_repair_json"
        return "failed"
    trajectory.repair = repair
    if repair.get("action") != "PATCH":
        trajectory.success = False
        trajectory.failure_reason = "repair_no_patch"
        return "failed"

    trajectory.messages.append({
        "role": "user",
        "content": regenerate_prompt(json.dumps(repair, ensure_ascii=False)),
    })
    # The old answer must not survive as the answer of the repaired segment.
    trajectory.final_answer = None
    trajectory.success = False
    status = _generate_solver_step(
        trajectory,
        teacher,
        solver_system_prompt,
        step_index=step_index,
        sandbox_timeout=sandbox_timeout,
    )
    if status == "failed":
        return "failed"

    # A corrected Solver segment must have its own verifier result.  A low
    # post-repair confidence is not silently accepted as a clean SFT label.
    post_verification = _append_verifier(trajectory, teacher, solver_system_prompt)
    if post_verification is None:
        return "failed"
    trajectory.repair_verification = post_verification
    if (
        float(post_verification.get("confidence", 0.0)) < REPAIR_THRESHOLD
        or post_verification.get("tool_check") is not True
    ):
        trajectory.success = False
        trajectory.failure_reason = "repair_post_verification_failed"
        return "failed"

    if status == "finished":
        trajectory.success = True
        return "finished"
    # The repaired segment was verified but did not finish the problem; the
    # outer Solver loop can continue with the next normal reasoning step.
    trajectory.success = False
    return "continue"


def rollout_solver(
    sample: Dict[str, Any],
    teacher: Any,
    solver_system_prompt: str,
    *,
    max_reasoning_steps: int = DEFAULT_MAX_REASONING_STEPS,
    sandbox_timeout: float = 10.0,
) -> Trajectory:
    """Run Solver to completion, then append verifier and confidence-gated repair turns."""

    trajectory = Trajectory(sample=sample, messages=initial_messages(sample))
    for step_index in range(1, max_reasoning_steps + 1):
        status = _generate_solver_step(
            trajectory,
            teacher,
            solver_system_prompt,
            step_index=step_index,
            sandbox_timeout=sandbox_timeout,
        )
        if status == "failed":
            return trajectory
        # A code-only Solver turn must receive its tool observation and another
        # Solver turn before verification; injecting Verifier here interrupts
        # the natural tool-use loop and often elicits another JSON verifier turn.
        if status != "finished":
            continue

        verification = _append_verifier(trajectory, teacher, solver_system_prompt)
        if verification is None:
            return trajectory
        if trajectory.initial_verification is None:
            trajectory.initial_verification = verification

        if float(verification.get("confidence", 0.0)) < REPAIR_THRESHOLD:
            repair_status = _repair_and_reverify(
                trajectory,
                teacher,
                solver_system_prompt,
                step_index=step_index,
                sandbox_timeout=sandbox_timeout,
            )
            if repair_status == "failed":
                return trajectory
            if repair_status == "finished" or trajectory.final_answer is not None:
                trajectory.success = True
                return trajectory
            trajectory.messages.append({
                "role": "user",
                "content": continue_solver_prompt(),
            })
            continue

        trajectory.success = True
        return trajectory

    trajectory.success = False
    trajectory.failure_reason = "max_reasoning_steps"
    return trajectory

def _normalized_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Keep normalization intentionally narrow: remove only a complete wrapper,
    # not arbitrary substrings or prose around an answer.
    while True:
        boxed = re.fullmatch(r"\\boxed\{([^{}]*)\}", text)
        if not boxed:
            break
        text = boxed.group(1).strip()
    if len(text) >= 2 and text[0] == "$" and text[-1] == "$":
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).casefold()


def _numeric_decimal(value: str) -> Optional[Decimal]:
    compact = value.replace(",", "") if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(?:\.\d+)?", value) else value
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", compact):
        return None
    try:
        number = Decimal(compact)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def reference_match(
    predicted: Optional[str],
    ground_truth: Optional[str],
    aliases: Optional[Sequence[str]] = None,
) -> Optional[bool]:
    """Return exact/numeric match, or ``None`` when no reliable reference exists."""

    pred = _normalized_answer(predicted)
    if pred is None:
        return None
    pred_number = _numeric_decimal(pred)
    references: List[Any] = [ground_truth]
    if aliases:
        references.extend(aliases)
    saw_reference = False
    for reference in references:
        truth = _normalized_answer(reference)
        if truth is None:
            continue
        saw_reference = True
        if pred == truth:
            return True
        truth_number = _numeric_decimal(truth)
        if pred_number is not None and truth_number is not None and pred_number == truth_number:
            return True
    if not saw_reference:
        return None
    return False


def _export_record(trajectory: Trajectory) -> Dict[str, Any]:
    if not trajectory.success or trajectory.final_answer is None:
        raise ValueError("Cannot export an unsuccessful trajectory")
    if any(message.get("role") == "system" for message in trajectory.messages):
        raise ValueError("System prompts must be supplied by the official SFT scripts, not data rows")
    images = list(trajectory.sample.get("images", []))
    initial = trajectory.messages[0]
    if initial.get("role") != "user" or str(initial.get("content", "")).count("<image>") != len(images):
        raise ValueError("Exported image marker count does not match images")
    return {"messages": trajectory.messages, "images": images}


def _run_one_sample(
    sample: Dict[str, Any],
    teacher: Any,
    solver_system_prompt: str,
    *,
    max_reasoning_steps: int,
    sandbox_timeout: float,
) -> Tuple[Trajectory, Optional[str]]:
    """Run one complete sample while preserving its internal step order.

    The returned phase lets the caller keep the same solver/verifier failure
    accounting as the sequential implementation.
    """

    try:
        trajectory = rollout_solver(
            sample,
            teacher,
            solver_system_prompt,
            max_reasoning_steps=max_reasoning_steps,
            sandbox_timeout=sandbox_timeout,
        )
    except Exception as exc:
        trajectory = Trajectory(sample=sample, messages=[])
        trajectory.failure_reason = (
            "teacher_request_error" if isinstance(exc, TeacherError) else
            f"sample_exception_{type(exc).__name__.lower()}"
        )
        return trajectory, "solver"
    if not trajectory.success:
        failure_reason = trajectory.failure_reason or "solver_failure"
        verifier_reason = (
            failure_reason.startswith("invalid_verifier")
            or failure_reason.startswith("invalid_repair")
            or failure_reason.startswith("repair_")
        )
        return trajectory, "verifier" if verifier_reason else "solver"

    return trajectory, None


def _quality_failure_reason(
    sample: Mapping[str, Any],
    trajectory: Trajectory,
    quality_profile: str,
) -> Optional[str]:
    """Apply stage-specific quality gates after rollout and verification."""

    if quality_profile == "basic":
        return None
    if quality_profile not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported quality profile: {quality_profile}")
    expected_stage = 1 if quality_profile == "stage1" else 2
    if int(sample.get("stage", -1)) != expected_stage:
        return "quality_wrong_stage"

    images = list(sample.get("images", []))
    if quality_profile == "stage1":
        if not images:
            return "quality_missing_image"
        if any(
            not str(image).startswith(("data:", "http://", "https://"))
            and not Path(str(image)).is_file()
            for image in images
        ):
            return "quality_missing_image_file"

    if trajectory.tool_call_count <= 0:
        return "quality_missing_tool_call"
    if trajectory.repair is not None and trajectory.repair_verification is None:
        return "quality_repair_not_reverified"
    if trajectory.successful_tool_call_count != trajectory.tool_call_count:
        return "quality_tool_execution_error"

    verification = trajectory.verification or {}
    try:
        confidence = float(verification.get("confidence", 0.0))
    except (TypeError, ValueError):
        return "quality_invalid_confidence"
    if confidence < REPAIR_THRESHOLD:
        return "quality_low_verifier_confidence"
    if verification.get("tool_check") is not True:
        return "quality_verifier_tool_check_false"
    return None


def build_records(
    samples: Iterable[Dict[str, Any]],
    teacher: Any,
    solver_system_prompt: str,
    *,
    max_tasks: int,
    max_reasoning_steps: int = DEFAULT_MAX_REASONING_STEPS,
    sandbox_timeout: float = 10.0,
    keep_unverified: bool = False,
    concurrency: int = 1,
    quality_profile: str = "basic",
    batch_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], BuildStats]:
    """Roll out, bridge, filter, and exact-deduplicate a source subset.

    ``concurrency`` parallelizes independent samples only.  The Solver,
    sandbox, Verifier, and optional Repair sequence for each sample remains
    strictly ordered.
    """

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if batch_size is None:
        batch_size = max(concurrency, concurrency * 4)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stats = BuildStats()
    candidates: List[Dict[str, Any]] = []
    sample_iter = iter(samples)
    remaining = max_tasks

    def process_outcome(sample: Dict[str, Any], outcome: Tuple[Trajectory, Optional[str]]) -> None:
        trajectory, failure_phase = outcome
        stats.attempted += 1
        if not trajectory.success:
            if failure_phase == "verifier":
                stats.verifier_failures += 1
            else:
                stats.solver_failures += 1
            stats.record_failure(trajectory.failure_reason or "solver_failure")
            return

        quality_failure = _quality_failure_reason(sample, trajectory, quality_profile)
        if quality_failure is not None:
            stats.quality_failures += 1
            stats.record_failure(quality_failure)
            return

        match = reference_match(
            trajectory.final_answer,
            sample.get("ground_truth"),
            sample.get("ground_truth_aliases"),
        )
        if match is False:
            stats.reference_failures += 1
            stats.record_failure("reference_mismatch")
            return
        if match is None and not keep_unverified:
            stats.unverified_skips += 1
            stats.record_failure("unverified_reference")
            return
        candidates.append(_export_record(trajectory))

    executor = None
    if concurrency > 1:
        executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        while remaining > 0:
            selected_samples = list(itertools.islice(sample_iter, min(batch_size, remaining)))
            if not selected_samples:
                break
            if executor is None:
                outcomes = [
                    _run_one_sample(
                        sample,
                        teacher,
                        solver_system_prompt,
                        max_reasoning_steps=max_reasoning_steps,
                        sandbox_timeout=sandbox_timeout,
                    )
                    for sample in selected_samples
                ]
            else:
                outcomes = list(
                    executor.map(
                        lambda sample: _run_one_sample(
                            sample,
                            teacher,
                            solver_system_prompt,
                            max_reasoning_steps=max_reasoning_steps,
                            sandbox_timeout=sandbox_timeout,
                        ),
                        selected_samples,
                    )
                )
            for sample, outcome in zip(selected_samples, outcomes):
                process_outcome(sample, outcome)
            remaining -= len(selected_samples)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    records = exact_deduplicate(candidates)
    stats.duplicate_skips = len(candidates) - len(records)
    stats.exported = len(records)
    return records, stats


def write_jsonl(records: Iterable[Dict[str, Any]], output: str | Path) -> None:
    """Write ms-swift-compatible one-row JSON objects."""

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_STAGES))
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-base-url", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-max-tokens", type=int, default=2048)
    parser.add_argument("--teacher-timeout", type=float, default=120.0)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--max-tasks", required=True, type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run independent samples concurrently; each sample's Solver/tool/Verifier loop remains sequential",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum number of samples held in one concurrent batch (default: 4 * concurrency)",
    )
    parser.add_argument("--max-reasoning-steps", type=int, default=DEFAULT_MAX_REASONING_STEPS)
    parser.add_argument("--sandbox-timeout", type=float, default=10.0)
    parser.add_argument(
        "--source-split",
        choices=("train", "dev", "test", "all"),
        default="train",
        help="Dataset split for sources that publish split manifests, such as GeoQA",
    )
    parser.add_argument(
        "--quality-profile",
        choices=("basic", "stage1", "stage2"),
        default="basic",
        help="Stage-specific gates requiring successful tool execution and tool_check=true",
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--keep-unverified",
        action="store_true",
        help="Export rows whose source has no reliable reference answer",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.max_tasks <= 0 or args.max_reasoning_steps <= 0 or args.concurrency <= 0:
        raise SystemExit("--max-tasks, --max-reasoning-steps, and --concurrency must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    try:
        sample_iter = iter_source_samples(
            args.source,
            args.source_path,
            stage=args.stage,
            source_split=args.source_split,
        )
        first = next(sample_iter)
    except (StopIteration, SourceFormatError) as exc:
        raise SystemExit(f"Could not read a source sample: {exc}") from exc

    print(json.dumps({
        "question": first["question"],
        "images": first["images"],
        "ground_truth": first["ground_truth"],
        "stage": first["stage"],
    }, ensure_ascii=False, indent=2))

    teacher = OpenAICompatibleTeacher(
        args.teacher_base_url,
        args.teacher_model,
        api_key_env=args.teacher_api_key_env,
        temperature=args.teacher_temperature,
        top_p=args.teacher_top_p,
        max_tokens=args.teacher_max_tokens,
        timeout=args.teacher_timeout,
        retries=args.teacher_retries,
    )
    solver_prompt = load_stage_solver_prompt(args.repo_root, args.stage)
    records, stats = build_records(
        itertools.chain([first], sample_iter),
        teacher,
        solver_prompt,
        max_tasks=args.max_tasks,
        concurrency=args.concurrency,
        max_reasoning_steps=args.max_reasoning_steps,
        sandbox_timeout=args.sandbox_timeout,
        keep_unverified=args.keep_unverified,
        quality_profile=args.quality_profile,
        batch_size=args.batch_size,
    )
    write_jsonl(records, args.output)
    print(json.dumps({
        "output": str(args.output),
        "attempted": stats.attempted,
        "exported": stats.exported,
        "solver_failures": stats.solver_failures,
        "verifier_failures": stats.verifier_failures,
        "reference_failures": stats.reference_failures,
        "unverified_skips": stats.unverified_skips,
        "duplicate_skips": stats.duplicate_skips,
        "quality_failures": stats.quality_failures,
        "failure_reasons": stats.failure_reasons,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TeacherError as exc:
        raise SystemExit(f"Teacher error: {exc}") from exc
