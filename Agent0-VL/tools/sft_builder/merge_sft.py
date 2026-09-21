"""Merge per-source SFT outputs and perform a final strict audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .build import extract_final_answer, parse_repair_output, parse_verification_output
from .dedup import record_hash


def _last_verification(messages: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        parsed = parse_verification_output(str(message.get("content", "")))
        if parsed is not None:
            return parsed
    return None


def _audit_repair_flow(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Require a complete Repair -> regeneration -> post-Verifier sequence."""

    repair_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and "Now switch to the Self-Repair role." in message.get("content", "")
    ]
    for repair_index in repair_positions:
        if repair_index + 1 >= len(messages) or messages[repair_index + 1].get("role") != "assistant":
            return "missing_repair_response"
        repair = parse_repair_output(messages[repair_index + 1].get("content", ""))
        if repair is None:
            return "invalid_repair_json"
        if repair.get("action") != "PATCH":
            return "repair_without_patch"

        regenerate_index = next(
            (index for index in range(repair_index + 2, len(messages))
             if messages[index].get("role") == "user"
             and "Apply the repair instruction" in messages[index].get("content", "")),
            None,
        )
        if regenerate_index is None:
            return "missing_repair_regeneration_prompt"
        regenerated_assistant = next(
            (message for message in messages[regenerate_index + 1:]
             if message.get("role") == "assistant"),
            None,
        )
        if regenerated_assistant is None:
            return "missing_repaired_solver_response"
        regenerated_content = regenerated_assistant.get("content", "")
        if "```" not in regenerated_content and extract_final_answer(regenerated_content) is None:
            return "invalid_repaired_solver_response"
        if not any(
            message.get("role") == "user"
            and "Now switch to the Verifier role." in message.get("content", "")
            for message in messages[regenerate_index + 1:]
        ):
            return "missing_post_repair_verifier"
    return None


def audit_record(record: Dict[str, Any], stage: int) -> Optional[str]:
    if set(record) != {"messages", "images"}:
        return "top_level_schema"
    messages = record.get("messages")
    images = record.get("images")
    if not isinstance(messages, list) or not isinstance(images, list) or len(messages) < 4:
        return "message_or_image_schema"
    if any(
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or not isinstance(message["role"], str)
        or not isinstance(message["content"], str)
        or not message["content"]
        for message in messages
    ):
        return "message_schema"
    if any(message["role"] == "system" for message in messages):
        return "system_message_present"
    if any(
        ord(char) < 32 and char not in "\n\t\r"
        for message in messages
        for char in message["content"]
    ):
        return "control_character"
    if messages[0]["role"] != "user":
        return "initial_message_not_user"
    if messages[0]["content"].lower().count("<image>") != len(images):
        return "image_marker_mismatch"
    if stage == 1:
        if not images:
            return "stage1_missing_image"
        if any(
            not str(image).startswith(("data:", "http://", "https://"))
            and not Path(str(image)).is_file()
            for image in images
        ):
            return "stage1_missing_image_file"
    repair_failure = _audit_repair_flow(messages)
    if repair_failure is not None:
        return repair_failure
    solver_answers = [
        extract_final_answer(message["content"])
        for message in messages
        if message["role"] == "assistant"
    ]
    if not any(answer is not None for answer in solver_answers):
        return "missing_final_answer"
    if not any("```" in message["content"] for message in messages if message["role"] == "assistant"):
        return "missing_code_block"
    if not any("[Code Execution Result]" in message["content"] for message in messages if message["role"] == "user"):
        return "missing_tool_observation"
    verification = _last_verification(messages)
    if verification is None:
        return "invalid_verifier_json"
    try:
        if float(verification.get("confidence", 0.0)) < 0.7:
            return "low_verifier_confidence"
    except (TypeError, ValueError):
        return "invalid_verifier_confidence"
    if verification.get("tool_check") is not True:
        return "verifier_tool_check_false"
    observations = [
        message["content"]
        for message in messages
        if message["role"] == "user" and "[Code Execution Result]" in message["content"]
    ]
    if any("Error:" in observation for observation in observations):
        return "tool_observation_error"
    return None


def _read_rows(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                yield path, row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    accepted: List[Dict[str, Any]] = []
    seen: set[str] = set()
    rejected: Dict[str, int] = {}
    input_rows = 0
    duplicate_rows = 0
    for source_path, row in _read_rows(args.input):
        input_rows += 1
        reason = audit_record(row, args.stage)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        digest = record_hash(row)
        if digest in seen:
            duplicate_rows += 1
            continue
        seen.add(digest)
        accepted.append(row)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "stage": args.stage,
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
        "input_rows": input_rows,
        "accepted_rows": len(accepted),
        "duplicate_rows": duplicate_rows,
        "rejected": rejected,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
