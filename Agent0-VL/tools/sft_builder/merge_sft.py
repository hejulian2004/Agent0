"""Merge per-source SFT outputs and perform a final strict audit."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from .build import extract_python_blocks, extract_final_answer, parse_repair_output, parse_verification_output
from .dedup import record_hash


def _solver_messages(messages):
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        previous = messages[index - 1].get("content", "") if index else ""
        if previous.startswith("Now switch to the Verifier role.") or previous.startswith("Now switch to the Self-Repair role."):
            continue
        yield index, message


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
             and "A repair instruction has been issued for the previous step:" in messages[index].get("content", "")),
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
        if not regenerated_content.strip():
            return "invalid_repaired_solver_response"
        verifier_index = next(
            (
                index
                for index in range(regenerate_index + 1, len(messages))
                if messages[index].get("role") == "user"
                and "Now switch to the Verifier role." in messages[index].get("content", "")
            ),
            None,
        )
        if verifier_index is None:
            return "missing_post_repair_verifier"
        if verifier_index + 1 >= len(messages) or messages[verifier_index + 1].get("role") != "assistant":
            return "missing_post_repair_verifier_response"
        post_verification = parse_verification_output(messages[verifier_index + 1].get("content", ""))
        if post_verification is None:
            return "invalid_post_repair_verifier_json"
        try:
            if float(post_verification.get("confidence", 0.0)) < 0.7:
                return "low_post_repair_verifier_confidence"
        except (TypeError, ValueError):
            return "invalid_post_repair_verifier_confidence"
        repaired_has_tool = any("[Code Execution Result]" in message.get("content", "")
                                for message in messages[regenerate_index + 1:verifier_index])
        if repaired_has_tool and post_verification.get("tool_check") is not True:
            return "post_repair_verifier_tool_check_false"
    return None


def _image_reference_is_valid(image: str) -> bool:
    if image.startswith("data:"):
        try:
            header, payload = image.split(",", 1)
            if ";base64" not in header:
                return False
            base64.b64decode(payload, validate=True)
            return True
        except (ValueError, binascii.Error):
            return False
    if image.startswith(("http://", "https://")):
        parsed = urlparse(image)
        return bool(parsed.netloc and parsed.path)
    path = Path(image)
    if not path.is_file():
        return False
    try:
        with Image.open(path) as opened:
            opened.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def audit_record(
    record: Any,
    stage: int,
    *,
    allow_stage2_images: bool = False,
) -> Optional[str]:
    if not isinstance(record, dict):
        return "top_level_schema"
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
    if any(message["role"] not in {"user", "assistant"} for message in messages):
        return "invalid_message_role"
    if any(
        ord(char) < 32 and char not in "\n\t\r"
        for message in messages
        for char in message["content"]
    ):
        return "control_character"
    if messages[0]["role"] != "user":
        return "initial_message_not_user"
    for previous, current in zip(messages, messages[1:]):
        if previous["role"] != current["role"]:
            continue
        # A code-only Solver step is followed by its tool observation. The
        # Verifier prompt may then immediately follow that observation.
        if (
            previous["role"] == "user"
            and "[Code Execution Result]" in previous["content"]
            and "Now switch to the Verifier role." in current["content"]
        ):
            continue
        return "message_role_order"
    if messages[-1]["role"] != "assistant":
        return "final_message_not_assistant"
    if any(not isinstance(image, str) or not image for image in images):
        return "image_schema"
    if messages[0]["content"].lower().count("<image>") != len(images):
        return "image_marker_mismatch"
    if stage == 1:
        if not images:
            return "stage1_missing_image"
        if any(not _image_reference_is_valid(image) for image in images):
            return "stage1_invalid_image"
    elif images:
        if not allow_stage2_images:
            return "stage2_unexpected_image"
        if any(not _image_reference_is_valid(image) for image in images):
            return "stage2_invalid_image"
    repair_failure = _audit_repair_flow(messages)
    if repair_failure is not None:
        return repair_failure
    solver_messages = list(_solver_messages(messages))
    if not any(extract_final_answer(message["content"]) is not None for _, message in solver_messages):
        return "missing_final_answer"
    has_code = False
    for index, message in solver_messages:
        if not extract_python_blocks(message["content"]):
            continue
        has_code = True
        if index + 1 >= len(messages) or messages[index + 1]["role"] != "user" or "[Code Execution Result]" not in messages[index + 1]["content"]:
            return "missing_tool_observation"
    has_observation = any("[Code Execution Result]" in message["content"] for message in messages if message["role"] == "user")
    if has_observation and not has_code:
        return "missing_code_block"
    verification = _last_verification(messages)
    if verification is None:
        return "invalid_verifier_json"
    try:
        if float(verification.get("confidence", 0.0)) < 0.7:
            return "low_verifier_confidence"
    except (TypeError, ValueError):
        return "invalid_verifier_confidence"
    last_solver_index = solver_messages[-1][0]
    latest_has_tool = any("[Code Execution Result]" in message["content"]
                          for message in messages[last_solver_index + 1:] if message["role"] == "user")
    if latest_has_tool and verification.get("tool_check") is not True:
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
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield path, row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--allow-stage2-images",
        action="store_true",
        help="Allow and validate local/URL images in multimodal Stage-2 rows",
    )
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
        reason = audit_record(
            row,
            args.stage,
            allow_stage2_images=args.allow_stage2_images,
        )
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
