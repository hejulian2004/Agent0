from __future__ import annotations

import pytest

from tools.local_sft_builder.protocol import (
    ProtocolError,
    extract_python_blocks,
    format_code_execution_observation,
    parse_repair_json,
    parse_verifier_json,
    reject_json_tool_call,
    validate_solver_final,
)
from tools.local_sft_builder.runtime import SourceRuntimeAdapter


def test_source_solver_protocol_extracts_fenced_python_only() -> None:
    text = "<think>calc</think>\n```python\nprint(2 + 2)\n```"
    assert extract_python_blocks(text) == ("print(2 + 2)\n",)
    with pytest.raises(ProtocolError):
        reject_json_tool_call('{"tool_name":"PythonExec","tool_input":{"code":"print(4)"}}')
    with pytest.raises(ProtocolError):
        SourceRuntimeAdapter().extract_tool_calls(
            '{"tool_name":"PythonExec","tool_input":{"code":"print(4)"}}'
        )


def test_solver_final_and_role_json_parsers() -> None:
    final = validate_solver_final(
        "<think>done</think>\nCONFIDENCE: 0.9\nFINAL_ANSWER: 4"
    )
    assert final.final_answer == "4"
    verifier = parse_verifier_json(
        '{"step_index":0,"score":-0.9,"confidence":0.95,"critique":"wrong","tool_check":true}'
    )
    repair = parse_repair_json(
        '{"action":"PATCH","target_step":0,"patch_type":"text","new_content":"right"}'
    )
    assert verifier["confidence"] == 0.95
    assert repair["action"] == "PATCH"


def test_solver_final_accepts_the_source_protocol_without_local_markers() -> None:
    """The source Solver protocol has no ``<think>``/``CONFIDENCE:``/``FINAL_ANSWER:``.

    ``agent0_evaluator._build_prompt`` asks only for fenced Python and a final
    answer in ``\\boxed{...}``, so requiring the local triple would reject every
    response the upstream runtime accepts.
    """

    response = validate_solver_final(
        "<think>volume</think>\n"
        "```python\nprint(32.54)\n```\n"
        "The volume is $\\boxed{32.54}$ cm^3."
    )

    assert response.final_answer == "32.54"
    assert response.confidence is None
    assert response.is_complete is True
    assert len(response.tool_calls) == 1


def test_solver_final_rejects_a_response_without_any_final_answer() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        validate_solver_final("```python\nprint(32.54)\n```")

    assert "final answer" in str(excinfo.value)


def test_observation_matches_upstream_runtime_wrapper() -> None:
    observation = format_code_execution_observation(
        [{"status": "success", "run_result": {"stdout": "4\n", "stderr": ""}}]
    )
    # The upstream evaluator appends a newline to stdout that already ends in
    # a newline, so this intentionally contains two trailing newlines.
    assert observation == "\n[Code Execution Result]\nOutput: 4\n\n"
