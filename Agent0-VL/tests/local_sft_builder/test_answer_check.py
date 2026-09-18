"""Tests for the deterministic reference-answer check."""

from __future__ import annotations

import pytest

from tools.local_sft_builder.answer_check import (
    ANSWER_CHECK_METHOD,
    EVIDENCE_SOURCE,
    INDETERMINATE,
    MATCH,
    MISMATCH,
    check_reference_answer,
    describe,
    extract_final_answer,
    normalize_answer,
    strip_final_answer,
)
from tools.local_sft_builder.manifest import build_manifest


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_extract_prefers_final_answer_marker() -> None:
    text = "<think>reason</think>\nCONFIDENCE: 0.9\nFINAL_ANSWER: 12"

    assert extract_final_answer(text) == "12"


def test_extract_falls_back_to_boxed() -> None:
    assert extract_final_answer(r"therefore \boxed{7}") == "7"


def test_extract_uses_last_marker() -> None:
    text = "FINAL_ANSWER: 1\nscratch\nFINAL_ANSWER: 2"

    assert extract_final_answer(text) == "2"


def test_extract_returns_none_without_answer() -> None:
    assert extract_final_answer("<think>still working</think>") is None
    assert extract_final_answer("") is None


def test_extract_marker_is_case_insensitive() -> None:
    """Align with ``agent0_evaluator._extract_answer`` and the frozen protocol."""

    assert extract_final_answer("final_answer: 12") == "12"
    assert extract_final_answer("Final_Answer: 12") == "12"
    assert check_reference_answer("final_answer: 12", "12").consistency == MATCH


def test_extract_accepts_the_mulberry_source_phrasing_as_a_last_resort() -> None:
    """``### The final answer is:`` is accepted, but only after the frozen markers.

    This used to be asserted as ``None`` on the premise that "the real rollout
    supplies the ``FINAL_ANSWER:`` protocol through the prompt scaffold".  That
    premise does not hold: ``SOLVER_SYSTEM_PROMPT`` asks for ``\\boxed{...}`` and
    the user turn carries the Mulberry template verbatim, so no part of the live
    scaffold ever mentions ``FINAL_ANSWER:``.  Measured on the real Teacher, the
    template shape is what a large share of responses actually use, and dropping
    them discarded otherwise-correct trajectories.
    """

    assert extract_final_answer("### The final answer is:\n9") == "9"
    assert check_reference_answer("### The final answer is:\n9", "9").consistency == MATCH


def test_extract_resolves_a_repeated_template_heading() -> None:
    """The Teacher sometimes restates the heading on the answer line."""

    text = "### The final answer is:\nThe final answer is: yes."
    assert extract_final_answer(text) == "yes."


def test_frozen_markers_still_outrank_the_template_phrasing() -> None:
    """A response carrying ``\\boxed{...}`` is never re-read as a template answer."""

    text = "### The final answer is:\nsee the reasoning\n\n\\boxed{A}"
    assert extract_final_answer(text) == "A"


# --------------------------------------------------------------------------
# Stripping (the rollout's salvage path)
# --------------------------------------------------------------------------


def test_strip_removes_the_answer_but_keeps_the_tool_call() -> None:
    """Salvaging a premature answer must never cost the code block.

    The whole point of the salvage path is to convert a turn that carries both
    code and an answer into a real tool call, so losing the code would defeat it.
    """

    text = (
        "<think>I will count with code.</think>\n"
        "```python\nprint(6 - 5)\n```\n"
        "Therefore the total is 1.\n"
        r"\boxed{1}"
    )
    stripped = strip_final_answer(text)

    assert extract_final_answer(stripped) is None
    assert "```python\nprint(6 - 5)\n```" in stripped
    assert "<think>I will count with code.</think>" in stripped
    assert "Therefore the total is 1." in stripped


def test_strip_removes_the_template_block_without_eating_neighbours() -> None:
    """The Mulberry heading is removed, but the preceding section is not."""

    text = (
        "### Rationales:\nBecause the bar is shortest.\n"
        "### The final answer is:\nThe final answer is: yes.\n"
    )
    stripped = strip_final_answer(text)

    assert extract_final_answer(stripped) is None
    assert "### Rationales:" in stripped
    assert "Because the bar is shortest." in stripped


def test_strip_removes_a_final_answer_line() -> None:
    text = "<think>x</think>\nFINAL_ANSWER: 12\ntrailing prose"
    stripped = strip_final_answer(text)

    assert extract_final_answer(stripped) is None
    assert "<think>x</think>" in stripped
    assert "trailing prose" in stripped


@pytest.mark.parametrize(
    "text",
    [
        "FINAL_ANSWER: 12",
        r"therefore \boxed{7}",
        "### The final answer is:\n9",
        "### The final answer is:\nThe final answer is: yes.",
        "### The final answer is: 9",
        "<think>t</think>\n```python\nprint(1)\n```\n" + r"\boxed{A}",
        "scratch\nFINAL_ANSWER: 1\nmore\nFINAL_ANSWER: 2",
    ],
)
def test_strip_leaves_nothing_extractable(text: str) -> None:
    """The next rollout pass must not re-terminate on the salvaged turn.

    This is the invariant ``strip_final_answer`` exists to provide: if any
    marker survived, the salvaged turn would terminate the very next iteration
    and the re-request would never happen.
    """

    assert extract_final_answer(text) is not None
    assert extract_final_answer(strip_final_answer(text)) is None


def test_strip_is_a_no_op_when_there_is_no_answer() -> None:
    text = "<think>t</think>\n```python\nprint(1)\n```"

    assert strip_final_answer(text) == text


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_normalize_strips_trailing_period_and_quotes() -> None:
    assert normalize_answer("2.") == "2"
    assert normalize_answer("'yes'") == "yes"
    assert normalize_answer('"42"') == "42"
    assert normalize_answer("  D.  Eggs  ") == "d. eggs"


def test_normalize_removes_thousands_separators_only_when_numeric() -> None:
    assert normalize_answer("1,234") == "1234"
    assert normalize_answer("a, b") == "a, b"


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_exact_normalized_match() -> None:
    result = check_reference_answer(
        "<think>x</think>\nFINAL_ANSWER: 2",
        "2.",
    )

    assert result.consistency == MATCH
    assert result.method == ANSWER_CHECK_METHOD
    assert result.independently_verified is False


def test_strict_numeric_match_with_tolerance() -> None:
    assert check_reference_answer("FINAL_ANSWER: 3.0", "3").consistency == MATCH
    assert check_reference_answer("FINAL_ANSWER: 1/2", "0.5").consistency == MATCH
    assert check_reference_answer("FINAL_ANSWER: 50%", "50%").consistency == MATCH


def test_numeric_unit_mismatch_is_not_a_match() -> None:
    assert check_reference_answer("FINAL_ANSWER: 50", "50%").consistency == MISMATCH


def test_numeric_tolerance_is_bounded() -> None:
    assert check_reference_answer("FINAL_ANSWER: 1.0000001", "1").consistency == MATCH
    assert check_reference_answer("FINAL_ANSWER: 1.01", "1").consistency == MISMATCH


def test_named_false_positive_cases_do_not_match() -> None:
    """The two cases called out in the Phase 2B review must be mismatches."""

    assert check_reference_answer("FINAL_ANSWER: 112", "12").consistency == MISMATCH
    assert (
        check_reference_answer(
            "FINAL_ANSWER: not enough information, no conclusion",
            "no",
        ).consistency
        == MISMATCH
    )


def test_option_label_equivalence_is_accepted_in_v2() -> None:
    """v2 accepts a bare option letter against a labelled reference.

    ``D. Eggs`` vs ``D`` was one of the shapes that made the Teacher's correct
    answers fail the export gate, so the option-label rule now accepts it.
    Substring equivalence is still rejected; see
    ``test_named_false_positive_cases_do_not_match``.
    """

    result = check_reference_answer("FINAL_ANSWER: D", "D. Eggs")

    assert result.consistency == MATCH
    assert result.reason == "option_label"
    assert result.reference_has_option_label is True
    assert result.extracted_has_option_label is True


def test_surface_form_rules_accept_the_observed_teacher_shapes() -> None:
    """Every shape from the 200-task smoke run that v1 threw away."""

    cases = [
        ("FINAL_ANSWER: D", "D. increase", "option_label"),
        ("FINAL_ANSWER: (B) israel.", "B", "option_label"),
        ("FINAL_ANSWER: A. Harper.", "A", "option_label"),
        ("FINAL_ANSWER: 4.4%", "4.4", "reference_prefix"),
        ("FINAL_ANSWER: 22 m.", "22", "reference_prefix"),
        ("FINAL_ANSWER: Kerr (Steve Kerr).", "Kerr", "reference_prefix"),
        ("FINAL_ANSWER: 79.27 billion U.S. dollars.", "79.27", "reference_prefix"),
        ("FINAL_ANSWER: 5.", "Five.", "notation_equal"),
        ("FINAL_ANSWER: $169,785.", "$ 169,785", "notation_equal"),
        (
            "FINAL_ANSWER: The Nutrition Foundation, Inc.",
            "THE NUTRITION FOUNDATION INC.",
            "notation_equal",
        ),
    ]
    for solver, reference, reason in cases:
        result = check_reference_answer(solver, reference)
        assert result.consistency == MATCH, (solver, reference)
        assert result.reason == reason, (solver, reference)


def test_surface_form_rules_still_reject_wrong_answers() -> None:
    """The relaxations must not swallow a genuinely different answer."""

    cases = [
        ("FINAL_ANSWER: 137.07", "137.04."),
        ("FINAL_ANSWER: 13.", "12"),
        ("FINAL_ANSWER: 5.", "3"),
        ("FINAL_ANSWER: apple.", "nokia"),
        ("FINAL_ANSWER: July 31, 1958.", "August 8, 1958"),
        ("FINAL_ANSWER: 1.01", "1"),
        ("FINAL_ANSWER: 7.0", "0.07"),
        ("FINAL_ANSWER: 1.5", "15"),
    ]
    for solver, reference in cases:
        result = check_reference_answer(solver, reference)
        assert result.consistency == MISMATCH, (solver, reference)


def test_empty_reference_is_indeterminate() -> None:
    result = check_reference_answer("FINAL_ANSWER: 5", "")

    assert result.consistency == INDETERMINATE
    assert result.reason == "empty_reference"


def test_missing_final_answer_is_indeterminate() -> None:
    result = check_reference_answer("<think>no answer yet</think>", "5")

    assert result.consistency == INDETERMINATE
    assert result.reason == "no_final_answer_in_solver_response"


def test_result_serialization_records_provenance() -> None:
    payload = check_reference_answer("FINAL_ANSWER: 4", "4").to_dict()

    assert payload["consistency"] == MATCH
    assert payload["answer_check_method"] == ANSWER_CHECK_METHOD
    assert payload["independently_verified"] is False


def test_describe_is_manifest_safe() -> None:
    manifest = build_manifest(
        run_id="run-1",
        base_sha="a" * 40,
        extra=describe(),
    )

    assert manifest["answer_check_method"] == ANSWER_CHECK_METHOD
    assert manifest["answer_check_evidence_source"] == EVIDENCE_SOURCE
    assert manifest["independently_verified"] is False


def test_non_text_reference_is_rejected() -> None:
    with pytest.raises(TypeError):
        normalize_answer(12)  # type: ignore[arg-type]
