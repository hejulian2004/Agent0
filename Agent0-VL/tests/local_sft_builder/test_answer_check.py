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


def test_extract_does_not_match_the_mulberry_source_phrasing() -> None:
    """``### The final answer is:`` is not the frozen marker on its own.

    The real rollout supplies the ``FINAL_ANSWER:`` protocol through the prompt
    scaffold; the raw source phrasing must not be silently accepted here,
    otherwise the exported row would not satisfy ``validate_solver_final``.
    """

    assert extract_final_answer("### The final answer is:\n9") is None


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


def test_no_substring_or_option_label_equivalence_in_v1() -> None:
    """v1 is deliberately conservative: 'D' is not 'D. Eggs'."""

    result = check_reference_answer("FINAL_ANSWER: D", "D. Eggs")

    assert result.consistency == MISMATCH
    # ...but the case is visible in the audit diagnostics.
    assert result.reference_has_option_label is True
    assert result.extracted_has_option_label is True


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
