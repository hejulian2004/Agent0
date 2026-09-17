"""Deterministic reference-answer checking for Phase 2B-P0.

The Mulberry ``ground_truth`` is a *reference label* copied from the source
artifact's own assistant message.  It is usable for an answer-consistency
check, but it is **not** independent correctness verification:

    reference_answer_match == True  !=  independently_verified == True

Therefore the evidence source recorded for a positive is
``reference_answer_match`` rather than ``deterministic_rule``, and
``independently_verified`` is always ``False``.

v1 is deliberately conservative.  Only two things can produce a ``match``:

1. exact equality after normalization, or
2. a strict numeric comparison with the documented tolerance below.

There is no fuzzy matching, no substring / "short answer contained in long
answer" rule, and no LLM judge.  In particular ``GT: 12`` vs ``Pred: 112`` and
``GT: no`` vs ``Pred: not enough information, no conclusion`` must not match.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

ANSWER_CHECK_METHOD = "reference_answer_match_v1"
EVIDENCE_SOURCE = "reference_answer_match"
INDEPENDENTLY_VERIFIED = False

# Documented numeric tolerance for the strict numeric comparison.
NUMERIC_REL_TOL = 1e-6
NUMERIC_ABS_TOL = 1e-9

MATCH = "match"
MISMATCH = "mismatch"
INDETERMINATE = "indeterminate"

_FINAL_ANSWER_RE = re.compile(
    r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_WHITESPACE_RE = re.compile(r"\s+")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_FRACTION_RE = re.compile(r"^([+-]?\d+)\s*/\s*(\d+)$")
_OPTION_LABEL_RE = re.compile(r"^([A-Ha-h])\s*[.):]?\s*(.*)$", re.DOTALL)
_QUOTE_CHARS = "\"'`\u2018\u2019\u201c\u201d"
_TRAILING_JUNK = ".\u3002!?\uff01\uff1f \t"


@dataclass(frozen=True)
class AnswerCheckResult:
    """Outcome of one reference-answer comparison."""

    consistency: str
    extracted_answer: str | None = None
    reference_answer: str | None = None
    reason: str | None = None
    method: str = ANSWER_CHECK_METHOD
    independently_verified: bool = INDEPENDENTLY_VERIFIED
    # Diagnostics only: these never change the verdict in v1.
    reference_has_option_label: bool = False
    extracted_has_option_label: bool = False

    @property
    def matched(self) -> bool:
        return self.consistency == MATCH

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistency": self.consistency,
            "answer_check_method": self.method,
            "independently_verified": self.independently_verified,
            "extracted_answer": self.extracted_answer,
            "reference_answer": self.reference_answer,
            "reason": self.reason,
            "reference_has_option_label": self.reference_has_option_label,
            "extracted_has_option_label": self.extracted_has_option_label,
        }


def extract_final_answer(text: str) -> str | None:
    """Extract the Solver's final answer using the frozen upstream order.

    ``FINAL_ANSWER:`` is preferred, with ``\\boxed{...}`` as the fallback,
    matching ``agent0_evaluator._extract_answer``.  The marker match is
    case-insensitive for the same reason: the upstream evaluator matches with
    ``re.IGNORECASE``, so a case-sensitive match here would reject responses the
    source runtime accepts.

    This is the single source of truth for "what is the final answer":
    ``protocol.parse_solver_response`` and the export gate both defer to it.
    """

    if not isinstance(text, str) or not text:
        return None
    matches = _FINAL_ANSWER_RE.findall(text)
    if matches:
        candidate = _strip_wrapping(matches[-1])
        if candidate:
            return candidate
    boxed = _BOXED_RE.findall(text)
    if boxed:
        candidate = _strip_wrapping(boxed[-1])
        if candidate:
            return candidate
    return None


def _strip_wrapping(value: str) -> str:
    text = unicodedata.normalize("NFC", value).strip()
    while text and text[0] in _QUOTE_CHARS:
        text = text[1:].lstrip()
    while text and text[-1] in _QUOTE_CHARS:
        text = text[:-1].rstrip()
    return text


def normalize_answer(value: str) -> str:
    """Normalize an answer string for exact comparison."""

    if not isinstance(value, str):
        raise TypeError("answer must be text")
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\u2212", "-").replace("\u00a0", " ").replace("\u2009", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _strip_wrapping(text)
    if _THOUSANDS_RE.fullmatch(text):
        text = text.replace(",", "")
    # A trailing sentence period is punctuation, not part of the answer.
    while text and text[-1] in _TRAILING_JUNK:
        text = text[:-1].rstrip()
    return text.casefold()


def _as_number(value: str) -> tuple[float, str] | None:
    """Parse ``value`` as a number, returning ``(number, unit)``."""

    text = value.strip()
    if not text:
        return None
    unit = ""
    if text.endswith("%"):
        unit = "%"
        text = text[:-1].strip()
    if _THOUSANDS_RE.fullmatch(text):
        text = text.replace(",", "")
    fraction = _FRACTION_RE.fullmatch(text)
    if fraction:
        denominator = int(fraction.group(2))
        if denominator == 0:
            return None
        return int(fraction.group(1)) / denominator, unit
    if not _NUMBER_RE.fullmatch(text):
        return None
    try:
        return float(text), unit
    except ValueError:  # pragma: no cover - regex already constrains the input
        return None


def _option_label(value: str) -> str | None:
    """Return the leading option letter for diagnostics only."""

    match = _OPTION_LABEL_RE.fullmatch(value.strip())
    if match is None:
        return None
    return match.group(1).casefold()


def check_reference_answer(
    solver_text: str,
    reference: str,
    *,
    rel_tol: float = NUMERIC_REL_TOL,
    abs_tol: float = NUMERIC_ABS_TOL,
) -> AnswerCheckResult:
    """Compare a Solver's final answer against the source reference label."""

    if not isinstance(reference, str) or not normalize_answer(reference):
        return AnswerCheckResult(
            consistency=INDETERMINATE,
            reference_answer=reference if isinstance(reference, str) else None,
            reason="empty_reference",
        )

    extracted = extract_final_answer(solver_text)
    normalized_reference = normalize_answer(reference)
    reference_has_label = _option_label(reference) is not None
    if extracted is None:
        return AnswerCheckResult(
            consistency=INDETERMINATE,
            reference_answer=normalized_reference,
            reason="no_final_answer_in_solver_response",
            reference_has_option_label=reference_has_label,
        )

    normalized_extracted = normalize_answer(extracted)
    extracted_has_label = _option_label(extracted) is not None
    diagnostics = {
        "extracted_answer": normalized_extracted,
        "reference_answer": normalized_reference,
        "reference_has_option_label": reference_has_label,
        "extracted_has_option_label": extracted_has_label,
    }

    if normalized_extracted == normalized_reference:
        return AnswerCheckResult(consistency=MATCH, **diagnostics)

    candidate_number = _as_number(normalized_extracted)
    reference_number = _as_number(normalized_reference)
    if candidate_number is not None and reference_number is not None:
        candidate_value, candidate_unit = candidate_number
        reference_value, reference_unit = reference_number
        if candidate_unit == reference_unit and math.isclose(
            candidate_value,
            reference_value,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            return AnswerCheckResult(consistency=MATCH, **diagnostics)

    return AnswerCheckResult(consistency=MISMATCH, **diagnostics)


def describe() -> dict[str, Any]:
    """Provenance fields for manifests and audit records."""

    return {
        "answer_check_method": ANSWER_CHECK_METHOD,
        "answer_check_evidence_source": EVIDENCE_SOURCE,
        "independently_verified": INDEPENDENTLY_VERIFIED,
        "numeric_rel_tol": NUMERIC_REL_TOL,
        "numeric_abs_tol": NUMERIC_ABS_TOL,
    }
