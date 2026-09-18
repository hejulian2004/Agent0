"""Deterministic reference-answer checking for Phase 2B-P0.

The Mulberry ``ground_truth`` is a *reference label* copied from the source
artifact's own assistant message.  It is usable for an answer-consistency
check, but it is **not** independent correctness verification:

    reference_answer_match == True  !=  independently_verified == True

Therefore the evidence source recorded for a positive is
``reference_answer_match`` rather than ``deterministic_rule``, and
``independently_verified`` is always ``False``.

The comparison is exact per axis, never fuzzy.  A ``match`` can come from:

1. exact equality after normalization,
2. a strict numeric comparison with the documented tolerance below, or
3. one of the v2 *surface-form* rules, which accept the same answer written in a
   different notation.

The v2 rules exist because the Teacher answers through the Mulberry user
template (``### The final answer is:``) while the reference label is the source
artifact's own terse string, so a correct answer routinely differs only in
notation: ``D. increase`` vs ``D``, ``4.4`` vs ``4.4%``, ``Five.`` vs ``5.``,
``$ 169,785`` vs ``$169,785.``, ``Kerr`` vs ``Kerr (Steve Kerr).``.

The surface-form rules are:

* **notation equality** - equal after casefolding and dropping every character
  except letters, digits, ``.``, ``%``, ``+`` and ``-``.  Keeping ``.`` is what
  makes this safe: ``1.5`` and ``15`` must not collide.
* **number word** - ``five`` and ``5`` are the same answer.
* **option label** - when the reference carries an option label (``D. increase``)
  a bare letter from the Solver is accepted, and a single-letter reference is
  accepted against the Solver's leading option label (``A`` vs ``A. Harper.``,
  ``B`` vs ``(B) israel.``).
* **word-boundary prefix** - the reference is a prefix of the Solver answer and
  the next character is not alphanumeric, which covers trailing units and
  parenthetical expansions.

There is still no substring / "short answer contained in long answer" rule and
no LLM judge, and there is deliberately no ``startswith`` without the boundary
check.  In particular ``GT: 12`` vs ``Pred: 112`` and ``GT: no`` vs
``Pred: not enough information, no conclusion`` must not match: neither is
notation-equal, neither is a number word, and in both the character after the
prefix is alphanumeric.  Both are pinned by tests.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

ANSWER_CHECK_METHOD = "reference_answer_match_v2"
EVIDENCE_SOURCE = "reference_answer_match"
INDEPENDENTLY_VERIFIED = False

# Documented numeric tolerance for the strict numeric comparison.
NUMERIC_REL_TOL = 1e-6
NUMERIC_ABS_TOL = 1e-9

# Spelled-out numbers the Teacher uses interchangeably with digits.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

# Everything that is not content-bearing for the notation-equality rule.
# ``.`` is deliberately retained so ``1.5`` never collapses onto ``15``.
_NOTATION_NOISE_RE = re.compile(r"[^0-9a-z.%+-]+")
# A leading option letter, tolerating the ``(B)``/``[B]`` wrappers.
_LEADING_OPTION_RE = re.compile(r"^([a-h])[\s.):\]]")
# The number an answer opens with, used to stop ``1`` from matching ``1.01``.
_LEADING_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)")

MATCH = "match"
MISMATCH = "mismatch"
INDETERMINATE = "indeterminate"

_FINAL_ANSWER_RE = re.compile(
    r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
# The Mulberry source template asks for its own answer shape:
#
#     ### The final answer is:
#     <answer>
#
# The Teacher follows whichever instruction it weights higher, and the user turn
# carries that template verbatim while the system turn asks for ``\boxed{...}``.
# This pattern is therefore the last-resort branch; see ``extract_final_answer``.
_TEMPLATE_ANSWER_RE = re.compile(
    r"#*[ \t]*the final answer is[ \t]*:?[ \t]*", re.IGNORECASE
)
# The removal patterns used by ``strip_final_answer``.  They are the inverse of
# the three extraction patterns above and live next to them so the marker
# vocabulary has a single home: adding a marker to the extractor without adding
# it here would leave the rollout's salvage path unable to remove it.
#
# ``FINAL_ANSWER:`` takes the whole line, because the marker and its value are
# one syntactic unit.
_FINAL_ANSWER_LINE_RE = re.compile(r"^.*FINAL_ANSWER:.*$", re.IGNORECASE | re.MULTILINE)
# The Mulberry heading takes the rest of its own line, plus the following line
# when that line is the answer rather than another ``###`` section heading.  The
# negative lookahead is what keeps ``### The final answer is:`` from swallowing
# a subsequent ``### Rationales:`` block.
_TEMPLATE_BLOCK_RE = re.compile(
    r"^[ \t]*#*[ \t]*the final answer is[ \t]*:?[^\n]*(?:\n(?![ \t]*#)[^\n]*)?",
    re.IGNORECASE | re.MULTILINE,
)
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

    A third, last-resort branch accepts the *Mulberry source template* answer
    shape (``### The final answer is:``) used by the user turn.  It is required
    because nothing in the live prompt scaffold supplies ``FINAL_ANSWER:``: the
    system turn asks for ``\\boxed{...}`` while the user turn carries the source
    template verbatim, so the Teacher legitimately answers in either shape.  The
    branch is deliberately ordered last, so a response that does carry
    ``FINAL_ANSWER:`` or ``\\boxed{...}`` is never re-interpreted.

    Unlike the upstream extractor this returns ``None`` when nothing is found:
    ``is_complete`` gates whether the rollout keeps executing code, and a
    never-``None`` extractor would terminate every rollout on its first turn.

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
    return _extract_template_answer(text)


def strip_final_answer(text: str) -> str:
    """Remove the final-answer markers from a Solver turn.

    The inverse of :func:`extract_final_answer`, and the reason it lives beside
    it: the rollout's salvage path needs to drop an answer without dropping the
    reasoning or the code that came with it.  When one turn carries both a code
    block and a final answer, that answer was produced *without* the sandbox
    output, so it is removed and the next request answers again with the
    observation actually in context.

    Everything that is not an answer marker is preserved verbatim -- the
    ``<think>`` block, the prose reasoning, and every fenced code block.

    Guarantee: whenever :func:`extract_final_answer` finds an answer,
    ``extract_final_answer(strip_final_answer(text))`` is ``None``.  The rollout
    relies on that to keep the salvaged turn from terminating the next pass.
    """

    if not isinstance(text, str) or not text:
        return text
    stripped = _FINAL_ANSWER_LINE_RE.sub("", text)
    stripped = _BOXED_RE.sub("", stripped)
    stripped = _TEMPLATE_BLOCK_RE.sub("", stripped)
    return stripped


def _extract_template_answer(text: str) -> str | None:
    """Read the answer that follows the last ``### The final answer is:`` heading.

    The Teacher sometimes repeats the heading on the answer line itself, so a
    second marker inside the first non-empty line is resolved rather than
    returned verbatim.
    """

    matches = list(_TEMPLATE_ANSWER_RE.finditer(text))
    if not matches:
        return None
    tail = text[matches[-1].end() :]
    for line in tail.split("\n"):
        candidate = line.strip()
        if not candidate:
            continue
        nested = list(_TEMPLATE_ANSWER_RE.finditer(candidate))
        if nested:
            candidate = candidate[nested[-1].end() :].strip()
        return _strip_wrapping(candidate) or None
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


def _notation_key(value: str) -> str:
    """Collapse an answer to a notation-insensitive key.

    Case, spacing, thousands separators, currency symbols and brackets all
    disappear.  ``.`` survives on purpose, so ``1.5`` can never collapse onto
    ``15``; ``%`` survives so ``4.4`` and ``4.4%`` stay distinct here and are
    left to the numeric and prefix rules.
    """

    text = normalize_answer(value)
    text = _NOTATION_NOISE_RE.sub("", text)
    return _NUMBER_WORDS.get(text, text)


def _leading_option_letter(value: str) -> str | None:
    """Return the option letter an answer opens with, if it really is one.

    A delimiter is required after the letter, so an ordinary word that merely
    starts with ``a``-``h`` (``answer``, ``plants``, ``Kerr``) is not mistaken
    for an option label.
    """

    text = normalize_answer(value).strip().lstrip("([")
    if len(text) == 1 and text in "abcdefgh":
        return text
    match = _LEADING_OPTION_RE.match(text)
    return match.group(1) if match else None


def _leading_number(value: str) -> float | None:
    """Parse the number an answer opens with, if any."""

    match = _LEADING_NUMBER_RE.match(normalize_answer(value))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:  # pragma: no cover - the regex already constrains it
        return None


def _prefix_at_boundary(reference: str, extracted: str) -> bool:
    """True when the reference is a prefix of the answer at a token boundary.

    The boundary test is what keeps ``12`` from matching ``112`` and ``no``
    from matching ``not enough information, no conclusion``.  A numeric
    reference additionally requires the Solver's leading number to be the same
    number, so ``1`` still does not match ``1.01``.
    """

    ref = normalize_answer(reference)
    pred = normalize_answer(extracted)
    if not ref or len(ref) >= len(pred) or not pred.startswith(ref):
        return False
    reference_number = _leading_number(reference)
    if reference_number is not None:
        candidate_number = _leading_number(extracted)
        if candidate_number is None or not math.isclose(
            reference_number,
            candidate_number,
            rel_tol=NUMERIC_REL_TOL,
            abs_tol=NUMERIC_ABS_TOL,
        ):
            return False
    return not pred[len(ref)].isalnum()


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

    # --- v2 surface-form stages -------------------------------------------
    # Each stage is exact on its own axis.  None of them is a substring test,
    # so ``12``/``112`` and ``no``/``not enough information, no conclusion``
    # stay mismatches (see the module docstring).
    notation_extracted = _notation_key(extracted)
    notation_reference = _notation_key(reference)
    if notation_extracted and notation_extracted == notation_reference:
        return AnswerCheckResult(
            consistency=MATCH, reason="notation_equal", **diagnostics
        )

    reference_option = _leading_option_letter(reference)
    if reference_option is not None and _leading_option_letter(extracted) == reference_option:
        return AnswerCheckResult(
            consistency=MATCH, reason="option_label", **diagnostics
        )

    if _prefix_at_boundary(reference, extracted):
        return AnswerCheckResult(
            consistency=MATCH, reason="reference_prefix", **diagnostics
        )

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
