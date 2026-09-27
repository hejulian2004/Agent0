"""Answer scoring for the Agent0-VL paper's public evaluation splits.

This module is used by the SERC validation path so process metrics and
benchmark-specific answer accuracy are collected from the same trajectories.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from verl.utils.reward_score.math_verify import compute_score as compute_math_score


def extract_boxed_answer(value: Any) -> str | None:
    """Return the last complete LaTeX boxed value, including nested braces."""
    text = "" if value is None else str(value)
    last_answer = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        for index in range(match.end(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    last_answer = text[match.end():index].strip()
                    break
    return last_answer


def _clean_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        text = boxed
    else:
        answer_tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
        if answer_tags:
            text = answer_tags[-1]
        else:
            final = re.findall(r"FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)", text, flags=re.I)
            if final:
                text = final[-1]
            else:
                answer_lines = re.findall(r"(?:final\s+)?answer\s*(?:is|:|=)\s*(.+?)(?:\n|$)", text,
                                          flags=re.I)
                if answer_lines:
                    text = answer_lines[-1]
    return text.strip().strip("` \t\r\n")


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _clean_answer(value)).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\(\[\{]+|[\s\)\]\}.,;:!?]+$", "", text)
    return text


def _choice_label(value: Any, options: list[str]) -> str | None:
    text = _clean_answer(value)
    if not text:
        return None
    letters = [chr(ord("A") + index) for index in range(min(len(options), 8))] if options else list("ABCDEFGH")

    # Prefer explicit answer/choice markers, then a final standalone label.
    marked = re.findall(
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*[\(\[]?([A-H])[\)\]]?\b",
        text, flags=re.I,
    )
    if marked and marked[-1].upper() in letters:
        return marked[-1].upper()
    boxed = re.findall(r"\\boxed\s*\{\s*([A-H])\s*\}", text, flags=re.I)
    if boxed and boxed[-1].upper() in letters:
        return boxed[-1].upper()
    short = re.fullmatch(r"\s*[\(\[]?([A-H])[\)\].,:!?]?\s*", text, flags=re.I)
    if short and short.group(1).upper() in letters:
        return short.group(1).upper()
    leading = re.match(r"\s*[\(\[]?([A-H])(?:[\)\].,:-])(?:\s+|$)", text, flags=re.I)
    if leading and leading.group(1).upper() in letters:
        return leading.group(1).upper()
    bracketed = re.findall(r"(?:^|\s)[\(\[]([A-H])[\)\]](?:$|\s)", text, flags=re.I)
    if bracketed and bracketed[-1].upper() in letters:
        return bracketed[-1].upper()

    normalized = _normalize_text(text)
    matches = []
    for index, option in enumerate(options[:8]):
        option_text = re.sub(r"^\s*[A-H][.)：:]\s*", "", str(option), flags=re.I)
        normalized_option = _normalize_text(option_text)
        if normalized_option and normalized == normalized_option:
            matches.append(chr(ord("A") + index))
    return matches[0] if len(matches) == 1 else None


def _decimal(value: Any) -> Decimal | None:
    text = unicodedata.normalize("NFKC", _clean_answer(value)).replace(",", "")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def _chartqa_correct(prediction: str, answers: list[str]) -> bool:
    predicted_text = _normalize_text(prediction)
    for answer in answers:
        if predicted_text == _normalize_text(answer):
            return True
        pred_number = _decimal(prediction)
        answer_number = _decimal(answer)
        if pred_number is None or answer_number is None:
            continue
        if answer_number == 0:
            if pred_number == 0:
                return True
        elif abs(pred_number - answer_number) / abs(answer_number) <= Decimal("0.05"):
            return True
    return False


def compute_paper_eval_score(
    *,
    solution_str: str,
    ground_truth: str,
    benchmark: str = "",
    score_type: str = "",
    answers: list[str] | None = None,
    options: list[str] | None = None,
    precision: float | None = None,
) -> dict[str, Any]:
    """Return ``score``, ``acc``, and a normalized ``pred`` for one answer."""
    prediction = _clean_answer(solution_str)
    gold_answers = [str(answer) for answer in (answers or [ground_truth]) if answer is not None]
    if not gold_answers:
        gold_answers = [str(ground_truth)]
    options = [str(option) for option in (options or [])]
    kind = score_type or "math_or_text"
    benchmark = benchmark.lower()

    if kind == "multiple_choice":
        pred_label = _choice_label(prediction, options)
        gold_label = _choice_label(ground_truth, options) if not re.fullmatch(r"\s*[A-H]\s*", ground_truth, re.I) else ground_truth.strip().upper()
        correct = bool(pred_label and gold_label and pred_label == gold_label)
        return {"score": float(correct), "acc": float(correct), "pred": pred_label or prediction}

    if kind == "yes_no":
        lower = _normalize_text(prediction)
        yes_positions = [match.start() for match in re.finditer(r"\b(?:yes|true|correct|affirmative)\b", lower)]
        no_positions = [match.start() for match in re.finditer(r"\b(?:no|false|incorrect|negative)\b", lower)]
        if yes_positions and (not no_positions or yes_positions[-1] > no_positions[-1]):
            pred_label = "yes"
        elif no_positions:
            pred_label = "no"
        else:
            pred_label = lower
        gold = "yes" if _normalize_text(ground_truth) in {"yes", "1", "true"} else "no"
        correct = pred_label == gold
        return {"score": float(correct), "acc": float(correct), "pred": pred_label}

    if kind == "chartqa" or benchmark == "chartqa":
        correct = _chartqa_correct(prediction, gold_answers)
        return {"score": float(correct), "acc": float(correct), "pred": prediction}

    if kind == "numeric":
        pred_number = _decimal(prediction)
        if pred_number is not None:
            for answer in gold_answers:
                gold_number = _decimal(answer)
                if gold_number is None:
                    continue
                if precision is not None:
                    decimal_places = max(0, min(12, int(precision)))
                    quantum = Decimal(1).scaleb(-decimal_places)
                    correct = pred_number.quantize(quantum) == gold_number.quantize(quantum)
                else:
                    correct = pred_number == gold_number
                if correct:
                    return {"score": 1.0, "acc": 1.0, "pred": prediction}
            return {"score": 0.0, "acc": 0.0, "pred": prediction}

    normalized_pred = _normalize_text(prediction)
    if any(normalized_pred == _normalize_text(answer) for answer in gold_answers):
        return {"score": 1.0, "acc": 1.0, "pred": prediction}

    math_result = compute_math_score(solution_str=solution_str, ground_truth=ground_truth)
    if float(math_result.get("acc", math_result.get("score", 0.0))) > 0:
        return math_result
    return {"score": 0.0, "acc": 0.0, "pred": prediction}
