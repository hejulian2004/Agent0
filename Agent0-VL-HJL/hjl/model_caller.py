"""Strictly typed model caller schemas, protocol, and implementations for HJL."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TypeVar

from PIL import Image

from agent0_protocol.adapters import ResponsesAdapter

from .schemas import (
    CandidateRegion,
    DirectPredictionResult,
    EvidenceVerificationResult,
    FailureDiagnosisResult,
    GenericVerificationResult,
    GlobalInspectionResult,
    HypothesisResult,
    ModelOutputError,
    ReactAction,
    ReactDecisionResult,
    RegionalEvidenceFinding,
    RegionalVerificationResult,
    require_bool,
    require_dict,
    require_float,
    require_list,
    require_str,
)
from .state import EvidenceRelation, EvidenceState
from .taxonomy import (
    EvidenceConclusion,
    EvidenceStatus,
    FailureType,
    RegionalStatus,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Re-export schemas for external callers
__all__ = [
    "CandidateRegion",
    "DirectPredictionResult",
    "EvidenceVerificationResult",
    "FailureDiagnosisResult",
    "GenericVerificationResult",
    "GlobalInspectionResult",
    "HypothesisResult",
    "ModelOutputError",
    "ReactAction",
    "ReactDecisionResult",
    "RegionalEvidenceFinding",
    "RegionalVerificationResult",
    "require_bool",
    "require_dict",
    "require_float",
    "require_list",
    "require_str",
    "HJLModelCaller",
    "ResponsesHJLModelCaller",
    "MockHJLModelCaller",
    "ScriptedHJLModelCaller",
]


class HJLModelCaller(Protocol):
    """Protocol defining model caller operations across HJL nodes and baselines."""

    def inspect_global(
        self,
        image_path: str,
        instruction: str,
        category: str,
    ) -> GlobalInspectionResult: ...

    def generate_hypothesis(
        self,
        image_path: str,
        candidate_region: list[int] | None,
        category: str,
    ) -> HypothesisResult: ...

    def verify_regional(
        self,
        observation: dict[str, Any],
        image_path: str,
    ) -> RegionalVerificationResult: ...

    def extract_regional_evidence(
        self,
        image_path: str,
        observation: dict[str, Any],
        active_hypothesis: dict[str, Any] | None,
    ) -> RegionalEvidenceFinding: ...

    def verify_evidence(
        self,
        evidence_state: EvidenceState,
        category: str,
        anomaly_threshold: float,
        normal_threshold: float,
    ) -> EvidenceVerificationResult: ...

    def diagnose_failure(
        self,
        verifier_judgment: Any,
        observation: dict[str, Any] | None,
        evidence_state: EvidenceState,
    ) -> FailureDiagnosisResult: ...

    def direct_inspect(
        self,
        image_path: str,
        instruction: str,
        category: str,
    ) -> DirectPredictionResult: ...

    def react_step(
        self,
        history: list[dict[str, Any]],
        image_path: str,
        enabled_tools: list[str],
        instruction: str = "Inspect this image using tools.",
        category: str = "visual_object",
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> ReactDecisionResult: ...

    def generic_verify_react(
        self,
        history: list[dict[str, Any]],
        decision: ReactDecisionResult,
        image_path: str,
        category: str,
    ) -> GenericVerificationResult: ...


def _image_to_base64_url(image_path: str | Path) -> str:
    raw = str(image_path)
    if raw.startswith("data:"):
        return raw
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _clean_json_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


class ResponsesHJLModelCaller:
    """Live single-turn inference caller communicating with OpenAI Responses endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.base_url = base_url or os.environ.get("AGENT0_RESPONSES_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("AGENT0_RESPONSES_API_KEY", "")
        self.model = model or os.environ.get("AGENT0_RESPONSES_MODEL", "gpt-4o")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("AGENT0_RESPONSES_API_KEY is required for ResponsesHJLModelCaller.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _call_and_validate(
        self,
        prompt: str,
        image_path: str | None,
        validator_fn: Callable[[dict[str, Any]], T],
    ) -> T:
        """Execute Responses call and validate typed output with a single retry."""
        last_error: Exception | None = None
        image_url = _image_to_base64_url(image_path) if image_path else None

        for attempt in range(2):
            try:
                content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
                if image_url:
                    content.append({
                        "type": "input_image",
                        "image_url": image_url,
                    })

                response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": content}],
                    max_output_tokens=1024,
                )

                adapter = ResponsesAdapter()
                items = adapter.output_items(response)
                text = ""
                for item in items:
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("type") in {"output_text", "input_text"}:
                                text += str(part.get("text", ""))

                cleaned = _clean_json_markdown(text)
                try:
                    data = json.loads(cleaned)
                except Exception:
                    from agent0_protocol.verifier import extract_json_dict
                    data = extract_json_dict(text)
                if not isinstance(data, dict):
                    raise ValueError(f"Model output root must be a JSON object, got {type(data).__name__}")

                return validator_fn(data)

            except Exception as exc:
                last_error = exc
                logger.warning(f"Responses attempt {attempt + 1} validation failed: {exc}")
                if attempt == 1:
                    raise ModelOutputError(f"Model response validation failed after retry: {exc}") from exc

        raise ModelOutputError(f"Model call failed: {last_error}")

    def inspect_global(self, image_path: str, instruction: str, category: str) -> GlobalInspectionResult:
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception:
            img_w, img_h = None, None

        prompt = (
            f"Inspect {category} image for key visual targets. Instruction: {instruction}\n"
            "Return JSON:\n"
            '{"observation": str, "is_normal": bool, "confidence": float, '
            '"candidate_regions": [{"bbox": [x1, y1, x2, y2], "confidence": float}]}'
        )

        def _validate(data: dict[str, Any]) -> GlobalInspectionResult:
            obs = require_str(data, "observation")
            is_normal = require_bool(data, "is_normal")
            conf = require_float(data, "confidence", 0.0, 1.0)
            raw_cands = require_list(data, "candidate_regions")

            candidates: list[CandidateRegion] = []
            for c in raw_cands:
                if not isinstance(c, dict):
                    raise ValueError(f"candidate item must be a dict, got {type(c).__name__}")
                raw_box = c.get("bbox") or c.get("box_2d") or c.get("bounding_box") or c.get("box")
                if not isinstance(raw_box, list):
                    raise ValueError(f"candidate bbox must be a list of 4 numbers, got {raw_box}")
                if len(raw_box) != 4 or not all(isinstance(v, (int, float)) for v in raw_box):
                    raise ValueError(f"candidate bbox must be a list of 4 numbers, got {raw_box}")

                is_box_2d = ("box_2d" in c and "bbox" not in c)
                c0, c1, c2, c3 = [float(v) for v in raw_box]

                # Case 1: Normalized [0.0, 1.0] floats
                if (
                    img_w is not None
                    and img_h is not None
                    and all(0.0 <= v <= 1.0 for v in (c0, c1, c2, c3))
                    and any(isinstance(v, float) and 0.0 < v < 1.0 for v in raw_box)
                ):
                    if is_box_2d:
                        y1, x1, y2, x2 = c0 * img_h, c1 * img_w, c2 * img_h, c3 * img_w
                    else:
                        x1, y1, x2, y2 = c0 * img_w, c1 * img_h, c2 * img_w, c3 * img_h
                # Case 2: box_2d scaled in [0, 1000] format (e.g. Gemini specification [ymin, xmin, ymax, xmax])
                elif (
                    is_box_2d
                    and img_w is not None
                    and img_h is not None
                    and all(0 <= v <= 1000 for v in (c0, c1, c2, c3))
                    and (img_w != 1000 or img_h != 1000)
                ):
                    y1, x1, y2, x2 = c0 * img_h / 1000.0, c1 * img_w / 1000.0, c2 * img_h / 1000.0, c3 * img_w / 1000.0
                elif is_box_2d:
                    y1, x1, y2, x2 = c0, c1, c2, c3
                else:
                    x1, y1, x2, y2 = c0, c1, c2, c3

                if img_w is not None and img_h is not None:
                    x_min = max(0, min(img_w, int(round(min(x1, x2)))))
                    y_min = max(0, min(img_h, int(round(min(y1, y2)))))
                    x_max = max(0, min(img_w, int(round(max(x1, x2)))))
                    y_max = max(0, min(img_h, int(round(max(y1, y2)))))
                    if x_max <= x_min:
                        if x_min < img_w:
                            x_max = x_min + 1
                        else:
                            x_min = max(0, x_max - 1)
                    if y_max <= y_min:
                        if y_min < img_h:
                            y_max = y_min + 1
                        else:
                            y_min = max(0, y_max - 1)
                else:
                    x_min = int(round(min(x1, x2)))
                    y_min = int(round(min(y1, y2)))
                    x_max = int(round(max(x1, x2)))
                    y_max = int(round(max(y1, y2)))
                    if x_max <= x_min:
                        x_max = x_min + 1
                    if y_max <= y_min:
                        y_max = y_min + 1

                bbox = [x_min, y_min, x_max, y_max]

                c_conf = require_float(c, "confidence", 0.0, 1.0)
                label = c.get("label")
                candidates.append(CandidateRegion(bbox=bbox, confidence=c_conf, label=label))

            return GlobalInspectionResult(
                observation=obs,
                candidate_regions=candidates,
                is_normal=is_normal,
                confidence=conf,
            )

        return self._call_and_validate(prompt, image_path, _validate)

    def generate_hypothesis(
        self, image_path: str, candidate_region: list[int] | None, category: str
    ) -> HypothesisResult:
        prompt = (
            f"Formulate visual hypothesis for {category} at ROI {candidate_region}.\n"
            'Return JSON: {"hypothesis_id": str, "type": str, "description": str, "confidence": float}'
        )

        def _validate(data: dict[str, Any]) -> HypothesisResult:
            hyp_id = require_str(data, "hypothesis_id")
            typ = require_str(data, "type")
            desc = require_str(data, "description")
            conf = require_float(data, "confidence", 0.0, 1.0)
            return HypothesisResult(
                hypothesis_id=hyp_id,
                type=typ,
                description=desc,
                confidence=conf,
                target_region=candidate_region,
            )

        return self._call_and_validate(prompt, image_path, _validate)

    def verify_regional(self, observation: dict[str, Any], image_path: str) -> RegionalVerificationResult:
        tool_name = observation.get("tool", "")
        meta = observation.get("metadata", {})
        prompt = (
            f"Verify regional ROI quality from {tool_name}. Metadata: {json.dumps(meta, ensure_ascii=False, separators=(',', ':'))}\n"
            "Evaluate ROI sharpness, resolution, and relevance (do not classify anomalies).\n"
            'Return JSON: {"status": "PASS"|"FAIL", "judgment_confidence": float (0-1), "reason": str}'
        )

        def _validate(data: dict[str, Any]) -> RegionalVerificationResult:
            status_str = require_str(data, "status")
            status = RegionalStatus(status_str)
            conf = require_float(data, "judgment_confidence", 0.0, 1.0)
            reason = require_str(data, "reason")
            return RegionalVerificationResult(
                status=status,
                judgment_confidence=conf,
                reason=reason,
            )

        return self._call_and_validate(prompt, image_path, _validate)

    def extract_regional_evidence(
        self,
        image_path: str,
        observation: dict[str, Any],
        active_hypothesis: dict[str, Any] | None,
    ) -> RegionalEvidenceFinding:
        prompt = (
            f"Analyze regional evidence for hypothesis: {active_hypothesis}.\n"
            'Return JSON: {"finding": str, "relation": "SUPPORT"|"CONTRADICT"|"NEUTRAL", '
            '"observation_type": str, "confidence": float (0-1)}'
        )

        def _validate(data: dict[str, Any]) -> RegionalEvidenceFinding:
            finding = require_str(data, "finding")
            rel_str = require_str(data, "relation")
            rel = EvidenceRelation(rel_str)
            obs_type = require_str(data, "observation_type")
            conf = require_float(data, "confidence", 0.0, 1.0)
            return RegionalEvidenceFinding(
                finding=finding,
                relation=rel,
                observation_type=obs_type,
                confidence=conf,
                metadata=dict(observation.get("metadata", {})),
            )

        return self._call_and_validate(prompt, image_path, _validate)

    def verify_evidence(
        self,
        evidence_state: EvidenceState,
        category: str,
        anomaly_threshold: float,
        normal_threshold: float,
    ) -> EvidenceVerificationResult:
        score = evidence_state.anomaly_score
        if score >= anomaly_threshold:
            return EvidenceVerificationResult(
                status=EvidenceStatus.PASS,
                conclusion=EvidenceConclusion.ANOMALY,
                judgment_confidence=0.90,
                reason=f"Accumulated evidence confirms visual target (score {score:.2f} >= {anomaly_threshold}).",
            )
        has_explicit_normal = any(
            e.observation_type in {"normal_reference_match", "reference_comparison", "verified_normal_feature"}
            and e.relation == EvidenceRelation.CONTRADICT
            for e in evidence_state.evidence_items
        )
        if (
            score <= normal_threshold
            and not evidence_state.unresolved_regions
            and len(evidence_state.evidence_items) >= 1
            and has_explicit_normal
        ):
            return EvidenceVerificationResult(
                status=EvidenceStatus.PASS,
                conclusion=EvidenceConclusion.NORMAL,
                judgment_confidence=0.85,
                reason=f"Evidence confirms normal component (score {score:.2f} <= {normal_threshold}).",
            )
        return EvidenceVerificationResult(
            status=EvidenceStatus.FAIL,
            conclusion=EvidenceConclusion.UNRESOLVED,
            judgment_confidence=0.65,
            reason=f"Evidence unresolved (score {score:.2f}, unresolved regions: {len(evidence_state.unresolved_regions)}).",
        )

    def diagnose_failure(
        self,
        verifier_judgment: Any,
        observation: dict[str, Any] | None,
        evidence_state: EvidenceState,
    ) -> FailureDiagnosisResult:
        ev_summary = {
            "anomaly_score": evidence_state.anomaly_score,
            "supporting_count": len(evidence_state.supporting_evidence),
            "contradicting_count": len(evidence_state.contradicting_evidence),
            "unresolved_regions_count": len(evidence_state.unresolved_regions),
            "normal_references_count": len(evidence_state.normal_references),
        }
        latest_obs_summary = {
            "tool": observation.get("tool") if observation else None,
            "success": observation.get("success") if observation else None,
            "metadata": observation.get("metadata") if observation else {},
            "error": observation.get("error") if observation else None,
        }
        prompt = (
            f"Inspection checkpoint failed. Reason: {getattr(verifier_judgment, 'reason', '')}\n"
            f"Observation: {json.dumps(latest_obs_summary, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Evidence: {json.dumps(ev_summary, ensure_ascii=False, separators=(',', ':'))}\n"
            "Classify root cause into: WRONG_REGION, LOW_RESOLUTION, MISSING_REFERENCE, "
            "VIEWPOINT_MISMATCH, CONTRADICTORY_EVIDENCE, LOCALIZATION_UNCERTAIN, PREMATURE_CONCLUSION.\n"
            'Return JSON: {"failure_type": str, "cause": str, "diagnosis_confidence": float (0-1)}'
        )

        def _validate(data: dict[str, Any]) -> FailureDiagnosisResult:
            ft_str = require_str(data, "failure_type")
            ft = FailureType(ft_str)
            cause = require_str(data, "cause")
            conf = require_float(data, "diagnosis_confidence", 0.0, 1.0)
            return FailureDiagnosisResult(
                failure_type=ft,
                cause=cause,
                diagnosis_confidence=conf,
            )

        return self._call_and_validate(prompt, None, _validate)

    def direct_inspect(self, image_path: str, instruction: str, category: str) -> DirectPredictionResult:
        prompt = (
            f"Directly inspect {category} image. Instruction: {instruction}\n"
            'Return JSON: {"is_anomaly": bool, "conclusion": "ANOMALY"|"NORMAL", '
            '"anomaly_score": float (0-1), "confidence": float (0-1), "explanation": str}'
        )

        def _validate(data: dict[str, Any]) -> DirectPredictionResult:
            is_anom = require_bool(data, "is_anomaly")
            conc = require_str(data, "conclusion")
            score = require_float(data, "anomaly_score", 0.0, 1.0)
            conf = require_float(data, "confidence", 0.0, 1.0)
            exp = require_str(data, "explanation")
            return DirectPredictionResult(
                is_anomaly=is_anom,
                conclusion=conc,
                anomaly_score=score,
                confidence=conf,
                explanation=exp,
            )

        return self._call_and_validate(prompt, image_path, _validate)

    def react_step(
        self,
        history: list[dict[str, Any]],
        image_path: str,
        enabled_tools: list[str],
        instruction: str = "Inspect this image using tools.",
        category: str = "visual_object",
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> ReactDecisionResult:
        tools_spec = (
            json.dumps(tool_definitions, ensure_ascii=False, separators=(",", ":"))
            if tool_definitions
            else json.dumps(enabled_tools, ensure_ascii=False, separators=(",", ":"))
        )
        history_spec = json.dumps(history[-4:] if history else [], ensure_ascii=False, separators=(",", ":"))
        prompt = (
            f"Visual reasoning agent inspecting {category}. Instruction: {instruction}\n"
            f"Tools:\n{tools_spec}\n"
            f"History:\n{history_spec}\n"
            "Decide next step. Return JSON:\n"
            '{"action": "TOOL_CALL", "tool_name": str, "tool_arguments": dict, "confidence": float (0-1)}\n'
            'or: {"action": "FINISH", "final_answer": str, "is_anomaly": bool, "confidence": float (0-1)}'
        )

        def _validate(data: dict[str, Any]) -> ReactDecisionResult:
            act_str = require_str(data, "action")
            act = ReactAction(act_str)

            if act == ReactAction.TOOL_CALL:
                tool_name = require_str(data, "tool_name")
                tool_arguments = require_dict(data, "tool_arguments")
                raw_conf = data.get("confidence")
                conf = float(raw_conf) if raw_conf is not None else 0.5
                if not (0.0 <= conf <= 1.0):
                    conf = 0.5
                return ReactDecisionResult(
                    action=act,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    confidence=conf,
                )
            else:
                conf = require_float(data, "confidence", 0.0, 1.0)
                is_anom = require_bool(data, "is_anomaly")
                final_answer = require_str(data, "final_answer")
                return ReactDecisionResult(
                    action=act,
                    final_answer=final_answer,
                    is_anomaly=is_anom,
                    confidence=conf,
                )

        return self._call_and_validate(prompt, image_path, _validate)

    def generic_verify_react(
        self,
        history: list[dict[str, Any]],
        decision: ReactDecisionResult,
        image_path: str,
        category: str,
    ) -> GenericVerificationResult:
        prompt = (
            f"Visual verification agent reviewing {category} inspection trajectory.\n"
            f"Claim: is_anomaly={decision.is_anomaly}, explanation={decision.final_answer}\n"
            f"History: {json.dumps(history[-4:] if history else [], ensure_ascii=False, separators=(',', ':'))}\n"
            'Return JSON: {"passed": bool, "confidence": float (0-1), "feedback": str}'
        )

        def _validate(data: dict[str, Any]) -> GenericVerificationResult:
            passed = require_bool(data, "passed")
            conf = require_float(data, "confidence", 0.0, 1.0)
            fb = require_str(data, "feedback")
            return GenericVerificationResult(passed=passed, confidence=conf, feedback=fb)

        return self._call_and_validate(prompt, image_path, _validate)


class MockHJLModelCaller:
    """Deterministic offline caller for unit tests, dry runs, and golden trajectory verification."""

    def __init__(self, mode: str = "default") -> None:
        self.mode = mode
        self.call_history: list[str] = []

    def inspect_global(self, image_path: str, instruction: str, category: str) -> GlobalInspectionResult:
        self.call_history.append("inspect_global")
        try:
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception:
            w, h = 100, 100

        if instruction and "force_normal" in instruction.lower():
            return GlobalInspectionResult(
                observation="Pristine global surface confirmed.",
                candidate_regions=[],
                is_normal=True,
                confidence=0.98,
            )

        cands = [
            CandidateRegion(
                bbox=[int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)],
                confidence=0.85,
                label="primary_visual_target_candidate",
                region_id="cand_0",
            )
        ]
        return GlobalInspectionResult(
            observation=f"Identified candidate target visual region on [{w}x{h}] surface.",
            candidate_regions=cands,
            is_normal=False,
            confidence=0.80,
        )

    def generate_hypothesis(
        self, image_path: str, candidate_region: list[int] | None, category: str
    ) -> HypothesisResult:
        self.call_history.append("generate_hypothesis")
        return HypothesisResult(
            hypothesis_id="hyp_mock_feature_01",
            type="visual_feature",
            description=f"Key visual feature on {category}.",
            confidence=0.80,
            target_region=candidate_region,
        )

    def verify_regional(self, observation: dict[str, Any], image_path: str) -> RegionalVerificationResult:
        self.call_history.append("verify_regional")
        metadata = observation.get("metadata", {})
        size = metadata.get("image_size")
        if size and (size[0] < 12 or size[1] < 12):
            return RegionalVerificationResult(
                status=RegionalStatus.FAIL,
                judgment_confidence=0.85,
                reason=f"ROI resolution [{size[0]}x{size[1]}] is insufficient (<12px).",
            )
        return RegionalVerificationResult(
            status=RegionalStatus.PASS,
            judgment_confidence=0.90,
            reason="ROI is sharp and verified.",
        )

    def extract_regional_evidence(
        self,
        image_path: str,
        observation: dict[str, Any],
        active_hypothesis: dict[str, Any] | None,
    ) -> RegionalEvidenceFinding:
        self.call_history.append("extract_regional_evidence")
        tool_name = observation.get("tool", "")

        if tool_name in {"crop_region", "zoom_region", "rotate_image"}:
            return RegionalEvidenceFinding(
                finding=f"Acquired focused observation using {tool_name}.",
                relation=EvidenceRelation.NEUTRAL,
                observation_type="inspected_roi",
                confidence=0.85,
                metadata=observation.get("metadata", {}),
            )

        return RegionalEvidenceFinding(
            finding=f"Observed feature via {tool_name}.",
            relation=EvidenceRelation.NEUTRAL,
            observation_type="visual_feature",
            confidence=0.80,
            metadata=observation.get("metadata", {}),
        )

    def verify_evidence(
        self,
        evidence_state: EvidenceState,
        category: str,
        anomaly_threshold: float,
        normal_threshold: float,
    ) -> EvidenceVerificationResult:
        self.call_history.append("verify_evidence")
        score = evidence_state.anomaly_score
        if score >= anomaly_threshold:
            return EvidenceVerificationResult(
                status=EvidenceStatus.PASS,
                conclusion=EvidenceConclusion.ANOMALY,
                judgment_confidence=0.92,
                reason=f"Definitive visual target confirmed (score {score:.2f} >= {anomaly_threshold}).",
            )
        has_explicit_normal = any(
            e.observation_type in {"normal_reference_match", "reference_comparison", "verified_normal_feature"}
            and e.relation == EvidenceRelation.CONTRADICT
            for e in evidence_state.evidence_items
        )
        if (
            score <= normal_threshold
            and not evidence_state.unresolved_regions
            and len(evidence_state.evidence_items) >= 1
            and has_explicit_normal
        ):
            return EvidenceVerificationResult(
                status=EvidenceStatus.PASS,
                conclusion=EvidenceConclusion.NORMAL,
                judgment_confidence=0.85,
                reason=f"Definitive normal confirmed (score {score:.2f} <= {normal_threshold}).",
            )
        return EvidenceVerificationResult(
            status=EvidenceStatus.FAIL,
            conclusion=EvidenceConclusion.UNRESOLVED,
            judgment_confidence=0.65,
            reason="Evidence inconclusive, entering resolution phase.",
        )

    def diagnose_failure(
        self,
        verifier_judgment: Any,
        observation: dict[str, Any] | None,
        evidence_state: EvidenceState,
    ) -> FailureDiagnosisResult:
        self.call_history.append("diagnose_failure")
        reason = getattr(verifier_judgment, "reason", "").lower()
        if "resolution" in reason or "<12px" in reason or "small" in reason:
            return FailureDiagnosisResult(
                failure_type=FailureType.LOW_RESOLUTION,
                cause="Crop resolution too low.",
                diagnosis_confidence=0.90,
            )
        if len(evidence_state.normal_references) == 0:
            return FailureDiagnosisResult(
                failure_type=FailureType.MISSING_REFERENCE,
                cause="Requires comparison against normal standard.",
                diagnosis_confidence=0.90,
            )
        return FailureDiagnosisResult(
            failure_type=FailureType.LOCALIZATION_UNCERTAIN,
            cause="Candidate ROI requires relocalization.",
            diagnosis_confidence=0.80,
        )

    def direct_inspect(self, image_path: str, instruction: str, category: str) -> DirectPredictionResult:
        self.call_history.append("direct_inspect")
        return DirectPredictionResult(
            is_anomaly=False,
            conclusion="NORMAL",
            anomaly_score=0.10,
            confidence=0.85,
            explanation="Mock direct visual inspection result.",
        )

    def react_step(
        self,
        history: list[dict[str, Any]],
        image_path: str,
        enabled_tools: list[str],
        instruction: str = "Inspect this image using tools.",
        category: str = "visual_object",
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> ReactDecisionResult:
        self.call_history.append("react_step")
        if not history:
            return ReactDecisionResult(
                action=ReactAction.TOOL_CALL,
                tool_name="crop_region",
                tool_arguments={"bbox": [10, 10, 50, 50]},
            )
        elif len(history) == 1:
            return ReactDecisionResult(
                action=ReactAction.TOOL_CALL,
                tool_name="zoom_region",
                tool_arguments={"scale": 2.0},
            )
        return ReactDecisionResult(
            action=ReactAction.FINISH,
            final_answer="Inspection completed via ReAct tool loop.",
            is_anomaly=False,
            confidence=0.80,
        )

    def generic_verify_react(
        self,
        history: list[dict[str, Any]],
        decision: ReactDecisionResult,
        image_path: str,
        category: str,
    ) -> GenericVerificationResult:
        self.call_history.append("generic_verify_react")
        return GenericVerificationResult(
            passed=True,
            confidence=0.85,
            feedback="ReAct inspection trajectory verified by mock verifier.",
        )


class ScriptedHJLModelCaller:
    """Deterministic scripted model caller for end-to-end testing of specific control-flow trajectories."""

    def __init__(
        self,
        global_inspection: GlobalInspectionResult | None = None,
        hypothesis: HypothesisResult | None = None,
        regional_verification: RegionalVerificationResult | None = None,
        evidence_finding: RegionalEvidenceFinding | None = None,
        evidence_verification: EvidenceVerificationResult | None = None,
        failure_diagnosis: FailureDiagnosisResult | None = None,
        direct_prediction: DirectPredictionResult | None = None,
        react_decision: ReactDecisionResult | None = None,
        react_verification: GenericVerificationResult | None = None,
    ) -> None:
        self.scripted_global = global_inspection
        self.scripted_hypothesis = hypothesis
        self.scripted_regional = regional_verification
        self.scripted_evidence_finding = evidence_finding
        self.scripted_evidence_verification = evidence_verification
        self.scripted_failure_diagnosis = failure_diagnosis
        self.scripted_direct = direct_prediction
        self.scripted_react = react_decision
        self.scripted_react_verification = react_verification
        self.call_history: list[str] = []

    def inspect_global(self, image_path: str, instruction: str, category: str) -> GlobalInspectionResult:
        self.call_history.append("inspect_global")
        if self.scripted_global:
            return self.scripted_global
        return MockHJLModelCaller().inspect_global(image_path, instruction, category)

    def generate_hypothesis(
        self, image_path: str, candidate_region: list[int] | None, category: str
    ) -> HypothesisResult:
        self.call_history.append("generate_hypothesis")
        if self.scripted_hypothesis:
            return self.scripted_hypothesis
        return MockHJLModelCaller().generate_hypothesis(image_path, candidate_region, category)

    def verify_regional(self, observation: dict[str, Any], image_path: str) -> RegionalVerificationResult:
        self.call_history.append("verify_regional")
        if self.scripted_regional:
            return self.scripted_regional
        return MockHJLModelCaller().verify_regional(observation, image_path)

    def extract_regional_evidence(
        self,
        image_path: str,
        observation: dict[str, Any],
        active_hypothesis: dict[str, Any] | None,
    ) -> RegionalEvidenceFinding:
        self.call_history.append("extract_regional_evidence")
        if self.scripted_evidence_finding:
            return self.scripted_evidence_finding
        return MockHJLModelCaller().extract_regional_evidence(image_path, observation, active_hypothesis)

    def verify_evidence(
        self,
        evidence_state: EvidenceState,
        category: str,
        anomaly_threshold: float,
        normal_threshold: float,
    ) -> EvidenceVerificationResult:
        self.call_history.append("verify_evidence")
        if self.scripted_evidence_verification:
            return self.scripted_evidence_verification
        return MockHJLModelCaller().verify_evidence(evidence_state, category, anomaly_threshold, normal_threshold)

    def diagnose_failure(
        self,
        verifier_judgment: Any,
        observation: dict[str, Any] | None,
        evidence_state: EvidenceState,
    ) -> FailureDiagnosisResult:
        self.call_history.append("diagnose_failure")
        if self.scripted_failure_diagnosis:
            return self.scripted_failure_diagnosis
        return MockHJLModelCaller().diagnose_failure(verifier_judgment, observation, evidence_state)

    def direct_inspect(self, image_path: str, instruction: str, category: str) -> DirectPredictionResult:
        self.call_history.append("direct_inspect")
        if self.scripted_direct:
            return self.scripted_direct
        return MockHJLModelCaller().direct_inspect(image_path, instruction, category)

    def react_step(
        self,
        history: list[dict[str, Any]],
        image_path: str,
        enabled_tools: list[str],
        instruction: str = "Inspect this image using tools.",
        category: str = "visual_object",
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> ReactDecisionResult:
        self.call_history.append("react_step")
        if self.scripted_react:
            return self.scripted_react
        return MockHJLModelCaller().react_step(
            history, image_path, enabled_tools, instruction, category, tool_definitions
        )

    def generic_verify_react(
        self,
        history: list[dict[str, Any]],
        decision: ReactDecisionResult,
        image_path: str,
        category: str,
    ) -> GenericVerificationResult:
        self.call_history.append("generic_verify_react")
        if self.scripted_react_verification:
            return self.scripted_react_verification
        return MockHJLModelCaller().generic_verify_react(history, decision, image_path, category)
