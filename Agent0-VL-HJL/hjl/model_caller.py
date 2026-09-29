"""Strictly typed model caller schemas, protocol, and implementations for HJL."""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from PIL import Image

from .state import EvidenceRelation, EvidenceState
from .taxonomy import (
    EvidenceConclusion,
    EvidenceStatus,
    FailureType,
    RegionalStatus,
)

logger = logging.getLogger(__name__)


class ModelOutputError(RuntimeError):
    """Raised when model response cannot be parsed or violates schema constraints."""


class ReactAction(str, Enum):
    TOOL_CALL = "TOOL_CALL"
    FINISH = "FINISH"


@dataclass
class CandidateRegion:
    bbox: list[int]
    confidence: float
    label: str | None = None
    region_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have length 4, got {self.bbox}")
        if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            raise ValueError(f"Invalid bbox dimensions [x1,y1,x2,y2]: {self.bbox}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def validate_bounds(self, width: int, height: int) -> None:
        """Validate bounding box falls within image boundaries."""
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(f"bbox {self.bbox} exceeds image dimensions [{width}x{height}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "label": self.label,
            "region_id": self.region_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateRegion":
        return cls(
            bbox=list(data["bbox"]),
            confidence=float(data.get("confidence", 0.8)),
            label=data.get("label"),
            region_id=data.get("region_id"),
        )


@dataclass
class GlobalInspectionResult:
    observation: str
    candidate_regions: list[CandidateRegion]
    is_normal: bool
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class HypothesisResult:
    hypothesis_id: str
    type: str
    description: str
    confidence: float
    target_region: list[int] | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class RegionalVerificationResult:
    status: RegionalStatus
    judgment_confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.judgment_confidence <= 1.0):
            raise ValueError(f"judgment_confidence must be in [0, 1], got {self.judgment_confidence}")


@dataclass
class RegionalEvidenceFinding:
    finding: str
    relation: EvidenceRelation
    observation_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class EvidenceVerificationResult:
    status: EvidenceStatus
    conclusion: EvidenceConclusion
    judgment_confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.judgment_confidence <= 1.0):
            raise ValueError(f"judgment_confidence must be in [0, 1], got {self.judgment_confidence}")
        if self.status == EvidenceStatus.PASS and self.conclusion == EvidenceConclusion.UNRESOLVED:
            raise ValueError("EvidenceStatus.PASS cannot have conclusion UNRESOLVED")
        if self.status == EvidenceStatus.FAIL and self.conclusion != EvidenceConclusion.UNRESOLVED:
            raise ValueError(f"EvidenceStatus.FAIL must have conclusion UNRESOLVED, got {self.conclusion}")


@dataclass
class FailureDiagnosisResult:
    failure_type: FailureType
    cause: str
    diagnosis_confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.diagnosis_confidence <= 1.0):
            raise ValueError(f"diagnosis_confidence must be in [0, 1], got {self.diagnosis_confidence}")


@dataclass
class DirectPredictionResult:
    is_anomaly: bool
    conclusion: str
    anomaly_score: float
    confidence: float
    explanation: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.anomaly_score <= 1.0):
            raise ValueError(f"anomaly_score must be in [0, 1], got {self.anomaly_score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class ReactDecisionResult:
    action: ReactAction
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    final_answer: str | None = None
    is_anomaly: bool | None = None
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.action == ReactAction.TOOL_CALL and not self.tool_name:
            raise ValueError("ReactAction.TOOL_CALL requires tool_name")
        if self.action == ReactAction.FINISH and self.tool_name:
            raise ValueError("ReactAction.FINISH must not specify tool_name")


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
    ) -> ReactDecisionResult: ...


def _image_to_base64_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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

    def _call_json(self, prompt: str, image_path: str | None = None) -> dict[str, Any]:
        """Perform a single-turn Responses call requesting JSON output."""
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if image_path:
            content.append({
                "type": "input_image",
                "image_url": _image_to_base64_url(image_path),
            })

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=1024,
        )

        text = ""
        for item in getattr(response, "output", []):
            if isinstance(item, dict) and item.get("type") == "message":
                msg_content = item.get("content", "")
                if isinstance(msg_content, str):
                    text += msg_content
                elif isinstance(msg_content, list):
                    text += "".join(str(p.get("text", "")) for p in msg_content if isinstance(p, dict))

        text = text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            return json.loads(text)
        except Exception as exc:
            raise ModelOutputError(f"Failed to parse JSON model output: {exc}. Raw text: {text[:200]}") from exc

    def inspect_global(self, image_path: str, instruction: str, category: str) -> GlobalInspectionResult:
        prompt = (
            f"You are an industrial visual anomaly inspector. Inspect this {category} image.\n"
            f"Instruction: {instruction}\n"
            "Return JSON with keys:\n"
            "- observation: text description of overall visual surface\n"
            "- is_normal: boolean (true if completely defect-free, false if suspicious)\n"
            "- confidence: float between 0.0 and 1.0\n"
            "- candidate_regions: list of objects with [x1, y1, x2, y2] bbox and confidence"
        )
        data = self._call_json(prompt, image_path)
        candidates = [
            CandidateRegion(
                bbox=c["bbox"],
                confidence=float(c.get("confidence", 0.8)),
                label=c.get("label", "suspicious_region"),
            )
            for c in data.get("candidate_regions", [])
        ]
        return GlobalInspectionResult(
            observation=data.get("observation", "Global visual inspection."),
            candidate_regions=candidates,
            is_normal=bool(data.get("is_normal", False)),
            confidence=float(data.get("confidence", 0.8)),
        )

    def generate_hypothesis(
        self, image_path: str, candidate_region: list[int] | None, category: str
    ) -> HypothesisResult:
        prompt = (
            f"Formulate a defect hypothesis for {category} at ROI {candidate_region}.\n"
            "Return JSON with: hypothesis_id, type, description, confidence (0.0-1.0)."
        )
        data = self._call_json(prompt, image_path)
        return HypothesisResult(
            hypothesis_id=data.get("hypothesis_id", "hyp_01"),
            type=data.get("type", "surface_abnormality"),
            description=data.get("description", "Potential defect observed."),
            confidence=float(data.get("confidence", 0.75)),
            target_region=candidate_region,
        )

    def verify_regional(self, observation: dict[str, Any], image_path: str) -> RegionalVerificationResult:
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
            reason="ROI is sharp and feature-relevant.",
        )

    def extract_regional_evidence(
        self,
        image_path: str,
        observation: dict[str, Any],
        active_hypothesis: dict[str, Any] | None,
    ) -> RegionalEvidenceFinding:
        tool_name = observation.get("tool", "")
        # Spatial operations default to neutral
        if tool_name in {"crop_region", "zoom_region", "rotate_image"}:
            return RegionalEvidenceFinding(
                finding=f"Acquired high-clarity inspection crop via {tool_name}.",
                relation=EvidenceRelation.NEUTRAL,
                observation_type="inspected_roi",
                confidence=0.85,
                metadata=observation.get("metadata", {}),
            )

        prompt = (
            f"Analyze this cropped region in regard to hypothesis: {active_hypothesis}.\n"
            "Return JSON with: finding (string), relation ('SUPPORT', 'CONTRADICT', 'NEUTRAL'), "
            "observation_type (string), confidence (0.0-1.0)."
        )
        data = self._call_json(prompt, image_path)
        rel = EvidenceRelation(data.get("relation", "NEUTRAL"))
        return RegionalEvidenceFinding(
            finding=data.get("finding", "Observed regional visual structure."),
            relation=rel,
            observation_type=data.get("observation_type", "regional_feature"),
            confidence=float(data.get("confidence", 0.8)),
            metadata=observation.get("metadata", {}),
        )

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
                reason=f"Accumulated evidence confirms anomaly (score {score:.2f} >= {anomaly_threshold}).",
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
        prompt = (
            f"Inspection checkpoint failed with reason: {getattr(verifier_judgment, 'reason', '')}.\n"
            "Classify root cause strictly into one of: WRONG_REGION, LOW_RESOLUTION, MISSING_REFERENCE, "
            "VIEWPOINT_MISMATCH, CONTRADICTORY_EVIDENCE, LOCALIZATION_UNCERTAIN, PREMATURE_CONCLUSION.\n"
            "Return JSON: failure_type, cause, diagnosis_confidence (0.0-1.0)."
        )
        data = self._call_json(prompt)
        ft = FailureType(data.get("failure_type", "LOCALIZATION_UNCERTAIN"))
        return FailureDiagnosisResult(
            failure_type=ft,
            cause=data.get("cause", "Diagnosis from model."),
            diagnosis_confidence=float(data.get("diagnosis_confidence", 0.85)),
        )

    def direct_inspect(self, image_path: str, instruction: str, category: str) -> DirectPredictionResult:
        prompt = (
            f"Inspect this {category} image directly.\n"
            f"Instruction: {instruction}\n"
            "Return JSON: is_anomaly (bool), conclusion ('ANOMALY'|'NORMAL'), anomaly_score (0.0-1.0), "
            "confidence (0.0-1.0), explanation (str)."
        )
        data = self._call_json(prompt, image_path)
        return DirectPredictionResult(
            is_anomaly=bool(data.get("is_anomaly", False)),
            conclusion=str(data.get("conclusion", "NORMAL")),
            anomaly_score=float(data.get("anomaly_score", 0.1)),
            confidence=float(data.get("confidence", 0.85)),
            explanation=str(data.get("explanation", "Direct single-pass visual assessment.")),
        )

    def react_step(
        self, history: list[dict[str, Any]], image_path: str, enabled_tools: list[str]
    ) -> ReactDecisionResult:
        prompt = (
            f"You are a ReAct agent inspecting an industrial component.\n"
            f"Enabled tools: {enabled_tools}\n"
            f"History: {json.dumps(history[-4:] if history else [])}\n"
            "Decide next step. Return JSON with:\n"
            "- action: 'TOOL_CALL' or 'FINISH'\n"
            "- tool_name: tool name if TOOL_CALL\n"
            "- tool_arguments: dict of arguments if TOOL_CALL\n"
            "- final_answer: str if FINISH\n"
            "- is_anomaly: bool if FINISH\n"
            "- confidence: float (0.0-1.0)"
        )
        data = self._call_json(prompt, image_path)
        act_str = data.get("action", "FINISH")
        act = ReactAction(act_str)
        return ReactDecisionResult(
            action=act,
            tool_name=data.get("tool_name"),
            tool_arguments=data.get("tool_arguments"),
            final_answer=data.get("final_answer"),
            is_anomaly=data.get("is_anomaly"),
            confidence=float(data.get("confidence", 0.8)),
        )


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

        # In mock mode, propose candidate region unless specifically instructed normal
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
                label="primary_surface_defect_candidate",
                region_id="cand_0",
            )
        ]
        return GlobalInspectionResult(
            observation=f"Identified candidate anomaly region on [{w}x{h}] surface.",
            candidate_regions=cands,
            is_normal=False,
            confidence=0.80,
        )

    def generate_hypothesis(
        self, image_path: str, candidate_region: list[int] | None, category: str
    ) -> HypothesisResult:
        self.call_history.append("generate_hypothesis")
        return HypothesisResult(
            hypothesis_id="hyp_mock_crack_01",
            type="crack",
            description=f"Suspected crack on {category} surface.",
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

        # Pure spatial crops and zooms are NEUTRAL transformations
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
                reason=f"Definitive anomaly confirmed (score {score:.2f} >= {anomaly_threshold}).",
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
        self, history: list[dict[str, Any]], image_path: str, enabled_tools: list[str]
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
