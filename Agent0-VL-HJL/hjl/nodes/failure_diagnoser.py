"""Failure Diagnoser node for HJL.

Diagnoses root cause of Regional or Evidence Checkpoint failure,
classifying into one of 7 semantic failure types and deriving the action mask.
"""

from __future__ import annotations

from typing import Any

from ..model_caller import HJLModelCaller
from ..routing import FailureRoutingPolicy
from ..schemas import FailureDiagnosisResult
from ..state import HJLState, StopReason
from ..taxonomy import (
    EvidenceStatus,
    FailureDiagnosis,
    FailureType,
    RegionalStatus,
)


class RuleBasedFailureDiagnoser:
    """Deterministic rule-based attribution mapping verifier reasons to the 7 semantic failure types."""

    @staticmethod
    def diagnose(state: HJLState) -> FailureDiagnosisResult:
        regional_fail = (
            state.regional_judgment is not None
            and state.regional_judgment.status == RegionalStatus.FAIL
        )
        evidence_fail = (
            state.evidence_judgment is not None
            and state.evidence_judgment.status == EvidenceStatus.FAIL
        )

        failure_type = FailureType.LOCALIZATION_UNCERTAIN
        cause = "Inspection requires candidate localization."
        conf = 0.85

        # 1. Attribution from Regional Checkpoint Failure
        if regional_fail and state.regional_judgment:
            reason = state.regional_judgment.reason.lower()
            if "resolution" in reason or "small" in reason or "<12px" in reason:
                failure_type = FailureType.LOW_RESOLUTION
                cause = "Current crop resolution is too low (<12px) to inspect fine structural anomalies."
            elif "narrow" in reason or "misses" in reason or "background" in reason:
                failure_type = FailureType.WRONG_REGION
                cause = "Selected bounding box did not capture the target surface feature."
            elif "angle" in reason or "orientation" in reason or "rotation" in reason:
                failure_type = FailureType.VIEWPOINT_MISMATCH
                cause = "Component perspective or orientation is misaligned."
            else:
                failure_type = FailureType.WRONG_REGION
                cause = state.regional_judgment.reason

        # 2. Attribution from Evidence Checkpoint Failure
        elif evidence_fail and state.evidence_judgment:
            ev = state.evidence_state
            if len(ev.supporting_evidence) > 0 and len(ev.contradicting_evidence) > 0:
                failure_type = FailureType.CONTRADICTORY_EVIDENCE
                cause = "Observations are contradictory; requires reference cross-validation or alternative ROI."
            elif len(ev.normal_references) == 0:
                failure_type = FailureType.MISSING_REFERENCE
                cause = "Cannot verify anomaly without comparing against a standard normal template."
            elif ev.unresolved_regions:
                failure_type = FailureType.PREMATURE_CONCLUSION
                cause = f"{len(ev.unresolved_regions)} candidate region(s) still remain uninspected."
            else:
                failure_type = FailureType.LOCALIZATION_UNCERTAIN
                cause = "Evidence is inconclusive; requires precise candidate relocalization."

        return FailureDiagnosisResult(
            failure_type=failure_type,
            cause=cause,
            diagnosis_confidence=conf,
        )


class VLMFailureDiagnoser:
    """Semantic failure diagnoser querying the VLM model caller."""

    @staticmethod
    def diagnose(
        state: HJLState,
        model_caller: HJLModelCaller,
    ) -> FailureDiagnosisResult:
        verifier_judgment = state.regional_judgment or state.evidence_judgment
        observation = state.observations[-1] if state.observations else None
        return model_caller.diagnose_failure(
            verifier_judgment=verifier_judgment,
            observation=observation,
            evidence_state=state.evidence_state,
        )


def failure_diagnoser_node(
    state: HJLState,
    model_caller: HJLModelCaller | None = None,
    live_mode: bool = False,
) -> dict[str, Any]:
    """Diagnose root cause of checkpoint failure and set allowed action mask."""
    if model_caller is not None:
        try:
            diag_result = VLMFailureDiagnoser.diagnose(state, model_caller)
        except Exception:
            if live_mode:
                return {"stop_reason": StopReason.MODEL_ERROR}
            diag_result = RuleBasedFailureDiagnoser.diagnose(state)
    else:
        diag_result = RuleBasedFailureDiagnoser.diagnose(state)

    failure_type = diag_result.failure_type
    cause = diag_result.cause

    allowed_actions = FailureRoutingPolicy.get_allowed_actions(failure_type)

    return {
        "failure_type": failure_type,
        "failure_reason": cause,
        "allowed_actions": allowed_actions,
    }
