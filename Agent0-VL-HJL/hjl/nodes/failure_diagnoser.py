"""Failure Diagnoser node for HJL.

Diagnoses why the current inspection or evidence chain is insufficient,
classifying into one of 7 semantic failure types and deriving the action mask.
"""

from __future__ import annotations

from typing import Any

from ..routing import FailureRoutingPolicy
from ..state import HJLPhase, HJLState
from ..taxonomy import (
    FailureDiagnosis,
    FailureType,
    RegionalStatus,
)


def failure_diagnoser_node(state: HJLState) -> dict[str, Any]:
    """Diagnose root cause of Regional or Evidence Checkpoint failure."""
    regional_fail = (
        state.regional_judgment is not None
        and state.regional_judgment.status == RegionalStatus.FAIL
    )
    evidence_fail = (
        state.evidence_judgment is not None
        and state.evidence_judgment.status == RegionalStatus.FAIL
    )

    failure_type = FailureType.LOCALIZATION_UNCERTAIN
    cause = "Inspection requires candidate localization."
    conf = 0.85

    # 1. Diagnosis from Regional Checkpoint Failure
    if regional_fail:
        reason = state.regional_judgment.reason.lower()
        if "resolution" in reason or "small" in reason:
            failure_type = FailureType.LOW_RESOLUTION
            cause = "Current crop resolution is too low to inspect fine structural anomalies."
        elif "narrow" in reason or "misses" in reason or "background" in reason:
            failure_type = FailureType.WRONG_REGION
            cause = "Selected bounding box did not capture the target surface feature."
        elif "angle" in reason or "orientation" in reason:
            failure_type = FailureType.VIEWPOINT_MISMATCH
            cause = "Component perspective or orientation is misaligned."
        else:
            failure_type = FailureType.WRONG_REGION
            cause = state.regional_judgment.reason

    # 2. Diagnosis from Evidence Checkpoint Failure
    elif evidence_fail:
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

    diagnosis = FailureDiagnosis(
        failure_type=failure_type,
        cause=cause,
        diagnosis_confidence=conf,
    )

    # Derive action mask from FailureRoutingPolicy
    allowed_actions = FailureRoutingPolicy.get_allowed_actions(failure_type)

    return {
        "failure_type": failure_type,
        "failure_reason": cause,
        "allowed_actions": allowed_actions,
    }
