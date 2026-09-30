"""Specialized state updater nodes for tool-type routing in HJL."""

from __future__ import annotations

import copy
from typing import Any

from ..schemas import RegionalEvidenceFinding
from ..state import EvidenceRelation, HJLPhase, HJLState, StopReason
from ..taxonomy import ActionType
from ..tools_adapter import validate_reference_metadata


def reference_state_updater_node(state: HJLState) -> dict[str, Any]:
    """Extract and validate retrieved train-normal reference, transitioning to CROSS_VALIDATE."""
    if not state.observations:
        return {}

    latest_obs = state.observations[-1]
    metadata = latest_obs.get("metadata", {})

    # Validate against dataset leakage (split == 'train' and is_normal == True)
    validate_reference_metadata(metadata)

    evidence_state = state.evidence_state
    evidence_state.normal_references.append(copy.deepcopy(metadata))

    # Deterministic transition to CROSS_VALIDATE
    return {
        "evidence_state": evidence_state,
        "selected_action": ActionType.CROSS_VALIDATE,
        "allowed_actions": [ActionType.CROSS_VALIDATE],
    }


def comparison_evidence_extractor_node(
    state: HJLState,
    reference_similarity_threshold: float = 0.80,
) -> dict[str, Any]:
    """Extract evidence finding from reference comparison metrics."""
    if not state.observations:
        return {}

    latest_obs = state.observations[-1]
    metadata = latest_obs.get("metadata", {})
    similarity = float(metadata.get("similarity", 1.0))

    if similarity < reference_similarity_threshold:
        relation = EvidenceRelation.SUPPORT
        confidence = round(max(0.80, 1.0 - similarity), 2)
        statement = f"Local appearance significantly deviates from reference (similarity {similarity:.2f} < {reference_similarity_threshold:.2f})."
    else:
        relation = EvidenceRelation.CONTRADICT
        confidence = round(max(0.80, similarity), 2)
        statement = f"Local appearance conforms to normal reference (similarity {similarity:.2f} >= {reference_similarity_threshold:.2f})."

    finding = RegionalEvidenceFinding(
        finding=statement,
        relation=relation,
        observation_type="reference_comparison",
        confidence=confidence,
        metadata=dict(metadata),
    )

    return {"extracted_finding": finding}


def candidate_state_updater_node(
    state: HJLState,
    max_discovery_attempts: int = 2,
) -> dict[str, Any]:
    """Update candidate regions from localize_candidate and bound discovery retries."""
    if not state.observations:
        return {}

    latest_obs = state.observations[-1]
    metadata = latest_obs.get("metadata", {})
    candidates = metadata.get("candidate_regions", [])

    candidate_regions = list(state.candidate_regions)
    discovery_attempts = state.discovery_attempts

    if candidates:
        candidate_regions = list(candidates)
        state.evidence_state.unresolved_regions = [c["bbox"] for c in candidates if "bbox" in c]
        discovery_attempts = 0
        return {
            "candidate_regions": candidate_regions,
            "discovery_attempts": discovery_attempts,
            "active_hypothesis": None,
            "hypotheses": [],
            "phase": HJLPhase.HYPOTHESIS_INSPECTION,
        }

    # If 0 candidates returned, bound discovery attempts
    discovery_attempts += 1
    if discovery_attempts >= max_discovery_attempts:
        return {
            "discovery_attempts": discovery_attempts,
            "stop_reason": StopReason.NO_VALID_ACTION,
        }

    return {
        "discovery_attempts": discovery_attempts,
    }
