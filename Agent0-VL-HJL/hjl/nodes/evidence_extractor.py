"""Evidence Extractor node for HJL.

Extracts structured visual findings (RegionalEvidenceFinding) from a verified ROI
strictly AFTER RegionalVerifier yields PASS.
"""

from __future__ import annotations

from typing import Any

from ..model_caller import HJLModelCaller, RegionalEvidenceFinding
from ..state import EvidenceRelation, HJLState


def evidence_extractor_node(
    state: HJLState,
    model_caller: HJLModelCaller | None = None,
) -> dict[str, Any]:
    """Extract factual visual finding from the latest verified regional observation."""
    if not state.observations:
        return {}

    latest_obs = state.observations[-1]
    tool_name = latest_obs.get("tool", "")
    metadata = latest_obs.get("metadata", {})

    # If model caller is available, request model extraction
    if model_caller is not None:
        try:
            finding = model_caller.extract_regional_evidence(
                image_path=state.image_path,
                observation=latest_obs,
                active_hypothesis=state.active_hypothesis,
            )
            return {"extracted_finding": finding}
        except Exception:
            pass  # Fall back to deterministic extraction

    # Spatial transformation operations are NEUTRAL by default
    if tool_name in {"crop_region", "zoom_region", "rotate_image"}:
        finding = RegionalEvidenceFinding(
            finding=f"Acquired focused observation of ROI {metadata.get('bbox')} via {tool_name}.",
            relation=EvidenceRelation.NEUTRAL,
            observation_type="inspected_roi",
            confidence=0.85,
            metadata=metadata,
        )
    elif tool_name == "localize_candidate":
        candidates = metadata.get("candidate_regions", [])
        finding = RegionalEvidenceFinding(
            finding=f"Localized {len(candidates)} candidate region(s).",
            relation=EvidenceRelation.NEUTRAL,
            observation_type="candidate_localization",
            confidence=0.80,
            metadata=metadata,
        )
    else:
        finding = RegionalEvidenceFinding(
            finding=f"Observed feature via {tool_name}.",
            relation=EvidenceRelation.NEUTRAL,
            observation_type="visual_feature",
            confidence=0.80,
            metadata=metadata,
        )

    return {"extracted_finding": finding}
