"""Anomaly hypothesis generation node for HJL."""

from __future__ import annotations

from typing import Any

from ..state import HJLState, HJLPhase


def hypothesis_generator_node(state: HJLState) -> dict[str, Any]:
    """Formulate initial defect hypotheses based on global observation."""
    hypotheses = list(state.hypotheses)

    if not hypotheses:
        category = state.category or "industrial_component"
        hypotheses.append({
            "hypothesis_id": "hyp_surface_defect_01",
            "type": "surface_abnormality",
            "description": f"Potential surface defect, scratch, crack, or contamination on {category}.",
            "confidence": 0.70,
            "target_region": state.candidate_regions[0]["bbox"] if state.candidate_regions else None,
        })

    active = hypotheses[0]
    state.evidence_state.hypotheses = copy_hypotheses = list(hypotheses)

    return {
        "hypotheses": copy_hypotheses,
        "active_hypothesis": active,
        "phase": HJLPhase.HYPOTHESIS_INSPECTION,
    }
