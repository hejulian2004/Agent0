"""Anomaly hypothesis generation node for HJL."""

from __future__ import annotations

from typing import Any

from ..model_caller import HJLModelCaller
from ..state import HJLPhase, HJLState, StopReason


def hypothesis_generator_node(
    state: HJLState,
    model_caller: HJLModelCaller | None = None,
    live_mode: bool = False,
) -> dict[str, Any]:
    """Formulate initial visual hypotheses based on global observation and candidate regions."""
    hypotheses = list(state.hypotheses)

    if not hypotheses:
        category = state.category or "visual_object"
        target_bbox = state.candidate_regions[0]["bbox"] if state.candidate_regions else None

        if model_caller is not None:
            try:
                res = model_caller.generate_hypothesis(
                    image_path=state.image_path,
                    candidate_region=target_bbox,
                    category=category,
                )
                hypotheses.append(res.to_dict())
            except Exception:
                if live_mode:
                    return {"stop_reason": StopReason.MODEL_ERROR}
                # Offline fallback

        if not hypotheses:
            hypotheses.append({
                "hypothesis_id": "hyp_visual_feature_01",
                "type": "visual_feature",
                "description": f"Target visual feature or pattern on {category}.",
                "confidence": 0.70,
                "target_region": target_bbox,
            })

    active = hypotheses[0]
    state.evidence_state.hypotheses = copy_hypotheses = list(hypotheses)

    return {
        "hypotheses": copy_hypotheses,
        "active_hypothesis": active,
        "phase": HJLPhase.HYPOTHESIS_INSPECTION,
    }
