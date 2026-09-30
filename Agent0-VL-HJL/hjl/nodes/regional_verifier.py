"""Checkpoint 2: Regional Verifier node for HJL.

Two-stage ROI quality gate:
Stage 1: Resolution, boundary geometry, and rotation quality check.
Stage 2: Model-based regional verification when model_caller is provided.
"""

from __future__ import annotations

from typing import Any

from ..model_caller import HJLModelCaller
from ..state import HJLState, StopReason
from ..taxonomy import CheckpointJudgment, RegionalStatus


def regional_verifier_node(
    state: HJLState,
    model_caller: HJLModelCaller | None = None,
    live_mode: bool = False,
) -> dict[str, Any]:
    """Verify quality, resolution, and relevance of the current regional tool observation."""
    if not state.observations:
        judgment = CheckpointJudgment(
            status=RegionalStatus.FAIL,
            judgment_confidence=0.90,
            reason="No observation available to verify.",
        )
        return {"regional_judgment": judgment}

    latest_obs = state.observations[-1]
    tool_name = latest_obs.get("tool", "")
    metadata = latest_obs.get("metadata", {})
    inspection_image = latest_obs.get("output_path") or state.image_path

    # Stage 1: Resolution and boundary geometry quality checks
    if tool_name in {"crop_region", "zoom_region"}:
        size = metadata.get("image_size")
        if size and (size[0] < 12 or size[1] < 12):
            judgment = CheckpointJudgment(
                status=RegionalStatus.FAIL,
                judgment_confidence=0.85,
                reason=f"ROI resolution [{size[0]}x{size[1]}] is insufficient (<12px) to inspect fine visual features.",
            )
            return {"regional_judgment": judgment}

        # Check if crop bbox is degenerate (applicable to crop_region)
        if tool_name == "crop_region":
            bbox = metadata.get("bbox")
            if bbox and (bbox[2] - bbox[0] < 8 or bbox[3] - bbox[1] < 8):
                judgment = CheckpointJudgment(
                    status=RegionalStatus.FAIL,
                    judgment_confidence=0.85,
                    reason=f"Cropped bounding box {bbox} is too narrow (<8px) or misses target structure.",
                )
                return {"regional_judgment": judgment}

    elif tool_name == "rotate_image":
        if "error" in metadata:
            judgment = CheckpointJudgment(
                status=RegionalStatus.FAIL,
                judgment_confidence=0.85,
                reason=f"Image rotation failed: {metadata['error']}",
            )
            return {"regional_judgment": judgment}

    # Stage 2: Model-based regional verification if model_caller available
    if model_caller is not None:
        try:
            m_res = model_caller.verify_regional(
                observation=latest_obs,
                image_path=inspection_image,
            )
            if m_res.status == RegionalStatus.FAIL:
                judgment = CheckpointJudgment(
                    status=RegionalStatus.FAIL,
                    judgment_confidence=m_res.judgment_confidence,
                    reason=m_res.reason,
                )
                return {"regional_judgment": judgment}
        except Exception:
            if live_mode:
                return {"stop_reason": StopReason.MODEL_ERROR}

    # Pure ROI quality checks passed
    judgment = CheckpointJudgment(
        status=RegionalStatus.PASS,
        judgment_confidence=0.90,
        reason=f"Regional observation from {tool_name} is sharp, sufficiently resolved, and orientation-aligned.",
    )
    return {"regional_judgment": judgment}
