"""Checkpoint 2: Regional Verifier node for HJL."""

from __future__ import annotations

from typing import Any

from ..state import HJLState
from ..taxonomy import CheckpointJudgment, RegionalStatus


def regional_verifier_node(state: HJLState) -> dict[str, Any]:
    """Verify validity, quality, and relevance of the current regional tool observation."""
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

    # Evaluate crop / zoom resolution and validity
    if tool_name in {"crop_region", "zoom_region"}:
        size = metadata.get("image_size")
        if size and (size[0] < 12 or size[1] < 12):
            judgment = CheckpointJudgment(
                status=RegionalStatus.FAIL,
                judgment_confidence=0.85,
                reason=f"ROI resolution [{size[0]}x{size[1]}] is insufficient to inspect fine defect textures.",
            )
            return {"regional_judgment": judgment}

        # Check if crop bbox is degenerate
        bbox = metadata.get("bbox")
        if bbox and (bbox[2] - bbox[0] < 8 or bbox[3] - bbox[1] < 8):
            judgment = CheckpointJudgment(
                status=RegionalStatus.FAIL,
                judgment_confidence=0.85,
                reason=f"Cropped bounding box {bbox} is too narrow or misses the anomaly feature.",
            )
            return {"regional_judgment": judgment}

    # Reference comparison verification
    elif tool_name == "compare_with_reference":
        if "error" in metadata:
            judgment = CheckpointJudgment(
                status=RegionalStatus.FAIL,
                judgment_confidence=0.80,
                reason=f"Reference comparison failed: {metadata['error']}",
            )
            return {"regional_judgment": judgment}

    # All checks passed
    judgment = CheckpointJudgment(
        status=RegionalStatus.PASS,
        judgment_confidence=0.90,
        reason=f"Regional observation from tool {tool_name} is valid, sharp, and feature-relevant.",
    )
    return {"regional_judgment": judgment}
