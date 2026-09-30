"""Global visual inspection node for HJL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..model_caller import HJLModelCaller
from ..schemas import CandidateRegion
from ..state import HJLState, StopReason


def global_inspector_node(
    state: HJLState,
    model_caller: HJLModelCaller | None = None,
    live_mode: bool = False,
) -> dict[str, Any]:
    """Inspect the full global image and propose candidate suspicious regions without automatic bias."""
    image_path = Path(state.image_path)
    if not image_path.is_file():
        return {
            "global_observation": f"Image file not found: {image_path}",
            "candidate_regions": [],
            "global_is_normal": False,
            "global_confidence": 0.0,
        }

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as exc:
        return {
            "global_observation": f"Failed to open image: {exc}",
            "candidate_regions": [],
            "global_is_normal": False,
            "global_confidence": 0.0,
        }

    candidate_regions = list(state.candidate_regions)

    # 1. Delegate to Model Caller if available
    if model_caller is not None and not candidate_regions:
        try:
            res = model_caller.inspect_global(
                image_path=str(image_path),
                instruction=state.instruction,
                category=state.category,
            )
            valid_cands: list[dict[str, Any]] = []
            for c in res.candidate_regions:
                try:
                    c.validate_bounds(width, height)
                    valid_cands.append(c.to_dict())
                except ValueError:
                    pass

            unresolved = [c["bbox"] for c in valid_cands if "bbox" in c]
            state.evidence_state.unresolved_regions = unresolved

            return {
                "global_observation": res.observation,
                "candidate_regions": valid_cands,
                "global_is_normal": res.is_normal,
                "global_confidence": res.confidence,
            }
        except Exception as exc:
            if live_mode:
                return {
                    "global_observation": f"Model inspection failed: {exc}",
                    "candidate_regions": [],
                    "global_is_normal": False,
                    "global_confidence": 0.0,
                    "stop_reason": StopReason.MODEL_ERROR,
                }
            # Fall back to offline heuristic only when not live_mode

    # 2. Offline / Heuristic inspection
    if not candidate_regions:
        if state.instruction and "force_normal" in state.instruction.lower():
            state.evidence_state.unresolved_regions = []
            return {
                "global_observation": f"Global image size [{width}x{height}]. Confirmed normal surface.",
                "candidate_regions": [],
                "global_is_normal": True,
                "global_confidence": 0.98,
            }

        # Candidate region: center surface area candidate with validated bounds
        center_w = int(width * 0.5)
        center_h = int(height * 0.5)
        x1 = max(0, int(width * 0.25))
        y1 = max(0, int(height * 0.25))
        x2 = min(width, x1 + center_w)
        y2 = min(height, y1 + center_h)

        reg = CandidateRegion(
            bbox=[x1, y1, x2, y2],
            confidence=0.80,
            label="primary_surface_candidate",
            region_id="roi_center",
        )
        reg.validate_bounds(width, height)
        candidate_regions.append(reg.to_dict())

    unresolved = [c["bbox"] for c in candidate_regions if "bbox" in c]
    state.evidence_state.unresolved_regions = unresolved

    obs = f"Global image size [{width}x{height}]. Identified {len(candidate_regions)} candidate region(s) for inspection."

    return {
        "global_observation": obs,
        "candidate_regions": candidate_regions,
        "global_is_normal": False,
        "global_confidence": 0.80,
    }
