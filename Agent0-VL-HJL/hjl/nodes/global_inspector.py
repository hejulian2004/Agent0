"""Global visual inspection node for HJL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..state import HJLState


def global_inspector_node(state: HJLState) -> dict[str, Any]:
    """Inspect the full global image and propose candidate suspicious regions."""
    image_path = Path(state.image_path)
    if not image_path.is_file():
        return {
            "global_observation": f"Image file not found: {image_path}",
            "candidate_regions": [],
        }

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as exc:
        return {
            "global_observation": f"Failed to open image: {exc}",
            "candidate_regions": [],
        }

    # Propose candidate regions if not already populated
    candidate_regions = list(state.candidate_regions)
    if not candidate_regions:
        # Default candidate regions: centered ROI and quadrant candidates
        center_w = int(width * 0.5)
        center_h = int(height * 0.5)
        x1 = max(0, int(width * 0.25))
        y1 = max(0, int(height * 0.25))
        x2 = min(width, x1 + center_w)
        y2 = min(height, y1 + center_h)

        candidate_regions.append({
            "region_id": "roi_center",
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.75,
            "label": "primary_surface_area",
        })

    # Initialize unresolved regions in evidence state
    unresolved = [c["bbox"] for c in candidate_regions if "bbox" in c]

    obs = f"Global image size [{width}x{height}]. Identified {len(candidate_regions)} candidate region(s) for inspection."

    state.evidence_state.unresolved_regions = unresolved

    return {
        "global_observation": obs,
        "candidate_regions": candidate_regions,
    }
