"""Evidence Updater node for HJL.

Accumulates, deduplicates, and fuses valid observations into persistent EvidenceState.
CRITICAL INVARIANT: Runs strictly after RegionalVerifier yields PASS to avoid evidence pollution.
"""

from __future__ import annotations

from typing import Any

from ..state import EvidenceItem, EvidenceRelation, HJLState


def _compute_iou(box_a: list[int] | None, box_b: list[int] | None) -> float:
    """Compute Intersection-over-Union between two bounding boxes [x1, y1, x2, y2]."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evidence_updater_node(state: HJLState) -> dict[str, Any]:
    """Fuse the latest verified regional observation into persistent EvidenceState."""
    if not state.observations:
        return {}

    latest_obs = state.observations[-1]
    tool_name = latest_obs.get("tool", "")
    metadata = latest_obs.get("metadata", {})
    step = latest_obs.get("step", state.current_step)

    evidence_state = state.evidence_state

    # 1. Determine region and observation type
    region = metadata.get("bbox")
    obs_type = "visual_feature"
    statement = f"Observed feature via {tool_name}"
    relation = EvidenceRelation.NEUTRAL
    confidence = 0.80

    if tool_name in {"crop_region", "zoom_region"}:
        obs_type = "local_roi"
        # If hypothesis is active, this crop inspects it
        if state.active_hypothesis:
            hyp_type = state.active_hypothesis.get("type", "defect")
            statement = f"Examined local ROI {region} targeting {hyp_type}."
            relation = EvidenceRelation.SUPPORT
            confidence = float(state.active_hypothesis.get("confidence", 0.75))
        else:
            statement = f"Examined local ROI {region}."
            relation = EvidenceRelation.NEUTRAL

    elif tool_name == "retrieve_normal_reference":
        obs_type = "normal_reference"
        statement = f"Retrieved reference standard with {metadata.get('count', 0)} matching documents."
        relation = EvidenceRelation.NEUTRAL
        confidence = 0.85
        evidence_state.normal_references.append(metadata)

    elif tool_name == "compare_with_reference":
        obs_type = "reference_comparison"
        sim = metadata.get("similarity", 1.0)
        if sim < 0.80:
            statement = f"Local appearance significantly deviates from reference (similarity {sim})."
            relation = EvidenceRelation.SUPPORT
            confidence = round(1.0 - sim, 2)
        else:
            statement = f"Local appearance strongly conforms to normal reference (similarity {sim})."
            relation = EvidenceRelation.CONTRADICT
            confidence = round(sim, 2)

    elif tool_name == "localize_candidate":
        obs_type = "candidate_localization"
        candidates = metadata.get("candidate_regions", [])
        statement = f"Localized {len(candidates)} candidate anomaly region(s)."
        relation = EvidenceRelation.NEUTRAL
        confidence = 0.80

    new_item = EvidenceItem(
        source_step=step,
        region=region,
        observation_type=obs_type,
        statement=statement,
        relation=relation,
        confidence=confidence,
        source_tool=tool_name,
        metadata=dict(metadata),
    )

    # 2. Refined deduplication:
    # Flag duplicate only if IoU > 0.7 AND same tool AND same observation_type
    is_duplicate = False
    for existing in evidence_state.evidence_items:
        if (
            existing.source_tool == new_item.source_tool
            and existing.observation_type == new_item.observation_type
            and _compute_iou(existing.region, new_item.region) > 0.7
        ):
            is_duplicate = True
            break

    if not is_duplicate:
        evidence_state.evidence_items.append(new_item)

        if relation == EvidenceRelation.SUPPORT:
            evidence_state.supporting_evidence.append(new_item)
        elif relation == EvidenceRelation.CONTRADICT:
            evidence_state.contradicting_evidence.append(new_item)
        else:
            evidence_state.neutral_evidence.append(new_item)

    # 3. Update inspected regions and unresolved regions
    if region and region not in evidence_state.inspected_regions:
        evidence_state.inspected_regions.append(region)
        # Remove matching box from unresolved_regions
        evidence_state.unresolved_regions = [
            r for r in evidence_state.unresolved_regions
            if _compute_iou(r, region) < 0.7
        ]

    # 4. Compute updated anomaly_score heuristic strictly from SUPPORT and CONTRADICT
    support_weight = sum(e.confidence for e in evidence_state.supporting_evidence)
    contradict_weight = sum(e.confidence for e in evidence_state.contradicting_evidence)
    evidence_state.anomaly_score = max(0.0, min(1.0, support_weight - contradict_weight))

    return {"evidence_state": evidence_state}
