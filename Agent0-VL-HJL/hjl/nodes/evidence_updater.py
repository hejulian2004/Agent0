"""Evidence Updater node for HJL.

Accumulates, deduplicates, and fuses valid observations into persistent EvidenceState.
CRITICAL INVARIANT: Runs strictly after RegionalVerifier yields PASS (or comparison extractor)
to avoid evidence pollution.
"""

from __future__ import annotations

from typing import Any

from ..schemas import RegionalEvidenceFinding
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


def evidence_updater_node(
    state: HJLState,
    finding: RegionalEvidenceFinding | None = None,
    reference_similarity_threshold: float = 0.80,
) -> dict[str, Any]:
    """Fuse verified observation or structured finding into persistent EvidenceState."""
    if not state.observations and finding is None:
        return {}

    evidence_state = state.evidence_state
    latest_obs = state.observations[-1] if state.observations else {}
    tool_name = latest_obs.get("tool", "")
    metadata = latest_obs.get("metadata", {})
    step = latest_obs.get("step", state.current_step)
    region = (
        metadata.get("bbox")
        or (finding.metadata.get("bbox") if finding and finding.metadata else None)
        or (state.extracted_finding.metadata.get("bbox") if getattr(state, "extracted_finding", None) and state.extracted_finding.metadata else None)
        or state.active_region_original_bbox
    )

    # 1. Determine finding attributes
    if finding is not None:
        obs_type = finding.observation_type
        statement = finding.finding
        relation = finding.relation
        confidence = finding.confidence
        meta = {**metadata, **finding.metadata}
    elif getattr(state, "extracted_finding", None) is not None:
        ext = state.extracted_finding
        obs_type = ext.observation_type
        statement = ext.finding
        relation = ext.relation
        confidence = ext.confidence
        meta = {**metadata, **ext.metadata}
    elif tool_name == "compare_with_reference":
        obs_type = "reference_comparison"
        sim = metadata.get("similarity", 1.0)
        if sim < reference_similarity_threshold:
            statement = f"Local appearance significantly deviates from reference (similarity {sim:.2f} < {reference_similarity_threshold:.2f})."
            relation = EvidenceRelation.SUPPORT
            confidence = round(max(0.80, 1.0 - sim), 2)
        else:
            statement = f"Local appearance conforms to normal reference (similarity {sim:.2f} >= {reference_similarity_threshold:.2f})."
            relation = EvidenceRelation.CONTRADICT
            confidence = round(max(0.80, sim), 2)
        meta = dict(metadata)
    else:
        # Spatial inspection tools (crop, zoom, rotate) are NEUTRAL by default
        obs_type = "inspected_roi"
        statement = f"Examined local ROI {region} via {tool_name}."
        relation = EvidenceRelation.NEUTRAL
        confidence = 0.80
        meta = dict(metadata)

    new_item = EvidenceItem(
        source_step=step,
        region=region,
        observation_type=obs_type,
        statement=statement,
        relation=relation,
        confidence=confidence,
        source_tool=tool_name,
        metadata=meta,
    )

    # 2. Refined deduplication:
    # Flag duplicate only if same tool AND same observation_type AND matching region
    is_duplicate = False
    for existing in evidence_state.evidence_items:
        same_tool = (existing.source_tool == new_item.source_tool)
        same_type = (existing.observation_type == new_item.observation_type)
        same_region = (
            (_compute_iou(existing.region, new_item.region) > 0.7)
            if (existing.region and new_item.region)
            else (existing.region == new_item.region)
        )
        if same_tool and same_type and same_region:
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
    if region:
        if region not in evidence_state.inspected_regions:
            evidence_state.inspected_regions.append(region)
        evidence_state.unresolved_regions = [
            r for r in evidence_state.unresolved_regions
            if _compute_iou(r, region) < 0.7
        ]

    # 4. Compute updated anomaly_score heuristic strictly from SUPPORT and CONTRADICT
    support_weight = sum(e.confidence for e in evidence_state.supporting_evidence)
    contradict_weight = sum(e.confidence for e in evidence_state.contradicting_evidence)
    evidence_state.anomaly_score = max(0.0, min(1.0, support_weight - contradict_weight))

    return {
        "evidence_state": evidence_state,
        "extracted_finding": None,
    }
