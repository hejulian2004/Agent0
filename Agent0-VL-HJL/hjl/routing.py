"""Deterministic failure-conditioned action mask routing for HJL."""

from __future__ import annotations

from .taxonomy import ActionType, FailureType, GlobalStatus


class GlobalRecoveryPolicy:
    """Action masking policy when Global Checkpoint fails (ambiguous or unlocalized)."""

    @staticmethod
    def get_recovery_actions(status: GlobalStatus) -> list[ActionType]:
        if status == GlobalStatus.FAIL:
            return [ActionType.GLOBAL_SCAN, ActionType.RELOCALIZE]
        return []


class FailureRoutingPolicy:
    """Action space mask mapping from FailureType to allowed ActionTypes."""

    _MAPPING: dict[FailureType, list[ActionType]] = {
        FailureType.TOOL_FAILURE: [ActionType.RETRY_TOOL],
        FailureType.WRONG_REGION: [ActionType.RELOCALIZE, ActionType.ENHANCE_REGION],
        FailureType.LOW_RESOLUTION: [ActionType.ENHANCE_REGION],
        FailureType.MISSING_REFERENCE: [ActionType.RETRIEVE_REFERENCE],
        FailureType.VIEWPOINT_MISMATCH: [ActionType.ALIGN_VIEW],
        FailureType.CONTRADICTORY_EVIDENCE: [ActionType.CROSS_VALIDATE, ActionType.ENHANCE_REGION],
        FailureType.LOCALIZATION_UNCERTAIN: [ActionType.RELOCALIZE],
        FailureType.PREMATURE_CONCLUSION: [ActionType.INSPECT_NEXT_REGION, ActionType.RELOCALIZE],
    }

    @classmethod
    def get_allowed_actions(cls, failure_type: FailureType) -> list[ActionType]:
        actions = cls._MAPPING.get(failure_type, [])
        return list(actions)
