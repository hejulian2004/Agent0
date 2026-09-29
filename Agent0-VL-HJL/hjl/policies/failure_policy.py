"""Failure escalation and retry limit policy for HJL."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..state import HJLState
from ..taxonomy import FailureType


class FailurePolicy:
    """Tracks failure frequency and enforces escalation when retries stall."""

    def __init__(self, max_repeated_failures: int = 2) -> None:
        self.max_repeated_failures = max_repeated_failures
        self._counts: Counter[FailureType] = Counter()

    def record_failure(self, failure_type: FailureType) -> None:
        self._counts[failure_type] += 1

    def is_escalated(self, failure_type: FailureType) -> bool:
        """Return True if the same failure has occurred repeatedly without resolution."""
        return self._counts[failure_type] > self.max_repeated_failures

    def reset(self) -> None:
        self._counts.clear()
