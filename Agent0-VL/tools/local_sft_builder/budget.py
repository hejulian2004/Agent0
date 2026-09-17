"""Persistent, crash-recoverable Teacher request-slot ledger."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TERMINAL_STATUSES = frozenset(
    {"successful", "timeout", "parse_failed", "other_failed"}
)
ALL_STATUSES = TERMINAL_STATUSES | {"pending"}


class TeacherBudgetExceeded(RuntimeError):
    """Raised before backend invocation when no request slot remains."""


class RequestStateError(RuntimeError):
    """Raised when a request is finalized more than once or is unknown."""


@dataclass(frozen=True)
class RequestLease:
    request_id: str
    task_id: str
    request_seq: int
    role: str
    retry_of_request_id: str | None = None
    retry_index: int = 0


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    status: str
    value: Any = None
    error: str | None = None


@dataclass(frozen=True)
class BudgetStats:
    task_id: str
    limit: int
    consumed_slot: int
    pending: int
    successful: int
    timeout: int
    parse_failed: int
    other_failed: int

    @property
    def terminal_total(self) -> int:
        return self.successful + self.timeout + self.parse_failed + self.other_failed

    def assert_consistent(self, require_no_pending: bool = False) -> None:
        if self.consumed_slot != self.terminal_total + self.pending:
            raise AssertionError(
                "budget ledger invariant failed: "
                f"consumed={self.consumed_slot}, terminal={self.terminal_total}, "
                f"pending={self.pending}"
            )
        if self.consumed_slot > self.limit:
            raise AssertionError("budget ledger exceeded its configured limit")
        if require_no_pending and self.pending:
            raise AssertionError(f"{self.pending} pending request(s) remain")


class TeacherRequestBudget:
    """One task's shared budget for Solver/Verifier/Repair/Regeneration calls.

    Each operation opens its own SQLite connection and uses ``BEGIN IMMEDIATE``
    for reservation/finalization. This permits multiple worker processes to
    share one database without bypassing the 32-slot guard.
    """

    def __init__(self, db_path: str | Path, task_id: str, limit: int = 32):
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.db_path = Path(db_path)
        self.task_id = task_id
        self.limit = limit
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS teacher_requests (
                    request_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    request_seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'pending', 'successful', 'timeout', 'parse_failed', 'other_failed'
                    )),
                    retry_of_request_id TEXT,
                    retry_index INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    backend_submission_known INTEGER,
                    backend_called INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    finalized_at TEXT,
                    UNIQUE(task_id, request_seq),
                    FOREIGN KEY(retry_of_request_id) REFERENCES teacher_requests(request_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_teacher_requests_task "
                "ON teacher_requests(task_id)"
            )

    def reserve_request(
        self,
        role: str,
        *,
        retry_of_request_id: str | None = None,
        retry_index: int = 0,
        request_id: str | None = None,
    ) -> RequestLease:
        """Atomically consume one slot and insert a pending request."""

        if not role:
            raise ValueError("role must not be empty")
        if retry_index < 0:
            raise ValueError("retry_index must be non-negative")

        request_id = request_id or f"req_{uuid.uuid4().hex}"
        created_at = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count_row = connection.execute(
                "SELECT COUNT(*) AS count FROM teacher_requests WHERE task_id = ?",
                (self.task_id,),
            ).fetchone()
            consumed = int(count_row["count"])
            if consumed >= self.limit:
                connection.rollback()
                raise TeacherBudgetExceeded(
                    f"task {self.task_id!r} exhausted budget: "
                    f"{consumed}/{self.limit} consumed slots"
                )

            seq_row = connection.execute(
                "SELECT COALESCE(MAX(request_seq), 0) + 1 AS next_seq "
                "FROM teacher_requests WHERE task_id = ?",
                (self.task_id,),
            ).fetchone()
            request_seq = int(seq_row["next_seq"])
            connection.execute(
                """
                INSERT INTO teacher_requests (
                    request_id, task_id, request_seq, role, status,
                    retry_of_request_id, retry_index, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    request_id,
                    self.task_id,
                    request_seq,
                    role,
                    retry_of_request_id,
                    retry_index,
                    created_at,
                ),
            )
            connection.commit()
            return RequestLease(
                request_id=request_id,
                task_id=self.task_id,
                request_seq=request_seq,
                role=role,
                retry_of_request_id=retry_of_request_id,
                retry_index=retry_index,
            )
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def mark_backend_started(self, request_id: str) -> None:
        """Record that execution crossed the backend call boundary."""

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE teacher_requests SET backend_called = 1, "
                "backend_submission_known = 1 "
                "WHERE request_id = ? AND task_id = ? AND status = 'pending'",
                (request_id, self.task_id),
            )
            if cursor.rowcount != 1:
                raise RequestStateError(f"unknown or non-pending request: {request_id}")

    def finalize_request(
        self,
        request_id: str,
        status: str,
        *,
        failure_reason: str | None = None,
        backend_submission_known: bool | None = None,
    ) -> None:
        """Move a pending request to exactly one terminal state."""

        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        finalized_at = self._now()
        known_value = (
            None if backend_submission_known is None else int(backend_submission_known)
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE teacher_requests
                SET status = ?, failure_reason = ?,
                    backend_submission_known = COALESCE(?, backend_submission_known),
                    finalized_at = ?
                WHERE request_id = ? AND task_id = ? AND status = 'pending'
                """,
                (
                    status,
                    failure_reason,
                    known_value,
                    finalized_at,
                    request_id,
                    self.task_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RequestStateError(
                    f"request is unknown or already finalized: {request_id}"
                )

    def recover_stale_pending(self) -> int:
        """Conservatively finalize every pending request for this task."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE teacher_requests
                SET status = 'other_failed',
                    failure_reason = 'recovered_stale_pending',
                    backend_submission_known = 0,
                    finalized_at = ?
                WHERE task_id = ? AND status = 'pending'
                """,
                (self._now(), self.task_id),
            )
            return cursor.rowcount

    def stats(self) -> BudgetStats:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM teacher_requests "
                "WHERE task_id = ? GROUP BY status",
                (self.task_id,),
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        stats = BudgetStats(
            task_id=self.task_id,
            limit=self.limit,
            consumed_slot=sum(counts.values()),
            pending=counts.get("pending", 0),
            successful=counts.get("successful", 0),
            timeout=counts.get("timeout", 0),
            parse_failed=counts.get("parse_failed", 0),
            other_failed=counts.get("other_failed", 0),
        )
        stats.assert_consistent()
        return stats

    def request_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM teacher_requests WHERE task_id = ? "
                "ORDER BY request_seq",
                (self.task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def assert_consistent(self, require_no_pending: bool = False) -> BudgetStats:
        stats = self.stats()
        stats.assert_consistent(require_no_pending=require_no_pending)
        return stats

    def execute(
        self,
        role: str,
        backend: Callable[[Any], Any],
        payload: Any,
        *,
        parser: Callable[[Any], Any] | None = None,
        retry_of_request_id: str | None = None,
        retry_index: int = 0,
    ) -> RequestResult:
        """Reserve, call, parse, and finalize one backend attempt.

        The slot is reserved before invoking ``backend``. A parser exception is
        a distinct ``parse_failed`` terminal state; backend exceptions become
        ``timeout`` for ``TimeoutError`` and ``other_failed`` otherwise.
        """

        lease = self.reserve_request(
            role,
            retry_of_request_id=retry_of_request_id,
            retry_index=retry_index,
        )
        try:
            self.mark_backend_started(lease.request_id)
            response = backend(payload)
        except TimeoutError as exc:
            self.finalize_request(
                lease.request_id,
                "timeout",
                failure_reason=str(exc) or "timeout",
                backend_submission_known=True,
            )
            return RequestResult(lease.request_id, "timeout", error=str(exc))
        except Exception as exc:  # pragma: no cover - exercised by integration test
            self.finalize_request(
                lease.request_id,
                "other_failed",
                failure_reason=str(exc) or type(exc).__name__,
                backend_submission_known=True,
            )
            return RequestResult(lease.request_id, "other_failed", error=str(exc))

        if parser is not None:
            try:
                value = parser(response)
            except Exception as exc:
                self.finalize_request(
                    lease.request_id,
                    "parse_failed",
                    failure_reason=str(exc) or type(exc).__name__,
                    backend_submission_known=True,
                )
                return RequestResult(lease.request_id, "parse_failed", error=str(exc))
        else:
            value = response

        self.finalize_request(
            lease.request_id,
            "successful",
            backend_submission_known=True,
        )
        return RequestResult(lease.request_id, "successful", value=value)
