from __future__ import annotations

import gc
import sqlite3

import pytest

from tools.local_sft_builder.budget import (
    TeacherBudgetExceeded,
    TeacherRequestBudget,
)


def _live_connections() -> int:
    return sum(1 for obj in gc.get_objects() if isinstance(obj, sqlite3.Connection))


def test_33rd_teacher_request_is_never_submitted(tmp_path) -> None:
    calls: list[int] = []
    budget = TeacherRequestBudget(tmp_path / "requests.sqlite3", "task-a", limit=32)

    for index in range(32):
        result = budget.execute(
            "solver",
            lambda payload, index=index: calls.append(index) or "ok",
            {"index": index},
        )
        assert result.status == "successful"

    with pytest.raises(TeacherBudgetExceeded):
        budget.execute("solver", lambda payload: calls.append(999), {})

    assert len(calls) == 32
    stats = budget.assert_consistent(require_no_pending=True)
    assert stats.consumed_slot == 32
    assert stats.successful == 32


def test_failed_teacher_requests_consume_budget_and_are_mutually_exclusive(tmp_path) -> None:
    budget = TeacherRequestBudget(tmp_path / "requests.sqlite3", "task-b", limit=4)

    timeout = budget.execute("solver", lambda _: (_ for _ in ()).throw(TimeoutError("late")), {})
    parse_failed = budget.execute(
        "verifier",
        lambda _: "not-json",
        {},
        parser=lambda _: (_ for _ in ()).throw(ValueError("bad json")),
    )
    other_failed = budget.execute(
        "repair",
        lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        {},
    )
    success = budget.execute("regeneration", lambda _: "ok", {})

    assert timeout.status == "timeout"
    assert parse_failed.status == "parse_failed"
    assert other_failed.status == "other_failed"
    assert success.status == "successful"
    stats = budget.assert_consistent(require_no_pending=True)
    assert (stats.successful, stats.timeout, stats.parse_failed, stats.other_failed) == (1, 1, 1, 1)
    assert stats.consumed_slot == 4
    assert all(row["status"] in {"successful", "timeout", "parse_failed", "other_failed"}
               for row in budget.request_rows())


def test_pending_recovery_records_unknown_backend_submission(tmp_path) -> None:
    db_path = tmp_path / "requests.sqlite3"
    budget = TeacherRequestBudget(db_path, "task-c")
    lease = budget.reserve_request("solver")
    assert budget.stats().pending == 1

    restarted = TeacherRequestBudget(db_path, "task-c")
    assert restarted.recover_stale_pending() == 1
    stats = restarted.assert_consistent(require_no_pending=True)
    assert stats.other_failed == 1
    row = restarted.request_rows()[0]
    assert row["failure_reason"] == "recovered_stale_pending"
    assert row["backend_submission_known"] == 0
    assert row["backend_called"] == 0
    assert row["request_id"] == lease.request_id


def test_retry_is_a_new_consumed_slot(tmp_path) -> None:
    budget = TeacherRequestBudget(tmp_path / "requests.sqlite3", "task-d", limit=2)
    first = budget.execute("solver", lambda _: "bad", {}, parser=lambda _: (_ for _ in ()).throw(ValueError()))
    second = budget.execute(
        "solver",
        lambda _: "good",
        {},
        retry_of_request_id=first.request_id,
        retry_index=1,
    )
    assert second.status == "successful"
    rows = budget.request_rows()
    assert rows[1]["retry_of_request_id"] == first.request_id
    assert rows[1]["retry_index"] == 1
    assert budget.stats().consumed_slot == 2


def test_budget_releases_every_sqlite_connection(tmp_path) -> None:
    """Regression: ``with self._connect() as c`` does not close ``c``.

    ``sqlite3.Connection.__exit__`` only ends the open transaction. Since the
    object sits in a reference cycle it is reclaimed solely by the cyclic
    collector, so a long run accumulates open handles. On Windows those
    handles keep the ``-wal``/``-shm`` files mapped and the run-end
    ``journal_mode = DELETE`` fails with ``database is locked`` -- which is
    exactly how this surfaced in the end-to-end ``--generate`` test.
    """

    db_path = tmp_path / "requests.sqlite3"
    budget = TeacherRequestBudget(db_path, "task-leak", limit=8)

    gc.collect()
    baseline = _live_connections()

    for _ in range(3):
        budget.execute("solver", lambda _: "ok", {})
    budget.stats()
    budget.request_rows()
    budget.recover_stale_pending()
    budget.assert_consistent(require_no_pending=True)

    # Deliberately no ``gc.collect()`` here: a deterministically closed
    # connection is released by refcounting alone, without the cyclic
    # collector's help.
    assert _live_connections() == baseline

    # The ledger must also be switchable out of WAL without the exclusive-lock
    # failure that the leaked handles caused.
    connection = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == (
            "delete"
        )
    finally:
        connection.close()
