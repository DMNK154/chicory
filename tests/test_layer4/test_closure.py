"""Tests for specialist closure — supersession and the reverse-order unwind."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.closure import ClosureLedger


@pytest.fixture
def ledger():
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test", llm_model="test", embedding_model="mock",
        embedding_dimension=32,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    yield config, db, ClosureLedger(config, db)
    db.close()


def _mem(db, mid, summary="s"):
    db.execute(
        "INSERT INTO memories (id, content, source_model, summary) "
        "VALUES (?, 'x', 't', ?)", (mid, summary),
    )


class TestSupersede:
    def test_records_and_reads_back(self, ledger):
        _config, db, led = ledger
        _mem(db, "old", "the old digest")
        _mem(db, "new", "the correcting report")
        res = led.supersede("old", "new", actor="specialist:x",
                            claim="three editions", reason="they differ")
        assert res["status"] == "recorded"
        marks = led.for_memories(["old", "new"])
        assert "new" not in marks  # the corrector is not itself superseded
        assert marks["old"][0]["superseding_id"] == "new"
        assert marks["old"][0]["claim"] == "three editions"
        assert marks["old"][0]["scope"] == "claim"

    def test_nothing_is_deleted(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="a")
        row = db.execute(
            "SELECT id, is_archived FROM memories WHERE id = 'old'"
        ).fetchone()
        assert row is not None and row["is_archived"] == 0

    def test_journals_a_custody_event(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="specialist:zohar", claim="C")
        ev = db.execute(
            "SELECT actor, reason FROM custody_events WHERE memory_id = 'old'"
        ).fetchone()
        assert ev["actor"] == "specialist:zohar"
        assert "superseded" in ev["reason"] and "C" in ev["reason"]

    def test_rejects_self_and_unknown(self, ledger):
        _config, db, led = ledger
        _mem(db, "a")
        assert "error" in led.supersede("a", "a", actor="x")
        assert "error" in led.supersede("a", "ghost", actor="x")
        assert "error" in led.supersede("a", "a", actor="x", scope="bogus")

    def test_archived_memory_cannot_be_a_corrector(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        db.execute("UPDATE memories SET is_archived = 1 WHERE id = 'new'")
        assert "error" in led.supersede("old", "new", actor="x")

    def test_repeat_updates_rather_than_duplicates(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="a", claim="C", reason="first")
        led.supersede("old", "new", actor="b", claim="C", reason="second")
        rows = db.execute("SELECT reason, actor FROM supersessions").fetchall()
        assert len(rows) == 1
        assert rows[0]["reason"] == "second" and rows[0]["actor"] == "b"

    def test_distinct_claims_coexist(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="a", claim="claim one")
        led.supersede("old", "new", actor="a", claim="claim two")
        assert len(led.for_memories(["old"])["old"]) == 2

    def test_retract(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="a")
        assert led.retract("old", "new")
        assert led.for_memories(["old"]) == {}
        assert not led.retract("old", "new")  # already retracted


class TestFaqInvalidation:
    def test_active_faq_resting_on_a_refuted_memory_goes_stale(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        db.execute(
            "INSERT INTO faq_entries (question_text, question_centroid, "
            "supporting_memory_ids, status) VALUES ('q', X'00', ?, 'active')",
            (json.dumps(["old", "other"]),),
        )
        res = led.supersede("old", "new", actor="a")
        assert res["faq_marked_stale"] == 1
        assert db.execute(
            "SELECT status FROM faq_entries"
        ).fetchone()["status"] == "stale"

    def test_unrelated_faq_untouched(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        db.execute(
            "INSERT INTO faq_entries (question_text, question_centroid, "
            "supporting_memory_ids, status) VALUES ('q', X'00', ?, 'active')",
            (json.dumps(["someone-else"]),),
        )
        assert led.supersede("old", "new", actor="a")["faq_marked_stale"] == 0
        assert db.execute(
            "SELECT status FROM faq_entries"
        ).fetchone()["status"] == "active"


class TestCloseChain:
    def test_full_unwind_in_reverse_order(self, ledger):
        _config, db, led = ledger
        _mem(db, "parent-digest", "parent digest")
        _mem(db, "report", "the report")
        db.execute(
            "INSERT INTO commissions (question_text, source, status) "
            "VALUES ('why?', 'recurring-weak', 'claimed')"
        )
        cid = db.execute("SELECT id FROM commissions").fetchone()["id"]
        db.execute(
            "INSERT INTO sweep_findings (memory_id, finding_type, detail) "
            "VALUES ('parent-digest', 'near_duplicate', 'a|b')"
        )
        fid = db.execute("SELECT id FROM sweep_findings").fetchone()["id"]
        db.execute(
            "INSERT INTO faq_entries (question_text, question_centroid, "
            "supporting_memory_ids, status) VALUES ('q', X'00', ?, 'active')",
            (json.dumps(["parent-digest"]),),
        )

        out = led.close_chain(
            report_memory_id="report", actor="specialist:zohar",
            commission_id=cid,
            supersedes=[{"memory_id": "parent-digest",
                         "claim": "three editions", "reason": "they differ"}],
            resolve_findings=[fid],
        )
        assert out["status"] == "closed"
        kinds = [s["step"] for s in out["steps"]]
        # reverse order: the report, what it overturned, then the ticket
        assert kinds == ["report", "supersede", "findings", "commission"]

        assert led.for_memories(["parent-digest"])["parent-digest"]
        assert db.execute(
            "SELECT resolved FROM sweep_findings WHERE id = ?", (fid,)
        ).fetchone()["resolved"] == 1
        assert db.execute(
            "SELECT status FROM faq_entries"
        ).fetchone()["status"] == "stale"
        crow = db.execute(
            "SELECT status, report_memory_id FROM commissions WHERE id = ?", (cid,)
        ).fetchone()
        assert crow["status"] == "closed"
        assert crow["report_memory_id"] == "report"

    def test_closes_a_claimed_ticket_and_names_its_source(self, ledger):
        _config, db, led = ledger
        _mem(db, "report")
        db.execute(
            "INSERT INTO commissions (question_text, source, status) "
            "VALUES ('q', 'recurring-weak', 'claimed')"
        )
        cid = db.execute("SELECT id FROM commissions").fetchone()["id"]
        out = led.close_chain("report", actor="a", commission_id=cid)
        step = [s for s in out["steps"] if s["step"] == "commission"][0]
        assert step["closed_from"] == "claimed"
        assert step["raised_by"] == "recurring-weak"

    def test_unknown_report_is_refused(self, ledger):
        _config, _db, led = ledger
        assert "error" in led.close_chain("ghost", actor="a")

    def test_total_failure_stops_before_the_ticket(self, ledger):
        # If nothing was superseded, the belief that raised the ticket is
        # still standing, so the ticket must stay open.
        _config, db, led = ledger
        _mem(db, "report")
        out = led.close_chain(
            "report", actor="a",
            supersedes=[{"memory_id": "does-not-exist"}],
            commission_id=999,
        )
        assert out["status"] == "incomplete"
        sup = [s for s in out["steps"] if s["step"] == "supersede"]
        assert any("error" in s for s in sup)
        assert not [s for s in out["steps"] if s["step"] == "commission"]

    def test_partial_success_still_unwinds_and_reports_the_failure(self, ledger):
        _config, db, led = ledger
        _mem(db, "report")
        _mem(db, "real")
        db.execute(
            "INSERT INTO commissions (question_text, source) VALUES ('q', 'manual')"
        )
        cid = db.execute("SELECT id FROM commissions").fetchone()["id"]
        out = led.close_chain(
            "report", actor="a", commission_id=cid,
            supersedes=[{"memory_id": "real", "claim": "A"},
                        {"memory_id": "ghost", "claim": "B"}],
        )
        assert out["status"] == "closed"
        sup = [s for s in out["steps"] if s["step"] == "supersede"]
        assert any(s.get("recorded") == ["real"] for s in sup)
        assert any("error" in s for s in sup)
        assert db.execute(
            "SELECT status FROM commissions WHERE id = ?", (cid,)
        ).fetchone()["status"] == "closed"


class TestReporting:
    def test_recent_and_count_respect_since(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="a")
        db.execute("UPDATE supersessions SET created_at = '2020-01-01T00:00:00'")
        assert led.count() == 1
        assert led.count(since="2026-01-01T00:00:00") == 0
        assert len(led.recent(since="2026-01-01T00:00:00")) == 0
        assert len(led.recent()) == 1


class TestReviewFixes:
    def test_claimless_supersession_does_not_duplicate(self, ledger):
        # NULL inside a UNIQUE tuple enforces nothing in SQLite; '' does.
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        for reason in ("first", "second", "third"):
            led.supersede("old", "new", actor="a", reason=reason, scope="whole")
        rows = db.execute("SELECT reason FROM supersessions").fetchall()
        assert len(rows) == 1
        assert rows[0]["reason"] == "third"  # updated, not appended
        assert len(led.for_memories(["old"])["old"]) == 1
        assert led.count() == 1

    def test_retract_withdraws_only_the_named_claim(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="a", claim="claim one")
        led.supersede("old", "new", actor="a", claim="claim two")
        assert led.retract("old", "new", claim="claim one")
        remaining = led.for_memories(["old"])["old"]
        assert len(remaining) == 1
        assert remaining[0]["claim"] == "claim two"
        # and the deliberate sweep still exists
        assert led.retract("old", "new", all_claims=True)
        assert led.for_memories(["old"]) == {}

    def test_revived_supersession_redates(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        led.supersede("old", "new", actor="a", claim="C")
        db.execute("UPDATE supersessions SET created_at = '2020-01-01T00:00:00'")
        led.retract("old", "new", claim="C")
        led.supersede("old", "new", actor="a", claim="C")
        # A revived correction is news again, so the since-window sees it
        assert led.count(since="2026-01-01T00:00:00") == 1

    def test_candidate_and_stale_faq_are_invalidated_too(self, ledger):
        _config, db, led = ledger
        _mem(db, "old")
        _mem(db, "new")
        for status in ("active", "candidate", "stale"):
            db.execute(
                "INSERT INTO faq_entries (question_text, question_centroid, "
                "supporting_memory_ids, status) VALUES (?, X'00', ?, ?)",
                (f"q-{status}", json.dumps(["old"]), status),
            )
        assert led.supersede("old", "new", actor="a")["faq_marked_stale"] == 3
        assert db.execute(
            "SELECT COUNT(*) AS n FROM faq_entries WHERE status = 'stale'"
        ).fetchone()["n"] == 3

    def test_close_refuses_to_close_when_every_supersede_failed(self, ledger):
        _config, db, led = ledger
        _mem(db, "report")
        db.execute(
            "INSERT INTO commissions (question_text, source, status) "
            "VALUES ('q', 'manual', 'claimed')"
        )
        cid = db.execute("SELECT id FROM commissions").fetchone()["id"]
        out = led.close_chain(
            "report", actor="a", commission_id=cid,
            supersedes=[{"memory_id": "ghost"}],
        )
        assert out["status"] == "incomplete"
        # the ticket must survive — the belief it was raised about still stands
        assert db.execute(
            "SELECT status FROM commissions WHERE id = ?", (cid,)
        ).fetchone()["status"] == "claimed"

    def test_missing_memory_id_does_not_raise(self, ledger):
        _config, db, led = ledger
        _mem(db, "report")
        _mem(db, "old")
        out = led.close_chain(
            "report", actor="a",
            supersedes=[{"claim": "no id here"}, {"memory_id": "old"}],
        )
        assert out["status"] == "closed"
        errs = [s for s in out["steps"]
                if s["step"] == "supersede" and "error" in s]
        assert errs and "no memory_id" in errs[0]["error"]
        assert led.for_memories(["old"])["old"]

    def test_superseded_among(self, ledger):
        _config, db, led = ledger
        _mem(db, "a")
        _mem(db, "b")
        _mem(db, "new")
        led.supersede("a", "new", actor="x")
        assert led.superseded_among(["a", "b"]) == {"a"}
        assert led.superseded_among([]) == set()
