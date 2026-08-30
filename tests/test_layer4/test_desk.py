"""Tests for the desk report."""

from __future__ import annotations

from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.desk import DeskReport


@pytest.fixture
def desk():
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test", llm_model="test", embedding_model="mock",
        embedding_dimension=32,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    yield config, db, DeskReport(config, db)
    db.close()


def _mem(db, mid, created="2026-08-30T12:00:00", author=None, tier="critical"):
    db.execute(
        "INSERT INTO memories (id, content, source_model, created_at, "
        "updated_at, author, ingestion_tier) VALUES (?, 'x', 't', ?, ?, ?, ?)",
        (mid, created, created, author, tier),
    )


class TestWatermark:
    def test_first_report_has_no_baseline(self, desk):
        _config, _db, d = desk
        r = d.report()
        assert r["first_report"] is True
        assert r["since"] is None
        assert "first report" in r["headline"]

    def test_reading_does_not_acknowledge(self, desk):
        _config, _db, d = desk
        d.report()
        d.report()
        # A report you only looked at must come back
        assert d.watermark() is None
        assert d.report()["first_report"] is True

    def test_ack_advances_the_window(self, desk):
        _config, db, d = desk
        d.acknowledge("2026-08-30T00:00:00")
        _mem(db, "after", created="2026-08-30T09:00:00")
        _mem(db, "before-ack-window", created="2026-08-29T09:00:00")
        r = d.report()
        assert r["first_report"] is False
        assert r["since"] == "2026-08-30T00:00:00"
        assert r["changed"]["memories_stored"] == 1

    def test_ack_is_idempotent_and_updates(self, desk):
        _config, _db, d = desk
        d.acknowledge("2026-08-01T00:00:00")
        d.acknowledge("2026-08-02T00:00:00")
        assert d.watermark() == "2026-08-02T00:00:00"


class TestNeedsYou:
    def test_surfaces_every_queue(self, desk):
        _config, db, d = desk
        _mem(db, "q1", tier="quarantine")
        db.execute(
            "INSERT INTO sweep_findings (memory_id, finding_type, detail) "
            "VALUES ('q1', 'near_duplicate', 'a|b')"
        )
        db.execute(
            "INSERT INTO commissions (question_text, source) "
            "VALUES ('what about X?', 'manual')"
        )
        db.execute(
            "INSERT INTO faq_entries (question_text, question_centroid, "
            "supporting_memory_ids, status) VALUES ('recurring?', X'00', '[]', 'candidate')"
        )
        db.execute(
            "INSERT INTO message_ingress (receipt, content, content_hash, "
            "arrived_at, dwell_until, status) VALUES "
            "('rcpt-1', 'suspicious text', 'h', '2026-08-30', '2026-08-30', 'flagged')"
        )
        needs = d.report()["needs_you"]
        assert needs["quarantined_memories"] == 1
        assert needs["open_findings"]["near_duplicate"] == 1
        assert len(needs["open_commissions"]) == 1
        assert len(needs["faq_awaiting_judgment"]) == 1
        assert len(needs["ingress_flagged"]) == 1

    def test_resolved_findings_are_not_pending(self, desk):
        _config, db, d = desk
        _mem(db, "m1")
        db.execute(
            "INSERT INTO sweep_findings (memory_id, finding_type, detail, resolved) "
            "VALUES ('m1', 'tag_drift', 'x', 1)"
        )
        assert d.report()["needs_you"]["open_findings"] == {}

    def test_quiet_library_says_so(self, desk):
        _config, _db, d = desk
        d.acknowledge()
        assert "quiet" in d.report()["headline"]


class TestLearnedAndHealth:
    def test_digests_are_reported(self, desk):
        _config, db, d = desk
        _mem(db, "dig", author="specialist:physics")
        db.execute("INSERT INTO tags (name) VALUES ('digest')")
        tid = db.execute("SELECT id FROM tags WHERE name='digest'").fetchone()["id"]
        db.execute(
            "INSERT INTO memory_tags (memory_id, tag_id) VALUES ('dig', ?)", (tid,),
        )
        learned = d.report()["learned"]
        assert len(learned["digests"]) == 1
        assert learned["digests"][0]["author"] == "specialist:physics"

    def test_health_counts_coverage_and_signatures(self, desk):
        _config, db, d = desk
        _mem(db, "a", author="specialist:x")
        _mem(db, "b")
        db.execute("UPDATE memories SET swept_at = '2026-08-30' WHERE id = 'a'")
        health = d.report()["health"]
        assert health["memories"] == 2
        assert health["patrol_coverage"] == 0.5
        assert health["signed_memories"] == 1
        assert health["gate_enforcing"] is False

    def test_headline_mentions_pending_work(self, desk):
        _config, db, d = desk
        d.acknowledge()
        db.execute(
            "INSERT INTO commissions (question_text, source) "
            "VALUES ('gap?', 'recurring-weak')"
        )
        _mem(db, "q", tier="quarantine")
        headline = d.report()["headline"]
        assert "commission" in headline
        assert "quarantined" in headline


class TestQueueHonesty:
    def test_headline_counts_the_backlog_not_the_page(self, desk):
        _config, db, d = desk
        d.acknowledge()
        for i in range(12):
            db.execute(
                "INSERT INTO commissions (question_text, source) VALUES (?, 'manual')",
                (f"question {i}?",),
            )
        for i in range(7):
            db.execute(
                "INSERT INTO faq_entries (question_text, question_centroid, "
                "supporting_memory_ids, status) VALUES (?, X'00', '[]', 'candidate')",
                (f"faq {i}?",),
            )
        r = d.report(limit=5)
        # Lists are capped for display...
        assert len(r["needs_you"]["open_commissions"]) == 5
        assert r["needs_you"]["lists_capped_at"] == 5
        # ...but the totals and headline tell the truth
        assert r["needs_you"]["totals"]["commissions"] == 12
        assert r["needs_you"]["totals"]["faq"] == 7
        assert "12 commissions" in r["headline"]
        assert "7 FAQ drafts" in r["headline"]

    def test_stalled_claimed_commissions_are_visible(self, desk):
        _config, db, d = desk
        db.execute(
            "INSERT INTO commissions (question_text, source, status) "
            "VALUES ('stranded?', 'manual', 'claimed')"
        )
        r = d.report()
        rows = r["needs_you"]["open_commissions"]
        assert len(rows) == 1
        assert rows[0]["status"] == "claimed"
        assert rows[0]["stalled"] is True
        assert r["needs_you"]["totals"]["commissions_stalled"] == 1
        assert "stalled mid-research" in r["headline"]

    def test_closed_commissions_stay_out(self, desk):
        _config, db, d = desk
        db.execute(
            "INSERT INTO commissions (question_text, source, status) "
            "VALUES ('done', 'manual', 'closed')"
        )
        r = d.report()
        assert r["needs_you"]["totals"]["commissions"] == 0

    def test_limit_is_clamped(self, desk):
        _config, db, d = desk
        db.execute(
            "INSERT INTO commissions (question_text, source) VALUES ('q', 'manual')"
        )
        # 0 would read as an all-clear; negative would be unbounded
        assert d.report(limit=0)["needs_you"]["totals"]["commissions"] == 1
        assert d.report(limit=0)["needs_you"]["lists_capped_at"] >= 1
        assert d.report(limit=-3)["needs_you"]["lists_capped_at"] >= 1
        assert d.report(limit=9999)["needs_you"]["lists_capped_at"] <= 50

    def test_ack_stamp_matches_the_schema_clock_format(self, desk):
        _config, db, d = desk
        d.acknowledge()
        wm = d.watermark()
        # SQLite's millisecond format, not Python's microseconds — the two
        # compare lexicographically, and a mismatch drops recent rows
        assert wm and len(wm.split(".")[-1]) == 3
        schema_stamp = db.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%f','now') AS s"
        ).fetchone()["s"]
        assert len(wm) == len(schema_stamp)

    def test_rows_after_the_ack_are_counted(self, desk):
        _config, db, d = desk
        d.acknowledge()
        wm = d.watermark()
        # Strictly later than the ack (same-millisecond rows count as seen,
        # which is what "acknowledged up to now" means)
        _mem(db, "later", created="2099-01-01T00:00:00.000")
        _mem(db, "earlier", created="2000-01-01T00:00:00.000")
        assert wm < "2099-01-01T00:00:00.000"
        assert d.report()["changed"]["memories_stored"] == 1
