"""Checks for drift that unit tests structurally cannot catch.

1. Schema drift: a database built fresh from TABLES must be identical to
   one walked up through every migration. With 37 versions those two
   paths are maintained by hand and diverge silently — a column added to
   TABLES but not to a migration works perfectly for new users and is
   missing for everyone who upgrades.
2. Documentation drift: the skills are prose, and prose naming a tool
   that no longer exists fails only when a specialist follows it, alone,
   in a scheduled session.
"""

from __future__ import annotations

import pathlib
import re
import sqlite3

import pytest

from chicory.db.schema import SCHEMA_VERSION, apply_schema
from chicory.orchestrator.tool_definitions import CHICORY_TOOLS

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _fingerprint(conn: sqlite3.Connection) -> dict[str, list[tuple]]:
    """Every table's column shape, normalised for comparison."""
    out: dict[str, list[tuple]] = {}
    names = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    for name in names:
        cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
        out[name] = sorted(
            (c[1], (c[2] or "").upper(), c[3], c[5]) for c in cols
        )
    return out


class _Engine:
    """Minimal DatabaseEngine stand-in over a raw sqlite3 connection."""

    def __init__(self, conn):
        conn.row_factory = sqlite3.Row  # as DatabaseEngine does
        self.connection = conn

    def execute(self, sql, params=None):
        return self.connection.execute(sql, params) if params is not None \
            else self.connection.execute(sql)

    def executemany(self, sql, seq):
        return self.connection.executemany(sql, seq)

    def transaction(self):
        class _T:
            def __enter__(_s): return None
            def __exit__(_s, *a): return False
        return _T()


def test_fresh_and_migrated_schemas_agree():
    fresh = sqlite3.connect(":memory:")
    apply_schema(_Engine(fresh))

    # An "old" database: only the version marker, forcing the full
    # migration ladder to run against a minimal starting point.
    migrated = sqlite3.connect(":memory:")
    migrated.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')), "
        "description TEXT)"
    )
    migrated.execute("INSERT INTO schema_version (version) VALUES (1)")
    apply_schema(_Engine(migrated))

    a, b = _fingerprint(fresh), _fingerprint(migrated)
    assert set(a) == set(b), (
        f"tables only in fresh: {sorted(set(a) - set(b))}; "
        f"only in migrated: {sorted(set(b) - set(a))}"
    )
    drift = {t: (a[t], b[t]) for t in a if a[t] != b[t]}
    assert not drift, f"column drift between fresh and migrated: {list(drift)}"
    fresh.close()
    migrated.close()


def test_migrated_database_reports_current_version():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')), "
        "description TEXT)"
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    apply_schema(_Engine(conn))
    v = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert v == SCHEMA_VERSION
    conn.close()


def test_apply_schema_is_idempotent():
    conn = sqlite3.connect(":memory:")
    eng = _Engine(conn)
    apply_schema(eng)
    before = _fingerprint(conn)
    apply_schema(eng)
    apply_schema(eng)
    assert _fingerprint(conn) == before
    rows = conn.execute(
        "SELECT COUNT(*) FROM schema_version WHERE version = ?",
        (SCHEMA_VERSION,),
    ).fetchone()[0]
    assert rows == 1, "re-applying the schema duplicated the version row"
    conn.close()


# ── Documentation drift ─────────────────────────────────────────────────

_SKILLS = [
    ".claude/skills/librarian/SKILL.md",
    ".claude/skills/teach-cycle/SKILL.md",
    ".claude/skills/teach-cycle/briefs.md",
]


def _known_tools() -> set[str]:
    names = {t["name"] for t in CHICORY_TOOLS}
    server = (_REPO / "chicory/mcp/server.py").read_text(encoding="utf-8")
    names |= set(re.findall(r"^def (\w+)\(", server, re.M))
    return names


@pytest.mark.parametrize("relpath", _SKILLS)
def test_skills_only_name_tools_that_exist(relpath):
    text = (_REPO / relpath).read_text(encoding="utf-8")
    known = _known_tools()
    # Tool names as the skills write them: bare or in backticks
    referenced = set(re.findall(r"`?\b((?:librarian|store|retrieve|deep|get|ingest|sync|compare|process)_\w+)`?", text))
    unknown = {r for r in referenced if r not in known}
    assert not unknown, (
        f"{relpath} names tools that do not exist: {sorted(unknown)}"
    )


# ── Background phases ───────────────────────────────────────────────────
# _sync_detection_background catches and logs each phase's exceptions, so
# a phase that is renamed, unbound, or throws on every call degrades
# silently: the library simply stops sweeping, settling or promoting, and
# nothing surfaces. These check the wiring the try/except would hide.

_EXPECTED_PHASES = {
    "deferred retrieval work",
    "sync detection",
    "linear sweep",
    "demand processing",
    "ingress settle",
    "lateral inhibition",
}


def test_background_phase_methods_all_exist():
    import inspect

    from chicory.orchestrator.orchestrator import Orchestrator

    src = inspect.getsource(Orchestrator._sync_detection_background)
    named = set(re.findall(r'\("([^"]+)",\s*self\.(\w+)\)', src))
    labels = {label for label, _ in named}
    assert _EXPECTED_PHASES <= labels, (
        f"background phases missing from the runner: "
        f"{sorted(_EXPECTED_PHASES - labels)}"
    )
    for label, method in named:
        assert callable(getattr(Orchestrator, method, None)), (
            f"phase '{label}' points at Orchestrator.{method}, which does not exist"
        )


def test_background_phases_are_rate_limited():
    """Each periodic phase must consult a checkpoint before doing work.

    Without one it runs on every retrieval, holding the foreground lock.
    """
    import inspect

    from chicory.orchestrator.orchestrator import Orchestrator

    for method, marker in (
        ("_maybe_run_linear_sweep", "_last_sweep_check"),
        ("_maybe_run_demand_processing", "_last_demand_check"),
        ("_maybe_settle_ingress", "_last_ingress_check"),
        ("_maybe_recompute_inhibition", "_last_inhibition_check"),
    ):
        src = inspect.getsource(getattr(Orchestrator, method))
        assert marker in src, f"{method} does not consult {marker}"
        assert "return" in src, f"{method} has no early-out path"


def test_an_existing_table_gains_its_new_columns_on_upgrade():
    """The upgrade path the drift test above cannot see.

    `test_fresh_and_migrated_schemas_agree` walks the ladder from a
    minimal v1 database, where `patrol_observations` does not exist yet —
    so `CREATE TABLE IF NOT EXISTS` in TABLES builds it complete and the
    ALTER never runs. Unhooking `_migrate_v45_to_v46` from the dispatcher
    leaves that test green.

    But a real install already HAS the table, without the columns, and
    for it the ALTER is the only thing that adds them. That is the case
    that matters: 12,612 standing observations on the live library, and a
    missing column there is an OperationalError on the first upsert.
    """
    conn = sqlite3.connect(":memory:")
    db = _Engine(conn)
    apply_schema(db)

    # Rewind to v45: an install that predates retraction provenance.
    conn.execute("ALTER TABLE patrol_observations DROP COLUMN retracted_by")
    conn.execute("ALTER TABLE patrol_observations DROP COLUMN revived_count")
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (45)")
    conn.execute(
        "INSERT INTO patrol_observations (errand, kind, subject, detail, "
        "weight, lap, retracted) VALUES ('e', 'k', 's', 'd', 0.1, 1, 1)")
    conn.commit()

    apply_schema(db)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(patrol_observations)")}
    assert {"retracted_by", "revived_count"} <= cols, (
        "an upgraded install is missing columns a fresh one has"
    )
    row = conn.execute(
        "SELECT retracted, retracted_by, revived_count "
        "FROM patrol_observations").fetchone()
    assert tuple(row) == (1, None, 0), (
        "a row retracted before the column existed must read as withdrawn "
        "by something that did not say, and stay revivable"
    )
    assert conn.execute(
        "SELECT MAX(version) FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()
