"""Database schema creation and migration."""

from __future__ import annotations

from chicory.db.engine import DatabaseEngine

SCHEMA_VERSION = 10

TABLES = [
    # -- Schema version tracking --
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER PRIMARY KEY,
        applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        description TEXT
    )
    """,

    # -- Layer 1: Memory Store --
    """
    CREATE TABLE IF NOT EXISTS memories (
        id                      TEXT PRIMARY KEY,
        content                 TEXT NOT NULL,
        summary                 TEXT,
        created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        access_count            INTEGER NOT NULL DEFAULT 0,
        last_accessed           TEXT,
        source_model            TEXT NOT NULL,
        salience_model          REAL NOT NULL DEFAULT 0.5,
        salience_usage          REAL NOT NULL DEFAULT 0.0,
        salience_composite      REAL NOT NULL DEFAULT 0.5,
        retrieval_success_count INTEGER NOT NULL DEFAULT 0,
        retrieval_total_count   INTEGER NOT NULL DEFAULT 0,
        is_archived             INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience_composite DESC)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)",

    """
    CREATE TABLE IF NOT EXISTS embeddings (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        chunk_index  INTEGER NOT NULL DEFAULT 0,
        embedding    BLOB NOT NULL,
        model_name   TEXT NOT NULL,
        dimension    INTEGER NOT NULL,
        generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        UNIQUE(memory_id, chunk_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_embeddings_memory ON embeddings(memory_id)",

    """
    CREATE TABLE IF NOT EXISTS tags (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        created_by  TEXT NOT NULL DEFAULT 'system',
        is_active   INTEGER NOT NULL DEFAULT 1,
        parent_id   INTEGER REFERENCES tags(id),
        merged_into INTEGER REFERENCES tags(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)",

    """
    CREATE TABLE IF NOT EXISTS memory_tags (
        memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        tag_id      INTEGER NOT NULL REFERENCES tags(id),
        assigned_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        assigned_by TEXT NOT NULL DEFAULT 'llm',
        confidence  REAL NOT NULL DEFAULT 1.0,
        PRIMARY KEY (memory_id, tag_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_tags_memory ON memory_tags(memory_id)",

    # -- Layer 2: Trend & Retrieval Tracking --
    """
    CREATE TABLE IF NOT EXISTS tag_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        tag_id      INTEGER NOT NULL REFERENCES tags(id),
        event_type  TEXT NOT NULL,
        occurred_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        memory_id   TEXT REFERENCES memories(id),
        weight      REAL NOT NULL DEFAULT 1.0,
        metadata    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tag_events_tag_time ON tag_events(tag_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_tag_events_type ON tag_events(event_type)",

    """
    CREATE TABLE IF NOT EXISTS retrieval_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text      TEXT NOT NULL,
        context_summary TEXT,
        method          TEXT NOT NULL,
        occurred_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        result_count    INTEGER NOT NULL DEFAULT 0,
        model_version   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_retrieval_events_time ON retrieval_events(occurred_at)",

    """
    CREATE TABLE IF NOT EXISTS retrieval_results (
        retrieval_id    INTEGER NOT NULL REFERENCES retrieval_events(id) ON DELETE CASCADE,
        memory_id       TEXT NOT NULL REFERENCES memories(id),
        rank            INTEGER NOT NULL,
        relevance_score REAL NOT NULL,
        was_useful      INTEGER,
        PRIMARY KEY (retrieval_id, memory_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_retrieval_results_memory ON retrieval_results(memory_id)",

    """
    CREATE TABLE IF NOT EXISTS retrieval_tag_hits (
        retrieval_id INTEGER NOT NULL REFERENCES retrieval_events(id) ON DELETE CASCADE,
        tag_id       INTEGER NOT NULL REFERENCES tags(id),
        hit_type     TEXT NOT NULL,
        PRIMARY KEY (retrieval_id, tag_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_retrieval_tag_hits_tag ON retrieval_tag_hits(tag_id)",

    """
    CREATE TABLE IF NOT EXISTS trend_snapshots (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        tag_id       INTEGER NOT NULL REFERENCES tags(id),
        computed_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        window_hours REAL NOT NULL,
        level        REAL NOT NULL,
        velocity     REAL NOT NULL,
        jerk         REAL NOT NULL,
        temperature  REAL NOT NULL,
        event_count  INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trend_snapshots_tag_time ON trend_snapshots(tag_id, computed_at)",

    # -- Layer 3: Synchronicity --
    """
    CREATE TABLE IF NOT EXISTS synchronicity_events (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        event_type           TEXT NOT NULL,
        description          TEXT NOT NULL,
        strength             REAL NOT NULL,
        quadrant             TEXT NOT NULL,
        involved_tags        TEXT NOT NULL,
        involved_memories    TEXT,
        trigger_retrieval_id INTEGER REFERENCES retrieval_events(id),
        acknowledged         INTEGER NOT NULL DEFAULT 0,
        last_reinforced      TEXT,
        reinforcement_count  INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sync_events_time ON synchronicity_events(detected_at)",
    "CREATE INDEX IF NOT EXISTS idx_sync_events_type ON synchronicity_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_sync_events_strength ON synchronicity_events(strength DESC)",

    # -- Layer 3.5: Prime Ramsey Lattice --
    """
    CREATE TABLE IF NOT EXISTS lattice_positions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        sync_event_id   INTEGER NOT NULL REFERENCES synchronicity_events(id) ON DELETE CASCADE,
        angle           REAL NOT NULL,
        prime_slots     TEXT NOT NULL,
        placed_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        UNIQUE(sync_event_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lattice_pos_event ON lattice_positions(sync_event_id)",
    "CREATE INDEX IF NOT EXISTS idx_lattice_pos_angle ON lattice_positions(angle)",

    """
    CREATE TABLE IF NOT EXISTS resonances (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        event_a_id          INTEGER NOT NULL,
        event_b_id          INTEGER NOT NULL,
        event_ids           TEXT NOT NULL,
        shared_primes       TEXT NOT NULL,
        resonance_strength  REAL NOT NULL,
        description         TEXT NOT NULL,
        detected_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        UNIQUE(event_a_id, event_b_id),
        CHECK(event_a_id < event_b_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_resonances_strength ON resonances(resonance_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_resonances_event_a ON resonances(event_a_id)",
    "CREATE INDEX IF NOT EXISTS idx_resonances_event_b ON resonances(event_b_id)",

    # -- Layer 3.5: Tag Relational Tensor --
    """
    CREATE TABLE IF NOT EXISTS tag_relational_tensor (
        tag_a_id               INTEGER NOT NULL REFERENCES tags(id),
        tag_b_id               INTEGER NOT NULL REFERENCES tags(id),
        cooccurrence_strength   REAL NOT NULL DEFAULT 0.0,
        synchronicity_strength  REAL NOT NULL DEFAULT 0.0,
        semantic_strength       REAL NOT NULL DEFAULT 0.0,
        semiotic_forward        REAL NOT NULL DEFAULT 0.0,
        semiotic_reverse        REAL NOT NULL DEFAULT 0.0,
        memory_ids             TEXT NOT NULL DEFAULT '[]',
        updated_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (tag_a_id, tag_b_id),
        CHECK (tag_a_id < tag_b_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tensor_tag_a ON tag_relational_tensor(tag_a_id)",
    "CREATE INDEX IF NOT EXISTS idx_tensor_tag_b ON tag_relational_tensor(tag_b_id)",
    "CREATE INDEX IF NOT EXISTS idx_tensor_cooccurrence ON tag_relational_tensor(cooccurrence_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tensor_synchronicity ON tag_relational_tensor(synchronicity_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tensor_semantic ON tag_relational_tensor(semantic_strength DESC)",

    # -- Layer 4: Meta-Patterns --
    """
    CREATE TABLE IF NOT EXISTS meta_patterns (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        description           TEXT NOT NULL,
        pattern_type          TEXT NOT NULL,
        confidence            REAL NOT NULL,
        involved_sync_ids     TEXT NOT NULL,
        involved_tag_clusters TEXT NOT NULL,
        actions_taken         TEXT,
        is_active             INTEGER NOT NULL DEFAULT 1,
        validated_by          TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_meta_patterns_time ON meta_patterns(detected_at)",
    "CREATE INDEX IF NOT EXISTS idx_meta_patterns_type ON meta_patterns(pattern_type)",

    """
    CREATE TABLE IF NOT EXISTS adaptive_thresholds (
        metric_name   TEXT PRIMARY KEY,
        current_value REAL NOT NULL,
        baseline_value REAL NOT NULL,
        sample_count  INTEGER NOT NULL DEFAULT 0,
        last_updated  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        burn_in_until TEXT,
        model_version TEXT NOT NULL
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS model_versions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name      TEXT NOT NULL,
        activated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        embedding_model TEXT NOT NULL,
        notes           TEXT
    )
    """,

    # -- Commons Layer: Pending Signals --
    """
    CREATE TABLE IF NOT EXISTS pending_signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id      TEXT NOT NULL,
        op_type         TEXT NOT NULL,
        tags            TEXT NOT NULL,
        strength        REAL,
        event_type      TEXT,
        created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        processed       INTEGER NOT NULL DEFAULT 0,
        tag_set_hash    TEXT NOT NULL,
        UNIQUE(project_id, op_type, tag_set_hash, created_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pending_signals_unprocessed ON pending_signals(processed) WHERE processed = 0",
]


def apply_schema(db: DatabaseEngine) -> None:
    """Create all tables if they don't exist, run migrations, and record schema version."""
    with db.transaction():
        # Ensure schema_version table exists so we can check current version
        db.execute(TABLES[0].strip())

        current = db.execute(
            "SELECT MAX(version) as v FROM schema_version"
        ).fetchone()
        current_version = current["v"] if current and current["v"] else 0

        if current_version == 1:
            _migrate_v1_to_v2(db)
        if current_version <= 2:
            _migrate_v2_to_v3(db)
        if current_version <= 3:
            _migrate_v3_to_v4(db)
        if current_version <= 4:
            _migrate_v4_to_v5(db)
        if current_version <= 5:
            _migrate_v5_to_v6(db)
        if current_version <= 6:
            _migrate_v6_to_v7(db)
        if current_version <= 7:
            _migrate_v7_to_v8(db)
        if current_version <= 8:
            _migrate_v8_to_v9(db)
        if current_version <= 9:
            _migrate_v9_to_v10(db)

        # Create all tables (IF NOT EXISTS handles idempotency)
        for stmt in TABLES:
            db.execute(stmt.strip())

        # Record version if not already present
        row = db.execute(
            "SELECT version FROM schema_version WHERE version = ?",
            (SCHEMA_VERSION,),
        ).fetchone()
        if not row:
            db.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (SCHEMA_VERSION, "Commons layer: cross-project signal federation"),
            )


def _migrate_v1_to_v2(db: DatabaseEngine) -> None:
    """Migrate embeddings table to support multiple chunks per memory."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    ).fetchone()
    if not table_exists:
        return

    # Check if already migrated
    cols = db.execute("PRAGMA table_info(embeddings)").fetchall()
    col_names = {c["name"] for c in cols}
    if "chunk_index" in col_names:
        return

    db.execute("ALTER TABLE embeddings RENAME TO _embeddings_v1")
    db.execute("""
        CREATE TABLE embeddings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            chunk_index  INTEGER NOT NULL DEFAULT 0,
            embedding    BLOB NOT NULL,
            model_name   TEXT NOT NULL,
            dimension    INTEGER NOT NULL,
            generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            UNIQUE(memory_id, chunk_index)
        )
    """)
    db.execute("""
        INSERT INTO embeddings (memory_id, chunk_index, embedding, model_name, dimension, generated_at)
        SELECT memory_id, 0, embedding, model_name, dimension, generated_at
        FROM _embeddings_v1
    """)
    db.execute("DROP TABLE _embeddings_v1")


def _migrate_v2_to_v3(db: DatabaseEngine) -> None:
    """Add lattice_positions and resonances tables for prime Ramsey lattice.

    Tables use IF NOT EXISTS so no data migration is needed.
    """
    pass


def _migrate_v3_to_v4(db: DatabaseEngine) -> None:
    """Multi-timescale context windows.

    No schema changes needed — logic-only feature (multi-tier salience decay,
    lattice-aware retrieval, extended meta-analysis).  Version bump for tracking.
    """
    pass


def _migrate_v4_to_v5(db: DatabaseEngine) -> None:
    """Add synchronicity decay columns: last_reinforced and reinforcement_count."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='synchronicity_events'"
    ).fetchone()
    if not table_exists:
        return  # Fresh database — table will be created with columns included

    cols = db.execute("PRAGMA table_info(synchronicity_events)").fetchall()
    col_names = {c["name"] for c in cols}
    if "last_reinforced" in col_names:
        return  # Already migrated

    db.execute(
        "ALTER TABLE synchronicity_events ADD COLUMN last_reinforced TEXT"
    )
    db.execute(
        "ALTER TABLE synchronicity_events ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0"
    )
    # Backfill: set last_reinforced = detected_at for all existing rows
    db.execute(
        "UPDATE synchronicity_events SET last_reinforced = detected_at WHERE last_reinforced IS NULL"
    )


def _migrate_v5_to_v6(db: DatabaseEngine) -> None:
    """Add memory_tags(memory_id) index for batched tag lookups.

    Index uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v6_to_v7(db: DatabaseEngine) -> None:
    """Add tag_relational_tensor table for three-network O(k) lattice-aware recall.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    Backfill is handled by SynchronicityEngine at boot time.
    """
    pass


def _migrate_v7_to_v8(db: DatabaseEngine) -> None:
    """Add event_a_id/event_b_id columns to resonances for unique constraint.

    The resonances table has never been populated, so safe to drop and recreate.
    """
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='resonances'"
    ).fetchone()
    if not table_exists:
        return  # Will be created by TABLES loop

    # Check if already migrated
    cols = db.execute("PRAGMA table_info(resonances)").fetchall()
    col_names = {c["name"] for c in cols}
    if "event_a_id" in col_names:
        return

    db.execute("DROP TABLE resonances")
    # Will be recreated by TABLES loop with new columns


def _migrate_v8_to_v9(db: DatabaseEngine) -> None:
    """Add semiotic_forward/semiotic_reverse columns to tag_relational_tensor."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_relational_tensor'"
    ).fetchone()
    if not table_exists:
        return  # Will be created by TABLES loop

    cols = db.execute("PRAGMA table_info(tag_relational_tensor)").fetchall()
    col_names = {c["name"] for c in cols}
    if "semiotic_forward" in col_names:
        return  # Already migrated

    db.execute(
        "ALTER TABLE tag_relational_tensor ADD COLUMN semiotic_forward REAL NOT NULL DEFAULT 0.0"
    )
    db.execute(
        "ALTER TABLE tag_relational_tensor ADD COLUMN semiotic_reverse REAL NOT NULL DEFAULT 0.0"
    )


def _migrate_v9_to_v10(db: DatabaseEngine) -> None:
    """Add pending_signals table for cross-project commons federation.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass
