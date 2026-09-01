"""Database schema creation and migration."""

from __future__ import annotations

from chicory.db.engine import DatabaseEngine

SCHEMA_VERSION = 38

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
        is_archived             INTEGER NOT NULL DEFAULT 0,
        content_hash            TEXT,
        source_path             TEXT,
        ingestion_tier          TEXT NOT NULL DEFAULT 'critical',
        author                  TEXT,
        origin                  TEXT,
        swept_at                TEXT,
        custody_status          TEXT NOT NULL DEFAULT 'unswept'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience_composite DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash) WHERE content_hash IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)",
    "CREATE INDEX IF NOT EXISTS idx_memories_source_path ON memories(source_path) WHERE source_path IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(ingestion_tier) WHERE ingestion_tier != 'critical'",
    "CREATE INDEX IF NOT EXISTS idx_memories_swept_at ON memories(swept_at)",

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

    # -- Layer 3: Synchronicity event ↔ memory junction --
    """
    CREATE TABLE IF NOT EXISTS sync_event_memories (
        event_id   INTEGER NOT NULL REFERENCES synchronicity_events(id) ON DELETE CASCADE,
        memory_id  TEXT NOT NULL,
        PRIMARY KEY (event_id, memory_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sync_event_memories_mid ON sync_event_memories(memory_id)",

    # -- Layer 3: Synchronicity event ↔ tag junction --
    """
    CREATE TABLE IF NOT EXISTS sync_event_tags (
        event_id   INTEGER NOT NULL REFERENCES synchronicity_events(id) ON DELETE CASCADE,
        tag_id     INTEGER NOT NULL REFERENCES tags(id),
        PRIMARY KEY (event_id, tag_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sync_event_tags_tid ON sync_event_tags(tag_id)",

    # -- Layer 3.5: Prime Ramsey Lattice --
    """
    CREATE TABLE IF NOT EXISTS lattice_positions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        sync_event_id   INTEGER NOT NULL REFERENCES synchronicity_events(id) ON DELETE CASCADE,
        angle           REAL NOT NULL,
        prime_slots     TEXT NOT NULL,
        poincare_x      REAL,
        poincare_y      REAL,
        placed_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        UNIQUE(sync_event_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lattice_pos_event ON lattice_positions(sync_event_id)",
    "CREATE INDEX IF NOT EXISTS idx_lattice_pos_angle ON lattice_positions(angle)",
    "CREATE INDEX IF NOT EXISTS idx_lattice_pos_placed ON lattice_positions(placed_at DESC)",

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
        glyph_strength          REAL NOT NULL DEFAULT 0.0,
        inhibition_strength     REAL NOT NULL DEFAULT 0.0,
        parallelness            REAL NOT NULL DEFAULT 0.0,
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

    # -- Layer 3.5: Centroid Sub-Graph --
    """
    CREATE TABLE IF NOT EXISTS tag_centroids (
        tag_id       INTEGER PRIMARY KEY REFERENCES tags(id),
        centroid     BLOB NOT NULL,
        memory_count INTEGER NOT NULL DEFAULT 0,
        updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS centroid_edges (
        tag_a_id           INTEGER NOT NULL REFERENCES tags(id),
        tag_b_id           INTEGER NOT NULL REFERENCES tags(id),
        edge_strength      REAL NOT NULL DEFAULT 0.0,
        co_retrieval_count INTEGER NOT NULL DEFAULT 0,
        updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (tag_a_id, tag_b_id),
        CHECK (tag_a_id < tag_b_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_centroid_edges_a ON centroid_edges(tag_a_id)",
    "CREATE INDEX IF NOT EXISTS idx_centroid_edges_b ON centroid_edges(tag_b_id)",
    "CREATE INDEX IF NOT EXISTS idx_centroid_edges_strength ON centroid_edges(edge_strength DESC)",

    # -- Layer 3.5: Glyph Ramsey Lattice --
    """
    CREATE TABLE IF NOT EXISTS glyph_positions (
        tag_id          INTEGER PRIMARY KEY REFERENCES tags(id),
        angle           REAL NOT NULL,
        prime_slots     TEXT NOT NULL,
        glyph_vector    BLOB NOT NULL,
        glyph_dimension INTEGER NOT NULL DEFAULT 26,
        placed_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_glyph_pos_angle ON glyph_positions(angle)",

    """
    CREATE TABLE IF NOT EXISTS glyph_resonances (
        tag_a_id           INTEGER NOT NULL REFERENCES tags(id),
        tag_b_id           INTEGER NOT NULL REFERENCES tags(id),
        shared_primes      TEXT NOT NULL,
        resonance_strength REAL NOT NULL,
        detected_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (tag_a_id, tag_b_id),
        CHECK (tag_a_id < tag_b_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_glyph_res_strength ON glyph_resonances(resonance_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_glyph_res_a ON glyph_resonances(tag_a_id)",
    "CREATE INDEX IF NOT EXISTS idx_glyph_res_b ON glyph_resonances(tag_b_id)",

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

    # -- Glyph Bridge Cache (GPT-GU integration) --
    """
    CREATE TABLE IF NOT EXISTS glyph_bridge_cache (
        tag_name      TEXT PRIMARY KEY,
        glyph_symbol  TEXT NOT NULL,
        glyph_concept TEXT,
        embedding     BLOB NOT NULL,
        embedding_dim INTEGER NOT NULL,
        source        TEXT NOT NULL,
        cached_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,

    # -- Glyph Content Analysis Metadata --
    """
    CREATE TABLE IF NOT EXISTS glyph_metadata (
        memory_id   TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
        glyph_json  TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,

    # -- Forest Layer: Co-occurrence --
    """
    CREATE TABLE IF NOT EXISTS cooccurrence_edges (
        left_type TEXT NOT NULL,
        left_id TEXT NOT NULL,
        right_type TEXT NOT NULL,
        right_id TEXT NOT NULL,
        scope_type TEXT NOT NULL,
        raw_count REAL NOT NULL DEFAULT 0.0,
        expected_count REAL NOT NULL DEFAULT 0.0,
        lift REAL NOT NULL DEFAULT 0.0,
        pmi REAL NOT NULL DEFAULT 0.0,
        co_strength REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (left_type, left_id, right_type, right_id, scope_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cooccurrence_edges_strength ON cooccurrence_edges(scope_type, co_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_cooccurrence_edges_left ON cooccurrence_edges(left_type, left_id)",
    "CREATE INDEX IF NOT EXISTS idx_cooccurrence_edges_right ON cooccurrence_edges(right_type, right_id)",

    # -- Forest Layer: Blocks & Memberships --
    """
    CREATE TABLE IF NOT EXISTS forest_blocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_key TEXT NOT NULL UNIQUE,
        block_type TEXT NOT NULL,
        forest_type TEXT NOT NULL,
        internal_density REAL NOT NULL DEFAULT 0.0,
        external_bridge_strength REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_forest_blocks_type_density ON forest_blocks(forest_type, internal_density DESC)",

    """
    CREATE TABLE IF NOT EXISTS block_memberships (
        block_id INTEGER NOT NULL REFERENCES forest_blocks(id) ON DELETE RESTRICT,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        membership_strength REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (block_id, target_type, target_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_block_memberships_target ON block_memberships(target_type, target_id)",

    # -- Forest Layer: Bridge Edges & Adjacency --
    """
    CREATE TABLE IF NOT EXISTS bridge_edges (
        left_block_id INTEGER NOT NULL REFERENCES forest_blocks(id) ON DELETE RESTRICT,
        right_block_id INTEGER NOT NULL REFERENCES forest_blocks(id) ON DELETE RESTRICT,
        connection_strength REAL NOT NULL DEFAULT 0.0,
        cluster_distance REAL NOT NULL DEFAULT 0.0,
        rarity_bonus REAL NOT NULL DEFAULT 0.0,
        bridge_strength REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (left_block_id, right_block_id),
        CHECK (left_block_id < right_block_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_bridge_edges_strength ON bridge_edges(bridge_strength DESC)",

    """
    CREATE TABLE IF NOT EXISTS block_adjacency (
        left_block_id INTEGER NOT NULL REFERENCES forest_blocks(id) ON DELETE RESTRICT,
        right_block_id INTEGER NOT NULL REFERENCES forest_blocks(id) ON DELETE RESTRICT,
        adjacency_type TEXT NOT NULL,
        cooccurrence_weight REAL NOT NULL DEFAULT 0.0,
        bridge_weight REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (left_block_id, right_block_id, adjacency_type),
        CHECK (left_block_id < right_block_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_block_adjacency_left ON block_adjacency(left_block_id)",

    # -- Forest Layer: Snapshots --
    """
    CREATE TABLE IF NOT EXISTS forest_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        trigger_type TEXT NOT NULL,
        trigger_id TEXT,
        touched_memory_ids TEXT NOT NULL DEFAULT '[]',
        touched_tag_ids TEXT NOT NULL DEFAULT '[]',
        touched_block_ids TEXT NOT NULL DEFAULT '[]',
        co_edge_count INTEGER NOT NULL DEFAULT 0,
        bridge_edge_count INTEGER NOT NULL DEFAULT 0,
        block_count INTEGER NOT NULL DEFAULT 0,
        notes TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_forest_snapshots_time ON forest_snapshots(snapshot_at DESC)",

    # -- Canopy Layer: Blocks --
    """
    CREATE TABLE IF NOT EXISTS canopy_blocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_key TEXT NOT NULL UNIQUE,
        block_type TEXT NOT NULL,
        layer_depth INTEGER NOT NULL DEFAULT 0,
        tag_ids TEXT NOT NULL DEFAULT '[]',
        memory_ids TEXT NOT NULL DEFAULT '[]',
        parent_block_keys TEXT NOT NULL DEFAULT '[]',
        source_event_types TEXT NOT NULL DEFAULT '[]',
        peak_bridge REAL NOT NULL DEFAULT 0.0,
        peak_heat REAL NOT NULL DEFAULT 0.0,
        peak_recurrence REAL NOT NULL DEFAULT 0.0,
        peak_cooccurrence REAL NOT NULL DEFAULT 0.0,
        peak_similarity REAL NOT NULL DEFAULT 0.0,
        peak_relevance REAL NOT NULL DEFAULT 0.0,
        peak_semantics REAL NOT NULL DEFAULT 0.0,
        peak_pressure REAL NOT NULL DEFAULT 0.0,
        peak_threshold REAL NOT NULL DEFAULT 0.0,
        peak_growth_potential REAL NOT NULL DEFAULT 0.0,
        peak_canopy_growth REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        first_growth_at TEXT,
        canonical_block_key TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_canopy_blocks_type_growth ON canopy_blocks(layer_depth, block_type, peak_canopy_growth DESC)",
    "CREATE INDEX IF NOT EXISTS idx_canopy_blocks_first_growth ON canopy_blocks(first_growth_at)",
    "CREATE INDEX IF NOT EXISTS idx_canopy_blocks_last_observed ON canopy_blocks(last_observed_at DESC)",

    # -- Canopy Layer: Cross-Layer Inhibition Edges --
    """
    CREATE TABLE IF NOT EXISTS canopy_cross_layer_edges (
        relevance_block_id INTEGER NOT NULL REFERENCES canopy_blocks(id) ON DELETE RESTRICT,
        semantic_block_id INTEGER NOT NULL REFERENCES canopy_blocks(id) ON DELETE RESTRICT,
        edge_inhibition REAL NOT NULL DEFAULT 0.0,
        a_heat REAL NOT NULL DEFAULT 0.0,
        a_similarity REAL NOT NULL DEFAULT 0.0,
        a_cooccurrence REAL NOT NULL DEFAULT 0.0,
        a_recurrence REAL NOT NULL DEFAULT 0.0,
        a_bridge REAL NOT NULL DEFAULT 0.0,
        edge_heat REAL NOT NULL DEFAULT 0.0,
        edge_similarity REAL NOT NULL DEFAULT 0.0,
        edge_cooccurrence REAL NOT NULL DEFAULT 0.0,
        edge_recurrence REAL NOT NULL DEFAULT 0.0,
        edge_bridge REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (relevance_block_id, semantic_block_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_canopy_cross_layer_edges_inhibition ON canopy_cross_layer_edges(edge_inhibition DESC)",

    # -- Canopy Layer: Support Edges --
    """
    CREATE TABLE IF NOT EXISTS canopy_support_edges (
        canopy_block_id INTEGER NOT NULL REFERENCES canopy_blocks(id) ON DELETE RESTRICT,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        strength REAL NOT NULL DEFAULT 0.0,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        PRIMARY KEY (canopy_block_id, target_type, target_id, edge_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_canopy_support_edges_target ON canopy_support_edges(target_type, target_id)",

    # -- Canopy Layer: Observations (append-only) --
    """
    CREATE TABLE IF NOT EXISTS canopy_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_key TEXT NOT NULL,
        observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        source TEXT NOT NULL,
        source_id TEXT,
        layer_depth INTEGER NOT NULL DEFAULT 0,
        tag_ids TEXT NOT NULL DEFAULT '[]',
        memory_ids TEXT NOT NULL DEFAULT '[]',
        source_canopy_block_ids TEXT NOT NULL DEFAULT '[]',
        bridge REAL NOT NULL,
        heat REAL NOT NULL,
        recurrence REAL NOT NULL,
        cooccurrence REAL NOT NULL,
        similarity REAL NOT NULL,
        relevance REAL NOT NULL,
        semantics REAL NOT NULL,
        pressure REAL NOT NULL,
        threshold REAL NOT NULL,
        growth_potential REAL NOT NULL,
        canopy_growth REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_canopy_observations_key_time ON canopy_observations(block_key, observed_at DESC)",

    # -- Episodic Relational Tensor: memory-to-memory edge cache --
    """
    CREATE TABLE IF NOT EXISTS memory_relational_tensor (
        memory_a_id              TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        memory_b_id              TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,

        semantic_strength        REAL NOT NULL DEFAULT 0.0,
        tag_projected_strength   REAL NOT NULL DEFAULT 0.0,
        co_retrieval_strength    REAL NOT NULL DEFAULT 0.0,
        temporal_proximity       REAL NOT NULL DEFAULT 0.0,
        source_proximity         REAL NOT NULL DEFAULT 0.0,

        tag_semantic_projected   REAL NOT NULL DEFAULT 0.0,
        tag_sync_projected       REAL NOT NULL DEFAULT 0.0,
        tag_cooccurrence_projected REAL NOT NULL DEFAULT 0.0,
        tag_inhibition_projected REAL NOT NULL DEFAULT 0.0,
        tag_glyph_projected      REAL NOT NULL DEFAULT 0.0,

        retrieval_reinforcement  REAL NOT NULL DEFAULT 0.0,
        narrative_continuity     REAL NOT NULL DEFAULT 0.0,
        supersession_strength    REAL NOT NULL DEFAULT 0.0,
        supersession_direction   INTEGER NOT NULL DEFAULT 0,
        contradiction_strength   REAL NOT NULL DEFAULT 0.0,
        bridge_strength          REAL NOT NULL DEFAULT 0.0,

        activation_count         INTEGER NOT NULL DEFAULT 0,
        edge_status              TEXT NOT NULL DEFAULT 'candidate',
        created_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_activated_at        TEXT,

        PRIMARY KEY (memory_a_id, memory_b_id),
        CHECK (memory_a_id < memory_b_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mrt_a ON memory_relational_tensor(memory_a_id)",
    "CREATE INDEX IF NOT EXISTS idx_mrt_b ON memory_relational_tensor(memory_b_id)",
    "CREATE INDEX IF NOT EXISTS idx_mrt_status ON memory_relational_tensor(edge_status)",
    "CREATE INDEX IF NOT EXISTS idx_mrt_tag_projected ON memory_relational_tensor(tag_projected_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mrt_bridge ON memory_relational_tensor(bridge_strength DESC)",

    # -- Temporal tag episodes --
    """
    CREATE TABLE IF NOT EXISTS temporal_episodes (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        centroid          BLOB NOT NULL,
        tag_ids           TEXT NOT NULL DEFAULT '[]',
        status            TEXT NOT NULL DEFAULT 'active',
        visit_count       INTEGER NOT NULL DEFAULT 1,
        operation_count   INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_active_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        snapshot_at       TEXT,
        mean_distance     REAL NOT NULL DEFAULT 0.3,
        variance_sum      REAL NOT NULL DEFAULT 0.0,
        distance_samples  INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_temporal_episodes_status ON temporal_episodes(status)",
    "CREATE INDEX IF NOT EXISTS idx_temporal_episodes_last_active ON temporal_episodes(last_active_at DESC)",

    """
    CREATE TABLE IF NOT EXISTS episode_transitions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        from_episode_id   INTEGER NOT NULL REFERENCES temporal_episodes(id),
        to_episode_id     INTEGER NOT NULL REFERENCES temporal_episodes(id),
        transition_type   TEXT NOT NULL,
        transition_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        metadata          TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_episode_transitions_from ON episode_transitions(from_episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_episode_transitions_to ON episode_transitions(to_episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_episode_transitions_type ON episode_transitions(transition_type)",

    """
    CREATE TABLE IF NOT EXISTS memory_episode_assignments (
        memory_id       TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        episode_id      INTEGER NOT NULL REFERENCES temporal_episodes(id),
        assigned_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        assignment_type TEXT NOT NULL DEFAULT 'store',
        PRIMARY KEY (memory_id, episode_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mem_episode_memory ON memory_episode_assignments(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_mem_episode_episode ON memory_episode_assignments(episode_id)",

    # -- Directional canopy: inflow/outflow tracking --
    """
    CREATE TABLE IF NOT EXISTS directional_retrieval_context (
        retrieval_id    INTEGER PRIMARY KEY REFERENCES retrieval_events(id) ON DELETE CASCADE,
        query_tag_ids   TEXT NOT NULL DEFAULT '[]',
        result_tag_ids  TEXT NOT NULL DEFAULT '[]',
        result_memory_ids TEXT NOT NULL DEFAULT '[]',
        query_tag_hash  TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_drc_query_hash ON directional_retrieval_context(query_tag_hash)",

    """
    CREATE TABLE IF NOT EXISTS inflow_canopy_blocks (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        block_key            TEXT NOT NULL UNIQUE,
        memory_ids           TEXT NOT NULL DEFAULT '[]',
        representative_tags  TEXT NOT NULL DEFAULT '[]',
        inflow_diversity     REAL NOT NULL DEFAULT 0.0,
        inflow_strength      REAL NOT NULL DEFAULT 0.0,
        unique_query_contexts INTEGER NOT NULL DEFAULT 0,
        total_activations    INTEGER NOT NULL DEFAULT 0,
        evidence_count       INTEGER NOT NULL DEFAULT 0,
        created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_observed_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_inflow_blocks_strength ON inflow_canopy_blocks(inflow_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_inflow_blocks_diversity ON inflow_canopy_blocks(inflow_diversity DESC)",

    """
    CREATE TABLE IF NOT EXISTS inflow_canopy_observations (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        block_key         TEXT NOT NULL,
        retrieval_id      INTEGER NOT NULL,
        query_tag_hash    TEXT NOT NULL,
        query_tag_ids     TEXT NOT NULL DEFAULT '[]',
        result_memory_ids TEXT NOT NULL DEFAULT '[]',
        inflow_diversity  REAL NOT NULL DEFAULT 0.0,
        inflow_strength   REAL NOT NULL DEFAULT 0.0,
        observed_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_inflow_obs_key_time ON inflow_canopy_observations(block_key, observed_at DESC)",

    """
    CREATE TABLE IF NOT EXISTS outflow_canopy_blocks (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        block_key             TEXT NOT NULL UNIQUE,
        query_tag_ids         TEXT NOT NULL DEFAULT '[]',
        query_tag_hash        TEXT NOT NULL,
        outflow_diversity     REAL NOT NULL DEFAULT 0.0,
        outflow_reach         REAL NOT NULL DEFAULT 0.0,
        outflow_strength      REAL NOT NULL DEFAULT 0.0,
        unique_result_clusters INTEGER NOT NULL DEFAULT 0,
        total_activations     INTEGER NOT NULL DEFAULT 0,
        evidence_count        INTEGER NOT NULL DEFAULT 0,
        created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_observed_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_outflow_blocks_strength ON outflow_canopy_blocks(outflow_strength DESC)",
    "CREATE INDEX IF NOT EXISTS idx_outflow_blocks_hash ON outflow_canopy_blocks(query_tag_hash)",

    """
    CREATE TABLE IF NOT EXISTS outflow_canopy_observations (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        block_key         TEXT NOT NULL,
        retrieval_id      INTEGER NOT NULL,
        result_memory_ids TEXT NOT NULL DEFAULT '[]',
        result_cluster_key TEXT NOT NULL,
        outflow_diversity REAL NOT NULL DEFAULT 0.0,
        outflow_reach     REAL NOT NULL DEFAULT 0.0,
        outflow_strength  REAL NOT NULL DEFAULT 0.0,
        observed_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_outflow_obs_key_time ON outflow_canopy_observations(block_key, observed_at DESC)",

    """
    CREATE TABLE IF NOT EXISTS directional_block_scores (
        block_id               INTEGER PRIMARY KEY REFERENCES forest_blocks(id) ON DELETE CASCADE,
        inflow_score           REAL NOT NULL DEFAULT 0.0,
        outflow_score          REAL NOT NULL DEFAULT 0.0,
        directional_hub_penalty REAL NOT NULL DEFAULT 1.0,
        evidence_count         INTEGER NOT NULL DEFAULT 0,
        updated_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,

    # -- Librarian: subject roster (named topical clusters over tags) --
    """
    CREATE TABLE IF NOT EXISTS subjects (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        name           TEXT NOT NULL UNIQUE,
        description    TEXT,
        centroid       BLOB,
        member_tag_ids TEXT NOT NULL,
        memory_count   INTEGER NOT NULL DEFAULT 0,
        status         TEXT NOT NULL DEFAULT 'active',
        parent_id      INTEGER REFERENCES subjects(id),
        depth          INTEGER NOT NULL DEFAULT 0,
        path           TEXT,
        display_name   TEXT,
        created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_subjects_status ON subjects(status)",
    "CREATE INDEX IF NOT EXISTS idx_subjects_parent ON subjects(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_subjects_depth ON subjects(depth)",
    "CREATE INDEX IF NOT EXISTS idx_subjects_path ON subjects(path)",

    # -- Librarian: write gate verdict log (shadow-mode calibration + audit) --
    """
    CREATE TABLE IF NOT EXISTS write_gate_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id   TEXT NOT NULL,
        best_tag_id INTEGER,
        score       REAL NOT NULL,
        verdict     TEXT NOT NULL,
        enforced    INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_write_gate_log_verdict ON write_gate_log(verdict)",
    "CREATE INDEX IF NOT EXISTS idx_write_gate_log_created ON write_gate_log(created_at)",

    # -- Librarian: linear sweep findings (shelf-reading patrol journal) --
    """
    CREATE TABLE IF NOT EXISTS sweep_findings (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id    TEXT NOT NULL,
        finding_type TEXT NOT NULL,
        detail       TEXT,
        resolved     INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        UNIQUE(memory_id, finding_type, detail)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sweep_findings_open ON sweep_findings(resolved) WHERE resolved = 0",

    # -- Librarian: custody transition journal (auditable status changes) --
    """
    CREATE TABLE IF NOT EXISTS custody_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id   TEXT NOT NULL,
        from_status TEXT NOT NULL,
        to_status   TEXT NOT NULL,
        actor       TEXT NOT NULL,
        reason      TEXT,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_custody_events_memory ON custody_events(memory_id)",

    # -- Librarian: demand ledger (queries as first-class archive) --
    """
    CREATE TABLE IF NOT EXISTS query_history (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text   TEXT NOT NULL,
        embedding    BLOB,
        result_count INTEGER NOT NULL DEFAULT 0,
        top_score    REAL NOT NULL DEFAULT 0.0,
        processed    INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_query_history_unprocessed ON query_history(processed) WHERE processed = 0",

    """
    CREATE TABLE IF NOT EXISTS query_clusters (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        centroid             BLOB NOT NULL,
        representative_query TEXT NOT NULL,
        hit_count            INTEGER NOT NULL DEFAULT 1,
        mean_top_score       REAL NOT NULL DEFAULT 0.0,
        created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_seen_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,

    # -- Librarian: FAQ desk (crystallized answers to recurring questions) --
    """
    CREATE TABLE IF NOT EXISTS faq_entries (
        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
        question_text          TEXT NOT NULL,
        question_centroid      BLOB NOT NULL,
        answer                 TEXT,
        supporting_memory_ids  TEXT NOT NULL,
        subject_ids            TEXT NOT NULL DEFAULT '[]',
        status                 TEXT NOT NULL DEFAULT 'candidate',
        hit_count              INTEGER NOT NULL DEFAULT 0,
        author                 TEXT,
        created_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        last_verified_at       TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_faq_entries_status ON faq_entries(status)",

    # -- Librarian: commission queue (assigned research on known gaps) --
    """
    CREATE TABLE IF NOT EXISTS commissions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        question_text    TEXT NOT NULL,
        centroid         BLOB,
        source           TEXT NOT NULL DEFAULT 'manual',
        subject_ids      TEXT NOT NULL DEFAULT '[]',
        status           TEXT NOT NULL DEFAULT 'open',
        report_memory_id TEXT,
        created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_commissions_status ON commissions(status)",

    # -- Librarian: penumbra ledger (near-misses per memory, aggregated) --
    """
    CREATE TABLE IF NOT EXISTS retrieval_penumbra (
        memory_id         TEXT PRIMARY KEY,
        near_count        INTEGER NOT NULL DEFAULT 0,
        retrieved_count   INTEGER NOT NULL DEFAULT 0,
        last_near_at      TEXT,
        last_retrieved_at TEXT
    )
    """,

    # -- Librarian: ingress buffer (receipted dwell before commitment) --
    """
    CREATE TABLE IF NOT EXISTS message_ingress (
        receipt        TEXT PRIMARY KEY,
        content        TEXT NOT NULL,
        content_hash   TEXT NOT NULL,
        embedding      BLOB,
        source         TEXT NOT NULL DEFAULT 'user',
        origin         TEXT,
        author         TEXT,
        arrived_at     TEXT NOT NULL,
        dwell_until    TEXT NOT NULL,
        status         TEXT NOT NULL DEFAULT 'held',
        examined_count INTEGER NOT NULL DEFAULT 0,
        memory_id      TEXT,
        settled_at     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingress_status ON message_ingress(status)",
    "CREATE INDEX IF NOT EXISTS idx_ingress_dwell ON message_ingress(dwell_until) WHERE status = 'held'",
    "CREATE INDEX IF NOT EXISTS idx_ingress_hash ON message_ingress(content_hash)",

    """
    CREATE TABLE IF NOT EXISTS ingress_findings (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        receipt      TEXT NOT NULL,
        finding_type TEXT NOT NULL,
        detail       TEXT,
        severity     REAL NOT NULL DEFAULT 0.0,
        -- '' (never NULL) for single-receipt findings: SQLite treats NULLs
        -- as distinct in UNIQUE, so a nullable column here would let every
        -- re-examination insert a duplicate of the same finding
        peer_receipt TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        UNIQUE(receipt, finding_type, peer_receipt)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingress_findings_receipt ON ingress_findings(receipt)",

    # -- Librarian: supersession (authored, never derived, never pruned) --
    """
    CREATE TABLE IF NOT EXISTS supersessions (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        superseded_id  TEXT NOT NULL,
        superseding_id TEXT NOT NULL,
        scope          TEXT NOT NULL DEFAULT 'claim',
        -- '' (never NULL) for "no specific claim named": SQLite treats
        -- NULLs as distinct in UNIQUE, so a nullable column here would let
        -- every re-record insert a duplicate and stop ON CONFLICT firing.
        -- Same trap as ingress_findings.peer_receipt above.
        claim          TEXT NOT NULL DEFAULT '',
        reason         TEXT,
        actor          TEXT NOT NULL,
        retracted      INTEGER NOT NULL DEFAULT 0,
        created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        UNIQUE(superseded_id, superseding_id, claim)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_supersessions_old ON supersessions(superseded_id) WHERE retracted = 0",
    "CREATE INDEX IF NOT EXISTS idx_supersessions_new ON supersessions(superseding_id)",

    # -- Librarian: durable key/value state (watermarks, checkpoints) --
    """
    CREATE TABLE IF NOT EXISTS librarian_state (
        key        TEXT PRIMARY KEY,
        value      TEXT,
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,

    # -- Wandering librarian: per-errand visit stamps (distributed cursor) --
    #
    # memories.swept_at is the same idea with one errand hardcoded into a
    # column.  This generalises it: the position of every errand lives on
    # the objects it visits rather than in a pointer, so each errand walks
    # its own lap, restart-safe, without a checkpoint to lose.  A memory
    # with no row here has never been visited BY THIS ERRAND and sorts
    # first, exactly as a NULL swept_at does.
    """
    CREATE TABLE IF NOT EXISTS patrol_visits (
        errand     TEXT NOT NULL,
        memory_id  TEXT NOT NULL,
        visited_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        lap        INTEGER NOT NULL DEFAULT 0,
        verdict    TEXT NOT NULL DEFAULT 'ok',
        detail     TEXT,
        PRIMARY KEY (errand, memory_id)
    )
    """,
    # The patrol's ordering index: least-recently-visited first, per errand.
    "CREATE INDEX IF NOT EXISTS idx_patrol_visits_order "
    "ON patrol_visits(errand, visited_at)",
    "CREATE INDEX IF NOT EXISTS idx_patrol_visits_memory "
    "ON patrol_visits(memory_id)",

    # -- Wandering librarian: what each errand is currently curious about --
    #
    # The small centralised half of the state.  Position is distributed
    # (patrol_visits); this holds the things that have no natural home on
    # an object: which lap the errand is on, where in the shelf tree it is
    # working, and the RNG seed — so a walk that chooses randomly can be
    # replayed exactly rather than merely described.
    """
    CREATE TABLE IF NOT EXISTS wander_state (
        errand        TEXT PRIMARY KEY,
        lap           INTEGER NOT NULL DEFAULT 0,
        cursor_hint   TEXT,
        seed          INTEGER NOT NULL DEFAULT 0,
        visited_count INTEGER NOT NULL DEFAULT 0,
        acted_count   INTEGER NOT NULL DEFAULT 0,
        runs          INTEGER NOT NULL DEFAULT 0,
        enabled       INTEGER NOT NULL DEFAULT 1,
        last_run_at   TEXT,
        last_result   TEXT,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
        updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
    )
    """,
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

        if current_version > 0:
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
            if current_version <= 10:
                _migrate_v10_to_v11(db)
            if current_version <= 11:
                _migrate_v11_to_v12(db)
            if current_version <= 12:
                _migrate_v12_to_v13(db)
            if current_version <= 13:
                _migrate_v13_to_v14(db)
            if current_version <= 14:
                _migrate_v14_to_v15(db)
            if current_version <= 15:
                _migrate_v15_to_v16(db)
            if current_version <= 16:
                _migrate_v16_to_v17(db)
            if current_version <= 17:
                _migrate_v17_to_v18(db)
            if current_version <= 18:
                _migrate_v18_to_v19(db)
            if current_version <= 19:
                _migrate_v19_to_v20(db)
            if current_version <= 20:
                _migrate_v20_to_v21(db)
            if current_version <= 21:
                _migrate_v21_to_v22(db)
            if current_version <= 22:
                _migrate_v22_to_v23(db)
            if current_version <= 23:
                _migrate_v23_to_v24(db)
            if current_version <= 24:
                _migrate_v24_to_v25(db)
            if current_version <= 25:
                _migrate_v25_to_v26(db)
            if current_version <= 26:
                _migrate_v26_to_v27(db)
            if current_version <= 27:
                _migrate_v27_to_v28(db)
            if current_version <= 28:
                _migrate_v28_to_v29(db)
            if current_version <= 29:
                _migrate_v29_to_v30(db)
            if current_version <= 30:
                _migrate_v30_to_v31(db)
            if current_version <= 31:
                _migrate_v31_to_v32(db)
            if current_version <= 32:
                _migrate_v32_to_v33(db)
            if current_version <= 33:
                _migrate_v33_to_v34(db)
            if current_version <= 34:
                _migrate_v34_to_v35(db)
            if current_version <= 35:
                _migrate_v35_to_v36(db)
            if current_version <= 36:
                _migrate_v36_to_v37(db)
            if current_version <= 37:
                _migrate_v37_to_v38(db)

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
                (SCHEMA_VERSION, "Wandering librarian: patrol_visits + wander_state"),
            )


def _has_table(db: DatabaseEngine, name: str) -> bool:
    """Whether a table exists yet.

    Migrations run BEFORE the TABLES loop creates anything, so a migration
    that reads a table it did not itself create must check first — the
    table may belong to a version the database has not reached.
    """
    return db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


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


def _migrate_v10_to_v11(db: DatabaseEngine) -> None:
    """Add tag_centroids and centroid_edges tables for retrieval-driven inhibition.

    Tables use IF NOT EXISTS in TABLES, so no explicit migration needed.
    Centroid backfill from existing embeddings is deferred to orchestrator boot.
    """
    pass


def _migrate_v11_to_v12(db: DatabaseEngine) -> None:
    """Add content_hash column to memories for O(1) dedup lookups."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    if not table_exists:
        return

    cols = db.execute("PRAGMA table_info(memories)").fetchall()
    col_names = {c["name"] for c in cols}
    if "content_hash" in col_names:
        return

    db.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")

    # Backfill from existing <!-- chicory:hash=XXXX --> markers
    import re
    rows = db.execute("SELECT id, content FROM memories").fetchall()
    for row in rows:
        m = re.search(r"<!-- chicory:hash=([0-9a-f]+) -->", row["content"])
        if m:
            db.execute(
                "UPDATE memories SET content_hash = ? WHERE id = ?",
                (m.group(1), row["id"]),
            )

    # Deduplicate: keep only the newest memory per content_hash, NULL the rest.
    # Duplicate hashes arise from repeated codebase ingestions.
    db.execute("""
        UPDATE memories SET content_hash = NULL
        WHERE content_hash IS NOT NULL
          AND rowid NOT IN (
              SELECT MAX(rowid) FROM memories
              WHERE content_hash IS NOT NULL
              GROUP BY content_hash
          )
    """)


def _migrate_v12_to_v13(db: DatabaseEngine) -> None:
    """Add glyph Ramsey lattice tables and glyph_strength column to tensor."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_relational_tensor'"
    ).fetchone()
    if not table_exists:
        return

    cols = db.execute("PRAGMA table_info(tag_relational_tensor)").fetchall()
    col_names = {c["name"] for c in cols}
    if "glyph_strength" in col_names:
        return

    db.execute(
        "ALTER TABLE tag_relational_tensor ADD COLUMN glyph_strength REAL NOT NULL DEFAULT 0.0"
    )


def _migrate_v13_to_v14(db: DatabaseEngine) -> None:
    """Convert glyph_vector from TEXT (JSON) to BLOB (float32 array).

    Glyph data is fully rebuildable from tag names, so drop and recreate.
    The orchestrator's _maybe_seed_glyph_lattice will rebuild on next boot.
    """
    db.execute("DROP TABLE IF EXISTS glyph_resonances")
    db.execute("DROP TABLE IF EXISTS glyph_positions")
    # Tables are recreated by the CREATE IF NOT EXISTS pass in apply_schema.
    # Reset glyph_strength in tensor so rebuild repopulates it.
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_relational_tensor'"
    ).fetchone()
    if table_exists:
        db.execute("UPDATE tag_relational_tensor SET glyph_strength = 0.0")


def _migrate_v14_to_v15(db: DatabaseEngine) -> None:
    """Add glyph_bridge_cache table for GPT-GU glyph translation caching.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v15_to_v16(db: DatabaseEngine) -> None:
    """Add inhibition_strength and parallelness columns to tag_relational_tensor."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_relational_tensor'"
    ).fetchone()
    if not table_exists:
        return  # Will be created by TABLES loop

    cols = db.execute("PRAGMA table_info(tag_relational_tensor)").fetchall()
    col_names = {c["name"] for c in cols}
    if "inhibition_strength" not in col_names:
        db.execute(
            "ALTER TABLE tag_relational_tensor ADD COLUMN inhibition_strength REAL NOT NULL DEFAULT 0.0"
        )
    if "parallelness" not in col_names:
        db.execute(
            "ALTER TABLE tag_relational_tensor ADD COLUMN parallelness REAL NOT NULL DEFAULT 0.0"
        )


def _migrate_v16_to_v17(db: DatabaseEngine) -> None:
    """Add glyph_metadata table for content-level glyph analysis.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v17_to_v18(db: DatabaseEngine) -> None:
    """Add Poincaré disk coordinates to lattice_positions."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lattice_positions'"
    ).fetchone()
    if not table_exists:
        return

    cols = db.execute("PRAGMA table_info(lattice_positions)").fetchall()
    col_names = {c["name"] for c in cols}
    if "poincare_x" in col_names:
        return

    db.execute("ALTER TABLE lattice_positions ADD COLUMN poincare_x REAL")
    db.execute("ALTER TABLE lattice_positions ADD COLUMN poincare_y REAL")

    # Backfill: place existing positions at r=0.5 using their existing angle.
    # The engine's reseed() will recompute proper Poincaré coords with depth.
    import math
    rows = db.execute("SELECT id, angle FROM lattice_positions").fetchall()
    for row in rows:
        angle = row["angle"]
        r = 0.5
        px = r * math.cos(angle)
        py = r * math.sin(angle)
        db.execute(
            "UPDATE lattice_positions SET poincare_x = ?, poincare_y = ? WHERE id = ?",
            (px, py, row["id"]),
        )


def _migrate_v18_to_v19(db: DatabaseEngine) -> None:
    """Create sync_event_memories junction table and backfill from JSON."""
    import json

    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sync_event_memories'"
    ).fetchone()
    if table_exists:
        return

    source_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='synchronicity_events'"
    ).fetchone()

    db.execute("""
        CREATE TABLE sync_event_memories (
            event_id   INTEGER NOT NULL REFERENCES synchronicity_events(id) ON DELETE CASCADE,
            memory_id  TEXT NOT NULL,
            PRIMARY KEY (event_id, memory_id)
        )
    """)
    db.execute(
        "CREATE INDEX idx_sync_event_memories_mid ON sync_event_memories(memory_id)"
    )

    if not source_exists:
        return

    cursor = db.execute(
        "SELECT id, involved_memories FROM synchronicity_events WHERE involved_memories IS NOT NULL"
    )
    rows_to_insert = []
    while True:
        batch = cursor.fetchmany(500)
        if not batch:
            break
        for row in batch:
            try:
                mids = json.loads(row["involved_memories"])
            except (json.JSONDecodeError, TypeError):
                continue
            for mid in mids:
                rows_to_insert.append((row["id"], str(mid)))

    if rows_to_insert:
        db.connection.executemany(
            "INSERT OR IGNORE INTO sync_event_memories (event_id, memory_id) VALUES (?, ?)",
            rows_to_insert,
        )


def _migrate_v19_to_v20(db: DatabaseEngine) -> None:
    """Add forest and canopy graph tables.

    All tables use IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v20_to_v21(db: DatabaseEngine) -> None:
    """Add memory_relational_tensor table for episodic edge cache.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v21_to_v22(db: DatabaseEngine) -> None:
    """Add source_path and ingestion_tier columns to memories for two-tier ingestion."""
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    if not table_exists:
        return

    cols = db.execute("PRAGMA table_info(memories)").fetchall()
    col_names = {c["name"] for c in cols}

    if "source_path" not in col_names:
        db.execute("ALTER TABLE memories ADD COLUMN source_path TEXT")
    if "ingestion_tier" not in col_names:
        db.execute(
            "ALTER TABLE memories ADD COLUMN ingestion_tier TEXT NOT NULL DEFAULT 'critical'"
        )


def _migrate_v28_to_v29(db: DatabaseEngine) -> None:
    """Add librarian provenance + sweep columns to memories.

    New tables (subjects, write_gate_log, sweep_findings) use IF NOT EXISTS
    in TABLES; only the memories columns need explicit migration.
    """
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    if not table_exists:
        return

    cols = db.execute("PRAGMA table_info(memories)").fetchall()
    col_names = {c["name"] for c in cols}

    if "author" not in col_names:
        db.execute("ALTER TABLE memories ADD COLUMN author TEXT")
    if "origin" not in col_names:
        db.execute("ALTER TABLE memories ADD COLUMN origin TEXT")
    if "swept_at" not in col_names:
        db.execute("ALTER TABLE memories ADD COLUMN swept_at TEXT")


def _migrate_v29_to_v30(db: DatabaseEngine) -> None:
    """Add custody_status to memories and backfill from existing state.

    New tables (custody_events, query_history, query_clusters, faq_entries,
    commissions) use IF NOT EXISTS in TABLES.
    """
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    if not table_exists:
        return

    cols = db.execute("PRAGMA table_info(memories)").fetchall()
    col_names = {c["name"] for c in cols}
    if "custody_status" in col_names:
        return

    db.execute(
        "ALTER TABLE memories ADD COLUMN custody_status TEXT NOT NULL DEFAULT 'unswept'"
    )

    # Backfill: quarantined tier wins; then swept memories split on open findings
    db.execute(
        "UPDATE memories SET custody_status = 'quarantined' "
        "WHERE ingestion_tier = 'quarantine'"
    )
    findings_exist = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sweep_findings'"
    ).fetchone()
    if findings_exist:
        db.execute(
            """
            UPDATE memories SET custody_status = 'flagged'
            WHERE custody_status = 'unswept' AND swept_at IS NOT NULL
              AND id IN (SELECT memory_id FROM sweep_findings WHERE resolved = 0)
            """
        )
    db.execute(
        "UPDATE memories SET custody_status = 'clean' "
        "WHERE custody_status = 'unswept' AND swept_at IS NOT NULL"
    )

    # Journal the backfill — custody transitions are never silent, even
    # migration-time ones.  (custody_events is normally created by the
    # TABLES loop, which runs AFTER migrations — create it here.)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS custody_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id   TEXT NOT NULL,
            from_status TEXT NOT NULL,
            to_status   TEXT NOT NULL,
            actor       TEXT NOT NULL,
            reason      TEXT,
            created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
        )
        """
    )
    db.execute(
        """
        INSERT INTO custody_events (memory_id, from_status, to_status, actor, reason)
        SELECT id, 'unswept', custody_status, 'migration', 'v29->v30 backfill'
        FROM memories WHERE custody_status != 'unswept'
        """
    )


def _migrate_v30_to_v31(db: DatabaseEngine) -> None:
    """Add retrieval_penumbra table.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v31_to_v32(db: DatabaseEngine) -> None:
    """Add message_ingress + ingress_findings tables.

    Tables use IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v36_to_v37(db: DatabaseEngine) -> None:
    """Rebuild supersessions with a NOT NULL claim.

    A nullable column inside a UNIQUE tuple enforces nothing in SQLite,
    so re-recording a supersession with no named claim duplicated the row
    instead of updating it.  Rebuild the table, collapsing any duplicates
    that were already written (keeping the most recent).
    """
    exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='supersessions'"
    ).fetchone()
    if not exists:
        return
    cols = db.execute("PRAGMA table_info(supersessions)").fetchall()
    if not any(c["name"] == "claim" and c["notnull"] for c in cols):
        db.execute("ALTER TABLE supersessions RENAME TO supersessions_old")
        db.execute(
            """
            CREATE TABLE supersessions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                superseded_id  TEXT NOT NULL,
                superseding_id TEXT NOT NULL,
                scope          TEXT NOT NULL DEFAULT 'claim',
                claim          TEXT NOT NULL DEFAULT '',
                reason         TEXT,
                actor          TEXT NOT NULL,
                retracted      INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
                UNIQUE(superseded_id, superseding_id, claim)
            )
            """
        )
        db.execute(
            """
            INSERT INTO supersessions
                (superseded_id, superseding_id, scope, claim, reason, actor,
                 retracted, created_at)
            SELECT superseded_id, superseding_id, scope,
                   COALESCE(claim, ''), reason, actor, retracted, MAX(created_at)
            FROM supersessions_old
            GROUP BY superseded_id, superseding_id, COALESCE(claim, '')
            """
        )
        db.execute("DROP TABLE supersessions_old")


def _migrate_v35_to_v36(db: DatabaseEngine) -> None:
    """Add the supersessions table.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v34_to_v35(db: DatabaseEngine) -> None:
    """Add subjects.display_name — an authored field, never derived.

    The roster rebuild refreshes derived columns but must leave this one
    alone, so a name a person or specialist gave a shelf survives.
    """
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='subjects'"
    ).fetchone()
    if not table_exists:
        return
    cols = {c["name"] for c in db.execute("PRAGMA table_info(subjects)").fetchall()}
    if "display_name" not in cols:
        db.execute("ALTER TABLE subjects ADD COLUMN display_name TEXT")


def _migrate_v33_to_v34(db: DatabaseEngine) -> None:
    """Add librarian_state.

    Table uses IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v32_to_v33(db: DatabaseEngine) -> None:
    """Add hierarchy columns to subjects.

    Existing subjects become depth-0 roots; the next roster rebuild
    derives the tree.
    """
    table_exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='subjects'"
    ).fetchone()
    if not table_exists:
        return

    cols = {c["name"] for c in db.execute("PRAGMA table_info(subjects)").fetchall()}
    if "parent_id" not in cols:
        db.execute("ALTER TABLE subjects ADD COLUMN parent_id INTEGER REFERENCES subjects(id)")
    if "depth" not in cols:
        db.execute("ALTER TABLE subjects ADD COLUMN depth INTEGER NOT NULL DEFAULT 0")
    if "path" not in cols:
        db.execute("ALTER TABLE subjects ADD COLUMN path TEXT")
    db.execute("UPDATE subjects SET path = name WHERE path IS NULL")


def _migrate_v22_to_v23(db: DatabaseEngine) -> None:
    """Add temporal tag episode tables.

    Tables use IF NOT EXISTS in TABLES, so no explicit migration needed.
    """
    pass


def _migrate_v23_to_v24(db: DatabaseEngine) -> None:
    """Replace composite PK on episode_transitions with autoincrement id."""
    tables = {
        r["name"]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "episode_transitions" not in tables:
        return

    db.execute(
        "CREATE TABLE IF NOT EXISTS episode_transitions_new ("
        "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  from_episode_id INTEGER NOT NULL REFERENCES temporal_episodes(id),"
        "  to_episode_id   INTEGER NOT NULL REFERENCES temporal_episodes(id),"
        "  transition_type TEXT NOT NULL,"
        "  transition_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),"
        "  metadata        TEXT"
        ")"
    )
    db.execute(
        "INSERT INTO episode_transitions_new "
        "(from_episode_id, to_episode_id, transition_type, transition_at, metadata) "
        "SELECT from_episode_id, to_episode_id, transition_type, transition_at, metadata "
        "FROM episode_transitions"
    )
    db.execute("DROP TABLE episode_transitions")
    db.execute("ALTER TABLE episode_transitions_new RENAME TO episode_transitions")


def _migrate_v24_to_v25(db: DatabaseEngine) -> None:
    """Add sync_event_tags junction table and backfill from involved_tags JSON."""
    import json as _json

    db.execute(
        "CREATE TABLE IF NOT EXISTS sync_event_tags ("
        "  event_id INTEGER NOT NULL REFERENCES synchronicity_events(id) ON DELETE CASCADE,"
        "  tag_id   INTEGER NOT NULL REFERENCES tags(id),"
        "  PRIMARY KEY (event_id, tag_id)"
        ")"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_sync_event_tags_tid ON sync_event_tags(tag_id)"
    )

    if not _has_table(db, "synchronicity_events"):
        return
    rows = db.execute(
        "SELECT id, involved_tags FROM synchronicity_events"
    ).fetchall()
    inserts = []
    for r in rows:
        try:
            tag_ids = _json.loads(r["involved_tags"])
        except (TypeError, _json.JSONDecodeError):
            continue
        for tid in tag_ids:
            inserts.append((r["id"], int(tid)))

    if inserts:
        for i in range(0, len(inserts), 500):
            chunk = inserts[i:i + 500]
            db.executemany(
                "INSERT OR IGNORE INTO sync_event_tags (event_id, tag_id) VALUES (?, ?)",
                chunk,
            )


def _migrate_v25_to_v26(db: DatabaseEngine) -> None:
    """Normalize resonance_strength to [0, 1] via min-max scaling.

    Uses batched updates by rowid range to avoid WAL journal overflow
    on large databases.
    """
    if not _has_table(db, "resonances"):
        return
    row = db.execute(
        "SELECT MIN(resonance_strength) AS mn, MAX(resonance_strength) AS mx "
        "FROM resonances"
    ).fetchone()
    if not row or row["mn"] is None:
        return
    mn, mx = row["mn"], row["mx"]

    BATCH = 50_000

    if mx - mn < 1e-9:
        bounds = db.execute(
            "SELECT MIN(rowid) AS lo, MAX(rowid) AS hi FROM resonances"
        ).fetchone()
        if not bounds or bounds["lo"] is None:
            return
        lo, hi = bounds["lo"], bounds["hi"]
        while lo <= hi:
            db.execute(
                "UPDATE resonances SET resonance_strength = 0.5 "
                "WHERE rowid BETWEEN ? AND ?",
                (lo, lo + BATCH - 1),
            )
            db.connection.commit()
            lo += BATCH
    else:
        bounds = db.execute(
            "SELECT MIN(rowid) AS lo, MAX(rowid) AS hi FROM resonances"
        ).fetchone()
        if not bounds or bounds["lo"] is None:
            return
        lo, hi = bounds["lo"], bounds["hi"]
        while lo <= hi:
            db.execute(
                "UPDATE resonances SET resonance_strength = "
                "(resonance_strength - ?) / ? "
                "WHERE rowid BETWEEN ? AND ?",
                (mn, mx - mn, lo, lo + BATCH - 1),
            )
            db.connection.commit()
            lo += BATCH


def _migrate_v26_to_v27(db: DatabaseEngine) -> None:
    """Reset episode distance stats for Euclidean metric switch."""
    if not _has_table(db, "temporal_episodes"):
        return
    db.execute(
        "UPDATE temporal_episodes SET mean_distance = 0, variance_sum = 0, "
        "distance_samples = 0"
    )
    db.connection.commit()


def _migrate_v27_to_v28(db: DatabaseEngine) -> None:
    """Directional canopy tables — all CREATE IF NOT EXISTS, no data migration."""
    pass


def _migrate_v37_to_v38(db: DatabaseEngine) -> None:
    """Wandering librarian tables.

    Both are CREATE IF NOT EXISTS in TABLES and start empty — there is no
    back-fill to do.  An existing install's memories.swept_at is NOT copied
    into patrol_visits: swept_at belongs to LinearSweep's errand, and
    seeding a different errand's cursor from it would claim coverage the
    new errand never performed.
    """
    pass
