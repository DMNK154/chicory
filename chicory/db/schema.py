"""Database schema creation and migration."""

from __future__ import annotations

from chicory.db.engine import DatabaseEngine

SCHEMA_VERSION = 24

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
        ingestion_tier          TEXT NOT NULL DEFAULT 'critical'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience_composite DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash) WHERE content_hash IS NOT NULL",
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

    # -- Layer 3: Synchronicity event ↔ memory junction --
    """
    CREATE TABLE IF NOT EXISTS sync_event_memories (
        event_id   INTEGER NOT NULL REFERENCES synchronicity_events(id) ON DELETE CASCADE,
        memory_id  TEXT NOT NULL,
        PRIMARY KEY (event_id, memory_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sync_event_memories_mid ON sync_event_memories(memory_id)",

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
                (SCHEMA_VERSION, "Temporal tag episodes"),
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
