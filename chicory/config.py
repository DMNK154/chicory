"""Configuration for Chicory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_GROK_MODEL = "grok-4.20-reasoning"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class ChicoryConfig(BaseModel):
    """All configuration for the Chicory memory system."""

    # Paths
    db_path: Path = Field(default_factory=lambda: Path.home() / ".chicory" / "chicory.db")
    # How long a statement waits on a write lock before giving up. Stores
    # trigger global network passes that hold locks for tens of seconds, so
    # the previous 5s meant a concurrent retrieval could fail an ingest.
    db_busy_timeout_ms: int = Field(default=30000)

    # LLM
    llm_provider: str = Field(default="auto")  # auto, anthropic, openai, grok, gemini, null
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    xai_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    llm_model: str = Field(default=DEFAULT_OPENAI_MODEL)
    grok_model: str = Field(default=DEFAULT_GROK_MODEL)
    gemini_model: str = Field(default=DEFAULT_GEMINI_MODEL)
    xai_base_url: str = Field(default="https://api.x.ai/v1")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai/")
    max_tokens: int = Field(default=4096)

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

    # Layer 1 — Salience
    salience_model_weight: float = Field(default=0.6)
    salience_usage_weight: float = Field(default=0.4)
    salience_recency_halflife_hours: float = Field(default=168.0)  # Legacy, superseded by multi-tier
    salience_recency_active_halflife_hours: float = Field(default=720.0)   # ~1 month
    salience_recency_longterm_halflife_hours: float = Field(default=4380.0)  # ~6 months
    salience_recency_active_weight: float = Field(default=0.6)
    salience_recency_longterm_weight: float = Field(default=0.4)
    max_retrieval_results: int = Field(default=10)
    similarity_threshold: float = Field(default=0.3)
    # Minimum hybrid relevance score required for a memory to be surfaced
    # to the LLM. Applied after all scoring (semantic + tag + lattice + glyph)
    # but BEFORE bookkeeping, so low-relevance noise doesn't reinforce itself.
    # 0.0 = no filtering (back-compat).
    #
    # Adaptive: the cap on possible scores differs depending on whether the
    # caller passed a `tags` argument, because the tag channel contributes
    # up to 0.3 to the total.  Without tags: theoretical max ~0.95.
    # With tags: theoretical max ~1.25.  Apply a stricter floor when tags
    # were passed (the signal is stronger) and a looser floor when not.
    retrieval_min_relevance_score_tagged: float = Field(default=0.5)
    retrieval_min_relevance_score_untagged: float = Field(default=0.25)

    # Layer 2 — Trends
    trend_window_hours: float = Field(default=168.0)  # 1 week
    trend_short_window_hours: float = Field(default=24.0)
    trend_snapshot_interval_minutes: float = Field(default=60.0)
    trend_level_weight: float = Field(default=0.5)
    trend_velocity_weight: float = Field(default=0.35)
    trend_jerk_weight: float = Field(default=0.15)

    # Layer 3 — Phase Space & Synchronicity
    phase_temperature_threshold: float = Field(default=0.5)
    phase_retrieval_threshold: float = Field(default=0.5)
    sync_detection_sigma: float = Field(default=2.0)
    sync_inactive_temp_ceiling: float = Field(default=0.3)
    cross_domain_surprise_threshold: float = Field(default=3.0)
    semantic_convergence_threshold: float = Field(default=0.7)
    sync_cross_domain_lookback_hours: float = Field(default=1.0)
    sync_semantic_convergence_lookback_hours: float = Field(default=24.0)

    # Synchronicity decay
    sync_decay_active_halflife_hours: float = Field(default=168.0)       # ~1 week
    sync_decay_longterm_halflife_hours: float = Field(default=2160.0)    # ~3 months
    sync_decay_active_weight: float = Field(default=0.6)
    sync_decay_longterm_weight: float = Field(default=0.4)
    sync_reinforcement_boost_factor: float = Field(default=0.15)
    sync_reinforcement_max_boost: float = Field(default=3.0)

    # Synchronicity velocity
    sync_velocity_window_hours: float = Field(default=168.0)  # 1 week

    # Layer 4 — Meta-Patterns
    meta_analysis_interval_hours: float = Field(default=24.0)
    meta_min_sync_events: int = Field(default=3)
    meta_use_lattice_resonances: bool = Field(default=True)
    meta_cross_domain_min_clusters: int = Field(default=2)
    base_rate_multiplier: float = Field(default=3.0)
    clustering_jaccard_threshold: float = Field(default=0.7)

    # Layer 3.5 — Prime Ramsey Lattice
    lattice_primes: list[int] = Field(
        default=[
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
            53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
            127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
            199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
            283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
            383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
            467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571,
            577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
            661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
            769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
        ]
    )
    lattice_min_resonance_primes: int = Field(default=30)
    lattice_void_radius: float = Field(default=0.3)
    lattice_retrieval_boost_enabled: bool = Field(default=True)
    lattice_retrieval_boost_weight: float = Field(default=0.15)

    # Poincaré projection — hyperbolic geometry for the Ramsey lattice
    poincare_curvature: float = Field(default=1.0)       # c > 0; sectional curvature is -c
    poincare_max_radius: float = Field(default=0.95)     # Clamp to open disk boundary
    poincare_depth_enabled: bool = Field(default=True)   # Ramsey prime-density radial adjustment

    # Tag relational tensor — sparse retention gating (PMI + bridge)
    tensor_retention_threshold: float = Field(default=0.1)
    tensor_bridge_bonus: float = Field(default=1.0)

    # Tag relational tensor — network combination weights for recall
    tensor_cooccurrence_weight: float = Field(default=0.5)
    tensor_synchronicity_weight: float = Field(default=0.3)
    tensor_semantic_weight: float = Field(default=0.2)
    tensor_semiotic_weight: float = Field(default=0.15)

    # Lateral inhibition weight — max suppression for antiparallel glyph pairs
    tensor_inhibition_weight: float = Field(default=0.20)

    # Glyph Ramsey lattice — letter-composition structural resonance
    glyph_lattice_primes: list[int] = Field(
        default=[
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
            53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
            127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
            199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        ]
    )
    tensor_glyph_weight: float = Field(default=0.2)
    tensor_meta_resonance_weight: float = Field(default=0.4)
    glyph_min_resonance_primes: int = Field(default=6)

    # Glyph Ramsey lattice — ByT5 encoder model (empty = letter-count fallback)
    glyph_model_dir: str = Field(default="")
    glyph_embedding_dimension: int = Field(default=1472)

    # Glyph Bridge — GPT-GU integration (optional)
    gptgu_path: str = Field(default="")              # Path to GPT-GU repo root
    glyph_bridge_enabled: bool = Field(default=False)
    glyph_symbolic_bonus: float = Field(default=2.0)   # Same-glyph tag pair bonus
    glyph_pair_bonus: float = Field(default=1.5)        # PAIR_TO_TEXT tag pair bonus
    glyph_retrieval_boost_weight: float = Field(default=0.10)  # Glyph association retrieval boost

    # Context tag limit — how many tags to include in the system prompt
    context_tag_limit: int = Field(default=30)

    # Resonance retrieval — minimum association strengths
    resonance_min_tensor_score: float = Field(default=0.3)
    resonance_min_edge_strength: float = Field(default=0.3)

    # Semiotic directed graph — IN→OUT / OUT→IN traversal
    semiotic_directed_min_strength: float = Field(default=0.05)
    semiotic_directed_weight: float = Field(default=0.3)
    semiotic_convergence_weight: float = Field(default=0.12)

    # Centroid sub-graph — retrieval-driven reweighting
    centroid_ema_alpha: float = Field(default=0.1)
    centroid_edge_ema_alpha: float = Field(default=0.15)
    centroid_inhibition_scale: float = Field(default=0.5)
    centroid_inhibition_enabled: bool = Field(default=True)

    # Canopy — memory cluster growth from episodic co-activation
    canopy_enabled: bool = Field(default=True)
    canopy_max_depth_per_pass: int = Field(default=1)
    canopy_growth_temperature: float = Field(default=1.0)
    canopy_use_soft_growth: bool = Field(default=False)
    # Pressure = co-activation + bridge (memories that fire together)
    canopy_pressure_coactivation_weight: float = Field(default=0.65)
    canopy_pressure_bridge_weight: float = Field(default=0.35)
    # Inhibition = tag Jaccard overlap (suppress what tags already explain)
    canopy_inhibition_tag_overlap_weight: float = Field(default=0.5)
    # Co-occurrence optimizer (forest layer)
    canopy_co_min_ppmi: float = Field(default=0.0)
    canopy_block_density_threshold: float = Field(default=0.5)
    # Bridge optimizer (forest layer)
    canopy_bridge_max_per_block: int = Field(default=5)
    canopy_bridge_hub_penalty: float = Field(default=0.5)
    # Ramsey adjacency filter (glyph lattice)
    canopy_ramsey_min_shared_primes: int = Field(default=7)

    # Directional canopy — inflow/outflow tracking via retrieval causality
    canopy_directional_enabled: bool = Field(default=True)
    canopy_directional_inflow_diversity_weight: float = Field(default=0.7)
    canopy_directional_inflow_frequency_weight: float = Field(default=0.3)
    canopy_directional_outflow_diversity_weight: float = Field(default=0.6)
    canopy_directional_outflow_reach_weight: float = Field(default=0.4)
    canopy_directional_inflow_centroid_boost: float = Field(default=0.3)
    canopy_directional_outflow_pressure_weight: float = Field(default=0.5)
    canopy_directional_ema_alpha: float = Field(default=0.25)
    canopy_directional_query_tag_similarity_threshold: float = Field(default=0.3)
    canopy_directional_inflow_context_gate_threshold: int = Field(default=5)
    canopy_directional_inflow_pressure_weight: float = Field(default=0.4)

    # Temporal tag episodes — drift-detected, revisitable tag-space clusters
    episode_enabled: bool = Field(default=True)
    episode_ema_alpha: float = Field(default=0.2)
    episode_drift_sigma: float = Field(default=2.0)
    episode_drift_base_threshold: float = Field(default=0.5)
    episode_revisit_max_candidates: int = Field(default=50)
    episode_sync_boundary_strength: float = Field(default=0.7)
    episode_min_samples_for_adaptive: int = Field(default=10)
    episode_dormant_after_hours: float = Field(default=168.0)
    episode_archive_after_hours: float = Field(default=4320.0)

    # Episodic relational tensor — memory-to-memory edge cache
    episodic_tag_affinity_threshold: float = Field(default=0.15)
    episodic_temporal_halflife_hours: float = Field(default=720.0)
    episodic_decay_inactive_hours: float = Field(default=2160.0)  # ~90 days
    episodic_min_edge_strength: float = Field(default=0.05)
    episodic_co_retrieval_ema_alpha: float = Field(default=0.2)

    # Migration
    burn_in_hours: float = Field(default=48.0)
    burn_in_threshold_multiplier: float = Field(default=1.5)

    # Retrieval
    hybrid_semantic_weight: float = Field(default=0.7)
    hybrid_tag_weight: float = Field(default=0.3)

    # Tag Space — independent tag-graph retrieval path
    tag_space_enabled: bool = Field(default=True)
    tag_space_weight: float = Field(default=0.15)
    tag_space_fan_depth: int = Field(default=1)
    tag_space_lexical_min_tag_length: int = Field(default=3)
    tag_space_min_inward_ratio: float = Field(default=0.01)
    tag_space_centroid_similarity_threshold: float = Field(default=0.4)
    tag_space_max_context_seed_edges: int = Field(default=25)

    # Deep Retrieve
    deep_retrieve_max_depth: int = Field(default=3)
    deep_retrieve_per_level_k: int = Field(default=5)
    deep_retrieve_depth_decay: float = Field(default=0.7)
    deep_retrieve_time_depth_weight: float = Field(default=0.3)

    # FAISS vector index.
    #
    # nprobe is how many of the IVF index's cells a query inspects, and it
    # sets recall, not just speed: at 351k chunks the index has 2370 cells,
    # so nprobe=8 searched 0.34% of it and returned 71.7% of the true
    # top-20 (measured against a brute-force scan over 200 queries drawn
    # from real memories).  Roughly one best match in four was never seen.
    # 64 recovers that to 89.3% for +0.6ms per query; 128 buys a further
    # 2.6 points for twice the time.  Recall plateaus near 94% regardless,
    # because search() over-fetches only top_k*3 chunks before
    # deduplicating to memories.
    faiss_nprobe: int = Field(default=64)
    faiss_rebuild_threshold: int = Field(default=500)
    # Cache the built index on disk. Rebuilding costs 39-57s on a 351k-chunk
    # corpus and ran on every server start, which the first retrieval then
    # waited out (measured: a 57.4s retrieve_memories, almost all lock wait).
    # Set False to disable without a code change; the cache is derived from
    # db_path so separate databases never share one.
    faiss_index_cache_enabled: bool = Field(default=True)
    # How long a query with NO index waits for one to be built. A query
    # that already holds a usable index never waits at all — it serves from
    # what it has and lets the rebuild happen. This bound only covers the
    # genuinely cold case, where returning early would mean reporting "no
    # matches" for a library that simply is not loaded yet.
    faiss_index_wait_seconds: float = Field(default=120.0)

    # Ingestion — two-tier criticality classification
    ingestion_criticality_threshold: float = Field(default=0.3)
    ingestion_critical_importance: float = Field(default=0.6)
    ingestion_reference_importance: float = Field(default=0.3)
    ingestion_reference_summary_max_chars: int = Field(default=2000)

    # Librarian — write gate, quarantine, linear sweep, subject roster
    librarian_gate_enabled: bool = Field(default=True)
    librarian_gate_enforce: bool = Field(default=False)  # False = shadow mode: log verdicts, never divert
    librarian_gate_threshold: float = Field(default=0.25)
    librarian_gate_min_centroids: int = Field(default=50)  # cold-start guard: gate stays open below this
    librarian_sweep_enabled: bool = Field(default=True)
    librarian_sweep_interval_minutes: float = Field(default=30.0)
    librarian_sweep_batch_size: int = Field(default=50)
    librarian_sweep_near_dup_threshold: float = Field(default=0.95)
    librarian_sweep_drift_threshold: float = Field(default=0.15)
    subject_min_edge_strength: float = Field(default=0.05)
    subject_min_cluster_size: int = Field(default=3)
    # Hysteresis on component formation.
    #
    # edge_strength is inert as a gate: record_co_retrieval creates every
    # edge at centroid_edge_ema_alpha (0.15), three times this floor, and
    # EMA only ever raises it.  Measured on a 124K library, the minimum
    # edge_strength in the whole table is exactly 0.15 — so *every* edge
    # that has ever existed is in the graph, and a SINGLE co-retrieval
    # permanently welds two tags into one component.  200 retrievals
    # created 1,560 such edges (+0.97% of the graph) and re-partitioned
    # 70% of the shelf tree, 45 shelves to 65.
    #
    # Evidence is the gate that matters, so these are counts, not
    # strengths, and they are asymmetric: holding two tags together takes
    # one co-retrieval, merging two previously-separate groups takes
    # several.  Three matches the bar the demand ledger uses for filing a
    # commission (commission_min_hits).
    subject_edge_keep_min_count: int = Field(default=1)
    subject_edge_merge_min_count: int = Field(default=3)
    subject_max_name_tags: int = Field(default=3)
    # Stop-tag rule: tags on more than this fraction of live memories are
    # format/hub tags (document, pdf), not topics — excluded from subjects
    subject_hub_tag_max_fraction: float = Field(default=0.2)
    # A connected component larger than this fraction of eligible tags is
    # a chained blob — re-clustered in centroid space instead
    subject_max_component_fraction: float = Field(default=0.25)
    subject_split_k: int = Field(default=0)  # 0 = auto (~sqrt(n/2))
    # A shelf is too big to teach when it holds this share of live memories.
    # Tag count alone misses this: a 99-tag subject can hold 77% of the
    # library and never trip a tag-fraction cap.
    subject_max_memory_fraction: float = Field(default=0.15)
    subject_max_depth: int = Field(default=3)
    # A split whose largest child holds this share of the parent's tags
    # separated nothing — treat the parent as a leaf rather than emitting
    # a near-identity child (the X / X~2 / X~3 chains).
    subject_split_min_separation: float = Field(default=0.9)
    # Routing consults coarse shelves; teaching works fine ones.
    # -1 = auto: the shallowest depth that actually has >1 shelf.  On a
    # densely connected corpus the root is a single "everything" node, and
    # routing to it is a no-op — auto skips past that to the first level
    # that presents a real choice.
    consult_route_depth: int = Field(default=-1)

    # Librarian front desk — demand ledger, FAQ, dispatch, commissions
    librarian_demand_enabled: bool = Field(default=True)
    librarian_demand_interval_minutes: float = Field(default=15.0)
    demand_cluster_threshold: float = Field(default=0.85)
    demand_process_batch: int = Field(default=200)
    demand_history_retention_days: int = Field(default=30)
    faq_serve_threshold: float = Field(default=0.90)
    # One ask is enough, because the SCORE now carries the judgment.
    #
    # These two gates were calibrated backwards.  Repetition was doing the
    # filtering (three asks) while quality was doing none (0.45, below the
    # noise floor of a 124K library).  Measured: retrieval scores 0.66-0.80
    # on questions the shelves can answer and 0.48-0.56 on ones they
    # cannot, so 0.45 admitted everything — a question about FAISS
    # internals was answered with untranscribed Tamil script at 0.483 and
    # promoted, while genuine one-off questions scoring 0.98 were held back
    # for want of a second asker.
    #
    # Inverted: any question the library can answer well is worth a card,
    # asked once.  Cards below the bar are not lost, they are simply not
    # crystallized; and a candidate serves nothing until a person approves
    # it, so the cost of a wrong one is a reader's time.
    faq_promote_min_hits: int = Field(default=1)
    # Fallback quality bar, used only when faq_promote_tiers is empty.
    faq_promote_min_score: float = Field(default=0.70)
    # Repetition and quality trade against each other, so the bar is a
    # ladder rather than one number: (minimum asks, minimum score),
    # evaluated from the most asks down, first match wins.
    #
    # A question asked once has to answer itself convincingly (0.95).
    # Asked three times it is demonstrably wanted, so a merely good answer
    # will do (0.70).  Two sits between (0.80) — and on the live library
    # that tier is bimodal, two clusters near 0.98 and a clean gap down to
    # the duds, so anything from 0.65 to 0.82 selects the same three.
    #
    # An empty list disables the ladder and falls back to the flat
    # faq_promote_min_hits / faq_promote_min_score pair above.
    faq_promote_tiers: list[tuple[int, float]] = Field(
        default_factory=lambda: [(3, 0.70), (2, 0.80), (1, 0.95)]
    )
    commission_min_hits: int = Field(default=3)
    commission_weak_score: float = Field(default=0.30)
    consult_subject_threshold: float = Field(default=0.30)
    consult_max_subjects: int = Field(default=3)
    # Retrieval attribution.
    #
    # Retrieval is not read-only: it records demand, and its bookkeeping
    # reweights the tag centroids and co-retrieval edges the shelf tree is
    # derived from. Until this existed, an agent researching on the
    # librarian's behalf was indistinguishable from a patron asking a
    # question, and three such research queries were promoted into FAQ
    # cards as though the library had been asked them.
    #
    # This is a DECLARED channel, not an enforced one. A caller states who
    # it is; nothing verifies the claim. That is worth stating plainly —
    # it makes traffic separable and auditable, which is all it can do.
    retrieval_default_actor: str = Field(default="patron")
    # Whose queries count as demand (FAQ promotion, commission filing).
    # Everything is still recorded; only these are clustered.
    demand_actors: list[str] = Field(default_factory=lambda: ["patron"])
    # Whose retrievals may reshape the centroid geometry. An automated
    # reader left out of this list can walk the library without rewriting
    # the map it walks — the precondition for the idle random-walk errand.
    geometry_actors: list[str] = Field(default_factory=lambda: ["patron"])

    # Ingress dwell — receipted hold before commitment (detection window)
    ingress_enabled: bool = Field(default=True)
    ingress_dwell_seconds: float = Field(default=180.0)
    ingress_window_lookback: int = Field(default=25)
    ingress_burst_threshold: float = Field(default=0.93)
    ingress_trajectory_min_history: int = Field(default=3)
    ingress_trajectory_deviation_threshold: float = Field(default=0.20)
    ingress_quarantine_resemblance_threshold: float = Field(default=0.90)
    ingress_retention_days: int = Field(default=30)
    ingress_settle_batch: int = Field(default=25)
    ingress_prune_interval_minutes: float = Field(default=60.0)

    penumbra_enabled: bool = Field(default=True)
    penumbra_keep: int = Field(default=20)
    penumbra_buried_min_near: int = Field(default=5)
    recall_max_subjects: int = Field(default=4)

    # Wandering librarian — the heartbeat and its errands.
    #
    # Every other background phase is traffic-paced: the thread that runs
    # them is spawned by a retrieval, so an idle library does no work at
    # all.  The heartbeat gives that thread a clock of its own.  It does
    # NOT bypass the foreground: it calls the same background pass, which
    # already yields the DB lock between phases.
    wander_enabled: bool = Field(default=True)
    wander_tick_seconds: float = Field(default=30.0)
    # Consecutive ticks that observed no foreground activity before the
    # library counts as idle.  Errands marked idle-only wait for this.
    wander_idle_ticks: int = Field(default=4)
    wander_batch_size: int = Field(default=50)
    wander_interval_minutes: float = Field(default=5.0)
    # Errands the patrol runs, in order.  Names must be registered in
    # chicory.layer4.wander.ERRANDS; unknown names are logged and skipped.
    wander_errands: list[str] = Field(
        default_factory=lambda: ["missing_embedding"]
    )
    # Only one process patrols a library at a time.  Every MCP client
    # session spawns its own server against the same SQLite file — 14 were
    # observed running on one install — and before this lease every one of
    # them walked the same cursor on the same 30s tick: ~38% duplicated
    # work, UNIQUE collisions in `embeddings`, and lock contention.  The
    # lease must outlast one patrol pass by a wide margin; a holder that
    # dies simply lets it expire.
    wander_lease_seconds: float = Field(default=120.0)

    # Commons Layer — cross-project signal federation
    commons_enabled: bool = Field(default=False)
    commons_db_path: Path = Field(
        default_factory=lambda: Path.home() / ".chicory" / "commons.db"
    )
    commons_project_id: str = Field(default="")
    commons_signal_buffer_size: int = Field(default=10)
    commons_flush_interval_seconds: float = Field(default=5.0)

    # Diagnostics — activation trace + context logging
    context_log_enabled: bool = Field(default=True)
    context_log_dir: Path = Field(
        default_factory=lambda: Path.home() / ".chicory" / "logs"
    )
    context_log_full: bool = Field(default=False)


def load_config(**overrides) -> ChicoryConfig:
    """Load config from .env file and environment variables, with overrides."""
    from dotenv import dotenv_values

    # Look for .env in cwd first, then next to the chicory package root
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    env = dotenv_values(str(env_path))

    kwargs: dict = {}
    mapping = {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "XAI_API_KEY": "xai_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "GOOGLE_API_KEY": "gemini_api_key",
        "CHICORY_LLM_PROVIDER": "llm_provider",
        "CHICORY_DB_PATH": "db_path",
        "CHICORY_LLM_MODEL": "llm_model",
        "CHICORY_GROK_MODEL": "grok_model",
        "CHICORY_GEMINI_MODEL": "gemini_model",
        "CHICORY_XAI_BASE_URL": "xai_base_url",
        "CHICORY_GEMINI_BASE_URL": "gemini_base_url",
        "CHICORY_EMBEDDING_MODEL": "embedding_model",
        "CHICORY_COMMONS_ENABLED": "commons_enabled",
        "CHICORY_COMMONS_DB_PATH": "commons_db_path",
        "CHICORY_COMMONS_PROJECT_ID": "commons_project_id",
        "CHICORY_GLYPH_MODEL_DIR": "glyph_model_dir",
        "CHICORY_GPTGU_PATH": "gptgu_path",
        "CHICORY_GLYPH_BRIDGE_ENABLED": "glyph_bridge_enabled",
        # The write gate ships in shadow mode; enforcement is a per-install
        # decision made after calibrating against that install's own corpus,
        # so it belongs in the environment rather than the shipped defaults.
        "CHICORY_GATE_ENFORCE": "librarian_gate_enforce",
        "CHICORY_GATE_THRESHOLD": "librarian_gate_threshold",
        # The heartbeat runs work while nobody is watching, so whether it
        # runs at all is a per-install decision like gate enforcement.
        "CHICORY_WANDER_ENABLED": "wander_enabled",
        "CHICORY_WANDER_TICK_SECONDS": "wander_tick_seconds",
    }
    for env_key, config_key in mapping.items():
        if env_key in env and env[env_key]:
            kwargs[config_key] = env[env_key]

    import os

    for env_key, config_key in mapping.items():
        val = os.environ.get(env_key)
        if val:
            kwargs[config_key] = val

    # Coerce string booleans from env vars
    for bool_key in (
        "commons_enabled", "glyph_bridge_enabled", "librarian_gate_enforce",
    ):
        if isinstance(kwargs.get(bool_key), str):
            kwargs[bool_key] = kwargs[bool_key].lower() in ("true", "1", "yes")

    kwargs.update(overrides)

    provider = str(kwargs.get("llm_provider", ChicoryConfig.model_fields["llm_provider"].default)).lower()
    model = kwargs.get("llm_model")
    if provider in ("grok", "xai") and (not model or not str(model).startswith("grok")):
        kwargs["llm_model"] = kwargs.get("grok_model") or DEFAULT_GROK_MODEL
    elif provider in ("gemini", "google") and (not model or not str(model).startswith("gemini")):
        kwargs["llm_model"] = kwargs.get("gemini_model") or DEFAULT_GEMINI_MODEL

    # Auto-detect project ID from git repo or cwd if not explicitly set
    if kwargs.get("commons_enabled") and not kwargs.get("commons_project_id"):
        kwargs["commons_project_id"] = _detect_project_id()

    return ChicoryConfig(**kwargs)


def _detect_project_id() -> str:
    """Detect project name from git repo root, falling back to cwd basename."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            name = Path(result.stdout.strip()).name
            return name.lower().replace(" ", "-")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return Path.cwd().name.lower().replace(" ", "-")
