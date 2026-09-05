"""Configuration for Chicory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


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
    # How much MORE separation a node must show to start splitting than to
    # keep a split it already has.  A Schmitt trigger on the split
    # decision, for the same reason the component pass has one.
    #
    # Without it the warm start feeds back on itself and oscillates.
    # Observed on the live roster after 2,522 attachments landed: a node
    # with no children seeds one arity, splits, and passes; with two
    # children it seeds another, splits differently, and fails — so
    # `life+wisdom+power` alternated between 0 and 2 children on every
    # single rebuild, a stable 2-cycle across eight runs. The inheritance
    # was driving the flip it exists to prevent.
    subject_split_hysteresis: float = Field(default=0.03)
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
    # Below this many memories, a tag exists but the library does not
    # meaningfully hold the subject — counted as a gap, not as coverage.
    faq_gap_thin_tag_memories: int = Field(default=20)
    # How often a capitalised term must recur WITHIN one memory before the
    # concept errand treats it as something that memory is about rather
    # than something it mentions.  Four was calibrated on this corpus:
    # three admitted 'that' and 'great' alongside malchut and tiferet.
    concept_min_mentions: int = Field(default=4)
    # A chunk that is "about" more terms than this is a list or a table,
    # not prose, and its terms all co-occur with each other by layout
    # rather than by meaning.  One CLDR locale file put 870 language names
    # in a single chunk and its clique swamped the net.  Measured on this
    # corpus the honest distribution is median 0, p99 4 — so 8 discards
    # 0.2% of memories and precisely the tabular ones.
    concept_max_terms_per_memory: int = Field(default=8)
    # A document raising more gap terms than this is a dump or a save
    # file, not a subject; pairing its terms would emit a dense clique
    # that says nothing (Rocket League telemetry alone contributes
    # hundreds of untagged JSON keys to one document).
    gap_edge_max_document_terms: int = Field(default=40)
    # Jaccard over document sets: how much of each gap's material the two
    # share, not whether they touched once.
    #
    # 0.10, and the number only makes sense against the metric it grades.
    # It was 0.6 when edges were Jaccard over DOCUMENTS; over MEMORIES the
    # same relationship scores far lower — malchut and anpin share 756
    # memories out of ~4,000, which is a strong association and a Jaccard
    # of 0.19. At 0.6 the entire Kabbalah vocabulary had no edges at all.
    #
    # Measured: 0.10 keeps malchut-anpin-binah and yields 456 clusters
    # that name individual works by their own vocabulary — the Tractatus
    # in German, Leviathan in 17th-century spelling, the Tibetan corpus.
    # 0.2 breaks the largest chain but loses those relationships with it.
    gap_edge_min_weight: float = Field(default=0.10)

    # ── Meta-shelf charters ────────────────────────────────────────────
    #
    # Direct co-occurrence Jaccard cannot assemble a subject, and the
    # ceiling is arithmetic rather than a tuning failure: Jaccard is bounded
    # above by min(m_a, m_b) / max(m_a, m_b), and the sefirot span 66 to
    # 3,948 mentions, so 17 of their 78 pairs CANNOT reach 0.10 however
    # perfectly they correlate.  Two second-order signals do work.
    #
    # arc — cosine over the `around` tag counters, after applying the
    # roster's own topical rule.  "Do these live in the same material."
    # Reusing _topical_tag_ids is load-bearing: unfiltered, every
    # component's top three around-tags are document / pdf / month-2026-05,
    # and the resulting similarity measures the import batch.
    #
    # sc — cosine over the raw `co` counters.  Scale-free, so it is immune
    # to the frequency ceiling above.  Measured: sefirot mean 0.436 against
    # a null of 0.0042.
    meta_shelf_arc_min: float = Field(default=0.90)
    meta_shelf_sc_min: float = Field(default=0.25)
    # Mutual top-k, not a threshold — the anti-chaining device, and not
    # optional.  Single linkage on any one signal blows up: document
    # Jaccard at 0.2 gives a 5,821-of-6,159 blob, and raw arc >= 0.97 gives
    # 480 nodes, which is simply "the Zohar".
    meta_shelf_mutual_k: int = Field(default=8)
    # A component too big to be one subject is re-clustered on its own
    # members at a stricter sc, recursively.  This is what separates the
    # Idra vocabulary from the Torah-portion vocabulary inside the
    # 450-node Zohar component.
    meta_shelf_split_cap: int = Field(default=100)
    meta_shelf_sc_step: float = Field(default=0.05)
    meta_shelf_sc_ceiling: float = Field(default=0.75)
    # Junk guards.  docJ is mean pairwise Jaccard of members' document
    # sets: a real vocabulary's terms share files, a citation convention's
    # terms each sit in a different reference list.  refHi is the fraction
    # of members appearing mostly in citation context — the only rule that
    # catches a bibliography that IS document-concentrated, i.e. one lab
    # where everyone cites everyone.
    #
    # Neither is sufficient and that is not a defect to tune away.
    # Measured on real groups: a League client config scores docJ 0.948 and
    # a lab's cellulose vocabulary 0.785 — both genuinely one source's
    # vocabulary, and neither a shelf anyone wants.  The filter is right
    # and the outcome is still wrong, which is why charters are confirmed
    # by a person rather than promoted automatically.
    meta_shelf_min_docj: float = Field(default=0.25)
    meta_shelf_max_ref_hi: float = Field(default=0.15)
    meta_shelf_min_size: int = Field(default=5)
    # A charter is persisted as a root and then offered to _should_split
    # like any other node.  Below this it is always a single leaf — its own
    # tight shelf, which is the geometry that avoids the measured breadth
    # bias.  Above it a charter could be split into children and stop being
    # one address, so confirm refuses it instead.
    #
    # A backstop, deliberately not a filter.  _should_split triggers at
    # subject_max_component_fraction of eligible tags — 0.25 of 1,855, so
    # 463 — and at 0.15 of the library by memory.  Set to 80 this rejected
    # the Upanishad vocabulary (99 members, docJ 0.812) and the
    # Torah-portion vocabulary (91), both real subjects nowhere near
    # splitting.  150 leaves a wide margin under the true threshold while
    # still refusing a "subject" that is really a whole corpus.
    meta_shelf_max_size: int = Field(default=150)
    # Transliteration and plural variants: `gevurah` and `gvurah` share
    # ZERO memories, so no co-occurrence signal of any strength reaches
    # them.  Only the spelling does, at ratio 0.923.
    meta_shelf_variant_ratio: float = Field(default=0.80)
    meta_shelf_ring_max: int = Field(default=60)
    # A `co` key held by more than this many concepts says nothing about
    # any of them; it is a stopword of the corpus rather than of English.
    meta_shelf_co_stopword_holders: int = Field(default=400)
    # Is the source human prose at all?
    #
    # The gate the other two could not be.  docJ measures CONCENTRATION,
    # which a config file aces — a League client config scored 0.948 and a
    # UI string dump 1.000, both better than real subjects.  What actually
    # separates them is that they are not discourse.
    #
    # Measured as the share of tokens that are English function words,
    # over a sample of each group's own documents.  Across all 93 standing
    # proposals the LARGEST gap in the whole distribution is 7.09 points
    # wide, between 11.35 and 18.44, and it falls exactly on the boundary:
    # below it sit lodbias, mobilecontentscalefactor, baseeventid,
    # aliasname, chicoryconfig, apply, authpsynetverbose, array, ffffff and
    # dipromosikan; above it, nothing but prose.  Real groups run 25.5 to
    # 44.3.  0.15 sits in the middle of that gap.
    #
    # It rejects machine text.  It cannot reject prose nobody wants
    # shelved — one lab's cellulose vocabulary is genuine discourse — so
    # it narrows the queue rather than emptying it.
    meta_shelf_min_prose: float = Field(default=0.15)
    # An indexed term's UNINDEXED spelling: a triage cut, not a verdict.
    #
    # Deliberately not meta_shelf_arc_min. That gate compares two `around`
    # counters; this compares a tag's co-occurring tags against a gap's
    # `around`, which is a different distribution shape, and 0.90 rejected
    # every true case measured — chochmah/chokhmah at 0.841,
    # shechinah/shekhinah at 0.795, malchut/malkhut at 0.889.
    #
    # Nor is there a threshold that WOULD work. Over 717 candidate pairs
    # the scores are continuous, the largest break is 0.033 wide, and true
    # and false are interleaved throughout: `mazal`/`azael` scores 0.884
    # and `power`/`tower` 0.839, both above `shechinah`/`shekhinah` at
    # 0.795, while `adsorption`/`absorption` and
    # `nitrocellulose`/`nanocellulose` — opposite processes and different
    # materials — sit at 0.801 and 0.798.
    #
    # So this only clears the floor (the tail sits at 0.000) and hands the
    # rest to a reader with both numbers attached, the way tag_twins does.
    variant_context_min: float = Field(default=0.50)

    # ── Bilingual extraction ───────────────────────────────────────────
    #
    # Twelve Zohar volumes carry Aramaic and Hebrew beside the English.
    # The English channel reached 92 of their 5,719 chunks; the Hebrew
    # reaches 977, and finds seven sefirot the English extractor cannot
    # see at all — הוד and דעת are below _CONCEPT_RE's five-letter floor.
    #
    # A term on nearly every page distinguishes nothing.  `ספר` and `זהר`
    # sit in 5,090 of 5,719 chunks and align to `Parashat` at dice 0.985
    # while meaning nothing whatever — the same failure the roster's hub
    # cap exists to prevent, so this is the same rule at 0.5 rather
    # than 0.2, because a facing text repeats its own title.
    bilingual_ubiquity_max: float = Field(default=0.50)
    # Below this a pairing is one coincidence rather than a convention.
    bilingual_min_chunks: int = Field(default=8)
    # Where the learned pairs stop being names and start being neighbours:
    # measured, משה/Moshe 0.947 and מטטרון/Metatron 0.932 sit far above,
    # and the tail below 0.45 is words that merely share a page.
    bilingual_min_dice: float = Field(default=0.45)
    # A repeated BLOCK aligns perfectly and means nothing.  A liturgical
    # preface sits in exactly 12 chunks beside the English word `Singer`
    # in the same 12, so every one of its 21 Hebrew words scores dice
    # 1.000 against it — higher than בינה/Binah at 0.903.  Co-occurrence
    # cannot tell a translation from a page that always travels together.
    #
    # What can: a word translates a word.  One English term drawing many
    # unrelated skeletons is boilerplate, not a lexicon entry, so the
    # whole cluster is dropped rather than trimmed — the block gives no
    # reason to prefer any member of it.
    bilingual_max_forms_per_term: int = Field(default=3)
    # How much of a chunk has to BE Hebrew before it counts as facing
    # text.  A volume can carry the English and be unable to carry the
    # Hebrew: the Pritzker Zohar is English commentary with inline
    # citations, median 2 Hebrew words per chunk against the bilingual
    # edition's 191.  Such a chunk passes "has a skeleton" on one word,
    # then contributes its whole English vocabulary to Dice's
    # denominator, where it can never contribute to the numerator.
    #
    # Measured when 722 of them arrived: `n` and `shared` for סנהדרין
    # did not move at all — 16 and 16 — while `m` went 21 to 107, and 86
    # of the 107 were commentary.  Dice fell 0.865 to 0.260 and the
    # entry was retracted, with 40 others.
    #
    # The two populations separate by two orders of magnitude in the
    # bulk — facing text p10 0.385, commentary p90 0.019 — but NOT in
    # the tails, which overlap: facing text runs down to 0.041 and
    # commentary up to 0.273.  So this is a minimum-error operating
    # point on a bimodal distribution, near its geometric midpoint, and
    # not a free choice anywhere in a wide band.  At 0.10 it discards 19
    # genuine facing chunks of 5,684 and admits 2 commentary chunks of
    # 722; at 0.20 it discards 31; by 0.29 it discards 354.
    bilingual_min_hebrew_share: float = Field(default=0.10)

    # ── Corpus asymmetry ───────────────────────────────────────────────
    #
    # Pairs of populations that describe the SAME thing, so that where
    # their vocabularies diverge means something.
    #
    # A side used to be a SQL LIKE over `source_path`, and naming files is
    # what made this module's own flagship example false.  Twenty-nine
    # works arrived on 2026-09-04; two were the Pritzker Zohar, which IS
    # primary text and matched no pattern, so "the primary text" quietly
    # became four files out of eight.  `companions` went on standing at
    # 39.2% of the scholarship against 0.03% of the primary text while
    # sitting in 5.79% of Pritzker — 95 chunks of what it was said to be
    # absent from.  `shekhinah` was reported absent from a primary text
    # holding it at 45.95%.
    #
    # So a side names ROLES, and a role is carried by the document rather
    # than by its filename.  `corpus_documents` below is the registry.
    corpus_documents: list[dict] = Field(default_factory=lambda: [
        # Identity is a digest over the document's chunk hashes, so the
        # same work under a new filename keeps its role and two copies of
        # it are one document.  That is not hypothetical: `A River Flows
        # from Eden` arrived a second time as `dokumen.pub_a-river-flows-
        # from-eden-...`, 546 chunks byte-identical to the copy already
        # shelved, and only the LIKE pattern's failure to match the new
        # name kept the scholarship side from doubling.
        #
        # A title is required because a digest is unreadable, and because
        # an entry that stops matching has to be nameable in the report
        # that says so.
        {"role": "zohar-kabbalah-centre", "digest": "4b3f5ca7acc12160",
         "title": "The Zohar, volumes 1-23 (Kabbalah Centre)"},

        {"role": "zohar-bilingual-12", "digest": "fe23c99249151aa6",
         "title": "The Holy Zohar, book 1"},
        {"role": "zohar-bilingual-12", "digest": "3782299de4af12dc",
         "title": "The Holy Zohar, book 2"},
        {"role": "zohar-bilingual-12", "digest": "75766b103c3d6bc2",
         "title": "The Holy Zohar, book 3"},
        {"role": "zohar-bilingual-12", "digest": "72357925e76a210a",
         "title": "The Holy Zohar, book 4"},
        {"role": "zohar-bilingual-12", "digest": "7b20057e024bf9f5",
         "title": "The Holy Zohar, book 5"},
        {"role": "zohar-bilingual-12", "digest": "5762b0dca1056629",
         "title": "The Holy Zohar, book 6"},
        {"role": "zohar-bilingual-12", "digest": "76b60a52636d3547",
         "title": "The Holy Zohar, book 7"},
        {"role": "zohar-bilingual-12", "digest": "2777a094c830b824",
         "title": "The Holy Zohar, book 8"},
        {"role": "zohar-bilingual-12", "digest": "363e2a3d4c9a5cb1",
         "title": "The Holy Zohar, book 9"},
        {"role": "zohar-bilingual-12", "digest": "cf6c7b7b14053696",
         "title": "The Holy Zohar, book 10"},
        {"role": "zohar-bilingual-12", "digest": "30cace067efc0300",
         "title": "The Holy Zohar, book 11"},
        {"role": "zohar-bilingual-12", "digest": "cfb8531c5b459979",
         "title": "The Holy Zohar, book 12"},

        {"role": "zohar-sefer", "digest": "cdfb673e549a8c49",
         "title": "Sefer ha-Zohar (English)"},

        # Ingested 2026-09-04, primary text, and in no pair until now.
        {"role": "zohar-pritzker", "digest": "46c3e7e6e23d6b56",
         "title": "The Zohar, Pritzker edition, volume 1"},
        {"role": "zohar-pritzker", "digest": "8242972f689d3ce8",
         "title": "The Zohar, Pritzker edition, volume 2"},

        # Also primary text, and also new — but Hebrew-script, so they
        # bear no capitalised Latin term at all: 0 of 1,571 chunks.  They
        # carry a role because they ARE the primary text; the comparison
        # drops them because a chunk with no term is denominator and
        # nothing else.  Declaring what a thing is and deciding what a
        # given measurement can use it for are different jobs.
        {"role": "zohar-hebrew", "digest": "77f7d1dfa091e0e3",
         "title": "Sefer ha-Zohar, volume 11 (Hebrew)"},
        {"role": "zohar-hebrew", "digest": "1679e4c99544fe0f",
         "title": "Zohar al ha-Torah, volume 1 (Hebrew)"},

        {"role": "zohar-scholarship", "digest": "f992c7b3e17ccbf4",
         "title": "Melila Hellner-Eshed, A River Flows from Eden"},
    ])
    # The defaults are the two comparisons that were measured rather than
    # guessed: three English Zohar editions whose key vocabulary is nearly
    # disjoint, and a body of scholarship against the primary texts it is
    # about, which share almost no words at all.
    asymmetry_pairs: list[dict] = Field(default_factory=lambda: [
        {
            "name": "zohar-editions",
            "left_label": "the Kabbalah Centre edition",
            "left_roles": ["zohar-kabbalah-centre"],
            "right_label": "the twelve-volume bilingual edition",
            "right_roles": ["zohar-bilingual-12"],
        },
        {
            "name": "zohar-scholarship",
            "left_label": "the scholarship",
            "left_roles": ["zohar-scholarship"],
            "right_label": "the primary text",
            "right_roles": ["zohar-kabbalah-centre", "zohar-bilingual-12",
                            "zohar-sefer", "zohar-pritzker", "zohar-hebrew"],
        },
    ])
    # A term must be this common on one side...
    asymmetry_min_rate: float = Field(default=0.02)
    # ...and this rare on the other.  Two orders of magnitude apart, which
    # is where the measured cases sit: 39.2% against 0.03%, 37.5% against
    # 0.00%.  A narrower gap is a difference of emphasis rather than of
    # vocabulary, and reads as noise.
    asymmetry_max_other: float = Field(default=0.002)
    # Below this a side is too small for a rate to mean anything.
    asymmetry_min_population: int = Field(default=50)
    asymmetry_max_per_pair: int = Field(default=60)

    # Content against the tag it was filed under.  A lexical diff of
    # content against the tag NAME is useless — `mysticism` never appears
    # in the material it labels — so the test is whether a tag's memories
    # share vocabulary the library at large does not.
    #
    # Measured: `plato` yields socrates, glaucon, republic; `biosemiotics`
    # yields pattee; `whirly-art` and `essay`, 1,552 memories apiece,
    # yield nothing whatever.
    tag_content_baseline_sample: int = Field(default=3000)
    tag_content_sample: int = Field(default=400)
    tag_content_min_memories: int = Field(default=25)
    # A term has to be this common in the tag's material...
    tag_content_min_rate: float = Field(default=0.15)
    # ...and this much commoner there than in the library at large.
    tag_content_lift: float = Field(default=4.0)

    # How many laps an observation may fall behind its errand before it is
    # taken to be one the errand no longer makes.
    #
    # Two, not one, because of an off-by-one in the walk itself: observe()
    # stamps the lap read at the START of a run and advance_lap increments
    # after it, so a perfectly current observation always sits exactly one
    # behind. Measured across 11,867 standing observations, a threshold of
    # one flags every single one and a threshold of two flags four.
    #
    # This is the only staleness test available to a kind with no tag to
    # check — corpus_asymmetry anchors on a term and two populations,
    # bilingual_term on a Hebrew skeleton — and re-deriving either costs a
    # full-population scan, which is what the errand's own lap already is.
    audit_stale_laps: int = Field(default=2)
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
    # Flat at 0.70, and the flatness is the point.
    #
    # The rungs were tuned one at a time and drifted into incoherence:
    # (3, 0.70), (2, 0.80), (1, 0.75) demanded MORE of a question asked
    # twice than of the same question asked once or three times.  A bar
    # that RISES with evidence has the idea backwards — repetition is
    # evidence of demand, so it can only ever lower what quality must
    # prove, never raise it.  The validator below now refuses that shape
    # instead of letting it be configured silently.
    #
    # 0.70 is where the evidence sits: retrieval scores 0.66-0.80 on
    # questions these shelves CAN answer and 0.48-0.56 on ones they
    # cannot, so it clears the noise floor with room and admits every
    # genuine candidate.  The two it still excludes scored 0.483 and
    # 0.627 — measured failures, not marginal calls.
    #
    # Kept as a ladder rather than collapsed to one number because the
    # SHAPE is worth being able to express; it simply has no reason to be
    # stepped on this corpus yet.
    #
    # An empty list disables the ladder and falls back to the flat
    # faq_promote_min_hits / faq_promote_min_score pair above.
    faq_promote_tiers: list[tuple[int, float]] = Field(
        default_factory=lambda: [(3, 0.70), (2, 0.70), (1, 0.70)]
    )

    @field_validator("faq_promote_tiers")
    @classmethod
    def _tiers_must_not_rise_with_evidence(
        cls, tiers: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """More asks must never demand a higher score.

        Repetition and quality are evidence for the same conclusion, so a
        question asked more often can be admitted on a weaker answer —
        never held to a stricter one.  This shape was configured by
        accident once, by tuning rungs independently, and nothing caught
        it: a card asked twice was refused at 0.79 while a card asked once
        passed at 0.76.
        """
        ordered = sorted(tiers, key=lambda t: t[0])
        for (h1, s1), (h2, s2) in zip(ordered, ordered[1:]):
            if s2 > s1:
                raise ValueError(
                    f"faq_promote_tiers must not rise with hit count: "
                    f"{h2} asks requires {s2} but {h1} asks only {s1}"
                )
        return tiers

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
    # Both lists take a plain name (exact match) or a namespace written
    # with a trailing colon: "specialist:" covers specialist:zohar and
    # every other shift. Subjects are derived and renameable, so listing
    # them individually would go stale at the next roster rebuild.
    #
    # A specialist is deliberately on NEITHER list. Its reading is research
    # on the library's own behalf: it should not register as patron demand,
    # and a shift that reads a shelf should not reshape the shelf it reads.
    # Its judgments are still attributed — through memories.author, on what
    # it writes, which is where a specialist's authority belongs.
    #
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
        default_factory=lambda: [
            "missing_embedding", "source_provenance", "concept_gaps",
        ]
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
