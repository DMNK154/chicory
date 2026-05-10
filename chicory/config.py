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

    # Temporal tag episodes — drift-detected, revisitable tag-space clusters
    episode_enabled: bool = Field(default=True)
    episode_ema_alpha: float = Field(default=0.2)
    episode_drift_sigma: float = Field(default=2.0)
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

    # Migration
    burn_in_hours: float = Field(default=48.0)
    burn_in_threshold_multiplier: float = Field(default=1.5)

    # Retrieval
    hybrid_semantic_weight: float = Field(default=0.7)
    hybrid_tag_weight: float = Field(default=0.3)

    # Deep Retrieve
    deep_retrieve_max_depth: int = Field(default=3)
    deep_retrieve_per_level_k: int = Field(default=5)
    deep_retrieve_depth_decay: float = Field(default=0.7)
    deep_retrieve_time_depth_weight: float = Field(default=0.3)

    # FAISS vector index
    faiss_nprobe: int = Field(default=8)
    faiss_rebuild_threshold: int = Field(default=500)

    # Ingestion — two-tier criticality classification
    ingestion_criticality_threshold: float = Field(default=0.3)
    ingestion_critical_importance: float = Field(default=0.6)
    ingestion_reference_importance: float = Field(default=0.3)
    ingestion_reference_summary_max_chars: int = Field(default=2000)

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
    for bool_key in ("commons_enabled", "glyph_bridge_enabled"):
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
