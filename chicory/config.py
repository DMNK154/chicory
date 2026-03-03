"""Configuration for Chicory."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ChicoryConfig(BaseModel):
    """All configuration for the Chicory memory system."""

    # Paths
    db_path: Path = Field(default_factory=lambda: Path.home() / ".chicory" / "chicory.db")

    # LLM
    anthropic_api_key: str = Field(default="")
    llm_model: str = Field(default="claude-sonnet-4-6")
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
        ]
    )
    lattice_min_resonance_primes: int = Field(default=4)
    lattice_void_radius: float = Field(default=0.3)
    lattice_retrieval_boost_enabled: bool = Field(default=True)
    lattice_retrieval_boost_weight: float = Field(default=0.15)

    # Tag relational tensor — network combination weights for recall
    tensor_cooccurrence_weight: float = Field(default=0.5)
    tensor_synchronicity_weight: float = Field(default=0.3)
    tensor_semantic_weight: float = Field(default=0.2)
    tensor_semiotic_weight: float = Field(default=0.15)

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

    # Commons Layer — cross-project signal federation
    commons_enabled: bool = Field(default=False)
    commons_db_path: Path = Field(
        default_factory=lambda: Path.home() / ".chicory" / "commons.db"
    )
    commons_project_id: str = Field(default="")
    commons_signal_buffer_size: int = Field(default=10)
    commons_flush_interval_seconds: float = Field(default=5.0)


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
        "CHICORY_DB_PATH": "db_path",
        "CHICORY_LLM_MODEL": "llm_model",
        "CHICORY_EMBEDDING_MODEL": "embedding_model",
        "CHICORY_COMMONS_ENABLED": "commons_enabled",
        "CHICORY_COMMONS_DB_PATH": "commons_db_path",
        "CHICORY_COMMONS_PROJECT_ID": "commons_project_id",
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
    if isinstance(kwargs.get("commons_enabled"), str):
        kwargs["commons_enabled"] = kwargs["commons_enabled"].lower() in ("true", "1", "yes")

    kwargs.update(overrides)

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
