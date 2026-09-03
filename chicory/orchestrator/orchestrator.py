"""Central coordinator — wires all layers, handles side effects."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Optional

# Module-level, not function-local: `from __future__ import annotations`
# means np-typed signatures never evaluate, so a missing import here is
# invisible until a runtime np.* call raises NameError.
import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.embedding_engine import EmbeddingEngine
from chicory.layer1.memory_store import MemoryStore
from chicory.layer1.salience import SalienceScorer
from chicory.layer1.tag_manager import TagManager
from chicory.layer2.retrieval_tracker import RetrievalTracker
from chicory.layer2.trend_engine import TrendEngine
from chicory.layer3.phase_space import PhaseSpace
from chicory.layer3.synchronicity_detector import SynchronicityDetector
from chicory.layer3.centroid_subgraph import CentroidSubgraph
from chicory.layer3.synchronicity_engine import SynchronicityEngine
from chicory.layer3.tag_space import TagSpace
from chicory.layer3.tensor_cache import TensorCache
from chicory.layer4.adaptive_thresholds import AdaptiveThresholds
from chicory.layer4.canopy import CanopyObserver
from chicory.layer3.directional_flow import InflowObserver, OutflowObserver
from chicory.layer4.episodic_tensor import EpisodicTensor
from chicory.layer4.temporal_episodes import TemporalEpisodeTracker
from chicory.layer4.feedback import FeedbackEngine
from chicory.layer4.closure import ClosureLedger
from chicory.layer4.desk import DeskReport
from chicory.layer4.ingress import IngressBuffer
from chicory.layer4.librarian import (
    CommissionDesk,
    DemandLedger,
    FaqDesk,
    LinearSweep,
    PenumbraLedger,
    SubjectRoster,
    WriteGate,
    analogic_target,
    parallelness,
    record_custody_transition,
)
from chicory.layer4.meta_analyzer import MetaAnalyzer
from chicory.layer4.forest import ForestReorganizer
from chicory.layer4.wander import Heartbeat, Patrol
from chicory.layer1.glyph_analyzer import analyze_text, extract_glyph_tags
from chicory.models.memory import Memory, Tag
from chicory.models.synchronicity import SynchronicityEvent


class Orchestrator:
    """Central coordinator. All LLM tool calls route through here."""

    def __init__(self, config: ChicoryConfig) -> None:
        self._config = config

        # Serializes the shared SQLite connection across foreground
        # (retrieve/store) and the background sync-detection thread.
        # Foreground always holds it during its DB work; the background
        # thread waits briefly and then skips if foreground stays busy,
        # so "database is locked" races on the shared connection are
        # avoided without blocking user-facing calls indefinitely.
        # RLock so any future nested foreground call doesn't deadlock.
        self._foreground_db_lock = threading.RLock()
        self._foreground_waiting = threading.Event()

        # Database
        self._db = DatabaseEngine(config)
        self._db.connect()
        apply_schema(self._db)

        # Layer 1 (partial — MemoryStore needs Layer 3 sync_engine)
        self._tag_manager = TagManager(self._db)
        self._embedding_engine = EmbeddingEngine(config, self._db)
        self._salience = SalienceScorer(config, self._db)

        # Layer 2
        self._trend_engine = TrendEngine(config, self._db)
        self._retrieval_tracker = RetrievalTracker(config, self._db)

        # Layer 3
        self._phase_space = PhaseSpace(
            config, self._db, self._trend_engine, self._retrieval_tracker,
        )
        self._sync_detector = SynchronicityDetector(
            config, self._db, self._phase_space,
            self._trend_engine, self._retrieval_tracker,
            self._tag_manager, self._embedding_engine,
        )
        self._glyph_encoder = None
        if config.glyph_bridge_enabled:
            try:
                from chicory.layer1.glyph_bridge import GlyphBridge
                self._glyph_encoder = GlyphBridge(config, self._db)
                if not self._glyph_encoder.is_available:
                    logging.getLogger(__name__).warning(
                        "GlyphBridge failed to load, trying GlyphEncoder")
                    self._glyph_encoder = None
            except ImportError:
                logging.getLogger(__name__).warning(
                    "GlyphBridge import failed, trying GlyphEncoder")
        if self._glyph_encoder is None and config.glyph_model_dir:
            from chicory.layer1.glyph_encoder import GlyphEncoder
            self._glyph_encoder = GlyphEncoder(config, self._db)

        self._tensor_cache = TensorCache(self._db)

        self._sync_engine = SynchronicityEngine(
            config, self._db, self._embedding_engine, self._tag_manager,
            glyph_encoder=self._glyph_encoder,
            tensor_cache=self._tensor_cache,
        )

        # Centroid sub-graph for retrieval-driven inhibition
        self._centroid_subgraph = CentroidSubgraph(
            config, self._db, self._embedding_engine,
        )

        # Seed the tag relational tensor if empty
        self._maybe_seed_tensor()

        # Seed glyph Ramsey lattice if empty
        self._maybe_seed_glyph_lattice()

        # Seed glyph cross-reference data (oppositions, transformations, semiotic)
        self._maybe_seed_glyph_cross_reference()

        # Seed centroid sub-graph if empty
        self._maybe_seed_centroid_subgraph()

        # Tag Space — independent tag-graph retrieval path
        self._tag_space = TagSpace(config, self._db, self._tag_manager,
                                   tensor_cache=self._tensor_cache)

        # Layer 1 (complete — now has access to Layer 3 for lattice-aware retrieval)
        self._memory_store = MemoryStore(
            config, self._db, self._embedding_engine,
            self._tag_manager, self._salience,
            sync_engine=self._sync_engine,
            tag_space=self._tag_space,
        )

        # Layer 4
        self._adaptive_thresholds = AdaptiveThresholds(config, self._db)
        self._meta_analyzer = MetaAnalyzer(
            config, self._db, self._adaptive_thresholds,
            sync_engine=self._sync_engine,
        )
        self._feedback = FeedbackEngine(self._db, self._tag_manager, self._salience)

        # Layer 4.5 — Forest + Canopy + Episodic Tensor
        self._forest = ForestReorganizer(self._db, config)
        self._canopy = CanopyObserver(self._db, config, self._forest)
        self._inflow_observer = InflowObserver(self._db, config)
        self._outflow_observer = OutflowObserver(self._db, config)
        self._episodic_tensor = EpisodicTensor(self._db, config)
        self._temporal_episodes = TemporalEpisodeTracker(
            self._db, config, self._centroid_subgraph,
        )
        self._maybe_seed_temporal_episodes()

        # Librarian — write gate, linear sweep patrol, subject roster
        self._write_gate = WriteGate(
            config, self._db, self._tag_manager, self._centroid_subgraph,
        )
        self._penumbra_ledger = PenumbraLedger(config, self._db)
        self._linear_sweep = LinearSweep(
            config, self._db, self._embedding_engine,
            self._tag_manager, self._centroid_subgraph,
            penumbra=self._penumbra_ledger,
        )
        self._subject_roster = SubjectRoster(
            config, self._db, self._centroid_subgraph,
        )

        # Ingress dwell — receipted hold in front of memory commitment
        self._ingress = IngressBuffer(config, self._db)

        # The desk report — the library's standing "since we last spoke"
        self._desk = DeskReport(config, self._db)

        # Closure — supersession, and unwinding the chain a report closes
        self._closure = ClosureLedger(config, self._db)

        # Librarian front desk — FAQ, demand ledger, commissions
        self._faq_desk = FaqDesk(config, self._db)
        self._commission_desk = CommissionDesk(config, self._db)
        self._demand_ledger = DemandLedger(
            config, self._db, self._memory_store,
            self._faq_desk, self._commission_desk, self._subject_roster,
            closure=self._closure,
        )

        # The wandering librarian — errands with their own cursors, and a
        # clock so they advance when nobody is talking to the library.
        self._patrol = Patrol(config, self._db, self._memory_store._embedding)
        self._heartbeat = Heartbeat(
            config,
            tick=self._heartbeat_tick,
            busy=self._foreground_waiting.is_set,
        )

        # Rate limiting for detection — initialise to "now" so the first
        # retrieval isn't blocked by expensive sync/meta detection.
        self._last_sync_check: datetime | None = datetime.utcnow()
        self._last_meta_check: datetime | None = datetime.utcnow()
        self._last_sweep_check: datetime | None = datetime.utcnow()
        self._last_demand_check: datetime | None = datetime.utcnow()
        self._last_ingress_check: datetime | None = None  # settle promptly
        self._last_ingress_prune: datetime | None = None
        # The patrol has its own per-errand intervals, so this only stops
        # every background pass from re-entering it.
        self._last_patrol_check: datetime | None = None
        self._background_idle = False

        # Diagnostics — activation trace + context log
        self._prev_trace = None
        self._last_trace = None
        self._context_logger = None
        if config.context_log_enabled:
            from chicory.diagnostics.context_log import ContextLogger
            self._context_logger = ContextLogger(
                config.context_log_dir, log_full=config.context_log_full,
            )

        # Commons signal emitter + auto-processor (optional)
        self._signal_emitter = None
        self._signal_processor = None
        if config.commons_enabled and config.commons_project_id:
            try:
                from chicory_commons import SignalEmitter
            except ImportError:
                SignalEmitter = None
            if SignalEmitter is not None:
                self._signal_emitter = SignalEmitter(config)
                self._signal_emitter.start()
            self._signal_processor = self._init_signal_processor()

    @property
    def db(self) -> DatabaseEngine:
        return self._db

    @property
    def glyph_encoder(self):
        """GlyphEncoder instance, or None if ByT5 model not configured."""
        return self._glyph_encoder

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def tag_manager(self) -> TagManager:
        return self._tag_manager

    @property
    def trend_engine(self) -> TrendEngine:
        return self._trend_engine

    @property
    def phase_space(self) -> PhaseSpace:
        return self._phase_space

    @property
    def sync_detector(self) -> SynchronicityDetector:
        return self._sync_detector

    @property
    def sync_engine(self) -> SynchronicityEngine:
        return self._sync_engine

    @property
    def meta_analyzer(self) -> MetaAnalyzer:
        return self._meta_analyzer

    @property
    def canopy(self) -> CanopyObserver:
        return self._canopy

    def get_relevant_tags(self, query: str) -> list[str]:
        """Return tag names most relevant to a query via centroid similarity.

        Embeds the query, loads all tag centroids, computes cosine similarity,
        and returns the top ``context_tag_limit`` tag names.  Falls back to
        all active tag names if no centroids exist or embedding fails.
        """
        import numpy as np

        limit = self._config.context_tag_limit

        # Load all centroids in one query
        rows = self._db.execute(
            "SELECT tc.tag_id, tc.centroid, t.name "
            "FROM tag_centroids tc "
            "JOIN tags t ON t.id = tc.tag_id "
            "WHERE t.is_active = 1"
        ).fetchall()

        if not rows:
            # No centroids yet — fall back to all active (capped)
            names = self._tag_manager.list_active_names()
            return names[:limit]

        try:
            query_vec = self._embedding_engine.embed(query)
        except Exception:
            names = self._tag_manager.list_active_names()
            return names[:limit]

        # Build matrix of centroid vectors and compute cosine similarities
        tag_names = [r["name"] for r in rows]
        centroid_matrix = np.stack([
            np.frombuffer(r["centroid"], dtype=np.float32) for r in rows
        ])

        # query_vec and centroids are already unit-normalized
        similarities = centroid_matrix @ query_vec

        # Top-k by similarity
        if len(similarities) <= limit:
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, limit)[:limit]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [tag_names[i] for i in top_indices]

    def _maybe_seed_tensor(self) -> None:
        """Seed the tag relational tensor on first boot after upgrade."""
        tensor_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM tag_relational_tensor"
        ).fetchone()["cnt"]

        if tensor_count > 0:
            return

        memory_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0"
        ).fetchone()["cnt"]
        if memory_count > 0:
            self._sync_engine.seed_tensor_from_associations()

        lattice_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM lattice_positions"
        ).fetchone()["cnt"]
        if lattice_count > 0:
            self._sync_engine.rebuild_tensor()

    def _maybe_seed_glyph_lattice(self) -> None:
        """Seed the glyph Ramsey lattice on first boot after upgrade."""
        glyph_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM glyph_positions"
        ).fetchone()["cnt"]
        if glyph_count > 0:
            return

        memory_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0"
        ).fetchone()["cnt"]
        if memory_count > 0:
            tag_names = self._tag_manager.list_active_names()
            word_tags = [n for n in tag_names if n]
            if word_tags:
                self._sync_engine.place_tags_on_glyph_lattice(word_tags)

    def _maybe_seed_glyph_cross_reference(self) -> None:
        """Seed glyph cross-reference data and parallelness on first boot.

        Runs when GlyphBridge is available and the tensor has no
        inhibition data yet (first boot after upgrade to v16).
        Auto-detects: if any tensor edge has inhibition_strength > 0,
        assumes seeding was already done.
        """
        from chicory.layer1.glyph_bridge import GlyphBridge

        if not isinstance(self._glyph_encoder, GlyphBridge):
            return

        # Check if already seeded
        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM tag_relational_tensor "
            "WHERE inhibition_strength > 0"
        ).fetchone()
        if row["cnt"] > 0:
            # Already seeded — just refresh parallelness from current angles
            self._sync_engine.compute_all_parallelness()
            return

        # Full seed: cross-reference data + semiotic pairs + parallelness
        logging.getLogger(__name__).info("Seeding glyph cross-reference data into tensor...")
        self._sync_engine.seed_from_cross_reference()
        self._sync_engine.seed_semiotic_from_pairs()
        self._sync_engine.compute_all_parallelness()

    def _maybe_seed_centroid_subgraph(self) -> None:
        """Seed centroids and co-retrieval edges on first boot after upgrade."""
        centroid_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM tag_centroids"
        ).fetchone()["cnt"]
        if centroid_count > 0:
            return

        memory_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0"
        ).fetchone()["cnt"]
        if memory_count > 0:
            self._centroid_subgraph.rebuild_centroids()

        retrieval_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM retrieval_tag_hits"
        ).fetchone()["cnt"]
        if retrieval_count > 0:
            self._centroid_subgraph.rebuild_edges_from_history()

    def _maybe_seed_temporal_episodes(self) -> None:
        """Bootstrap temporal episodes on first boot or resume partial bootstrap."""
        if not self._config.episode_enabled:
            return
        memory_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0"
        ).fetchone()["cnt"]
        if memory_count == 0:
            return
        assigned_count = self._db.execute(
            "SELECT COUNT(DISTINCT memory_id) as cnt FROM memory_episode_assignments"
        ).fetchone()["cnt"]
        if assigned_count >= memory_count:
            return
        self._temporal_episodes.bootstrap()

    def _init_signal_processor(self):
        """Initialize the commons SignalProcessor with optional glyph integration."""
        try:
            from chicory_commons import SignalProcessor
        except ImportError:
            return None

        glyph_bridge = None
        heat_tracker = None
        sync_detector = None
        try:
            from chicory_commons import (
                CommonsGlyphBridge,
                FederationHeatTracker,
                FederationSyncDetector,
            )

            if CommonsGlyphBridge is not None:
                glyph_bridge = CommonsGlyphBridge(
                    gptgu_path=self._config.gptgu_path or None,
                )
                if glyph_bridge.is_available and self._config.commons_db_path:
                    import sqlite3

                    commons_conn = sqlite3.connect(
                        str(self._config.commons_db_path),
                        check_same_thread=False,
                    )
                    commons_conn.execute("PRAGMA journal_mode = WAL")
                    commons_conn.execute("PRAGMA busy_timeout = 5000")
                    self._commons_conn = commons_conn
                    heat_tracker = FederationHeatTracker(commons_conn)
                    sync_detector = FederationSyncDetector(
                        commons_conn, heat_tracker, glyph_bridge,
                    )
        except Exception:
            logging.getLogger(__name__).debug(
                "Glyph integration not available for commons (tag-only mode)",
            )

        return SignalProcessor(
            self,
            glyph_bridge=glyph_bridge,
            heat_tracker=heat_tracker,
            sync_detector=sync_detector,
        )

    def _commons_auto_process(self) -> None:
        """Process pending commons signals in a background thread."""
        try:
            self._signal_processor.maybe_auto_process()
        except Exception:
            logging.getLogger(__name__).exception(
                "Error in commons auto-processing",
            )

    def run_sync(
        self,
        on_step: "Optional[Any]" = None,
    ) -> dict[str, Any]:
        """Run full network synchronization across all memories.

        Executes the expensive network-building passes that are normally
        incremental/rate-limited during live usage. Designed to be run
        after a large bulk ingest to bring the network up to date.

        Args:
            on_step: Optional callback ``(step_name: str, detail: str) -> None``
                called before each major phase for progress reporting.

        Returns:
            Dict with counts and timing for each phase.
        """
        import time

        def _report(step: str, detail: str = "") -> None:
            if on_step:
                on_step(step, detail)

        stats: dict[str, Any] = {}
        t_total = time.time()

        # 1. Rebuild tag centroids from embeddings
        _report("centroids", "Rebuilding tag centroids from embeddings")
        t0 = time.time()
        if self._config.centroid_inhibition_enabled:
            stats["centroids_rebuilt"] = self._centroid_subgraph.rebuild_centroids()
            stats["centroid_edges_rebuilt"] = (
                self._centroid_subgraph.rebuild_edges_from_history()
            )
        else:
            stats["centroids_rebuilt"] = 0
            stats["centroid_edges_rebuilt"] = 0
        stats["centroids_seconds"] = round(time.time() - t0, 1)

        # 2. Place all tags on the glyph Ramsey lattice
        _report("glyph_lattice", "Placing tags on glyph Ramsey lattice")
        t0 = time.time()
        all_tag_names = self._tag_manager.list_active_names()
        word_tags = [t for t in all_tag_names if len(t) > 1]
        glyph_positions = self._sync_engine.place_tags_on_glyph_lattice(word_tags)
        stats["glyph_positions"] = len(glyph_positions)
        stats["glyph_lattice_seconds"] = round(time.time() - t0, 1)

        # 3. Compute lateral inhibition (parallelness) for all tensor edges
        _report("parallelness", "Computing lateral inhibition across tensor")
        t0 = time.time()
        stats["parallelness_updated"] = self._sync_engine.compute_all_parallelness()
        stats["parallelness_seconds"] = round(time.time() - t0, 1)

        # 4. Seed tensor from tag associations
        _report("tensor_seed", "Seeding tensor from tag co-occurrence associations")
        t0 = time.time()
        self._sync_engine.seed_tensor_from_associations()
        stats["tensor_seed_seconds"] = round(time.time() - t0, 1)

        # 5. Update co-occurrence tensor (PMI + bridge gated)
        _report("cooccurrence", "Updating co-occurrence tensor (PMI + bridge gated)")
        t0 = time.time()
        stats["cooccurrence_pairs"] = self._sync_engine.update_cooccurrence_tensor()
        stats["cooccurrence_seconds"] = round(time.time() - t0, 1)

        # 6. Update semantic tensor
        _report("semantic", "Updating semantic tensor from embedding similarities")
        t0 = time.time()
        stats["semantic_pairs"] = self._sync_engine.update_semantic_tensor()
        stats["semantic_seconds"] = round(time.time() - t0, 1)

        # 7. Update glyph tensor
        _report("glyph_tensor", "Updating glyph tensor strengths")
        t0 = time.time()
        stats["glyph_pairs"] = self._sync_engine.update_glyph_tensor()
        stats["glyph_tensor_seconds"] = round(time.time() - t0, 1)

        # 8. Seed semiotic directionality from glyph pairs
        _report("semiotic", "Seeding semiotic tensor from glyph pair directionality")
        t0 = time.time()
        stats["semiotic_pairs"] = self._sync_engine.seed_semiotic_from_pairs()
        stats["semiotic_seconds"] = round(time.time() - t0, 1)

        # 9. Cross-reference seeding (GPT-GU if available)
        _report("cross_reference", "Seeding cross-reference data")
        t0 = time.time()
        stats["cross_reference_pairs"] = self._sync_engine.seed_from_cross_reference()
        stats["cross_reference_seconds"] = round(time.time() - t0, 1)

        # 10. Bootstrap episodic tensor (memory-to-memory edges)
        _report("episodic", "Bootstrapping episodic tensor (memory-to-memory edges)")
        t0 = time.time()
        if self._config.canopy_enabled:
            stats["episodic_edges"] = self._episodic_tensor.bootstrap()
            stats["episodic_bridges_backfilled"] = (
                self._episodic_tensor.backfill_bridge_strength()
            )
        else:
            stats["episodic_edges"] = 0
            stats["episodic_bridges_backfilled"] = 0
        stats["episodic_seconds"] = round(time.time() - t0, 1)

        # 10b. Episodic tensor pruning (standalone, for edges not from bootstrap)
        _report("episodic_prune", "Pruning episodic tensor (weak, caps, decay)")
        t0 = time.time()
        if self._config.canopy_enabled:
            prune_stats = self._episodic_tensor.prune()
            stats["episodic_weak_pruned"] = prune_stats["weak_pruned"]
            stats["episodic_cap_pruned"] = prune_stats["cap_pruned"]
            stats["episodic_decayed"] = prune_stats["decayed"]
            stats["episodic_archived"] = prune_stats["archived"]
        else:
            stats["episodic_weak_pruned"] = 0
            stats["episodic_cap_pruned"] = 0
            stats["episodic_decayed"] = 0
            stats["episodic_archived"] = 0
        stats["episodic_prune_seconds"] = round(time.time() - t0, 1)

        # 11. Canopy global recursive pass (merge blocks upward)
        _report("canopy", "Running canopy global recursive merge")
        t0 = time.time()
        if self._config.canopy_enabled:
            grown_keys = self._canopy.global_recursive_pass()
            stats["canopy_grown_blocks"] = len(grown_keys)
        else:
            stats["canopy_grown_blocks"] = 0
        stats["canopy_seconds"] = round(time.time() - t0, 1)

        # 12. Deduplicate sync events (before detection to prevent runaway multiplication)
        _report("sync_dedup", "Deduplicating synchronicity events")
        t0 = time.time()
        dedup_cursor = self._db.execute(
            """DELETE FROM synchronicity_events
               WHERE id NOT IN (
                   SELECT id FROM (
                       SELECT id, ROW_NUMBER() OVER (
                           PARTITION BY event_type, involved_tags, quadrant
                           ORDER BY strength DESC, detected_at DESC
                       ) AS rn
                       FROM synchronicity_events
                   ) WHERE rn = 1
               )"""
        )
        stats["sync_dedup_removed"] = dedup_cursor.rowcount
        if dedup_cursor.rowcount:
            self._db.connection.commit()
        stats["sync_dedup_seconds"] = round(time.time() - t0, 1)

        # 13. Synchronicity detection (unrestricted — bypass rate limit)
        _report("sync_detection", "Running synchronicity detection")
        t0 = time.time()
        new_events = self._sync_detector.check_for_synchronicities()
        if new_events:
            self._sync_engine.place_events_batch(new_events)
        stats["sync_events_detected"] = len(new_events) if new_events else 0
        stats["sync_detection_seconds"] = round(time.time() - t0, 1)

        # 14. Meta-pattern analysis
        _report("meta_analysis", "Running meta-pattern analysis")
        t0 = time.time()
        patterns = self._meta_analyzer.run_analysis()
        for pattern in patterns:
            self._feedback.apply_pattern_actions(pattern)
        stats["meta_patterns"] = len(patterns)
        stats["meta_analysis_seconds"] = round(time.time() - t0, 1)

        # 15. Episode lifecycle
        _report("episode_lifecycle", "Running episode lifecycle pass")
        t0 = time.time()
        if self._config.episode_enabled:
            ep_stats = self._temporal_episodes.lifecycle_pass()
            stats["episodes_dormant"] = ep_stats.get("dormant", 0)
            stats["episodes_archived"] = ep_stats.get("archived", 0)
        else:
            stats["episodes_dormant"] = 0
            stats["episodes_archived"] = 0
        stats["episode_lifecycle_seconds"] = round(time.time() - t0, 1)

        stats["total_seconds"] = round(time.time() - t_total, 1)

        # Commit all changes
        self._db.connection.commit()
        self._tensor_cache.invalidate()

        _report("done", f"Sync complete in {stats['total_seconds']:.1f}s")
        return stats

    def close(self) -> None:
        """Close database connection and signal emitter."""
        # Stop the clock before the connection goes: a beat landing on a
        # closed database would log an exception per tick forever.
        self._heartbeat.stop()
        if self._signal_emitter:
            self._signal_emitter.stop()
        if hasattr(self, "_commons_conn"):
            self._commons_conn.close()
        self._db.close()

    # ── Tool call handlers ──────────────────────────────────────────

    def handle_store_memory(
        self,
        content: str,
        tags: list[str],
        importance: float | None = None,
        summary: str | None = None,
        skip_embedding: bool = False,
        content_hash: str | None = None,
        defer_side_effects: bool = False,
        source_path: str | None = None,
        ingestion_tier: str = "critical",
        author: str | None = None,
        origin: str | None = None,
    ) -> dict[str, Any]:
        """Store a memory and fire side effects.

        Automatically scans content for glyph concept words and adds
        them as tags so that glyph definitions flow through the memory
        the same way GPT-GU's analyze_text works.

        When defer_side_effects=True, skips _on_memory_stored (trend events,
        lattice placement, canopy, etc). Callers must handle side effects
        themselves — used by bulk ingestion to avoid O(n) expensive work.

        The librarian write gate scores the incoming embedding against tag
        centroids; below-threshold writes are diverted to the 'quarantine'
        tier (when enforcement is on) and skip network side effects so a
        suspect memory can't reshape the centroids it failed against.
        """
        salience = importance if importance is not None else 0.5

        # CPU-bound work outside the lock so retrievals aren't blocked
        glyph_analysis = analyze_text(content)
        glyph_derived_tags = extract_glyph_tags(content)

        merged_tags = list(tags)
        for gt in glyph_derived_tags:
            if gt not in merged_tags:
                merged_tags.append(gt)

        # Pre-compute embedding outside the lock (model forward pass is CPU-bound)
        precomputed: list[np.ndarray] | None = None
        if not skip_embedding:
            from chicory.ingest.chunker import chunk_text_for_embedding
            chunks = chunk_text_for_embedding(content)
            if len(chunks) <= 1:
                precomputed = [self._embedding_engine.embed(content)]
            else:
                precomputed = list(self._embedding_engine.embed_batch(chunks))

        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                # Librarian write gate — only gates default-tier writes where
                # the embedding is in hand; bulk ingest paths classify tier
                # themselves and pass skip_embedding=True.
                gate_verdict = None
                if (
                    self._config.librarian_gate_enabled
                    and precomputed is not None
                    and ingestion_tier == "critical"
                ):
                    try:
                        gate_verdict = self._write_gate.evaluate_content(
                            precomputed, merged_tags,
                        )
                        if gate_verdict.diverted:
                            ingestion_tier = "quarantine"
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Write gate evaluation failed; storing ungated"
                        )
                        gate_verdict = None

                memory = self._memory_store.store(
                    content=content,
                    tags=merged_tags,
                    salience_model=salience,
                    summary=summary,
                    skip_embedding=True,
                    content_hash=content_hash,
                    source_path=source_path,
                    ingestion_tier=ingestion_tier,
                    author=author,
                    origin=origin,
                    precomputed_embeddings=precomputed,
                )

                if gate_verdict is not None:
                    self._write_gate.log(memory.id, gate_verdict)

                if glyph_analysis["words"]:
                    self._store_glyph_metadata(memory.id, glyph_analysis)

                quarantined = ingestion_tier == "quarantine"
                if quarantined:
                    # store_cached incrementally added this embedding to any
                    # live FAISS index; drop the index so the next search
                    # rebuilds from the tier-filtered load query.
                    self._embedding_engine._clear_memory_cache()
                    record_custody_transition(
                        self._db, memory.id, "quarantined", actor="gate",
                        reason=f"score={gate_verdict.score:.3f}" if gate_verdict else "",
                    )
                if not defer_side_effects and not quarantined:
                    self._on_memory_stored(memory, glyph_analysis=glyph_analysis)

                response: dict[str, Any] = {
                    "status": "quarantined" if quarantined else "stored",
                    "memory_id": memory.id,
                    "tags": memory.tags,
                    "glyph_concepts": [
                        {"concept": w["word"], "glyph": w["glyph"]}
                        for w in glyph_analysis["words"]
                    ],
                    "glyph_pairs": glyph_analysis["pairs"],
                    "salience": memory.salience_composite,
                }
                if gate_verdict is not None:
                    response["gate"] = {
                        "verdict": gate_verdict.verdict,
                        "score": round(gate_verdict.score, 3),
                    }
                return response
        finally:
            self._foreground_waiting.clear()

    def handle_retrieve_memories(
        self,
        query: str,
        tags: list[str] | None = None,
        method: str = "hybrid",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Retrieve memories, log the event, and fire side effects."""
        # Signal background thread that foreground needs the lock,
        # then acquire it.  The background thread checks the event
        # and yields if foreground is waiting.
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                return self._handle_retrieve_memories_locked(query, tags, method, top_k)
        finally:
            self._foreground_waiting.clear()

    def _handle_retrieve_memories_locked(
        self,
        query: str,
        tags: list[str] | None,
        method: str,
        top_k: int,
    ) -> dict[str, Any]:
        import time as _time
        _t0 = _time.time()

        # Reset the shared query-vector and penumbra slots so stale values
        # from a previous call (or a background promotion pass) can never
        # be paired with this query in the ledgers.
        self._memory_store._last_query_vec = None
        self._memory_store._last_penumbra = []

        if method == "semantic":
            results = self._memory_store.retrieve_semantic(query, top_k, tags)
        elif method == "tag" and tags:
            tag_results = self._memory_store.retrieve_by_tags(tags)
            results = [(m, 1.0) for m in tag_results[:top_k]]
        else:
            results = self._memory_store.retrieve_hybrid(query, tags, top_k)

        query_vec_for_ledger = getattr(self._memory_store, "_last_query_vec", None)
        penumbra_for_ledger = list(getattr(self._memory_store, "_last_penumbra", []))

        # Min-score gate: drop low-relevance results before they reach the LLM
        # or get reinforced by bookkeeping. Tag method always scores 1.0 so
        # it's unaffected; semantic/hybrid scores are the post-weighting total.
        # Threshold adapts to whether a tags arg was passed — without tags the
        # tag channel contributes 0, so the achievable score cap is lower and
        # the floor needs to drop with it.
        min_score = (
            self._config.retrieval_min_relevance_score_tagged
            if tags else self._config.retrieval_min_relevance_score_untagged
        )
        if min_score > 0:
            results = [(m, s) for m, s in results if s >= min_score]

        _t_search = _time.time()

        # Log retrieval event
        result_tuples = [
            (mem.id, rank + 1, score)
            for rank, (mem, score) in enumerate(results)
        ]
        retrieval_id = self._retrieval_tracker.log_retrieval(
            query_text=query,
            method=method,
            results=result_tuples,
            model_version=self._config.llm_model,
        )

        _t_log = _time.time()

        # Side effects (background thread)
        self._on_retrieval_completed(
            retrieval_id, results, query, query_vec=query_vec_for_ledger,
            penumbra=penumbra_for_ledger,
        )

        _t_side = _time.time()

        # Annotate results with live glyph scanning (batched metadata)
        memory_ids_for_glyph = [mem.id for mem, _ in results]
        glyph_meta_batch = self._get_glyph_metadata_batch(memory_ids_for_glyph)

        annotated = []
        for mem, score in results:
            entry: dict[str, Any] = {
                "memory_id": mem.id,
                "content": mem.content,
                "summary": mem.summary,
                "tags": mem.tags,
                "salience": mem.salience_composite,
                "relevance_score": round(score, 3),
                "created_at": mem.created_at.isoformat(),
                "tier": mem.ingestion_tier,
                "author": mem.author,
                "custody": mem.custody_status,
            }
            self._annotate_glyph(entry, mem.content, glyph_meta_batch.get(mem.id))
            annotated.append(entry)
        self._annotate_supersession(annotated)

        _t_annot = _time.time()

        hybrid_timing = getattr(self._memory_store, "_last_hybrid_timing", None)
        bk_breakdown = getattr(self, "_last_bookkeeping_breakdown", None)
        bk_total = getattr(self, "_last_bookkeeping_time", None)

        timing: dict[str, Any] = {
            "search": round(_t_search - _t0, 3),
            "log": round(_t_log - _t_search, 3),
            "side_effects": round(_t_side - _t_log, 3),
            "annotate": round(_t_annot - _t_side, 3),
            "total": round(_t_annot - _t0, 3),
        }
        if hybrid_timing:
            timing["hybrid_breakdown"] = {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in hybrid_timing.items()
            }
        if bk_breakdown:
            timing["bookkeeping_breakdown"] = {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in bk_breakdown.items()
            }
        if bk_total is not None:
            timing["bookkeeping_total"] = round(bk_total, 3)

        # Build activation trace
        from chicory.diagnostics.activation_trace import ActivationTrace, ScoreBreakdown

        score_components = getattr(self._memory_store, "_last_score_components", {})
        breakdowns = []
        for mem, score in results:
            comp = score_components.get(mem.id, {})
            breakdowns.append(ScoreBreakdown(
                memory_id=mem.id,
                semantic=comp.get("semantic", 0.0),
                tag=comp.get("tag", 0.0),
                lattice=comp.get("lattice", 0.0),
                glyph=comp.get("glyph", 0.0),
                convergence=comp.get("convergence", 0.0),
                total=comp.get("total", score),
            ))

        tag_activations: dict[str, float] = {}
        raw_partner_scores = getattr(self._sync_engine, "_last_partner_tag_scores", {})
        if raw_partner_scores:
            id_to_name = self._tag_manager.get_names_by_ids(list(raw_partner_scores.keys()))
            tag_activations = {
                id_to_name[tid]: strength
                for tid, strength in raw_partner_scores.items()
                if tid in id_to_name
            }

        trace = ActivationTrace(
            query=query,
            method=method,
            tags_requested=tags or [],
            tag_activations=tag_activations,
            score_breakdowns=breakdowns,
            timing=timing,
            context_entries=annotated if self._config.context_log_full else None,
        )
        self._prev_trace = self._last_trace
        self._last_trace = trace

        if self._context_logger:
            self._context_logger.log(trace)

        return {
            "results": annotated,
            "count": len(results),
            "method": method,
            "_timing": timing,
        }

    def handle_deep_retrieve(
        self,
        query: str,
        tags: list[str] | None = None,
        max_depth: int | None = None,
        per_level_k: int | None = None,
    ) -> dict[str, Any]:
        """Recursively retrieve memories with time-depth scoring.

        Level 0: Standard hybrid retrieval for the initial query.
        Level 1+: Generate expansion queries from previous level's results,
        retrieve excluding already-seen IDs, apply time-depth scoring that
        progressively favors older memories at deeper recursion levels.
        """
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                return self._handle_deep_retrieve_locked(
                    query, tags, max_depth, per_level_k,
                )
        finally:
            self._foreground_waiting.clear()

    def _handle_deep_retrieve_locked(
        self,
        query: str,
        tags: list[str] | None,
        max_depth: int | None,
        per_level_k: int | None,
    ) -> dict[str, Any]:
        from chicory.layer2.time_series import exponential_decay

        depth_limit = max_depth or self._config.deep_retrieve_max_depth
        level_k = per_level_k or self._config.deep_retrieve_per_level_k
        depth_decay = self._config.deep_retrieve_depth_decay
        time_depth_w = self._config.deep_retrieve_time_depth_weight

        now = datetime.utcnow()
        seen_ids: set[str] = set()
        all_results: list[dict[str, Any]] = []
        all_retrieval_results: list[tuple[Memory, float]] = []

        # -- Level 0: Standard hybrid retrieval --
        self._memory_store._last_query_vec = None
        self._memory_store._last_penumbra = []
        level_0_results = self._memory_store.retrieve_hybrid(query, tags, level_k)
        # Capture the ORIGINAL query's vector and penumbra now — expansion
        # retrievals below overwrite the shared slots with their own.
        query_vec_for_ledger = getattr(self._memory_store, "_last_query_vec", None)
        penumbra_for_ledger = list(getattr(self._memory_store, "_last_penumbra", []))

        for mem, score in level_0_results:
            seen_ids.add(mem.id)
            all_results.append(self._format_deep_result(mem, score, depth=0))
            all_retrieval_results.append((mem, score))

        if level_0_results:
            result_tuples = [
                (mem.id, rank + 1, score)
                for rank, (mem, score) in enumerate(level_0_results)
            ]
            self._retrieval_tracker.log_retrieval(
                query_text=query,
                method="deep_retrieve_L0",
                results=result_tuples,
                model_version=self._config.llm_model,
            )

        # -- Levels 1..depth_limit: Recursive expansion --
        prev_level_results = level_0_results

        for depth in range(1, depth_limit + 1):
            if not prev_level_results:
                break

            expansion_queries = self._generate_expansion_queries(prev_level_results)
            if not expansion_queries:
                break

            level_results: list[tuple[Memory, float]] = []

            for exp_query in expansion_queries:
                raw_results = self._memory_store.retrieve_semantic(
                    exp_query, top_k=level_k,
                )

                for mem, sim_score in raw_results:
                    if mem.id in seen_ids:
                        continue
                    seen_ids.add(mem.id)

                    # Time-depth scoring: older memories get bonus at deeper levels
                    age_hours = (now - mem.created_at).total_seconds() / 3600
                    age_factor = 1.0 - exponential_decay(
                        age_hours,
                        self._config.salience_recency_longterm_halflife_hours,
                    )
                    age_bonus = time_depth_w * depth * age_factor

                    final_score = sim_score * (depth_decay ** depth) * (1.0 + age_bonus)
                    level_results.append((mem, final_score))

            # Keep top per_level_k for this depth
            level_results.sort(key=lambda x: x[1], reverse=True)
            level_results = level_results[:level_k]

            for mem, score in level_results:
                all_results.append(self._format_deep_result(mem, score, depth=depth))
                all_retrieval_results.append((mem, score))

            if level_results:
                result_tuples = [
                    (mem.id, rank + 1, score)
                    for rank, (mem, score) in enumerate(level_results)
                ]
                self._retrieval_tracker.log_retrieval(
                    query_text=f"[deep_retrieve_L{depth}] {'; '.join(expansion_queries[:3])}",
                    method=f"deep_retrieve_L{depth}",
                    results=result_tuples,
                    model_version=self._config.llm_model,
                )

            prev_level_results = level_results

        # Sort all results by score descending
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Min-score gate (same adaptive policy as handle_retrieve_memories).
        # Filter before annotation/side-effects so low-relevance memories
        # don't get reinforced or sent to the LLM.  Threshold is based on
        # whether the initial query had tags — level-1+ expansions don't
        # carry their own tags, so the policy follows the root call.
        min_score = (
            self._config.retrieval_min_relevance_score_tagged
            if tags else self._config.retrieval_min_relevance_score_untagged
        )
        if min_score > 0:
            all_results = [r for r in all_results if r["relevance_score"] >= min_score]
            kept_ids = {r["memory_id"] for r in all_results}
            all_retrieval_results = [
                (mem, score) for mem, score in all_retrieval_results
                if mem.id in kept_ids
            ]

        # Batch glyph annotation (single DB query for all results)
        all_mem_ids = [r["memory_id"] for r in all_results]
        glyph_meta_batch = self._get_glyph_metadata_batch(all_mem_ids)
        for entry in all_results:
            self._annotate_glyph(
                entry, entry["content"], glyph_meta_batch.get(entry["memory_id"]),
            )

        # Fire side effects once for all retrieved memories
        if all_retrieval_results:
            combined_tuples = [
                (mem.id, rank + 1, score)
                for rank, (mem, score) in enumerate(all_retrieval_results)
            ]
            combined_id = self._retrieval_tracker.log_retrieval(
                query_text=query,
                method="deep_retrieve",
                results=combined_tuples,
                model_version=self._config.llm_model,
            )
            self._on_retrieval_completed(
                combined_id, all_retrieval_results, query,
                query_vec=query_vec_for_ledger,
                penumbra=penumbra_for_ledger,
            )

        self._annotate_supersession(all_results)
        return {
            "results": all_results,
            "count": len(all_results),
            "method": "deep_retrieve",
            "max_depth_reached": max(
                (r["depth"] for r in all_results), default=0
            ),
        }

    @staticmethod
    def _format_deep_result(mem: Memory, score: float, depth: int) -> dict[str, Any]:
        """Format a single deep retrieval result with depth annotation."""
        return {
            "memory_id": mem.id,
            "content": mem.content,
            "summary": mem.summary,
            "tags": mem.tags,
            "salience": mem.salience_composite,
            "relevance_score": round(score, 3),
            "created_at": mem.created_at.isoformat(),
            "tier": mem.ingestion_tier,
            "author": mem.author,
            "custody": mem.custody_status,
            "depth": depth,
        }

    def handle_get_last_trace(self) -> dict[str, Any]:
        """Return the activation trace from the most recent retrieval."""
        if self._last_trace is None:
            return {"error": "No retrieval trace available yet"}
        return self._last_trace.to_dict(include_context=self._config.context_log_full)

    def handle_compare_traces(self) -> dict[str, Any]:
        """Compare the last two retrieval traces."""
        if self._last_trace is None:
            return {"error": "No retrieval traces available yet"}
        if self._prev_trace is None:
            return {"error": "Only one retrieval so far — need at least two to compare"}
        from chicory.diagnostics.activation_trace import compare_traces
        return compare_traces(self._prev_trace, self._last_trace)

    def _generate_expansion_queries(
        self, results: list[tuple[Memory, float]],
    ) -> list[str]:
        """Generate expansion queries from retrieval results.

        Combines summary/content with tags. Purely extractive — no LLM call.
        """
        queries: list[str] = []
        seen_tag_sets: list[set[str]] = []

        for mem, _score in results:
            text_part = mem.summary if mem.summary else mem.content[:200]
            tag_phrase = " ".join(mem.tags) if mem.tags else ""

            expansion = f"{text_part} {tag_phrase}".strip()
            if not expansion:
                continue

            # Deduplicate by tag set
            tag_set = set(mem.tags)
            if tag_set and any(tag_set == s for s in seen_tag_sets):
                continue
            if tag_set:
                seen_tag_sets.append(tag_set)

            queries.append(expansion)

        return queries

    # ── Librarian handlers ──────────────────────────────────────────

    def handle_librarian_status(self) -> dict[str, Any]:
        """Gate calibration stats, sweep progress, quarantine and subject counts."""
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                gate_rows = self._db.execute(
                    """
                    SELECT verdict, COUNT(*) AS n, AVG(score) AS avg_score,
                           MIN(score) AS min_score
                    FROM write_gate_log GROUP BY verdict
                    """
                ).fetchall()
                quarantine_row = self._db.execute(
                    "SELECT COUNT(*) AS n FROM memories "
                    "WHERE ingestion_tier = 'quarantine' AND is_archived = 0"
                ).fetchone()
                subject_rows = self._db.execute(
                    "SELECT status, COUNT(*) AS n FROM subjects GROUP BY status"
                ).fetchall()
                custody_rows = self._db.execute(
                    "SELECT custody_status, COUNT(*) AS n FROM memories "
                    "WHERE is_archived = 0 GROUP BY custody_status"
                ).fetchall()
                faq_rows = self._db.execute(
                    "SELECT status, COUNT(*) AS n FROM faq_entries GROUP BY status"
                ).fetchall()
                commission_rows = self._db.execute(
                    "SELECT status, COUNT(*) AS n FROM commissions GROUP BY status"
                ).fetchall()
                demand_row = self._db.execute(
                    "SELECT COUNT(*) AS total, "
                    "SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) AS pending "
                    "FROM query_history"
                ).fetchone()
                cluster_row = self._db.execute(
                    "SELECT COUNT(*) AS n FROM query_clusters"
                ).fetchone()
                return {
                    "gate": {
                        "enabled": self._config.librarian_gate_enabled,
                        "enforce": self._config.librarian_gate_enforce,
                        "threshold": self._config.librarian_gate_threshold,
                        "verdicts": {
                            r["verdict"]: {
                                "count": r["n"],
                                "avg_score": round(r["avg_score"], 3) if r["avg_score"] is not None else None,
                                "min_score": round(r["min_score"], 3) if r["min_score"] is not None else None,
                            }
                            for r in gate_rows
                        },
                    },
                    "sweep": self._linear_sweep.progress(),
                    "custody": {r["custody_status"]: r["n"] for r in custody_rows},
                    "quarantined_memories": quarantine_row["n"] if quarantine_row else 0,
                    "subjects": {r["status"]: r["n"] for r in subject_rows},
                    "faq": {r["status"]: r["n"] for r in faq_rows},
                    "commissions": {r["status"]: r["n"] for r in commission_rows},
                    "demand": {
                        "queries_recorded": demand_row["total"] or 0,
                        "queries_pending": demand_row["pending"] or 0,
                        "clusters": cluster_row["n"] or 0,
                    },
                    "penumbra": self._penumbra_ledger.stats(),
                    "ingress": self._ingress.stats(),
                    "wander": self.wander_status(),
                }
        finally:
            self._foreground_waiting.clear()

    def handle_librarian_review(self, limit: int = 20) -> dict[str, Any]:
        """Quarantined memories and open sweep findings awaiting judgment."""
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                q_rows = self._db.execute(
                    """
                    SELECT m.id, m.summary, m.content, m.author, m.origin, m.created_at,
                           g.score AS gate_score, g.verdict AS gate_verdict
                    FROM memories m
                    LEFT JOIN write_gate_log g ON g.memory_id = m.id
                    WHERE m.ingestion_tier = 'quarantine' AND m.is_archived = 0
                    ORDER BY m.created_at DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                quarantined = []
                tags_map = self._tag_manager.get_tags_for_memories(
                    [r["id"] for r in q_rows]
                )
                for r in q_rows:
                    quarantined.append({
                        "memory_id": r["id"],
                        "summary": r["summary"],
                        "content_preview": (r["content"] or "")[:400],
                        "tags": tags_map.get(r["id"], []),
                        "author": r["author"],
                        "origin": r["origin"],
                        "created_at": r["created_at"],
                        "gate_score": r["gate_score"],
                        "gate_verdict": r["gate_verdict"],
                    })

                f_rows = self._db.execute(
                    """
                    SELECT id, memory_id, finding_type, detail, created_at
                    FROM sweep_findings WHERE resolved = 0
                    ORDER BY created_at ASC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                findings = [dict(r) for r in f_rows]
                return {
                    "quarantined": quarantined,
                    "open_findings": findings,
                    "faq_candidates": self._faq_desk.candidates(limit),
                    "open_commissions": self._commission_desk.list("open", limit),
                }
        finally:
            self._foreground_waiting.clear()

    def handle_librarian_resolve(
        self,
        memory_id: str | None = None,
        action: str | None = None,
        finding_id: int | None = None,
    ) -> dict[str, Any]:
        """Apply a judgment: release/archive a quarantined memory, or
        mark a sweep finding resolved.

        Releasing to 'critical' runs the deferred store side effects so the
        memory finally joins the network it was held out of.
        """
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                if finding_id is not None:
                    frow = self._db.execute(
                        "SELECT memory_id FROM sweep_findings WHERE id = ?",
                        (finding_id,),
                    ).fetchone()
                    if frow is None:
                        return {"error": f"Finding {finding_id} not found"}
                    self._db.execute(
                        "UPDATE sweep_findings SET resolved = 1 WHERE id = ?",
                        (finding_id,),
                    )
                    # No open findings left on a flagged memory -> clean
                    remaining = self._db.execute(
                        "SELECT COUNT(*) AS n FROM sweep_findings "
                        "WHERE memory_id = ? AND resolved = 0 "
                        "AND finding_type != 'buried'",
                        (frow["memory_id"],),
                    ).fetchone()
                    if (remaining["n"] or 0) == 0:
                        srow = self._db.execute(
                            "SELECT custody_status FROM memories WHERE id = ?",
                            (frow["memory_id"],),
                        ).fetchone()
                        if srow and srow["custody_status"] == "flagged":
                            record_custody_transition(
                                self._db, frow["memory_id"], "clean",
                                actor="judge",
                                reason=f"finding {finding_id} resolved",
                            )
                    return {"status": "finding_resolved", "finding_id": finding_id}

                if not memory_id or action not in (
                    "release_critical", "release_reference", "archive",
                ):
                    return {
                        "error": "Provide finding_id, or memory_id with action "
                                 "in {release_critical, release_reference, archive}"
                    }

                row = self._db.execute(
                    "SELECT ingestion_tier FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if not row:
                    return {"error": f"Memory {memory_id} not found"}
                if row["ingestion_tier"] != "quarantine":
                    return {"error": f"Memory {memory_id} is not quarantined"}

                if action == "archive":
                    self._db.execute(
                        "UPDATE memories SET is_archived = 1 WHERE id = ?",
                        (memory_id,),
                    )
                    # Archiving settles the memory's open findings — the
                    # sweep never revisits archived memories, so leaving
                    # them open would orphan them in the review queue.
                    self._db.execute(
                        "UPDATE sweep_findings SET resolved = 1 "
                        "WHERE memory_id = ? AND resolved = 0",
                        (memory_id,),
                    )
                    self._embedding_engine._clear_memory_cache()
                    return {"status": "archived", "memory_id": memory_id}

                tier = "critical" if action == "release_critical" else "reference"
                self._db.execute(
                    "UPDATE memories SET ingestion_tier = ? WHERE id = ?",
                    (tier, memory_id),
                )
                # Rebuild retrieval caches so the released memory enters FAISS
                self._embedding_engine._clear_memory_cache()
                if tier == "critical":
                    memory = self._memory_store.get_by_id(memory_id)
                    # Replay side effects with the llm-assigned word tags only —
                    # get_by_id returns ALL tags (temporal, letter, split), and
                    # _on_memory_stored would mint centroids/lattice positions
                    # for bookkeeping tags it never sees on the normal path.
                    word_rows = self._db.execute(
                        "SELECT t.name FROM memory_tags mt "
                        "JOIN tags t ON t.id = mt.tag_id "
                        "WHERE mt.memory_id = ? AND mt.assigned_by = 'llm'",
                        (memory_id,),
                    ).fetchall()
                    memory = memory.model_copy(
                        update={"tags": [r["name"] for r in word_rows]},
                    )
                    self._on_memory_stored(memory)
                record_custody_transition(
                    self._db, memory_id, "certified",
                    actor="judge", reason=action,
                )
                return {"status": "released", "memory_id": memory_id, "tier": tier}
        finally:
            self._foreground_waiting.clear()

    def handle_librarian_maintain(self, cold_rebuild: bool = False) -> dict[str, Any]:
        """Manual maintenance: sweep a batch, rebuild subjects, process demand.

        The roster rebuild is warm by default — it inherits the standing
        tree's shape so shelf paths stay stable as addresses.  Pass
        cold_rebuild to re-derive the tree from the geometry instead,
        which WILL move paths and retire shelves; it is the deliberate
        escape hatch, not routine maintenance.
        """
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                sweep_stats = self._linear_sweep.run_batch()
                roster_stats = self._subject_roster.rebuild(cold=cold_rebuild)
                # Same guard as the background phase: FAQ promotion does a
                # live retrieval, and a FAISS rebuild must never happen
                # while holding the foreground lock.
                demand_stats = self._demand_ledger.process_batch(
                    allow_retrieval=self._index_is_built(),
                )
                return {
                    "sweep": sweep_stats,
                    "subjects": roster_stats,
                    "demand": demand_stats,
                    "roster": self._subject_roster.list_subjects(limit=20),
                }
        finally:
            self._foreground_waiting.clear()

    def _index_is_built(self) -> bool:
        vi = getattr(self._embedding_engine, "_vector_index", None)
        return vi is not None and getattr(vi, "is_built", False)

    def handle_librarian_consult(
        self, query: str, top_k: int = 5,
    ) -> dict[str, Any]:
        """The front desk: FAQ fast path, then subject dispatch, then the
        general shelf — with open research tickets surfaced alongside.

        Response structure: faq (a served card or null), subjects (each
        consulted shelf with its own results), general (whole-library
        hybrid results), research_pending (open commissions near the query).
        """
        # Embed outside the lock (model forward pass is CPU-bound)
        query_vec = self._embedding_engine.embed(query)

        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                response: dict[str, Any] = {
                    "faq": None,
                    "subjects": [],
                    "general": [],
                    "research_pending": self._commission_desk.match_open(query_vec),
                }

                faq_hit = self._faq_desk.serve(query_vec)
                if faq_hit is not None:
                    # A cached answer built on a refuted memory must not be
                    # served as canonical — and the fast path returns before
                    # every other channel, so this is the one place to catch
                    # it.  Stale the card and fall through to the shelves.
                    refuted = self._closure.superseded_among(
                        faq_hit.get("supporting_memory_ids", [])
                    )
                    if refuted:
                        self._faq_desk.retire_to_stale(faq_hit["faq_id"])
                        response["faq_withdrawn"] = {
                            "faq_id": faq_hit["faq_id"],
                            "reason": "rests on superseded memories",
                            "superseded": sorted(refuted),
                        }
                        faq_hit = None
                if faq_hit is not None:
                    response["faq"] = faq_hit
                    # A cached answer must not starve the supporting
                    # memories of retrieval signal — run the normal
                    # bookkeeping (salience, trends, reinforcement, demand)
                    # over them, or FAQ-served shelves decay into burial
                    # while being read daily.
                    supporting = faq_hit["supporting_memory_ids"]
                    hydrated = self._memory_store._get_by_ids(supporting)
                    faq_results = [
                        (hydrated[mid], faq_hit["match_score"])
                        for mid in supporting if mid in hydrated
                    ]
                    if faq_results:
                        retrieval_id = self._retrieval_tracker.log_retrieval(
                            query_text=query,
                            method="faq",
                            results=[
                                (mem.id, rank + 1, score)
                                for rank, (mem, score) in enumerate(faq_results)
                            ],
                            model_version=self._config.llm_model,
                        )
                        self._on_retrieval_completed(
                            retrieval_id, faq_results, query, query_vec=query_vec,
                        )
                    return response

                # Subject dispatch: consult matching shelves individually
                matches = self._subject_roster.match(
                    query_vec,
                    self._config.consult_subject_threshold,
                    self._config.consult_max_subjects,
                    max_depth=self._subject_roster.route_depth(),
                )
                for subject, sim in matches:
                    # Cap member tags per shelf: a giant cluster would
                    # otherwise build an unbounded SQL IN clause under
                    # the foreground lock.
                    tag_names = list(
                        self._tag_manager.get_names_by_ids(
                            subject["member_tag_ids"][:200]
                        ).values()
                    )
                    if not tag_names:
                        continue
                    shelf = self._memory_store.retrieve_semantic(
                        query, top_k, tag_filter=tag_names, query_vec=query_vec,
                    )
                    # Shelf-served memories earn 'retrieved' credit in the
                    # penumbra ledger — otherwise routing's win case (served
                    # on the shelf, runner-up globally) manufactures false
                    # 'buried' findings.
                    if shelf and self._config.penumbra_enabled:
                        try:
                            self._penumbra_ledger.record(
                                [], [m.id for m, _ in shelf],
                            )
                        except Exception:
                            logging.getLogger(__name__).exception(
                                "Penumbra shelf credit failed"
                            )
                    shelf_entries = [
                        {
                            "memory_id": mem.id,
                            "summary": mem.summary,
                            "content_preview": mem.content[:300],
                            "tags": mem.tags,
                            "relevance_score": round(score, 3),
                            "tier": mem.ingestion_tier,
                            "author": mem.author,
                            "custody": mem.custody_status,
                        }
                        for mem, score in shelf
                    ]
                    self._annotate_supersession(shelf_entries)
                    response["subjects"].append({
                        "name": subject["name"],
                        "label": subject.get("label", subject["name"]),
                        "description": subject["description"],
                        "match_score": round(sim, 3),
                        "results": shelf_entries,
                    })

                # General channel: full hybrid retrieval with the normal
                # bookkeeping/side effects (this also records demand).
                general = self._handle_retrieve_memories_locked(
                    query, None, "hybrid", top_k,
                )
                response["general"] = general.get("results", [])
                return response
        finally:
            self._foreground_waiting.clear()

    def handle_librarian_faq(
        self,
        action: str = "list",
        faq_id: int | None = None,
        answer: str | None = None,
        author: str | None = None,
        status: str = "candidate",
        limit: int = 20,
        supporting_memory_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """FAQ desk management: list candidates/entries, approve, retire.

        Approval is the judgment step — a candidate's mechanical draft
        becomes servable, optionally rewritten, and gets a byline.
        """
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                if action == "list":
                    return self._faq_list(status, limit)
                if action == "show":
                    if faq_id is None:
                        return {"error": "show requires faq_id"}
                    return self._faq_show(faq_id)
                if action == "stale":
                    if faq_id is None:
                        return {"error": "stale requires faq_id"}
                    ok = self._faq_desk.retire_to_stale(faq_id)
                    return (
                        {"status": "stale", "faq_id": faq_id} if ok
                        else {"error": f"FAQ {faq_id} not found or retired"}
                    )
                if action == "approve":
                    if faq_id is None:
                        return {"error": "approve requires faq_id"}
                    ok = self._faq_desk.approve(
                        faq_id, answer=answer, author=author,
                        supporting_memory_ids=supporting_memory_ids,
                    )
                    if not ok:
                        return {"error": f"FAQ {faq_id} not found or retired"}
                    return {
                        "status": "approved",
                        "faq_id": faq_id,
                        "regrounded": supporting_memory_ids is not None,
                    }
                if action == "retire":
                    if faq_id is None:
                        return {"error": "retire requires faq_id"}
                    ok = self._faq_desk.retire(faq_id)
                    return (
                        {"status": "retired", "faq_id": faq_id} if ok
                        else {"error": f"FAQ {faq_id} not found"}
                    )
                return {"error": f"Unknown action: {action}"}
        finally:
            self._foreground_waiting.clear()

    _FAQ_STATUSES = ("candidate", "active", "stale", "retired")

    def _faq_list(self, status: str, limit: int) -> dict[str, Any]:
        """The desk view: how many cards sit in each state, then one queue.

        The tally rides along with every listing because the states are a
        pipeline, not independent buckets — a patron looking at an empty
        candidate queue needs to know whether that means "nothing to judge"
        or "everything already went stale".
        """
        counts = {
            r["status"]: r["n"]
            for r in self._db.execute(
                "SELECT status, COUNT(*) AS n FROM faq_entries GROUP BY status"
            ).fetchall()
        }
        rows = self._db.execute(
            "SELECT id, question_text, answer, status, hit_count, author, "
            "supporting_memory_ids, created_at, last_verified_at "
            "FROM faq_entries WHERE status = ? "
            "ORDER BY hit_count DESC, created_at ASC LIMIT ?",
            (status, limit),
        ).fetchall()
        entries = []
        for r in rows:
            try:
                supporting = json.loads(r["supporting_memory_ids"])
            except (json.JSONDecodeError, TypeError, ValueError):
                supporting = []
            entries.append({
                "faq_id": r["id"],
                "question": r["question_text"],
                "status": r["status"],
                "hit_count": r["hit_count"],
                "author": r["author"],
                "supporting_count": len(supporting),
                "created_at": r["created_at"],
                "last_verified_at": r["last_verified_at"],
                # Enough to recognise a card, never enough to judge one.
                # Saying so here is deliberate: the draft is a citation
                # list, and a preview of it reads far more finished than
                # the card actually is.
                "draft_preview": (r["answer"] or "")[:180],
            })
        return {
            "counts": {s: counts.get(s, 0) for s in self._FAQ_STATUSES},
            "listing": status,
            "entries": entries,
            "next": (
                "librarian_faq action=show faq_id=<id> for the full card"
                if entries else f"nothing in state {status!r}"
            ),
        }

    def _faq_show(self, faq_id: int) -> dict[str, Any]:
        """One card, hydrated with what an approver actually has to read.

        `list` hands back supporting memory *ids*, so judging a draft meant
        going and fetching each one yourself.  This returns the card with
        its sources' text, provenance and custody attached, plus the two
        checks that should block an approval outright: a supporting memory
        that has gone (archived or quarantined), and one that a later
        report has superseded.
        """
        r = self._db.execute(
            "SELECT * FROM faq_entries WHERE id = ?", (faq_id,),
        ).fetchone()
        if r is None:
            return {"error": f"FAQ {faq_id} not found"}
        try:
            supporting_ids = json.loads(r["supporting_memory_ids"])
        except (json.JSONDecodeError, TypeError, ValueError):
            supporting_ids = []

        mems = {}
        if supporting_ids:
            ph = ",".join("?" * len(supporting_ids))
            for m in self._db.execute(
                f"SELECT id, summary, content, author, ingestion_tier, "
                f"custody_status, is_archived, salience_composite, source_path "
                f"FROM memories WHERE id IN ({ph})", tuple(supporting_ids),
            ).fetchall():
                mems[m["id"]] = m
        refuted = self._closure.superseded_among(supporting_ids)
        tags_map = self._tag_manager.get_tags_for_memories(list(mems))

        supporting, missing = [], []
        for mid in supporting_ids:
            m = mems.get(mid)
            if m is None:
                missing.append(mid)
                continue
            gone = bool(m["is_archived"]) or m["ingestion_tier"] == "quarantine"
            supporting.append({
                "memory_id": mid,
                "summary": m["summary"],
                "content": m["content"],
                "source_path": m["source_path"],
                "tags": tags_map.get(mid, []),
                "author": m["author"],
                "tier": m["ingestion_tier"],
                "custody": m["custody_status"],
                "salience": round(m["salience_composite"] or 0.0, 3),
                "archived": bool(m["is_archived"]),
                "superseded": mid in refuted,
                "usable": not gone and mid not in refuted,
            })

        try:
            subject_ids = json.loads(r["subject_ids"])
        except (json.JSONDecodeError, TypeError, ValueError):
            subject_ids = []
        subjects = []
        if subject_ids:
            ph = ",".join("?" * len(subject_ids))
            found = {
                row["id"]: row for row in self._db.execute(
                    f"SELECT id, name, display_name, status, path FROM subjects "
                    f"WHERE id IN ({ph})", tuple(subject_ids),
                ).fetchall()
            }
            for sid in subject_ids:
                row = found.get(sid)
                subjects.append({
                    "subject_id": sid,
                    # Shelves are re-derived on every roster rebuild, so an
                    # id captured at promotion time can name a retired node
                    # or nothing at all.  Say which, rather than printing a
                    # number that reads as authoritative.
                    "label": (
                        (row["display_name"] or row["name"]) if row else None
                    ),
                    "status": row["status"] if row else "gone",
                })

        blockers = []
        if missing:
            blockers.append(f"{len(missing)} supporting memories no longer exist")
        n_gone = sum(
            1 for x in supporting if x["archived"] or x["tier"] == "quarantine"
        )
        if n_gone:
            blockers.append(f"{n_gone} archived or quarantined")
        if refuted:
            blockers.append(f"{len(refuted)} superseded by a later report")

        return {
            "faq_id": r["id"],
            "question": r["question_text"],
            "status": r["status"],
            "author": r["author"],
            "hit_count": r["hit_count"],
            "created_at": r["created_at"],
            "last_verified_at": r["last_verified_at"],
            "draft_answer": r["answer"],
            "mechanical": r["author"] == "librarian:mechanical",
            "subjects": subjects,
            "supporting": supporting,
            "supporting_missing": missing,
            "usable_sources": sum(1 for x in supporting if x["usable"]),
            "total_sources": len(supporting_ids),
            "blockers": blockers,
            "verdict_hint": (
                "blocked — resolve the blockers before approving"
                if blockers else
                "approve with a rewritten answer, or retire if unsupported"
            ),
        }

    def handle_librarian_commissions(
        self,
        action: str = "list",
        commission_id: int | None = None,
        question: str | None = None,
        report_memory_id: str | None = None,
        status: str = "open",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Commission queue: list, file, claim, report, close.

        Specialists store their report as a normal memory (with author and
        origin set), then attach it via report before closing.
        """
        question_vec = None
        if action == "file" and question:
            question_vec = self._embedding_engine.embed(question)

        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                if action == "list":
                    return {"commissions": self._commission_desk.list(status, limit)}
                if action == "file":
                    if not question:
                        return {"error": "file requires question"}
                    subject_ids = [
                        s["id"] for s, _ in self._subject_roster.match(
                            question_vec,
                            self._config.consult_subject_threshold,
                            self._config.consult_max_subjects,
                        )
                    ] if question_vec is not None else []
                    return self._commission_desk.file(
                        question, question_vec,
                        source="manual", subject_ids=subject_ids,
                    )
                if action in ("claim", "report", "close"):
                    if commission_id is None:
                        return {"error": f"{action} requires commission_id"}
                    target = {
                        "claim": "claimed", "report": "reported", "close": "closed",
                    }[action]
                    if action == "report":
                        if not report_memory_id:
                            return {"error": "report requires report_memory_id"}
                        # A report must be a real, live, unquarantined memory
                        rrow = self._db.execute(
                            "SELECT is_archived, ingestion_tier FROM memories "
                            "WHERE id = ?",
                            (report_memory_id,),
                        ).fetchone()
                        if rrow is None:
                            return {"error": f"Report memory {report_memory_id} not found"}
                        if rrow["is_archived"] or rrow["ingestion_tier"] == "quarantine":
                            return {
                                "error": "Report memory is archived or quarantined. "
                                         "A patron must review and release it before "
                                         "it can be attached; leave the ticket "
                                         "claimed and note it for review."
                            }
                    ok = self._commission_desk.set_status(
                        commission_id, target, report_memory_id,
                    )
                    return (
                        {"status": target, "commission_id": commission_id} if ok
                        else {"error": f"Commission {commission_id} not found"}
                    )
                return {"error": f"Unknown action: {action}"}
        finally:
            self._foreground_waiting.clear()

    def handle_librarian_desk(
        self, action: str = "report", limit: int = 5,
    ) -> dict[str, Any]:
        """The standing desk report: what changed, what needs you, what
        was learned, and the state of the building.

        Reading does NOT acknowledge — call action='ack' when you have
        actually dealt with what the report showed, and the next report's
        "since" window starts from there.
        """
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                if action == "report":
                    return self._desk.report(limit)
                if action == "ack":
                    return self._desk.acknowledge()
                return {"error": f"Unknown action: {action}"}
        finally:
            self._foreground_waiting.clear()

    def handle_librarian_subjects(
        self,
        action: str = "tree",
        path: str | None = None,
        name: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Address the subject hierarchy.

        tree  — the whole active roster as an outline (path, depth, size).
        show  — one node, addressed loosely: an exact path, a bare name, a
                given name, or just a topic ("gnosticism") resolves to the
                most specific shelf holding it.  Ambiguity comes back as
                candidates rather than an error.
        children — the direct children of a node.
        rename — give a shelf a human name (authored; survives rebuilds).

        Routing (librarian_consult) deliberately stays at the coarse
        depth; teaching addresses deeper nodes by path.
        """
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                if action == "tree":
                    nodes = self._subject_roster.tree(limit)
                    return {
                        "route_depth": self._subject_roster.route_depth(),
                        "max_depth": self._config.subject_max_depth,
                        "nodes": [
                            {
                                "path": n["path"],
                                "depth": n["depth"],
                                "label": n["label"],
                                "name": n["name"],
                                "named": n["display_name"] is not None,
                                "memory_count": n["memory_count"],
                                "tags": len(n["member_tag_ids"]),
                            }
                            for n in nodes
                        ],
                    }

                if action == "rename":
                    if not path:
                        return {"error": "rename requires path"}
                    node = self._subject_roster.rename(path, name)
                    if node is None:
                        return self._subject_miss(path)
                    return {
                        "status": "renamed",
                        "path": node["path"],
                        "label": node["label"],
                        "derived_name": node["name"],
                    }

                if action in ("show", "children"):
                    if not path:
                        return {"error": f"{action} requires path"}
                    resolution = self._subject_roster.resolve(path)
                    node = resolution.get("match")
                    if node is None:
                        return self._subject_miss(path, resolution)
                    children = self._subject_roster.children_of(node["path"])
                    if action == "children":
                        return {
                            "path": node["path"],
                            "label": node["label"],
                            "children": [
                                {"path": c["path"], "label": c["label"],
                                 "depth": c["depth"],
                                 "memory_count": c["memory_count"]}
                                for c in children
                            ],
                        }
                    tag_names = list(
                        self._tag_manager.get_names_by_ids(
                            node["member_tag_ids"][:300]
                        ).values()
                    )
                    return {
                        "path": node["path"],
                        "label": node["label"],
                        "name": node["name"],
                        "named": node["display_name"] is not None,
                        "resolved_by": resolution.get("how", "exact"),
                        "depth": node["depth"],
                        "description": node["description"],
                        "memory_count": node["memory_count"],
                        "tags": tag_names,
                        "children": [
                            {"path": c["path"], "label": c["label"],
                             "memory_count": c["memory_count"]}
                            for c in children
                        ],
                    }

                return {"error": f"Unknown action: {action}"}
        finally:
            self._foreground_waiting.clear()

    def _annotate_supersession(self, entries: list[dict[str, Any]]) -> None:
        """Mark any result whose claims have since been overturned.

        A refuted memory keeps its byline and its place; it must simply
        never arrive looking as authoritative as the thing that corrected
        it.  Silent equality is how a library serves its own mistakes.
        """
        if not entries:
            return
        try:
            marks = self._closure.for_memories(
                [e["memory_id"] for e in entries]
            )
        except Exception:
            logging.getLogger(__name__).exception("Supersession lookup failed")
            return
        for entry in entries:
            found = marks.get(entry["memory_id"])
            if found:
                entry["superseded_by"] = found

    def handle_librarian_closure(
        self,
        action: str = "list",
        memory_id: str | None = None,
        superseding_id: str | None = None,
        report_memory_id: str | None = None,
        commission_id: int | None = None,
        claim: str | None = None,
        all_claims: bool = False,
        reason: str | None = None,
        scope: str = "claim",
        actor: str | None = None,
        supersedes: list[dict] | None = None,
        resolve_findings: list[int] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Specialist closure — settle what a piece of work overturned.

        supersede — record that one memory overturns a claim in another.
                    Nothing is deleted; the older memory keeps its byline
                    and afterwards arrives marked.
        close     — the full reverse-order unwind for a finished report:
                    supersede what it refuted, stale dependent FAQ answers,
                    resolve findings it explains, close the ticket.
        list      — recent supersessions.
        retract   — withdraw one (the correction was itself wrong).
        """
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                who = actor or "librarian"
                if action == "supersede":
                    if not (memory_id and superseding_id):
                        return {"error": "supersede requires memory_id and "
                                         "superseding_id"}
                    return self._closure.supersede(
                        superseded_id=memory_id, superseding_id=superseding_id,
                        actor=who, claim=claim, reason=reason, scope=scope,
                    )
                if action == "close":
                    if not report_memory_id:
                        return {"error": "close requires report_memory_id"}
                    return self._closure.close_chain(
                        report_memory_id=report_memory_id, actor=who,
                        commission_id=commission_id, supersedes=supersedes,
                        resolve_findings=resolve_findings,
                    )
                if action == "retract":
                    if not (memory_id and superseding_id):
                        return {"error": "retract requires memory_id and "
                                         "superseding_id"}
                    # Forward the claim. Without it retract() falls back to
                    # matching claim = '', so it could only ever withdraw a
                    # whole-scope supersession — every claim-scoped one, which
                    # is the documented default, was unretractable while the
                    # schema advertised the argument.
                    ok = self._closure.retract(
                        memory_id, superseding_id,
                        claim=claim, all_claims=all_claims,
                    )
                    return ({"status": "retracted"} if ok
                            else {"error": "no such standing supersession"})
                if action == "list":
                    return {"supersessions": self._closure.recent(limit=limit)}
                return {"error": f"Unknown action: {action}"}
        finally:
            self._foreground_waiting.clear()

    def _subject_miss(self, query: str, resolution: dict | None = None) -> dict:
        """Turn a failed lookup into a useful answer, not a dead end."""
        res = resolution if resolution is not None else self._subject_roster.resolve(query)
        if res.get("ambiguous"):
            return {
                "error": f"'{query}' matches several shelves — name one",
                "matched_by": res.get("how"),
                "candidates": [
                    {"path": n["path"], "label": n["label"],
                     "memory_count": n["memory_count"]}
                    for n in res["ambiguous"]
                ],
            }
        return {
            "error": f"No shelf matches '{query}'",
            "did_you_mean": [
                {"path": n["path"], "label": n["label"],
                 "memory_count": n["memory_count"]}
                for n in res.get("suggestions", [])
            ],
        }

    def handle_librarian_ingress(
        self,
        action: str = "review",
        content: str | None = None,
        receipt: str | None = None,
        source: str = "user",
        origin: str | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        summary: str | None = None,
        status: str = "held",
        limit: int = 25,
    ) -> dict[str, Any]:
        """Receipted ingress dwell: hold, examine, settle, commit.

        hold    — take content into custody, issue a receipt, and examine
                  the whole window (so this arrival is compared against
                  everything still dwelling).  Commits nothing.
        examine — re-run detection over held content (all, or one receipt).
        review  — list receipts by status with their findings.
        settle  — close one dwell (or every elapsed dwell) with a verdict.
        commit  — promote a settled receipt into a memory (needs tags).
        discard — settle without shelving; the receipt and findings remain.
        stats   — buffer counts by status and finding type.
        """
        if not self._config.ingress_enabled and action in ("hold", "examine"):
            return {"error": "Ingress dwell is disabled (ingress_enabled=False)"}

        # Embed outside the lock — CPU-bound model forward pass
        embedding = None
        if action == "hold" and content:
            try:
                embedding = self._embedding_engine.embed(content)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Ingress embedding failed; holding without a vector "
                    "(cross-message comparison degrades to lexical only)"
                )

        commit_payload = None
        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                if action == "hold":
                    if not content:
                        return {"error": "hold requires content"}
                    record = self._ingress.hold(
                        content, embedding=embedding, source=source,
                        origin=origin, author=author,
                    )
                    # Examine the whole window: the new arrival and every
                    # message still dwelling are compared against each other.
                    findings = self._ingress.examine()
                    record["findings"] = [
                        f for f in findings
                        if record["receipt"] in (f["receipt"], f.get("peer_receipt"))
                    ]
                    record["window_findings"] = len(findings)
                    return record

                if action == "examine":
                    return {"findings": self._ingress.examine(receipt)}

                if action == "review":
                    return {
                        "status": status,
                        "receipts": self._ingress.list_receipts(status, limit),
                    }

                if action == "settle":
                    if receipt:
                        return self._ingress.settle(receipt)
                    return self._ingress.settle_due()

                if action == "discard":
                    if not receipt:
                        return {"error": "discard requires receipt"}
                    ok = self._ingress.mark_discarded(receipt)
                    return (
                        {"status": "discarded", "receipt": receipt} if ok
                        else {"error": f"Unknown receipt: {receipt}"}
                    )

                if action == "stats":
                    return self._ingress.stats()

                if action == "commit":
                    if not receipt:
                        return {"error": "commit requires receipt"}
                    record = self._ingress.get(receipt)
                    if record is None:
                        return {"error": f"Unknown receipt: {receipt}"}
                    if record["status"] == "held":
                        return {
                            "error": "Receipt is still dwelling — settle it "
                                     "first so detection has run",
                            "dwell_until": record["dwell_until"],
                        }
                    if record["status"] == "committed":
                        return {
                            "error": "Already committed",
                            "memory_id": record["memory_id"],
                        }
                    if record["status"] == "discarded":
                        return {
                            "error": "Receipt was discarded — committing it "
                                     "would silently reverse that decision",
                        }
                    if not tags:
                        return {"error": "commit requires tags"}
                    commit_payload = record
                else:
                    return {"error": f"Unknown action: {action}"}
        finally:
            self._foreground_waiting.clear()

        # Store outside the lock block — handle_store_memory takes the lock
        # itself (and runs the write gate on the content).
        result = self.handle_store_memory(
            content=commit_payload["content"],
            tags=tags or [],
            importance=importance,
            summary=summary,
            author=commit_payload.get("author"),
            origin=commit_payload.get("origin") or f"ingress:{receipt}",
        )
        memory_id = result.get("memory_id")
        if memory_id:
            self._foreground_waiting.set()
            try:
                with self._foreground_db_lock:
                    self._ingress.mark_committed(receipt, memory_id)
            finally:
                self._foreground_waiting.clear()
        return {
            "status": "committed",
            "receipt": receipt,
            "memory_id": memory_id,
            "store_status": result.get("status"),
            "gate": result.get("gate"),
            "findings_at_commit": commit_payload["findings"],
        }

    def handle_librarian_recall(
        self, query: str, top_k: int = 3,
    ) -> dict[str, Any]:
        """Analogic recall: cross-domain counterparts by structural role.

        Resolves the query's home subject, computes its offset from the
        home centroid, and searches other shelves at the transported
        position (centroid_B + offset).  Results are CONJECTURES — the
        same role in a different domain, not established associations —
        and every entry says so.
        """
        query_vec = self._embedding_engine.embed(query)

        self._foreground_waiting.set()
        try:
            with self._foreground_db_lock:
                home_matches = self._subject_roster.match(
                    query_vec,
                    self._config.consult_subject_threshold, 1,
                    max_depth=self._subject_roster.route_depth(),
                )
                if not home_matches:
                    return {
                        "note": "No home subject matched this query — "
                                "analogic recall needs a resolvable domain. "
                                "Run librarian_maintain to rebuild the roster.",
                        "counterparts": [],
                    }
                home, home_sim = home_matches[0]
                subjects = self._subject_roster.active_with_centroids(
                    max_depth=self._subject_roster.route_depth(),
                )
                home_centroid = next(
                    s["centroid"] for s in subjects if s["id"] == home["id"]
                )

                counterparts = []
                others = [s for s in subjects if s["id"] != home["id"]]
                for other in others[: self._config.recall_max_subjects]:
                    target = analogic_target(
                        home_centroid, query_vec, other["centroid"],
                    )
                    tag_names = list(
                        self._tag_manager.get_names_by_ids(
                            other["member_tag_ids"][:200]
                        ).values()
                    )
                    if not tag_names:
                        continue
                    shelf = self._memory_store.retrieve_semantic(
                        query, top_k, tag_filter=tag_names, query_vec=target,
                    )
                    if not shelf:
                        continue
                    mem_ids = [m.id for m, _ in shelf]
                    ph = ",".join("?" * len(mem_ids))
                    emb_rows = self._db.execute(
                        f"SELECT memory_id, embedding FROM embeddings "
                        f"WHERE memory_id IN ({ph}) AND chunk_index = 0",
                        tuple(mem_ids),
                    ).fetchall()
                    emb_map = {
                        r["memory_id"]: np.frombuffer(
                            r["embedding"], dtype=np.float32
                        ).copy()
                        for r in emb_rows
                    }
                    results = []
                    for mem, score in shelf:
                        entry = {
                            "memory_id": mem.id,
                            "summary": mem.summary,
                            "content_preview": mem.content[:300],
                            "tags": mem.tags,
                            "target_similarity": round(score, 3),
                            "tier": mem.ingestion_tier,
                            "author": mem.author,
                            "custody": mem.custody_status,
                            "register": "analogy",
                        }
                        if mem.id in emb_map:
                            entry["parallelness"] = round(parallelness(
                                emb_map[mem.id], other["centroid"],
                                home_centroid, query_vec,
                            ), 3)
                        results.append(entry)
                    self._annotate_supersession(results)
                    counterparts.append({
                        "subject": other["name"],
                        "counterpart_of": home["name"],
                        "results": results,
                    })

                return {
                    "home_subject": home["name"],
                    "home_match_score": round(home_sim, 3),
                    "register": "analogy",
                    "note": "Counterparts by structural role — conjectures, "
                            "not established associations. Verify against "
                            "the shelves before treating as evidence.",
                    "counterparts": counterparts,
                }
        finally:
            self._foreground_waiting.clear()

    def handle_get_trends(
        self, tag_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Get current trend data."""
        if tag_names:
            trends = {}
            for name in tag_names:
                tag = self._tag_manager.get_by_name(name)
                if tag:
                    trends[tag.id] = self._trend_engine.compute_trend(tag.id)
        else:
            trends = self._trend_engine.compute_all_trends()

        result = []
        for tag_id, tv in trends.items():
            tag = self._tag_manager.get_by_id(tag_id)
            result.append({
                "tag": tag.name,
                "temperature": round(tv.temperature, 3),
                "level": round(tv.level, 3),
                "velocity": round(tv.velocity, 3),
                "jerk": round(tv.jerk, 3),
                "event_count": tv.event_count,
            })

        result.sort(key=lambda x: x["temperature"], reverse=True)
        return {"trends": result}

    def handle_get_phase_space(self) -> dict[str, Any]:
        """Get full phase space state."""
        populations = self._phase_space.get_quadrant_populations()

        quadrants = {}
        for quadrant, coords in populations.items():
            quadrants[quadrant.value] = [
                {
                    "tag": c.tag_name,
                    "temperature": round(c.temperature, 3),
                    "retrieval_freq": round(c.retrieval_freq, 3),
                    "off_diagonal": round(c.off_diagonal_distance, 3),
                }
                for c in coords
            ]

        return {"phase_space": quadrants}

    def handle_get_synchronicities(
        self, limit: int = 10, unacknowledged_only: bool = False
    ) -> dict[str, Any]:
        """Get recent synchronicity events with effective strength and velocity."""
        events = self._sync_detector.get_recent(limit, unacknowledged_only)
        velocity_data = self._compute_sync_velocity()

        return {
            "synchronicities": [
                {
                    "id": e.id,
                    "type": e.event_type,
                    "description": e.description,
                    "strength": round(e.strength, 2),
                    "effective_strength": round(
                        self._sync_detector.effective_strength(e), 3
                    ),
                    "quadrant": e.quadrant,
                    "detected_at": e.detected_at.isoformat() if e.detected_at else None,
                    "last_reinforced": e.last_reinforced.isoformat() if e.last_reinforced else None,
                    "reinforcement_count": e.reinforcement_count,
                    "acknowledged": e.acknowledged,
                }
                for e in events
            ],
            "count": len(events),
            "velocity": velocity_data,
        }

    def handle_get_meta_patterns(self) -> dict[str, Any]:
        """Get active meta-patterns."""
        patterns = self._meta_analyzer.get_active_patterns()
        return {
            "meta_patterns": [
                {
                    "id": p.id,
                    "type": p.pattern_type,
                    "description": p.description,
                    "confidence": round(p.confidence, 2),
                    "detected_at": p.detected_at.isoformat() if p.detected_at else None,
                    "actions_taken": p.actions_taken,
                }
                for p in patterns
            ],
            "count": len(patterns),
        }

    def handle_get_lattice_resonances(self, max_items: int = 10) -> dict[str, Any]:
        """Get lattice state including positions, resonances, and void profile."""
        return self._sync_engine.get_lattice_state(
            max_positions=max_items,
            max_resonances=max_items,
            max_glyphs=max_items * 2,
        )

    def handle_ingest_codebase(
        self,
        path: str,
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ingest a codebase directory, storing structural summaries as memories.

        Walks the directory, extracts code structure (classes, functions,
        signatures, imports) via AST/regex parsing, and stores compact
        summaries. These summaries can later be retrieved via
        retrieve_memories instead of re-reading the full codebase.

        The returned stats include per-file listings: ``summaries`` (paths
        ingested), ``skipped`` (files no summary could be extracted from)
        and ``already_ingested`` (files whose summary hash is already
        stored), the skip lists being ``{"path": ..., "reason": ...}``.
        """
        import hashlib
        import fnmatch
        from pathlib import Path

        from chicory.ingest.code_summarizer import CODE_EXTENSIONS, summarize_file

        directory = Path(path).resolve()
        if not directory.is_dir():
            return {"error": f"Not a directory: {path}"}

        # Determine which extensions to scan
        if file_patterns:
            allowed_suffixes = None  # Use glob matching instead
        else:
            allowed_suffixes = CODE_EXTENSIONS

        # Load .chicoryignore + defaults + caller exclusions
        from chicory.ingest.ignore import is_ignored, load_ignore_patterns
        patterns = load_ignore_patterns(directory)
        if exclude_patterns:
            patterns.update(exclude_patterns)

        # Collect files
        all_files: list[Path] = []
        for f in directory.rglob("*"):
            if not f.is_file():
                continue
            if is_ignored(f, directory, patterns):
                continue
            # Check extension / pattern match
            if file_patterns:
                rel_str = str(f.relative_to(directory))
                if not any(fnmatch.fnmatch(rel_str, pat) for pat in file_patterns):
                    continue
            elif allowed_suffixes and f.suffix.lower() not in allowed_suffixes:
                continue
            all_files.append(f)

        all_files.sort()

        stats = {
            "files_scanned": len(all_files),
            "files_summarized": 0,
            "memories_created": 0,
            "files_skipped": 0,
            "files_already_ingested": 0,
            "summaries": [],
            "skipped": [],
            "already_ingested": [],
        }

        # ── Phase 1: Store summaries (no embeddings) ──────────────
        new_memory_ids: list[str] = []

        for filepath in all_files:
            rel_path = str(filepath.relative_to(directory)).replace("\\", "/")

            summary_text = summarize_file(filepath, base_dir=directory)
            if not summary_text:
                stats["files_skipped"] += 1
                stats["skipped"].append({
                    "path": rel_path,
                    "reason": "no_summary_extracted",
                })
                continue

            # Dedup by content hash
            content_hash = hashlib.sha256(
                summary_text.encode("utf-8")
            ).hexdigest()[:16]

            existing = self._db.execute(
                "SELECT id FROM memories WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing:
                stats["files_already_ingested"] += 1
                stats["already_ingested"].append({
                    "path": rel_path,
                    "reason": "duplicate_content_hash",
                })
                continue

            # Derive tags from file path
            tags = self._derive_code_tags(filepath, directory)

            # Store with hash for dedup — skip embedding in this phase
            content = summary_text + f"\n\n<!-- chicory:code_hash={content_hash} -->"
            short_summary = rel_path

            result = self.handle_store_memory(
                content=content,
                tags=tags,
                importance=0.6,
                summary=short_summary,
                skip_embedding=True,
                content_hash=content_hash,
            )

            new_memory_ids.append(result["memory_id"])
            stats["files_summarized"] += 1
            stats["memories_created"] += 1
            stats["summaries"].append(rel_path)

        # ── Phase 1b: Generate project overview ─────────────────────
        # Build a high-level summary so broad queries like "what does
        # this project do?" have something to match against.
        overview_id = self._generate_project_overview(
            directory, all_files, stats,
        )
        if overview_id:
            new_memory_ids.append(overview_id)
            stats["memories_created"] += 1

        # ── Phase 2: Batch embed all new memories ─────────────────
        if new_memory_ids:
            self._batch_embed_memories(new_memory_ids)

        stats["embeddings_created"] = len(new_memory_ids)

        # ── Phase 3: Update tensor networks and centroids ─────────
        if new_memory_ids:
            # Backfill centroids that were skipped during Phase 1
            # (skip_embedding=True meant no embedding was available)
            if self._config.centroid_inhibition_enabled:
                placeholders = ",".join("?" * len(new_memory_ids))
                tag_rows = self._db.execute(
                    f"SELECT mt.tag_id, mt.memory_id FROM memory_tags mt "
                    f"WHERE mt.memory_id IN ({placeholders})",
                    tuple(new_memory_ids),
                ).fetchall()

                pairs: list[tuple[int, "np.ndarray"]] = []
                embedding_cache: dict[str, "np.ndarray"] = {}
                for r in tag_rows:
                    mid = r["memory_id"]
                    if mid not in embedding_cache:
                        vec = self._embedding_engine.get_cached(mid)
                        embedding_cache[mid] = vec
                    vec = embedding_cache[mid]
                    if vec is not None:
                        pairs.append((r["tag_id"], vec))

                if pairs:
                    self._centroid_subgraph.update_centroids_batch(pairs)

            # Update co-occurrence, semantic, and semiotic tensor networks
            self._sync_engine.seed_tensor_from_associations()
            self._tensor_cache.invalidate()

        return stats

    def handle_ingest_documents(
        self,
        path: str,
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        critical: bool | None = None,
    ) -> dict[str, Any]:
        """Ingest documents with auto-classified two-tier depth.

        When critical is None (default), each file is auto-classified:
        - **critical**: full content stored, chunked, embedded, and deeply
          integrated into the tensor/lattice/centroid/canopy network.
        - **reference**: extractive summary stored with source_path for
          later reading. Tags assigned and summary embedded, but no deep
          network integration.

        Set critical=True/False to force all files into one tier.

        The returned stats include per-file listings: ``ingested``
        (path + tier), ``skipped`` (unreadable/empty files) and
        ``already_ingested`` (files whose content hash is already stored),
        each entry being ``{"path": ..., "reason": ...}`` for the skip lists.
        """
        import hashlib
        import fnmatch
        from pathlib import Path

        from chicory.ingest.document_ingestor import (
            read_document, chunk_document,
            derive_document_tags,
        )
        from chicory.ingest.criticality import classify_documents
        from chicory.ingest.summarizer import summarize_document

        target = Path(path).resolve()

        # Single file mode
        if target.is_file():
            files = [target]
            base_dir = target.parent
        elif target.is_dir():
            base_dir = target
            from chicory.ingest.ignore import is_ignored, load_ignore_patterns
            patterns = load_ignore_patterns(target)
            if exclude_patterns:
                patterns.update(exclude_patterns)

            files = []
            for f in target.rglob("*"):
                if not f.is_file():
                    continue
                if is_ignored(f, target, patterns):
                    continue
                if file_patterns:
                    rel_str = str(f.relative_to(target))
                    if not any(fnmatch.fnmatch(rel_str, pat) for pat in file_patterns):
                        continue
                files.append(f)
            files.sort()
        else:
            return {"error": f"Path not found: {path}"}

        stats: dict[str, Any] = {
            "files_scanned": len(files),
            "files_ingested": 0,
            "chunks_created": 0,
            "files_skipped": 0,
            "files_already_ingested": 0,
            "critical_count": 0,
            "reference_count": 0,
            "ingested": [],
            "skipped": [],
            "already_ingested": [],
        }

        # Phase 0: Read all files and derive tags for classification
        file_data: list[tuple] = []  # (filepath, text, content_hash, rel_path, tags)
        file_tags_map: dict[str, list[str]] = {}

        for filepath in files:
            rel_path = str(filepath.relative_to(base_dir)).replace("\\", "/")

            text = read_document(filepath)
            if not text:
                stats["files_skipped"] += 1
                stats["skipped"].append({
                    "path": rel_path,
                    "reason": "unreadable_or_empty",
                })
                continue

            content_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()[:16]

            existing = self._db.execute(
                "SELECT id FROM memories WHERE content_hash = ? OR content_hash LIKE ?",
                (content_hash, f"{content_hash}:chunk:%"),
            ).fetchone()
            if existing:
                stats["files_already_ingested"] += 1
                stats["already_ingested"].append({
                    "path": rel_path,
                    "reason": "duplicate_content_hash",
                })
                continue

            tags = derive_document_tags(filepath, base_dir)

            file_data.append((filepath, text, content_hash, rel_path, tags))
            file_tags_map[str(filepath)] = tags

        if not file_data:
            return stats

        # Phase 0.5: Auto-classify criticality
        if critical is None:
            classifications = classify_documents(
                file_tags_map,
                self._db,
                threshold=self._config.ingestion_criticality_threshold,
            )
        else:
            tier = "critical" if critical else "reference"
            classifications = {fp: tier for fp in file_tags_map}

        critical_memory_ids: list[str] = []
        reference_memory_ids: list[str] = []

        for filepath, text, content_hash, rel_path, tags in file_data:
            tier = classifications.get(str(filepath), "reference")
            abs_path = str(filepath.resolve()).replace("\\", "/")

            if tier == "critical":
                # Phase 1a: Store full content, chunked
                chunks = chunk_document(text, rel_path)
                for chunk in chunks:
                    chunk_content = chunk["content"]
                    if chunk["total_chunks"] == 1:
                        chunk_hash = content_hash
                        chunk_content += f"\n\n<!-- chicory:doc_hash={content_hash} -->"
                    else:
                        chunk_hash = f"{content_hash}:chunk:{chunk['chunk_index']}"
                        chunk_content += (
                            f"\n\n<!-- chicory:doc_hash={content_hash}"
                            f":chunk={chunk['chunk_index']} -->"
                        )

                    summary = rel_path
                    if chunk["total_chunks"] > 1:
                        summary = f"{rel_path} (part {chunk['chunk_index'] + 1}/{chunk['total_chunks']})"

                    result = self.handle_store_memory(
                        content=chunk_content,
                        tags=tags,
                        importance=self._config.ingestion_critical_importance,
                        summary=summary,
                        skip_embedding=True,
                        defer_side_effects=True,
                        source_path=abs_path,
                        ingestion_tier="critical",
                        content_hash=chunk_hash,
                    )
                    critical_memory_ids.append(result["memory_id"])
                    stats["chunks_created"] += 1

                stats["critical_count"] += 1

            else:
                # Phase 1b: Store extractive summary only
                summary_text = summarize_document(
                    text, rel_path,
                    max_chars=self._config.ingestion_reference_summary_max_chars,
                )
                summary_text += f"\n\n<!-- chicory:doc_hash={content_hash} -->"

                result = self.handle_store_memory(
                    content=summary_text,
                    tags=tags + ["reference-document"],
                    importance=self._config.ingestion_reference_importance,
                    summary=f"{rel_path} (reference summary)",
                    skip_embedding=True,
                    defer_side_effects=True,
                    source_path=abs_path,
                    ingestion_tier="reference",
                    content_hash=content_hash,
                )
                reference_memory_ids.append(result["memory_id"])
                stats["reference_count"] += 1

            stats["files_ingested"] += 1
            stats["ingested"].append({"path": rel_path, "tier": tier})

        # Phase 2: Batch embed all memories (critical chunks + reference summaries)
        all_memory_ids = critical_memory_ids + reference_memory_ids
        if all_memory_ids:
            self._batch_embed_memories(all_memory_ids)

        # Phase 3: Deep integration for critical memories only
        if critical_memory_ids:
            self._finalize_ingested_memories(critical_memory_ids)

        stats["embeddings_created"] = len(all_memory_ids)
        return stats

    def _batch_embed_memories(self, memory_ids: list[str]) -> None:
        """Batch-compute and store embeddings for memories that have none."""
        from chicory.ingest.chunker import chunk_text_for_embedding

        # Load content for all memories
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT id, content FROM memories WHERE id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()

        if not rows:
            return

        # Collect all texts to embed (one per memory for short content,
        # multiple chunks for long content)
        embed_tasks: list[tuple[str, list[str]]] = []  # (memory_id, chunks)
        all_texts: list[str] = []

        for row in rows:
            chunks = chunk_text_for_embedding(row["content"])
            if not chunks:
                continue
            embed_tasks.append((row["id"], chunks))
            all_texts.extend(chunks)

        if not all_texts:
            return

        # Single batch embed call — one model load, one forward pass
        all_vecs = self._embedding_engine.embed_batch(all_texts)

        # Distribute vectors back to their memories (single cache invalidation)
        with self._embedding_engine.batch_write():
            vec_idx = 0
            for memory_id, chunks in embed_tasks:
                n_chunks = len(chunks)
                vecs = [all_vecs[vec_idx + i] for i in range(n_chunks)]
                vec_idx += n_chunks

                if n_chunks == 1:
                    self._embedding_engine.store_cached(memory_id, vecs[0])
                else:
                    self._embedding_engine.store_chunks(memory_id, vecs)

    def _finalize_ingested_memories(self, memory_ids: list[str]) -> None:
        """Batch side effects for memories stored with defer_side_effects=True.

        Runs centroids, glyph lattice, and tensor seeding once for the
        whole batch instead of per-memory.
        """
        if not memory_ids:
            return

        placeholders = ",".join("?" * len(memory_ids))

        # Centroid backfill
        if self._config.centroid_inhibition_enabled:
            tag_rows = self._db.execute(
                f"SELECT mt.tag_id, mt.memory_id FROM memory_tags mt "
                f"WHERE mt.memory_id IN ({placeholders})",
                tuple(memory_ids),
            ).fetchall()

            pairs: list[tuple[int, Any]] = []
            embedding_cache: dict[str, Any] = {}
            for r in tag_rows:
                mid = r["memory_id"]
                if mid not in embedding_cache:
                    embedding_cache[mid] = self._embedding_engine.get_cached(mid)
                vec = embedding_cache[mid]
                if vec is not None:
                    pairs.append((r["tag_id"], vec))
            if pairs:
                self._centroid_subgraph.update_centroids_batch(pairs)

        # Glyph lattice placement — once per unique tag set
        all_tags: set[str] = set()
        tag_rows = self._db.execute(
            f"SELECT DISTINCT t.name FROM tags t "
            f"JOIN memory_tags mt ON mt.tag_id = t.id "
            f"WHERE mt.memory_id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        for r in tag_rows:
            all_tags.add(r["name"])
        if all_tags:
            new_positions = self._sync_engine.place_tags_on_glyph_lattice(list(all_tags))
            if new_positions:
                self._sync_engine.compute_all_parallelness()

        # Tensor network seeding
        self._sync_engine.seed_tensor_from_associations()
        self._tensor_cache.invalidate()

    def _generate_project_overview(
        self,
        directory: "Path",
        all_files: list["Path"],
        stats: dict[str, Any],
    ) -> str | None:
        """Build a project-level overview memory from the ingested files.

        Aggregates directory structure, languages, purpose lines from
        module docstrings, and key classes/functions so that broad queries
        like 'what does this project do?' have something to match against.
        """
        import hashlib
        import ast
        from collections import Counter

        project_name = directory.name

        # Dedup: skip if overview already exists
        overview_hash = hashlib.sha256(
            f"project_overview:{project_name}:{stats['files_scanned']}".encode()
        ).hexdigest()[:16]
        existing = self._db.execute(
            "SELECT id FROM memories WHERE content_hash = ?",
            (overview_hash,),
        ).fetchone()
        if existing:
            return None

        # Gather structure
        lang_counts: Counter[str] = Counter()
        top_dirs: set[str] = set()
        purposes: list[str] = []
        classes: list[str] = []
        ext_to_lang = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".jsx": "JavaScript", ".tsx": "TypeScript",
            ".java": "Java", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
            ".c": "C", ".cpp": "C++", ".h": "C/C++ header",
            ".sh": "Shell", ".md": "Markdown", ".sql": "SQL",
            ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
            ".json": "JSON", ".css": "CSS", ".html": "HTML",
        }

        for f in all_files:
            lang = ext_to_lang.get(f.suffix.lower(), f.suffix)
            lang_counts[lang] += 1
            rel = f.relative_to(directory)
            if len(rel.parts) > 1:
                top_dirs.add(rel.parts[0])

            # Extract module docstrings and classes from Python files
            if f.suffix.lower() == ".py":
                try:
                    source = f.read_text(encoding="utf-8", errors="replace")
                    tree = ast.parse(source)
                    docstring = ast.get_docstring(tree)
                    if docstring:
                        first_line = docstring.strip().split("\n")[0]
                        if len(first_line) > 10:
                            rel_path = str(rel).replace("\\", "/")
                            purposes.append(f"- {rel_path}: {first_line}")
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef):
                            cls_doc = ast.get_docstring(node)
                            desc = f" — {cls_doc.split(chr(10))[0]}" if cls_doc else ""
                            rel_path = str(rel).replace("\\", "/")
                            classes.append(f"- {node.name}{desc} ({rel_path})")
                except Exception:
                    pass

        # Check for README
        readme_summary = ""
        for name in ("README.md", "readme.md", "README.rst", "README.txt"):
            readme = directory / name
            if readme.exists():
                try:
                    text = readme.read_text(encoding="utf-8", errors="replace")
                    # Grab first meaningful paragraph
                    for para in text.split("\n\n"):
                        stripped = para.strip()
                        if stripped and not stripped.startswith("#") and len(stripped) > 20:
                            readme_summary = stripped[:500]
                            break
                except Exception:
                    pass
                break

        # Compose overview
        lines = [f"# Project Overview: {project_name}"]
        if readme_summary:
            lines.append(f"\n## Description\n{readme_summary}")
        lines.append(f"\n## Structure")
        lines.append(f"Files: {stats['files_scanned']}")
        if top_dirs:
            lines.append(f"Top-level directories: {', '.join(sorted(top_dirs))}")
        lines.append(f"Languages: {', '.join(f'{lang} ({n})' for lang, n in lang_counts.most_common())}")
        if purposes:
            lines.append(f"\n## Module Purposes")
            lines.extend(purposes[:20])
        if classes:
            lines.append(f"\n## Key Classes")
            lines.extend(classes[:20])

        content = "\n".join(lines)
        content += f"\n\n<!-- chicory:code_hash={overview_hash} -->"

        result = self.handle_store_memory(
            content=content,
            tags=["code-summary", "project-overview", project_name.lower()],
            importance=0.8,
            summary=f"Project overview: {project_name}",
            skip_embedding=True,
            content_hash=overview_hash,
        )
        return result["memory_id"]

    @staticmethod
    def _derive_code_tags(filepath: "Path", base_dir: "Path") -> list[str]:
        """Derive tags for a code file from its path and extension."""
        from pathlib import Path

        tags = ["code-summary"]

        ext_tags = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript",
            ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
            ".c": "c", ".cpp": "cpp", ".h": "c",
            ".sh": "shell", ".md": "markdown", ".sql": "sql",
            ".yaml": "config", ".yml": "config", ".toml": "config",
            ".json": "json", ".css": "css", ".html": "html",
        }
        ext_tag = ext_tags.get(filepath.suffix.lower())
        if ext_tag:
            tags.append(ext_tag)

        # Directory-based tags (skip generic ones)
        try:
            rel = filepath.relative_to(base_dir)
            skip = {"src", "lib", "dist", ".", "app"}
            for part in rel.parent.parts:
                tag = part.lower().replace(" ", "-").replace("_", "-")
                if tag and len(tag) > 1 and tag not in skip:
                    tags.append(tag)
        except ValueError:
            pass

        return tags

    # ── Velocity computation ────────────────────────────────────────

    def _compute_sync_velocity(self) -> dict[str, Any]:
        """Compute level/velocity/jerk for synchronicity event activity."""
        from chicory.layer2.time_series import (
            three_part_jerk,
            split_window_derivative,
            weighted_sum_with_decay,
        )

        W = self._config.sync_velocity_window_hours
        halflife = W / 2
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=W)

        rows = self._db.execute(
            """
            SELECT detected_at, strength, last_reinforced, reinforcement_count
            FROM synchronicity_events
            WHERE detected_at > ?
            ORDER BY detected_at
            """,
            (cutoff.isoformat(),),
        ).fetchall()

        if not rows:
            return {"level": 0.0, "velocity": 0.0, "jerk": 0.0, "event_count": 0}

        events = []
        for r in rows:
            detected = datetime.fromisoformat(r["detected_at"])
            age_hours = (now - detected).total_seconds() / 3600

            evt = SynchronicityEvent(
                detected_at=detected,
                event_type="",
                description="",
                strength=r["strength"],
                quadrant="",
                involved_tags="[]",
                last_reinforced=datetime.fromisoformat(r["last_reinforced"]) if r["last_reinforced"] else None,
                reinforcement_count=r["reinforcement_count"],
            )
            weight = self._sync_detector.effective_strength(evt)
            events.append((age_hours, weight))

        level = weighted_sum_with_decay(events, halflife)
        velocity = split_window_derivative(events, W, halflife)
        jerk = three_part_jerk(events, W, halflife)

        return {
            "level": round(level, 3),
            "velocity": round(velocity, 3),
            "jerk": round(jerk, 3),
            "event_count": len(events),
        }

    # ── Glyph metadata ──────────────────────────────────────────────

    def _store_glyph_metadata(self, memory_id: str, analysis: dict) -> None:
        """Persist glyph analysis results for a memory."""
        import json as _json

        concepts = [
            {"concept": w["word"], "glyph": w["glyph"]}
            for w in analysis.get("words", [])
        ]
        meta = {
            "concepts": concepts,
            "pairs": analysis.get("pairs", []),
            "glyph_line": analysis.get("glyph_line"),
            "formula": analysis.get("formula"),
        }
        self._db.execute(
            """
            INSERT INTO glyph_metadata (memory_id, glyph_json)
            VALUES (?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                glyph_json = excluded.glyph_json
            """,
            (memory_id, _json.dumps(meta, ensure_ascii=False)),
        )
        self._db.connection.commit()

    _formula_to_text = None
    _formula_to_text_checked = False

    def _annotate_glyph(
        self,
        entry: dict[str, Any],
        content: str,
        glyph_meta: dict | None,
    ) -> None:
        """Attach a glyph reading — prose interpretation of detected glyphs."""
        result = analyze_text(content)
        if not result["words"] and not glyph_meta:
            return

        formula = result["formula"]
        if not formula and glyph_meta:
            formula = glyph_meta.get("formula")
        if not formula:
            return

        if not Orchestrator._formula_to_text_checked:
            Orchestrator._formula_to_text_checked = True
            try:
                from app.guardrails import formula_to_text
                Orchestrator._formula_to_text = formula_to_text
            except ImportError:
                pass

        if Orchestrator._formula_to_text is not None:
            try:
                entry["glyph_reading"] = Orchestrator._formula_to_text(formula)
            except Exception:
                entry["glyph_reading"] = result["glyph_line"] or formula
        else:
            entry["glyph_reading"] = result["glyph_line"] or formula

    def _get_glyph_metadata_batch(self, memory_ids: list[str]) -> dict[str, dict]:
        """Load glyph metadata for multiple memories in a single query."""
        if not memory_ids:
            return {}
        ph = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT memory_id, glyph_json FROM glyph_metadata WHERE memory_id IN ({ph})",
            memory_ids,
        ).fetchall()
        return {r["memory_id"]: json.loads(r["glyph_json"]) for r in rows}

    # ── Side-effect triggers ────────────────────────────────────────

    def _on_memory_stored(self, memory: Memory, glyph_analysis: dict | None = None) -> None:
        """Fire after a memory is stored."""
        self._sync_engine.invalidate_pca_cache()

        # Resolve word tags once — reused for trends, centroids, forest/canopy
        word_tag_map: dict[str, Tag] = {}
        for tag_name in memory.tags:
            tag = self._tag_manager.get_by_name(tag_name)
            if tag:
                word_tag_map[tag_name] = tag

        # Resolve temporal + letter tags once
        now_dt = datetime.utcnow()
        temporal_names = [
            f"month-{now_dt.strftime('%Y-%m')}",
            f"day-{now_dt.strftime('%Y-%m-%d')}",
        ]
        letter_counts = self._tag_manager.decompose_to_letters(memory.tags)

        temporal_tags: list[Tag] = []
        for tname in temporal_names:
            tag = self._tag_manager.get_by_name(tname)
            if tag:
                temporal_tags.append(tag)

        letter_tags: list[Tag] = []
        for letter in letter_counts:
            tag = self._tag_manager.get_by_name(letter)
            if tag:
                letter_tags.append(tag)

        # Batch all trend events into one executemany
        trend_events: list[tuple[int, str, str | None, float]] = []
        for tag in word_tag_map.values():
            trend_events.append((tag.id, "assignment", memory.id, 1.0))
        for tag in temporal_tags:
            trend_events.append((tag.id, "assignment", memory.id, 1.0))
        for tag in letter_tags:
            trend_events.append((tag.id, "assignment", memory.id, 1.0))
        if trend_events:
            self._trend_engine.record_events_batch(trend_events)

        # Update tag centroids incrementally (reuse resolved tags)
        if self._config.centroid_inhibition_enabled:
            embedding = self._embedding_engine.get_cached(memory.id)
            if embedding is not None:
                for tag in word_tag_map.values():
                    self._centroid_subgraph.update_centroid_on_store(
                        tag.id, embedding,
                    )
                self._centroid_cache = None
                self._tag_space.invalidate_caches()
                self._tensor_cache.invalidate()
                self._write_gate.invalidate_cache()

        # Place word tags on the glyph Ramsey lattice
        word_tags = [t for t in memory.tags if t]
        if word_tags:
            new_positions = self._sync_engine.place_tags_on_glyph_lattice(word_tags)
            if new_positions:
                self._sync_engine.compute_all_parallelness()

        # Emit commons signal and auto-process
        if self._signal_emitter:
            glyphs = None
            if glyph_analysis and glyph_analysis.get("words"):
                glyphs = glyph_analysis["words"]
            self._signal_emitter.emit_store(memory.tags, glyphs=glyphs)
        if self._signal_processor:
            threading.Thread(
                target=self._commons_auto_process, daemon=True,
            ).start()

        # Forest reorganization + canopy observation on store (reuse resolved tags)
        store_tag_ids = [tag.id for tag in word_tag_map.values()]

        if self._config.episode_enabled and store_tag_ids:
            self._temporal_episodes.update_on_store(memory.id, store_tag_ids)

        if self._config.canopy_enabled and store_tag_ids:
            self._forest.update_on_store(memory.id, store_tag_ids)
            self._episodic_tensor.update_on_store(memory.id)
            self._canopy.observe_single_memory(
                source="store",
                source_id=memory.id,
                memory_id=memory.id,
            )

    def _on_retrieval_completed(
        self,
        retrieval_id: int,
        results: list[tuple[Memory, float]],
        query: str = "",
        query_vec=None,
        penumbra: list | None = None,
    ) -> None:
        """Fire after a retrieval is completed.

        Split into two phases:
          1. Cheap bookkeeping (synchronous): salience, trends,
             tag hits, reinforcement — all batch SQL, O(results).
          2. Expensive detection (background thread, rate-limited):
             sync detection and meta-analysis — runs without blocking
             the retrieval response.
        """
        import time as _time
        _bk_start = _time.perf_counter()
        try:
            self._retrieval_bookkeeping(
                retrieval_id, results, query, query_vec, penumbra,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Error in retrieval bookkeeping"
            )
        self._last_bookkeeping_time = _time.perf_counter() - _bk_start

        # Fire-and-forget: don't block the response on sync detection
        thread = threading.Thread(
            target=self._sync_detection_background,
            daemon=True,
        )
        thread.start()

    def _sync_detection_background(self, idle: bool = False) -> None:
        """Run deferred retrieval work, sync detection, and commons in background.

        Waits briefly for the foreground lock so we don't write to the
        shared SQLite connection while a retrieve/store is mid-flight.
        If foreground stays busy past the timeout, skip this round — the
        next retrieval will spawn another background pass, and the heavy
        work inside (sync detection / commons processing) is rate-limited
        internally so it'll catch up when the system goes idle.
        """
        # Run each phase under a short-lived lock acquisition so the
        # foreground can preempt between phases.  Each phase re-acquires
        # the lock independently — if a foreground retrieval arrives, it
        # gets served between phases instead of waiting for the entire
        # background pass to finish.
        # Whether this pass came from the heartbeat's quiet run or from a
        # retrieval — read by the patrol, which holds idle-only errands back
        # while somebody is using the library.
        self._background_idle = idle

        phases = [
            ("deferred retrieval work", self._run_deferred_retrieval_work),
            ("sync detection", self._maybe_run_sync_detection),
            ("linear sweep", self._maybe_run_linear_sweep),
            ("demand processing", self._maybe_run_demand_processing),
            ("ingress settle", self._maybe_settle_ingress),
            ("patrol", self._maybe_run_patrol),
        ]
        if self._signal_processor:
            phases.append(("commons auto-process", self._commons_auto_process))

        for name, fn in phases:
            if self._foreground_waiting.is_set():
                logging.getLogger(__name__).debug(
                    "background %s deferred: foreground waiting", name
                )
                return
            acquired = self._foreground_db_lock.acquire(timeout=10.0)
            if not acquired:
                logging.getLogger(__name__).info(
                    "background %s skipped: foreground busy >10s", name
                )
                return
            try:
                fn()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Error in background %s", name
                )
            finally:
                self._foreground_db_lock.release()

    def _run_deferred_retrieval_work(self) -> None:
        """Forest, canopy, episodic tensor, and episode updates from last retrieval."""
        bg = getattr(self, "_pending_bg_retrieval", None)
        if not bg:
            return
        self._pending_bg_retrieval = None

        retrieval_id = bg["retrieval_id"]
        memory_ids = bg["memory_ids"]
        tag_ids_map = bg["tag_ids_map"]

        query_tag_ids = bg.get("query_tag_ids", [])
        outflow_boost = 0.0
        if self._config.canopy_directional_enabled and query_tag_ids:
            from chicory.layer3.directional_flow import _hash_sorted_ids as _hsi
            q_hash = _hsi(query_tag_ids)
            outflow_score = self._outflow_observer.observe(
                retrieval_id=retrieval_id,
                query_tag_ids=query_tag_ids,
                query_tag_hash=q_hash,
                result_memory_ids=memory_ids,
            )
            outflow_boost = outflow_score.outflow_strength

        if self._config.canopy_enabled and len(memory_ids) >= 2:
            for i in range(len(memory_ids)):
                for j in range(i + 1, len(memory_ids)):
                    self._episodic_tensor.activate_edge(
                        memory_ids[i], memory_ids[j],
                        outflow_strength=outflow_boost,
                    )

        inflow_canopy_boost = 0.0
        inflow_score_bg = bg.get("inflow_score")
        if inflow_score_bg and inflow_score_bg.unique_query_contexts > 0:
            import math
            max_row = self._db.execute(
                "SELECT MAX(unique_query_contexts) as mx FROM inflow_canopy_blocks"
            ).fetchone()
            max_contexts = max(
                (max_row["mx"] or 1) if max_row else 1,
                inflow_score_bg.unique_query_contexts,
            )
            inflow_canopy_boost = (
                math.log(1.0 + inflow_score_bg.unique_query_contexts)
                / math.log(1.0 + max_contexts)
            )

        if self._config.canopy_enabled:
            all_retrieval_tags: list[int] = []
            for tids in tag_ids_map.values():
                all_retrieval_tags.extend(tids)
            unique_retrieval_tags = list(set(all_retrieval_tags))
            if unique_retrieval_tags:
                self._forest.update_on_retrieval(
                    retrieval_id, memory_ids, unique_retrieval_tags,
                )
            if len(memory_ids) >= 2:
                self._canopy.observe(
                    source="retrieval",
                    source_id=str(retrieval_id),
                    memory_ids=memory_ids,
                    outflow_boost=outflow_boost,
                    inflow_boost=inflow_canopy_boost,
                )

        if self._config.episode_enabled:
            all_ep_tags = list(set(
                tid for tids in tag_ids_map.values() for tid in tids
            ))
            if all_ep_tags:
                self._temporal_episodes.update_on_retrieval(
                    memory_ids, all_ep_tags,
                )

    def _retrieval_bookkeeping(
        self,
        retrieval_id: int,
        results: list[tuple[Memory, float]],
        query: str = "",
        query_vec=None,
        penumbra: list | None = None,
    ) -> None:
        """Phase 1: cheap per-retrieval bookkeeping (batch SQL writes).

        Tags from retrieved memories are filtered by cosine similarity
        to the query embedding — only tags whose centroid is above
        ``similarity_threshold`` participate in downstream analysis.
        """
        import time as _time
        _bt = {}

        # Demand ledger: record the initiation BEFORE the empty-results
        # bail — a query the store couldn't serve is the gap signal the
        # ledger exists to accumulate.
        if self._config.librarian_demand_enabled and query:
            try:
                self._demand_ledger.record(
                    query_text=query,
                    query_vec=query_vec,
                    result_count=len(results),
                    top_score=max((s for _, s in results), default=0.0),
                )
            except Exception:
                logging.getLogger(__name__).exception("Demand ledger record failed")

        # Penumbra ledger: near-misses recorded even for empty result sets
        if self._config.penumbra_enabled and (penumbra or results):
            try:
                self._penumbra_ledger.record(
                    penumbra or [], [mem.id for mem, _ in results],
                )
            except Exception:
                logging.getLogger(__name__).exception("Penumbra record failed")

        if not results:
            return

        memory_ids = [mem.id for mem, _ in results]

        _s = _time.perf_counter()
        self._salience.update_on_access_batch(memory_ids)
        _bt["salience"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        tag_ids_map = self._tag_manager.get_tag_ids_for_memories(memory_ids)
        _bt["tag_ids_lookup"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        trend_events: list[tuple[int, str, str | None, float]] = []
        tag_hits: list[tuple[int, str]] = []
        for mid in memory_ids:
            for tid in tag_ids_map.get(mid, []):
                trend_events.append((tid, "retrieval", mid, 1.0))
                tag_hits.append((tid, "direct_match"))
        self._trend_engine.record_events_batch(trend_events)
        if tag_hits:
            self._retrieval_tracker.log_tag_hits(retrieval_id, tag_hits)
        _bt["trends_and_hits"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        events_map = self._sync_detector.get_events_for_memories(memory_ids)
        reinforced_event_ids: set[int] = set()
        for mid in memory_ids:
            reinforced_event_ids.update(events_map.get(mid, []))
        if reinforced_event_ids:
            self._sync_detector.reinforce_events_batch(list(reinforced_event_ids))
        _bt["sync_reinforce"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        if self._signal_emitter:
            all_tag_ids: set[int] = set()
            for tids in tag_ids_map.values():
                all_tag_ids.update(tids)
            if all_tag_ids:
                id_to_name = self._tag_manager.get_names_by_ids(list(all_tag_ids))
                self._signal_emitter.emit_retrieve(list(id_to_name.values()))
        _bt["commons_signal"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        query_tag_ids: list[int] = []
        if self._config.canopy_directional_enabled and query:
            query_tag_ids = self._compute_query_side_tags(query)
        _bt["query_side_tags"] = _time.perf_counter() - _s

        if self._config.canopy_directional_enabled and query_tag_ids:
            import hashlib as _hl
            import json as _json
            q_hash = _hl.sha256(
                _json.dumps(sorted(query_tag_ids)).encode()
            ).hexdigest()[:32]
            all_result_tag_ids = list(set(
                tid for tids in tag_ids_map.values() for tid in tids
            ))
            self._db.execute(
                """INSERT INTO directional_retrieval_context
                   (retrieval_id, query_tag_ids, result_tag_ids,
                    result_memory_ids, query_tag_hash)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    retrieval_id,
                    _json.dumps(sorted(query_tag_ids)),
                    _json.dumps(sorted(all_result_tag_ids)),
                    _json.dumps(sorted(memory_ids)),
                    q_hash,
                ),
            )

        _s = _time.perf_counter()
        from chicory.models.canopy import InflowScore
        inflow_score = InflowScore()
        if self._config.canopy_directional_enabled and len(memory_ids) >= 2:
            all_result_tag_ids_for_inflow = list(set(
                tid for tids in tag_ids_map.values() for tid in tids
            ))
            from chicory.layer3.directional_flow import _hash_sorted_ids as _hsi
            q_hash_inflow = _hsi(query_tag_ids) if query_tag_ids else "empty"
            inflow_score = self._inflow_observer.observe(
                retrieval_id=retrieval_id,
                query_tag_ids=query_tag_ids,
                query_tag_hash=q_hash_inflow,
                result_memory_ids=memory_ids,
                result_tag_ids=all_result_tag_ids_for_inflow,
            )
        _bt["inflow_canopy"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        if self._config.centroid_inhibition_enabled:
            all_tag_ids_flat: list[int] = []
            for tids in tag_ids_map.values():
                all_tag_ids_flat.extend(tids)
            all_unique = set(all_tag_ids_flat)

            # Filter to query-relevant tags via centroid similarity.
            # Prefer the vector threaded in for THIS retrieval — the shared
            # slot may hold another call's vector (consult FAQ path, recall).
            qv = query_vec if query_vec is not None else getattr(
                self._memory_store, "_last_query_vec", None,
            )
            if qv is not None:
                relevant = self._tag_space.centroid_match(qv)
                unique_tags = sorted(all_unique & relevant)
            else:
                unique_tags = sorted(all_unique)

            if len(unique_tags) >= 2:
                mean_relevance = (
                    sum(score for _, score in results) / len(results)
                )

                pair_diversity: dict[tuple[int, int], float] | None = None
                if (self._config.canopy_directional_enabled
                        and inflow_score.block_key):
                    import itertools
                    sorted_tags = sorted(set(unique_tags))
                    tag_pairs = list(itertools.combinations(sorted_tags, 2))
                    if tag_pairs:
                        raw_diversity = self._inflow_observer.compute_pair_diversity(
                            inflow_score.block_key, tag_pairs,
                        )
                        gate_threshold = self._config.canopy_directional_inflow_context_gate_threshold
                        ctx_count = inflow_score.unique_query_contexts
                        if gate_threshold > 1:
                            context_gate = min(1.0, max(0.0,
                                (ctx_count - 1) / (gate_threshold - 1),
                            ))
                        else:
                            context_gate = 1.0 if ctx_count >= 1 else 0.0
                        boost_w = self._config.canopy_directional_inflow_centroid_boost
                        pair_diversity = {
                            pair: div * context_gate * boost_w
                            for pair, div in raw_diversity.items()
                        }

                self._centroid_subgraph.record_co_retrieval(unique_tags)
                self._centroid_subgraph.update_on_retrieval(
                    unique_tags, mean_relevance,
                    pair_diversity=pair_diversity,
                )
        _bt["centroid_reweight"] = _time.perf_counter() - _s

        self._last_bookkeeping_breakdown = _bt
        logging.getLogger(__name__).info("bookkeeping_timing: %s", _bt)

        self._pending_bg_retrieval = {
            "retrieval_id": retrieval_id,
            "memory_ids": memory_ids,
            "tag_ids_map": tag_ids_map,
            "query_tag_ids": query_tag_ids,
            "inflow_score": inflow_score,
        }

    def _compute_query_side_tags(self, query: str) -> list[int]:
        """Find all tags whose centroids are similar to the query embedding."""
        query_vec = getattr(self._memory_store, "_last_query_vec", None)
        if query_vec is None:
            try:
                query_vec = self._embedding_engine.embed(query)
            except Exception:
                return []

        return list(self._tag_space.centroid_match(query_vec))

    def _maybe_run_sync_detection(self) -> None:
        """Run synchronicity detection if enough time has passed.

        Releases the foreground lock during the expensive detection
        pass so user-facing queries aren't blocked.  Re-acquires to
        write results.
        """
        now = datetime.utcnow()
        if self._last_sync_check and (now - self._last_sync_check).total_seconds() < 60:
            return  # At most once per minute

        self._last_sync_check = now

        # Release lock during the expensive computation so foreground
        # queries aren't blocked.  check_for_synchronicities does DB
        # reads + record_event writes, but those are small row inserts
        # that don't conflict with foreground WAL reads.
        self._foreground_db_lock.release()
        try:
            new_events = self._sync_detector.check_for_synchronicities()
        finally:
            self._foreground_db_lock.acquire()

        # Place new events on the prime Ramsey lattice (under lock)
        if new_events:
            self._sync_engine.place_events_batch(new_events)

            # Emit synchronicity signals to commons (immediate flush)
            if self._signal_emitter:
                for event in new_events:
                    involved_tag_ids = json.loads(event.involved_tags)
                    id_to_name = self._tag_manager.get_names_by_ids(involved_tag_ids)
                    tag_names = list(id_to_name.values())
                    if tag_names:
                        self._signal_emitter.emit_synchronicity(
                            tags=tag_names,
                            strength=event.strength,
                            event_type=event.event_type,
                        )

        # Canopy observation for new sync events
        if new_events and self._config.canopy_enabled:
            for event in new_events:
                try:
                    involved_tag_ids = [int(t) for t in json.loads(event.involved_tags)]
                except (json.JSONDecodeError, TypeError, ValueError):
                    involved_tag_ids = []
                if involved_tag_ids:
                    involved_mids = []
                    if event.involved_memories:
                        try:
                            involved_mids = json.loads(event.involved_memories)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    self._forest.update_on_sync_event(
                        event.id, involved_tag_ids, involved_mids,
                    )
                    if len(involved_mids) >= 2:
                        self._canopy.observe(
                            source="synchronicity",
                            source_id=str(event.id),
                            memory_ids=involved_mids,
                        )

        # Force episode boundary on strong sync events
        if new_events and self._config.episode_enabled:
            for event in new_events:
                eff = self._sync_detector.effective_strength(event)
                try:
                    ep_tag_ids = [int(t) for t in json.loads(event.involved_tags)]
                except (json.JSONDecodeError, TypeError, ValueError):
                    ep_tag_ids = []
                if ep_tag_ids:
                    self._temporal_episodes.force_sync_boundary(
                        event.id, ep_tag_ids, eff,
                    )

        # Also maybe run meta-analysis
        self._maybe_run_meta_analysis()

    def _maybe_run_linear_sweep(self) -> None:
        """Advance the librarian's shelf-reading patrol if enough time has passed.

        Each pass sweeps one batch of the least-recently-visited memories;
        the cursor (memories.swept_at) is durable, so coverage survives
        restarts.  Traffic-paced: only advances when retrievals arrive.
        """
        if not self._config.librarian_sweep_enabled:
            return
        now = datetime.utcnow()
        interval = timedelta(minutes=self._config.librarian_sweep_interval_minutes)
        if self._last_sweep_check and (now - self._last_sweep_check) < interval:
            return
        self._last_sweep_check = now

        stats = self._linear_sweep.run_batch()
        if stats.get("findings"):
            logging.getLogger(__name__).info(
                "Linear sweep: %d memories, findings %s",
                stats["swept"], stats["findings"],
            )

    # ── The wandering librarian ─────────────────────────────────────

    def _maybe_run_patrol(self) -> None:
        """Advance the patrol's errands, each on its own cursor.

        Unlike the sweep, this is not one walk with one question: each
        errand carries its own visit stamps in patrol_visits, so a repair
        errand emptying its queue and a survey errand walking the whole
        library do not share a position.
        """
        if not self._config.wander_enabled:
            return
        now = datetime.utcnow()
        if (
            self._last_patrol_check is not None
            and (now - self._last_patrol_check).total_seconds() < 30
        ):
            return
        self._last_patrol_check = now

        stats = self._patrol.run_due(
            idle=self._background_idle,
            # The patrol is the last phase, so it holds the lock longest;
            # let it drop a batch mid-way when a request arrives rather
            # than making the patron wait for the whole errand.
            should_stop=self._foreground_waiting.is_set,
        )
        worked = [e for e in stats.get("errands", []) if e.get("examined")]
        if worked:
            logging.getLogger(__name__).info("Patrol: %s", worked)

    def _heartbeat_tick(self, idle: bool) -> None:
        """One beat: run the ordinary background pass on our own clock."""
        self._sync_detection_background(idle=idle)

    def start_heartbeat(self) -> bool:
        """Start the background clock.  Idempotent; False if disabled."""
        return self._heartbeat.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat.stop()

    def wander_status(self) -> dict:
        """What the patrol is doing and how far each errand has walked."""
        from chicory.layer4.wander import ERRANDS

        records = {r.errand: r for r in self._patrol.state.all()}
        errands = []
        for name in self._config.wander_errands:
            errand = ERRANDS.get(name)
            rec = records.get(name)
            errands.append({
                "errand": name,
                "known": errand is not None,
                "when": errand.when if errand else None,
                # Capped: an exact count is a full scan of the memories
                # table per errand, which a status call cannot afford.
                "remaining_at_least": (
                    self._patrol.population_size(errand, cap=1000)
                    if errand else None
                ),
                "lap": rec.lap if rec else 0,
                "visited": rec.visited_count if rec else 0,
                "acted": rec.acted_count if rec else 0,
                "runs": rec.runs if rec else 0,
                "enabled": rec.enabled if rec else True,
                "last_run_at": rec.last_run_at if rec else None,
                "last_result": rec.last_result if rec else None,
            })
        holder = self._patrol.lease.holder() or {}
        return {
            "enabled": self._config.wander_enabled,
            "tick_seconds": self._config.wander_tick_seconds,
            "ticks": self._heartbeat.ticks,
            "idle": self._heartbeat.idle,
            # Which process is patrolling right now, if any.  With one
            # server per client session, several may share this library.
            "lease_owner": self._patrol.lease.owner,
            "lease_held_by": holder.get("owner"),
            "errands": errands,
        }

    def _maybe_settle_ingress(self) -> None:
        """Close dwells whose window has elapsed, and prune old receipts.

        Runs every background pass (no rate limit beyond 30s) — a dwell
        that has elapsed should settle promptly, since nothing downstream
        can act on the content until it does.
        """
        if not self._config.ingress_enabled:
            return
        now = datetime.utcnow()
        if self._last_ingress_check and (now - self._last_ingress_check).total_seconds() < 30:
            return
        self._last_ingress_check = now

        stats = self._ingress.settle_due(
            limit=self._config.ingress_settle_batch,
        )
        if stats.get("flagged") or stats.get("remaining"):
            logging.getLogger(__name__).info(
                "Ingress settle: %s", stats,
            )

        # Retention is a slow janitorial pass — running it every 30s would
        # full-scan the buffer under the foreground lock to delete nothing.
        prune_interval = timedelta(
            minutes=self._config.ingress_prune_interval_minutes
        )
        if (
            self._last_ingress_prune is None
            or (now - self._last_ingress_prune) >= prune_interval
        ):
            self._last_ingress_prune = now
            self._ingress.prune()

    def _maybe_run_demand_processing(self) -> None:
        """Cluster the demand ledger and promote FAQs / file commissions.

        FAQ promotion runs a live retrieval to gather supporting memories,
        so it's gated on an already-built vector index — never rebuild
        FAISS while holding the foreground lock.
        """
        if not self._config.librarian_demand_enabled:
            return
        now = datetime.utcnow()
        interval = timedelta(minutes=self._config.librarian_demand_interval_minutes)
        if self._last_demand_check and (now - self._last_demand_check) < interval:
            return
        self._last_demand_check = now

        stats = self._demand_ledger.process_batch(
            allow_retrieval=self._index_is_built(),
        )
        if stats.get("faq_candidates") or stats.get("commissions_filed"):
            logging.getLogger(__name__).info(
                "Demand processing: %s", stats,
            )

    def _maybe_run_meta_analysis(self) -> None:
        """Run meta-pattern analysis if enough time has passed."""
        now = datetime.utcnow()
        interval = timedelta(hours=self._config.meta_analysis_interval_hours)

        if self._last_meta_check and (now - self._last_meta_check) < interval:
            return

        self._last_meta_check = now
        patterns = self._meta_analyzer.run_analysis()

        # Apply feedback for any new patterns
        for pattern in patterns:
            self._feedback.apply_pattern_actions(pattern)
