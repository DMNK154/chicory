"""Central coordinator — wires all layers, handles side effects."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Optional

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
from chicory.layer4.adaptive_thresholds import AdaptiveThresholds
from chicory.layer4.canopy import CanopyObserver
from chicory.layer4.episodic_tensor import EpisodicTensor
from chicory.layer4.temporal_episodes import TemporalEpisodeTracker
from chicory.layer4.feedback import FeedbackEngine
from chicory.layer4.meta_analyzer import MetaAnalyzer
from chicory.layer4.forest import ForestReorganizer
from chicory.layer1.glyph_analyzer import analyze_text, extract_glyph_tags
from chicory.models.memory import Memory
from chicory.models.synchronicity import SynchronicityEvent


class Orchestrator:
    """Central coordinator. All LLM tool calls route through here."""

    def __init__(self, config: ChicoryConfig) -> None:
        self._config = config

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

        self._sync_engine = SynchronicityEngine(
            config, self._db, self._embedding_engine, self._tag_manager,
            glyph_encoder=self._glyph_encoder,
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

        # Layer 1 (complete — now has access to Layer 3 for lattice-aware retrieval)
        self._memory_store = MemoryStore(
            config, self._db, self._embedding_engine,
            self._tag_manager, self._salience,
            sync_engine=self._sync_engine,
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
        self._episodic_tensor = EpisodicTensor(self._db, config)
        self._temporal_episodes = TemporalEpisodeTracker(
            self._db, config, self._centroid_subgraph,
        )
        self._maybe_seed_temporal_episodes()

        # Rate limiting for detection — initialise to "now" so the first
        # retrieval isn't blocked by expensive sync/meta detection.
        self._last_sync_check: datetime | None = datetime.utcnow()
        self._last_meta_check: datetime | None = datetime.utcnow()

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
        """Bootstrap temporal episodes on first boot after upgrade."""
        if not self._config.episode_enabled:
            return
        episode_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM temporal_episodes"
        ).fetchone()["cnt"]
        if episode_count > 0:
            return
        memory_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0"
        ).fetchone()["cnt"]
        if memory_count > 0:
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

        # 12. Synchronicity detection (unrestricted — bypass rate limit)
        _report("sync_detection", "Running synchronicity detection")
        t0 = time.time()
        new_events = self._sync_detector.check_for_synchronicities()
        if new_events:
            self._sync_engine.place_events_batch(new_events)
        stats["sync_events_detected"] = len(new_events) if new_events else 0
        stats["sync_detection_seconds"] = round(time.time() - t0, 1)

        # 13. Meta-pattern analysis
        _report("meta_analysis", "Running meta-pattern analysis")
        t0 = time.time()
        patterns = self._meta_analyzer.run_analysis()
        for pattern in patterns:
            self._feedback.apply_pattern_actions(pattern)
        stats["meta_patterns"] = len(patterns)
        stats["meta_analysis_seconds"] = round(time.time() - t0, 1)

        # 14. Episode lifecycle
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

        _report("done", f"Sync complete in {stats['total_seconds']:.1f}s")
        return stats

    def close(self) -> None:
        """Close database connection and signal emitter."""
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
    ) -> dict[str, Any]:
        """Store a memory and fire side effects.

        Automatically scans content for glyph concept words and adds
        them as tags so that glyph definitions flow through the memory
        the same way GPT-GU's analyze_text works.

        When defer_side_effects=True, skips _on_memory_stored (trend events,
        lattice placement, canopy, etc). Callers must handle side effects
        themselves — used by bulk ingestion to avoid O(n) expensive work.
        """
        salience = importance if importance is not None else 0.5

        # Glyph content analysis: extract concept-derived tags from content
        glyph_analysis = analyze_text(content)
        glyph_derived_tags = extract_glyph_tags(content)

        # Merge glyph-derived tags into the tag list (no duplicates)
        merged_tags = list(tags)
        for gt in glyph_derived_tags:
            if gt not in merged_tags:
                merged_tags.append(gt)

        memory = self._memory_store.store(
            content=content,
            tags=merged_tags,
            salience_model=salience,
            summary=summary,
            skip_embedding=skip_embedding,
            content_hash=content_hash,
            source_path=source_path,
            ingestion_tier=ingestion_tier,
        )

        # Persist glyph analysis metadata on the memory row
        if glyph_analysis["words"]:
            self._store_glyph_metadata(memory.id, glyph_analysis)

        if not defer_side_effects:
            self._on_memory_stored(memory)

        return {
            "status": "stored",
            "memory_id": memory.id,
            "tags": memory.tags,
            "glyph_concepts": [
                {"concept": w["word"], "glyph": w["glyph"]}
                for w in glyph_analysis["words"]
            ],
            "glyph_pairs": glyph_analysis["pairs"],
            "salience": memory.salience_composite,
        }

    def handle_retrieve_memories(
        self,
        query: str,
        tags: list[str] | None = None,
        method: str = "hybrid",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Retrieve memories, log the event, and fire side effects."""
        if method == "semantic":
            results = self._memory_store.retrieve_semantic(query, top_k, tags)
        elif method == "tag" and tags:
            tag_results = self._memory_store.retrieve_by_tags(tags)
            results = [(m, 1.0) for m in tag_results[:top_k]]
        else:
            results = self._memory_store.retrieve_hybrid(query, tags, top_k)

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

        # Side effects (background thread)
        self._on_retrieval_completed(retrieval_id, results, query)

        # Annotate results with live glyph scanning
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
            }
            self._annotate_glyph_definitions(entry, mem.id, mem.content)
            annotated.append(entry)

        return {
            "results": annotated,
            "count": len(results),
            "method": method,
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
        level_0_results = self._memory_store.retrieve_hybrid(query, tags, level_k)

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
            self._on_retrieval_completed(combined_id, all_retrieval_results, query)

        return {
            "results": all_results,
            "count": len(all_results),
            "method": "deep_retrieve",
            "max_depth_reached": max(
                (r["depth"] for r in all_results), default=0
            ),
        }

    def _format_deep_result(
        self, mem: Memory, score: float, depth: int,
    ) -> dict[str, Any]:
        """Format a single deep retrieval result with depth annotation."""
        entry: dict[str, Any] = {
            "memory_id": mem.id,
            "content": mem.content,
            "summary": mem.summary,
            "tags": mem.tags,
            "salience": mem.salience_composite,
            "relevance_score": round(score, 3),
            "created_at": mem.created_at.isoformat(),
            "depth": depth,
        }
        self._annotate_glyph_definitions(entry, mem.id, mem.content)
        return entry

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

    def handle_get_lattice_resonances(self) -> dict[str, Any]:
        """Get lattice state including positions, resonances, and void profile."""
        return self._sync_engine.get_lattice_state()

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

        # Directories to always skip
        skip_dirs = {
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            ".env", ".tox", ".mypy_cache", ".pytest_cache", "dist",
            "build", ".eggs", "*.egg-info",
        }
        if exclude_patterns:
            skip_dirs.update(exclude_patterns)

        # Collect files
        all_files: list[Path] = []
        for f in directory.rglob("*"):
            if not f.is_file():
                continue
            # Skip hidden and excluded directories
            parts = f.relative_to(directory).parts
            if any(
                part.startswith(".") or part in skip_dirs
                for part in parts[:-1]
            ):
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
        }

        # ── Phase 1: Store summaries (no embeddings) ──────────────
        new_memory_ids: list[str] = []

        for filepath in all_files:
            summary_text = summarize_file(filepath, base_dir=directory)
            if not summary_text:
                stats["files_skipped"] += 1
                continue

            # Dedup by content hash
            content_hash = hashlib.sha256(
                summary_text.encode("utf-8")
            ).hexdigest()[:16]

            existing = self._db.execute(
                "SELECT id FROM memories WHERE content LIKE ?",
                (f"%chicory:code_hash={content_hash}%",),
            ).fetchone()
            if existing:
                stats["files_already_ingested"] += 1
                continue

            # Derive tags from file path
            rel_path = str(filepath.relative_to(directory)).replace("\\", "/")
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
            skip_dirs = {
                ".git", ".venv", "venv", "node_modules", "__pycache__",
                ".env", ".tox", ".mypy_cache", ".pytest_cache", "dist",
                "build", ".eggs",
            }
            if exclude_patterns:
                skip_dirs.update(exclude_patterns)

            files = []
            for f in target.rglob("*"):
                if not f.is_file():
                    continue
                parts = f.relative_to(target).parts
                if any(
                    part.startswith(".") or part in skip_dirs
                    for part in parts[:-1]
                ):
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
        }

        # Phase 0: Read all files and derive tags for classification
        file_data: list[tuple] = []  # (filepath, text, content_hash, rel_path, tags)
        file_tags_map: dict[str, list[str]] = {}

        for filepath in files:
            text = read_document(filepath)
            if not text:
                stats["files_skipped"] += 1
                continue

            content_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()[:16]

            existing = self._db.execute(
                "SELECT id FROM memories WHERE content LIKE ?",
                (f"%chicory:doc_hash={content_hash}%",),
            ).fetchone()
            if existing:
                stats["files_already_ingested"] += 1
                continue

            rel_path = str(filepath.relative_to(base_dir)).replace("\\", "/")
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
                        chunk_content += f"\n\n<!-- chicory:doc_hash={content_hash} -->"
                    else:
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

        # Distribute vectors back to their memories
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
            "SELECT id FROM memories WHERE content LIKE ?",
            (f"%chicory:code_hash={overview_hash}%",),
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

    def _annotate_glyph_definitions(
        self, entry: dict[str, Any], memory_id: str, content: str,
    ) -> None:
        """Scan content for glyph concepts and merge with stored metadata.

        Runs ``analyze_text`` on the raw content — pure string matching,
        no model call — so every retrieval result carries the concept↔symbol
        mappings the LLM needs to interpret glyphs.  Also includes any
        pre-stored glyph metadata from ingestion time.
        """
        result = analyze_text(content)

        # Live-scanned definitions
        if result["words"]:
            entry["glyph_definitions"] = {
                w["word"]: w["glyph"] for w in result["words"]
            }
            entry["glyph_formula"] = result["formula"]
            entry["glyph_line"] = result["glyph_line"]

            if result["pairs"]:
                entry["glyph_pairs"] = [
                    {
                        "glyphs": p["glyphs"],
                        "concepts": p["concepts"],
                        "is_known_pair": p["is_known_pair"],
                    }
                    for p in result["pairs"]
                ]

        # Merge stored metadata (may have richer info from ingestion)
        glyph_meta = self._get_glyph_metadata(memory_id)
        if glyph_meta:
            entry["glyph_concepts"] = glyph_meta.get("concepts", [])
            if not entry.get("glyph_line"):
                entry["glyph_line"] = glyph_meta.get("glyph_line")
            if not entry.get("glyph_formula"):
                entry["glyph_formula"] = glyph_meta.get("formula")

    def _get_glyph_metadata(self, memory_id: str) -> dict | None:
        """Load glyph analysis metadata for a memory, or None."""
        import json as _json

        row = self._db.execute(
            "SELECT glyph_json FROM glyph_metadata WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row:
            return _json.loads(row["glyph_json"])
        return None

    # ── Side-effect triggers ────────────────────────────────────────

    def _on_memory_stored(self, memory: Memory) -> None:
        """Fire after a memory is stored."""
        self._sync_engine.invalidate_pca_cache()

        # Record tag assignment events for word tags
        for tag_name in memory.tags:
            tag = self._tag_manager.get_by_name(tag_name)
            if tag:
                self._trend_engine.record_event(
                    tag_id=tag.id,
                    event_type="assignment",
                    memory_id=memory.id,
                )

        # Record trend events for temporal tags
        now_dt = datetime.utcnow()
        for temporal_name in [
            f"month-{now_dt.strftime('%Y-%m')}",
            f"day-{now_dt.strftime('%Y-%m-%d')}",
        ]:
            tag = self._tag_manager.get_by_name(temporal_name)
            if tag:
                self._trend_engine.record_event(
                    tag_id=tag.id,
                    event_type="assignment",
                    memory_id=memory.id,
                )

        # Record tag assignment events for derived letter tags
        letter_counts = self._tag_manager.decompose_to_letters(memory.tags)
        for letter in letter_counts:
            tag = self._tag_manager.get_by_name(letter)
            if tag:
                self._trend_engine.record_event(
                    tag_id=tag.id,
                    event_type="assignment",
                    memory_id=memory.id,
                )

        # Update tag centroids incrementally
        if self._config.centroid_inhibition_enabled:
            embedding = self._embedding_engine.get_cached(memory.id)
            if embedding is not None:
                for tag_name in memory.tags:
                    tag = self._tag_manager.get_by_name(tag_name)
                    if tag:
                        self._centroid_subgraph.update_centroid_on_store(
                            tag.id, embedding,
                        )

        # Place word tags on the glyph Ramsey lattice
        word_tags = [t for t in memory.tags if t]
        if word_tags:
            new_positions = self._sync_engine.place_tags_on_glyph_lattice(word_tags)
            # Refresh parallelness if new glyph positions were created
            if new_positions:
                self._sync_engine.compute_all_parallelness()

        # Emit commons signal and auto-process
        if self._signal_emitter:
            self._signal_emitter.emit_store(memory.tags)
        if self._signal_processor:
            threading.Thread(
                target=self._commons_auto_process, daemon=True,
            ).start()

        # Forest reorganization + canopy observation on store
        store_tag_ids: list[int] = []
        for tag_name in memory.tags:
            tag = self._tag_manager.get_by_name(tag_name)
            if tag:
                store_tag_ids.append(tag.id)

        if self._config.canopy_enabled and store_tag_ids:
            self._forest.update_on_store(memory.id, store_tag_ids)
            self._episodic_tensor.update_on_store(memory.id)
            self._canopy.observe_single_memory(
                source="store",
                source_id=memory.id,
                memory_id=memory.id,
            )

        if self._config.episode_enabled and store_tag_ids:
            self._temporal_episodes.update_on_store(memory.id, store_tag_ids)

    def _on_retrieval_completed(
        self,
        retrieval_id: int,
        results: list[tuple[Memory, float]],
        query: str = "",
    ) -> None:
        """Fire after a retrieval is completed.

        Split into two phases:
          1. Cheap bookkeeping (synchronous): salience, trends,
             tag hits, reinforcement — all batch SQL, O(results).
          2. Expensive detection (background thread, rate-limited):
             sync detection and meta-analysis — runs without blocking
             the retrieval response.
        """
        try:
            self._retrieval_bookkeeping(retrieval_id, results, query)
        except Exception:
            logging.getLogger(__name__).exception(
                "Error in retrieval bookkeeping"
            )

        # Fire-and-forget: don't block the response on sync detection
        thread = threading.Thread(
            target=self._sync_detection_background,
            daemon=True,
        )
        thread.start()

    def _sync_detection_background(self) -> None:
        """Run sync detection and commons auto-processing in a background thread."""
        try:
            self._maybe_run_sync_detection()
        except Exception:
            logging.getLogger(__name__).exception(
                "Error in background sync detection"
            )
        if self._signal_processor:
            self._commons_auto_process()

    def _retrieval_bookkeeping(
        self,
        retrieval_id: int,
        results: list[tuple[Memory, float]],
        query: str = "",
    ) -> None:
        """Phase 1: cheap per-retrieval bookkeeping (batch SQL writes).

        Tags from retrieved memories are filtered by cosine similarity
        to the query embedding — only tags whose centroid is above
        ``similarity_threshold`` participate in downstream analysis.
        """
        if not results:
            return

        memory_ids = [mem.id for mem, _ in results]

        # Batch update access salience
        self._salience.update_on_access_batch(memory_ids)

        # Batch get tag IDs for all result memories
        tag_ids_map = self._tag_manager.get_tag_ids_for_memories(memory_ids)

        # Gate: keep only tags whose centroid is relevant to the query
        tag_ids_map = self._filter_relevant_tags(tag_ids_map, query)

        # Build trend events and tag hits in bulk
        trend_events: list[tuple[int, str, str | None, float]] = []
        tag_hits: list[tuple[int, str]] = []
        for mid in memory_ids:
            for tid in tag_ids_map.get(mid, []):
                trend_events.append((tid, "retrieval", mid, 1.0))
                tag_hits.append((tid, "direct_match"))

        self._trend_engine.record_events_batch(trend_events)

        if tag_hits:
            self._retrieval_tracker.log_tag_hits(retrieval_id, tag_hits)

        # Reinforce synchronicity events — collect IDs then single batch UPDATE
        events_map = self._sync_detector.get_events_for_memories(memory_ids)
        reinforced_event_ids: set[int] = set()
        for mid in memory_ids:
            reinforced_event_ids.update(events_map.get(mid, []))
        if reinforced_event_ids:
            self._sync_detector.reinforce_events_batch(list(reinforced_event_ids))

        # Record co-retrieval and apply centroid sub-graph reweighting
        if self._config.centroid_inhibition_enabled:
            all_tag_ids_flat: list[int] = []
            for tids in tag_ids_map.values():
                all_tag_ids_flat.extend(tids)
            unique_tags = list(set(all_tag_ids_flat))
            if len(unique_tags) >= 2:
                mean_relevance = (
                    sum(score for _, score in results) / len(results)
                )
                self._centroid_subgraph.record_co_retrieval(unique_tags)
                self._centroid_subgraph.update_on_retrieval(
                    unique_tags, mean_relevance,
                )

        # Emit commons retrieve signal with all result tag names
        if self._signal_emitter:
            all_tag_ids: set[int] = set()
            for tids in tag_ids_map.values():
                all_tag_ids.update(tids)
            if all_tag_ids:
                id_to_name = self._tag_manager.get_names_by_ids(list(all_tag_ids))
                self._signal_emitter.emit_retrieve(list(id_to_name.values()))

        # Activate episodic edges for co-retrieved memory pairs
        if self._config.canopy_enabled and len(memory_ids) >= 2:
            for i in range(len(memory_ids)):
                for j in range(i + 1, len(memory_ids)):
                    self._episodic_tensor.activate_edge(memory_ids[i], memory_ids[j])

        # Forest reorganization + canopy observation on retrieval
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
                )

        if self._config.episode_enabled:
            all_ep_tags = list(set(
                tid for tids in tag_ids_map.values() for tid in tids
            ))
            if all_ep_tags:
                self._temporal_episodes.update_on_retrieval(
                    memory_ids, all_ep_tags,
                )

    def _filter_relevant_tags(
        self,
        tag_ids_map: dict[str, list[int]],
        query: str,
    ) -> dict[str, list[int]]:
        """Keep only tags whose centroid is similar to the query embedding."""
        import numpy as np

        if not query:
            return tag_ids_map

        all_tag_ids: set[int] = set()
        for tids in tag_ids_map.values():
            all_tag_ids.update(tids)
        if not all_tag_ids:
            return tag_ids_map

        try:
            query_vec = self._embedding_engine.embed(query)
        except Exception:
            return tag_ids_map

        placeholders = ",".join("?" * len(all_tag_ids))
        rows = self._db.execute(
            f"SELECT tag_id, centroid FROM tag_centroids "
            f"WHERE tag_id IN ({placeholders})",
            list(all_tag_ids),
        ).fetchall()

        if not rows:
            return tag_ids_map

        row_tag_ids = [r["tag_id"] for r in rows]
        centroid_matrix = np.stack([
            np.frombuffer(r["centroid"], dtype=np.float32) for r in rows
        ])
        similarities = centroid_matrix @ query_vec

        threshold = self._config.similarity_threshold
        relevant_ids = {
            row_tag_ids[i]
            for i in range(len(row_tag_ids))
            if similarities[i] >= threshold
        }

        # Tags without centroids pass through (new tags not yet embedded)
        tags_with_centroids = set(row_tag_ids)
        filtered: dict[str, list[int]] = {}
        for mid, tids in tag_ids_map.items():
            kept = [
                t for t in tids
                if t in relevant_ids or t not in tags_with_centroids
            ]
            if kept:
                filtered[mid] = kept

        return filtered

    def _maybe_run_sync_detection(self) -> None:
        """Run synchronicity detection if enough time has passed."""
        now = datetime.utcnow()
        if self._last_sync_check and (now - self._last_sync_check).total_seconds() < 60:
            return  # At most once per minute

        self._last_sync_check = now
        new_events = self._sync_detector.check_for_synchronicities()

        # Place new events on the prime Ramsey lattice
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
