"""Memory store — CRUD and retrieval (semantic, tag, hybrid)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.exceptions import MemoryNotFoundError
from chicory.ingest.chunker import chunk_text_for_embedding
from chicory.layer1.embedding_engine import EmbeddingEngine
from chicory.layer1.salience import SalienceScorer
from chicory.layer1.tag_manager import TagManager
from chicory.models.memory import Memory

if TYPE_CHECKING:
    from chicory.layer3.synchronicity_engine import SynchronicityEngine


class MemoryStore:
    """CRUD operations for memories with tag management and retrieval."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        embedding_engine: EmbeddingEngine,
        tag_manager: TagManager,
        salience: SalienceScorer,
        sync_engine: SynchronicityEngine | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._embedding = embedding_engine
        self._tags = tag_manager
        self._salience = salience
        self._sync_engine = sync_engine

    def store(
        self,
        content: str,
        tags: list[str],
        salience_model: float = 0.5,
        source_model: str | None = None,
        summary: str | None = None,
        skip_embedding: bool = False,
        content_hash: str | None = None,
        source_path: str | None = None,
        ingestion_tier: str = "critical",
    ) -> Memory:
        """Store a new memory with tags and (optionally) embedding."""
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        model = source_model or self._config.llm_model

        salience_composite = self._salience.compute_composite(salience_model, 0.0)

        with self._db.transaction():
            self._db.execute(
                """
                INSERT INTO memories
                    (id, content, summary, created_at, updated_at, source_model,
                     salience_model, salience_usage, salience_composite, content_hash,
                     source_path, ingestion_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, ?, ?, ?, ?)
                """,
                (memory_id, content, summary, now, now, model,
                 salience_model, salience_composite, content_hash,
                 source_path, ingestion_tier),
            )

            # Assign word tags
            tag_objects = self._tags.validate_tags(tags)

            # Split compound tags into constituent words
            split_words = self._tags.split_compound_tags(tags)
            split_tag_objects = self._tags.validate_tags(split_words) if split_words else []

            # Temporal date-period tags
            now_dt = datetime.utcnow()
            temporal_names = [
                f"month-{now_dt.strftime('%Y-%m')}",
                f"day-{now_dt.strftime('%Y-%m-%d')}",
                f"minute-{now_dt.strftime('%Y-%m-%dt%H-%M')}",
            ]
            temporal_tags = self._tags.validate_tags(temporal_names)

            # Batch insert all memory_tags in one executemany
            mt_params: list[tuple[str, int, str]] = []
            for tag in tag_objects:
                mt_params.append((memory_id, tag.id, "llm"))
            for tag in split_tag_objects:
                mt_params.append((memory_id, tag.id, "system"))
            for tag in temporal_tags:
                mt_params.append((memory_id, tag.id, "system"))
            if mt_params:
                self._db.executemany(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id, assigned_by) VALUES (?, ?, ?)",
                    mt_params,
                )

            # Derive and assign single-letter tags from word tags + split words
            letter_counts = self._tags.decompose_to_letters(tags + split_words)
            if letter_counts:
                self._tags.assign_letter_tags(memory_id, letter_counts)

        # Generate and cache embeddings (outside transaction — not critical).
        # Chunk long content so each chunk fits the embedding model's token limit.
        if not skip_embedding:
            chunks = chunk_text_for_embedding(content)
            if len(chunks) <= 1:
                embedding = self._embedding.embed(content)
                self._embedding.store_cached(memory_id, embedding)
            else:
                vecs = list(self._embedding.embed_batch(chunks))
                self._embedding.store_chunks(memory_id, vecs)

        tag_names = [t.name for t in tag_objects]
        return Memory(
            id=memory_id,
            content=content,
            summary=summary,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            source_model=model,
            salience_model=salience_model,
            salience_usage=0.0,
            salience_composite=salience_composite,
            tags=tag_names,
        )

    def get_by_id(self, memory_id: str) -> Memory:
        """Retrieve a memory by ID."""
        row = self._db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            raise MemoryNotFoundError(f"Memory {memory_id} not found")
        return self._row_to_memory(row)

    def retrieve_semantic(
        self,
        query: str,
        top_k: int | None = None,
        tag_filter: list[str] | None = None,
        query_vec=None,
    ) -> list[tuple[Memory, float]]:
        """Retrieve memories by semantic similarity.

        Uses a FAISS IVFFlat index for approximate nearest-neighbor search
        over chunk embeddings, then deduplicates by memory (keeping the
        best chunk score).  When a tag_filter is provided, results are
        post-filtered with oversampling to compensate.
        """
        k = top_k or self._config.max_retrieval_results
        if query_vec is None:
            query_vec = self._embedding.embed(query)

        # Tag-filtered path: over-fetch then post-filter
        if tag_filter:
            tag_memory_ids = self._get_memory_ids_by_tags(tag_filter)
            if not tag_memory_ids:
                return []
            # Over-fetch to account for filtered-out results
            candidates = self._embedding.search_similar(
                query_vec, top_k=k * 5,
                threshold=self._config.similarity_threshold,
            )
            filtered = [
                (mid, score) for mid, score in candidates
                if mid in tag_memory_ids
            ][:k]
        else:
            filtered = self._embedding.search_similar(
                query_vec, top_k=k,
                threshold=self._config.similarity_threshold,
            )

        if not filtered:
            return []

        top_ids = [mid for mid, _ in filtered]
        score_map = dict(filtered)
        memories = self._get_by_ids(top_ids)
        return [(memories[mid], score_map[mid]) for mid in top_ids if mid in memories]

    def retrieve_by_tags(
        self,
        tags: list[str],
        operator: str = "OR",
    ) -> list[Memory]:
        """Retrieve memories by tag names."""
        tag_objects = [self._tags.get_by_name(t) for t in tags]
        tag_ids = [t.id for t in tag_objects if t is not None]
        if not tag_ids:
            return []

        if operator == "AND":
            placeholders = ",".join("?" * len(tag_ids))
            rows = self._db.execute(
                f"""
                SELECT m.* FROM memories m
                JOIN memory_tags mt ON m.id = mt.memory_id
                WHERE mt.tag_id IN ({placeholders}) AND m.is_archived = 0
                GROUP BY m.id
                HAVING COUNT(DISTINCT mt.tag_id) = ?
                ORDER BY m.salience_composite DESC
                """,
                (*tag_ids, len(tag_ids)),
            ).fetchall()
        else:
            placeholders = ",".join("?" * len(tag_ids))
            rows = self._db.execute(
                f"""
                SELECT DISTINCT m.* FROM memories m
                JOIN memory_tags mt ON m.id = mt.memory_id
                WHERE mt.tag_id IN ({placeholders}) AND m.is_archived = 0
                ORDER BY m.salience_composite DESC
                """,
                tuple(tag_ids),
            ).fetchall()

        return self._rows_to_memories(rows)

    def retrieve_hybrid(
        self,
        query: str,
        tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[Memory, float]]:
        """Hybrid retrieval combining semantic similarity and tag matching."""
        import time as _time
        _t = {}
        _t0 = _time.perf_counter()

        k = top_k or self._config.max_retrieval_results
        w_sem = self._config.hybrid_semantic_weight
        w_tag = self._config.hybrid_tag_weight

        query_vec = self._embedding.embed(query)
        _t["embed"] = _time.perf_counter() - _t0

        _s = _time.perf_counter()
        semantic_results = self.retrieve_semantic(query, top_k=k * 3, query_vec=query_vec)
        _t["semantic_search"] = _time.perf_counter() - _s
        scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for mem, sim in semantic_results:
            sem_val = w_sem * sim
            scores[mem.id] = sem_val
            components[mem.id] = {"semantic": sem_val, "tag": 0.0, "lattice": 0.0, "glyph": 0.0}

        if tags:
            _s = _time.perf_counter()
            tag_semantic = self.retrieve_semantic(query, top_k=k, tag_filter=tags, query_vec=query_vec)
            _t["tag_search"] = _time.perf_counter() - _s
            for mem, sim in tag_semantic:
                tag_val = w_tag * sim
                scores[mem.id] = scores.get(mem.id, 0.0) + tag_val
                if mem.id not in components:
                    components[mem.id] = {"semantic": 0.0, "tag": 0.0, "lattice": 0.0, "glyph": 0.0}
                components[mem.id]["tag"] = tag_val

        if self._sync_engine and self._config.lattice_retrieval_boost_enabled:
            top5_ids = [mem.id for mem, _ in semantic_results[:5]]
            candidate_ids = [mem.id for mem, _ in semantic_results]

            _s = _time.perf_counter()
            # Single batch lookup: top5_ids ⊂ candidate_ids, so one query
            # gives us both maps. Old code paid the IN-clause cost twice.
            candidate_tag_map = self._tags.get_tag_ids_for_memories(candidate_ids)
            top5_set = set(top5_ids)
            context_tag_ids: set[int] = set()
            for mid, tids in candidate_tag_map.items():
                if mid in top5_set:
                    context_tag_ids.update(tids)
            _t["tag_id_lookup"] = _time.perf_counter() - _s

            if context_tag_ids:
                ctx_list = list(context_tag_ids)

                _s = _time.perf_counter()
                resonant = self._sync_engine.get_resonant_memory_ids_fast(
                    ctx_list,
                    candidate_memory_ids=candidate_ids,
                    candidate_tag_ids_map=candidate_tag_map,
                    seed_memory_ids=top5_ids,
                )
                _t["resonant_lookup"] = _time.perf_counter() - _s
                _t["resonant_count"] = len(resonant)
                _t["context_tag_count"] = len(ctx_list)

                w_lattice = self._config.lattice_retrieval_boost_weight
                for mid, strength in resonant:
                    scores[mid] = scores.get(mid, 0.0) + w_lattice * strength
                    if mid in components:
                        components[mid]["lattice"] = w_lattice * strength

                w_glyph_ret = self._config.glyph_retrieval_boost_weight
                if w_glyph_ret > 0:
                    _s = _time.perf_counter()
                    glyph_assoc = self._sync_engine.get_glyph_association_scores(
                        ctx_list,
                    )
                    _t["glyph_assoc"] = _time.perf_counter() - _s
                    _t["glyph_assoc_count"] = len(glyph_assoc)
                    for mid, strength in glyph_assoc:
                        scores[mid] = scores.get(mid, 0.0) + w_glyph_ret * strength
                        if mid in components:
                            components[mid]["glyph"] = w_glyph_ret * strength

                # ── High-level OUT→IN convergence re-ranking ──
                w_conv = self._config.semiotic_convergence_weight
                if w_conv > 0 and candidate_tag_map:
                    _s = _time.perf_counter()
                    conv_scores = (
                        self._sync_engine.semiotic_graph
                        .convergence_scores_batch(ctx_list, candidate_tag_map)
                    )
                    _t["semiotic_convergence"] = _time.perf_counter() - _s
                    _t["semiotic_convergence_count"] = len(conv_scores)
                    if conv_scores:
                        max_conv = max(conv_scores.values())
                        if max_conv > 0:
                            for mid, raw in conv_scores.items():
                                norm = raw / max_conv
                                scores[mid] = scores.get(mid, 0.0) + w_conv * norm
                                if mid in components:
                                    components[mid]["convergence"] = w_conv * norm

        _s = _time.perf_counter()
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        if not sorted_ids:
            _t["total"] = _time.perf_counter() - _t0
            import logging
            logging.getLogger("chicory.hybrid").info("hybrid_timing: %s", _t)
            self._last_hybrid_timing = _t
            self._last_score_components = {}
            return []
        top_ids = [mid for mid, _ in sorted_ids]
        score_map = dict(sorted_ids)
        memories = self._get_by_ids(top_ids)
        _t["load_memories"] = _time.perf_counter() - _s
        _t["total"] = _time.perf_counter() - _t0

        import logging
        logging.getLogger("chicory.hybrid").info("hybrid_timing: %s", _t)
        self._last_hybrid_timing = _t
        self._last_score_components = {
            mid: {**components.get(mid, {}), "total": score_map[mid]}
            for mid in top_ids if mid in memories
        }

        return [(memories[mid], score_map[mid]) for mid in top_ids if mid in memories]

    def list_recent(self, limit: int = 20) -> list[Memory]:
        """List most recent memories."""
        rows = self._db.execute(
            "SELECT * FROM memories WHERE is_archived = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return self._rows_to_memories(rows)

    def archive(self, memory_id: str) -> None:
        """Soft-delete a memory."""
        self._db.execute(
            "UPDATE memories SET is_archived = 1, updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), memory_id),
        )
        self._db.connection.commit()

    def count(self) -> int:
        """Count non-archived memories."""
        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0"
        ).fetchone()
        return row["cnt"]

    def _get_memory_ids_by_tags(self, tags: list[str]) -> set[str]:
        """Get memory IDs that have any of the given tags."""
        tag_objects = [self._tags.get_by_name(t) for t in tags]
        tag_ids = [t.id for t in tag_objects if t is not None]
        if not tag_ids:
            return set()
        placeholders = ",".join("?" * len(tag_ids))
        rows = self._db.execute(
            f"SELECT DISTINCT memory_id FROM memory_tags WHERE tag_id IN ({placeholders})",
            tuple(tag_ids),
        ).fetchall()
        return {r["memory_id"] for r in rows}

    def _get_by_ids(self, memory_ids: list[str]) -> dict[str, Memory]:
        """Batch-load memories by IDs with a single tag query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        tags_map = self._tags.get_tags_for_memories(memory_ids)
        result: dict[str, Memory] = {}
        for row in rows:
            mid = row["id"]
            result[mid] = self._row_to_memory_with_tags(row, tags_map.get(mid, []))
        return result

    def _rows_to_memories(self, rows) -> list[Memory]:
        """Convert multiple rows to Memory objects with a single batched tag query."""
        if not rows:
            return []
        memory_ids = [r["id"] for r in rows]
        tags_map = self._tags.get_tags_for_memories(memory_ids)
        return [
            self._row_to_memory_with_tags(row, tags_map.get(row["id"], []))
            for row in rows
        ]

    def _row_to_memory(self, row) -> Memory:
        tags = self._tags.get_tags_for_memory(row["id"])
        return self._row_to_memory_with_tags(row, tags)

    @staticmethod
    def _row_to_memory_with_tags(row, tags: list[str]) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            source_model=row["source_model"],
            salience_model=row["salience_model"],
            salience_usage=row["salience_usage"],
            salience_composite=row["salience_composite"],
            retrieval_success_count=row["retrieval_success_count"],
            retrieval_total_count=row["retrieval_total_count"],
            is_archived=bool(row["is_archived"]),
            tags=tags,
        )
