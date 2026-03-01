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
from chicory.layer3.synchronicity_engine import SynchronicityEngine
from chicory.layer4.adaptive_thresholds import AdaptiveThresholds
from chicory.layer4.feedback import FeedbackEngine
from chicory.layer4.meta_analyzer import MetaAnalyzer
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
        self._sync_engine = SynchronicityEngine(
            config, self._db, self._embedding_engine, self._tag_manager,
        )

        # Seed the tag relational tensor if empty
        self._maybe_seed_tensor()

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

        # Rate limiting for detection — initialise to "now" so the first
        # retrieval isn't blocked by expensive sync/meta detection.
        self._last_sync_check: datetime | None = datetime.utcnow()
        self._last_meta_check: datetime | None = datetime.utcnow()

    @property
    def db(self) -> DatabaseEngine:
        return self._db

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

    def close(self) -> None:
        """Close database connection."""
        self._db.close()

    # ── Tool call handlers ──────────────────────────────────────────

    def handle_store_memory(
        self,
        content: str,
        tags: list[str],
        importance: float | None = None,
        summary: str | None = None,
        skip_embedding: bool = False,
    ) -> dict[str, Any]:
        """Store a memory and fire side effects."""
        salience = importance if importance is not None else 0.5

        memory = self._memory_store.store(
            content=content,
            tags=tags,
            salience_model=salience,
            summary=summary,
            skip_embedding=skip_embedding,
        )

        # Side effects: record tag assignment events
        self._on_memory_stored(memory)

        return {
            "status": "stored",
            "memory_id": memory.id,
            "tags": memory.tags,
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
        self._on_retrieval_completed(retrieval_id, results)

        return {
            "results": [
                {
                    "memory_id": mem.id,
                    "content": mem.content,
                    "summary": mem.summary,
                    "tags": mem.tags,
                    "salience": mem.salience_composite,
                    "relevance_score": round(score, 3),
                    "created_at": mem.created_at.isoformat(),
                }
                for mem, score in results
            ],
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
            self._on_retrieval_completed(combined_id, all_retrieval_results)

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
        return {
            "memory_id": mem.id,
            "content": mem.content,
            "summary": mem.summary,
            "tags": mem.tags,
            "salience": mem.salience_composite,
            "relevance_score": round(score, 3),
            "created_at": mem.created_at.isoformat(),
            "depth": depth,
        }

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

    def _on_retrieval_completed(
        self,
        retrieval_id: int,
        results: list[tuple[Memory, float]],
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
            self._retrieval_bookkeeping(retrieval_id, results)
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
        """Run sync detection in a background thread."""
        try:
            self._maybe_run_sync_detection()
        except Exception:
            logging.getLogger(__name__).exception(
                "Error in background sync detection"
            )

    def _retrieval_bookkeeping(
        self,
        retrieval_id: int,
        results: list[tuple[Memory, float]],
    ) -> None:
        """Phase 1: cheap per-retrieval bookkeeping (batch SQL writes)."""
        if not results:
            return

        memory_ids = [mem.id for mem, _ in results]

        # Batch update access salience
        self._salience.update_on_access_batch(memory_ids)

        # Batch get tag IDs for all result memories
        tag_ids_map = self._tag_manager.get_tag_ids_for_memories(memory_ids)

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
