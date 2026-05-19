# Chicory Changelog — Performance & TagSpace

## Performance: 141s → 0.12s warm retrieval

A series of architectural changes that eliminated scattered I/O, dead-weight computation, and background thread contention. No schema changes — all improvements are in-memory and algorithmic.

### In-Memory Tensor Cache

New `TensorCache` class (`chicory/layer3/tensor_cache.py`) loads the entire `tag_relational_tensor` table (~74K rows, ~6MB) into Python dicts indexed by `tag_a_id` and `tag_b_id`. All tensor lookups during retrieval become dict operations (microseconds) instead of random SQLite page reads across a 10GB database (seconds).

- Lazy-loaded on first access, invalidated after tensor writes
- Wired into `SynchronicityEngine`, `SemioticDirectedGraph`, `TagSpace`, and `CentroidSubgraph`
- Replaces all per-query SQL against `tag_relational_tensor`

### Candidate-Scoped Post-FAISS Scoring

Post-FAISS re-ranking layers (glyph associations, resonant lookup, semiotic convergence) now score only the ~400 FAISS candidates instead of scanning the full `memory_tags` table:

- `get_glyph_association_scores` accepts `candidate_memory_ids` and `candidate_tag_ids_map` — resolves tags from the pre-fetched map instead of querying 54K memory-tag rows
- `get_resonant_memory_ids_fast` uses tensor cache for all pair lookups
- `signifies_batch` / `signified_by_batch` in the semiotic graph use tensor cache

### Centroid Fan-Out Removal

The post-FAISS centroid fan-out (`score_centroid`) scored 32K+ memories per query (0.25s) with none ever surfacing in results. Removed entirely — `centroid_match` still used by the orchestrator for query-side tag computation, but the expensive fan-out → resolve pipeline is gone.

### Background Thread Preemption

Background sync detection (`check_for_synchronicities`) could hold the foreground DB lock for minutes, blocking user queries. Fixed by:

1. Splitting background work into per-phase lock acquisitions
2. Adding `_foreground_waiting` event — background thread checks before each phase and yields
3. Releasing lock during expensive sync detection computation, re-acquiring for writes
4. Moving CPU-bound work (glyph analysis, embedding) outside the lock in `store_memory`

### FAISS Deduplication

`retrieve_hybrid` now calls `search_similar` once and reuses results for both unfiltered and tag-filtered channels, eliminating a redundant FAISS search per retrieval.

### Preloaded Tag Resolution

Final memory loading (`_get_by_ids`) accepts `preloaded_tag_ids` to resolve tag names from the already-fetched candidate tag map, avoiding a redundant `memory_tags` query.

### Timing Breakdown (warm query)

| Phase | Before | After |
|-------|--------|-------|
| Centroid fan-out | 250ms | removed |
| Resonant lookup | 43,000ms | 7ms |
| Glyph associations | 200ms | 3ms |
| Semiotic convergence | 5,400ms | 7ms |
| FAISS search | 2ms | 1ms |
| Embed | 95ms | 9ms |
| **Total search** | **~49,000ms** | **35ms** |
| **Total with side effects** | **~59,000ms** | **120ms** |

## Feature: TagSpace — Independent Tag-Graph Retrieval Path

New `TagSpace` class (`chicory/layer3/tag_space.py`) provides a retrieval path independent of FAISS embeddings, navigating tag relationships to find memories the embedding model might miss.

### Architecture

```
query text
     ├──→ lexical tag match → fan-out → inward ratio → memory scores  (instant)
     └──→ embed(query) → FAISS scores                                 (model inference)
                    └──→ centroid similarity → seed tags for hub filtering
```

### Lexical Match

Tokenizes the query and matches against an in-memory `{word: [tag_id]}` index built from active tags. Compound tags register under each component: "embedding-engine" → "embedding", "engine". Bigram compounds are also checked.

### Tensor Fan-Out

Expands seed tags through the tag relational tensor using the same weighted scoring as `_centroid_path`: co-occurrence, synchronicity, semantic, glyph weights with inhibition subtraction. Uses `TensorCache` for all lookups.

### Inward Ratio

Scores fan-out tags by two multiplied signals:
1. **Convergence**: fraction of seed tags that reach this tag (BFS on the search subgraph)
2. **Inward fraction**: seed edges / total degree — how much connectivity points toward the query

Tags need both multi-seed agreement and a non-trivial fraction of edges pointing inward.

### Hub Filtering

Context tags from FAISS top-5 results are filtered by seed edge count. Tags with degree exceeding `tag_space_max_context_seed_edges` (default 25) are pruned as hubs — they connect to everything and add noise to downstream re-ranking.

### Config additions

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `tag_space_enabled` | `True` | Master switch |
| `tag_space_weight` | `0.15` | Lexical path weight in hybrid scoring |
| `tag_space_fan_depth` | `1` | Tensor fan-out hops |
| `tag_space_lexical_min_tag_length` | `3` | Skip very short tags in lexical matching |
| `tag_space_min_inward_ratio` | `0.01` | Minimum inward ratio to keep a fan-out tag |
| `tag_space_centroid_similarity_threshold` | `0.4` | Cosine threshold for centroid-based tag matching |
| `tag_space_max_context_seed_edges` | `25` | Hub pruning: max edges from context tag to seeds |

## Refactor: Directional Canopy → Directional Flow

Directional canopy split from monolithic `DirectionalCanopy` class into lighter `InflowObserver` and `OutflowObserver` in `chicory/layer3/directional_flow.py`. Bridge optimizer config keys renamed:

| Old | New |
|-----|-----|
| `canopy_directional_inflow_rescue_weight` | `canopy_directional_inflow_centroid_boost` |
| `canopy_directional_outflow_rescue_weight` | `canopy_directional_outflow_pressure_weight` |

New config keys: `canopy_directional_ema_alpha`, `canopy_directional_inflow_context_gate_threshold`, `canopy_directional_inflow_pressure_weight`.

## Other Changes

- `episodic_co_retrieval_ema_alpha` config key added (default 0.2)
- `embedding_engine.py`: `store_chunks` method for pre-computed chunk embeddings
- `memory_store.py`: `precomputed_embeddings` parameter on `store()` to skip redundant embedding inside the lock
- Ingest pipeline: `ignore.py` for `.gitignore`-style path filtering

### Files changed

| File | Change |
|------|--------|
| `chicory/layer3/tensor_cache.py` | **New** — In-memory cache for `tag_relational_tensor` |
| `chicory/layer3/tag_space.py` | **New** — Independent tag-graph retrieval path |
| `chicory/layer3/directional_flow.py` | **New** — `InflowObserver`, `OutflowObserver` (replaces `DirectionalCanopy`) |
| `chicory/ingest/ignore.py` | **New** — `.gitignore`-style path filtering |
| `chicory/config.py` | TagSpace config, directional canopy config renames, episodic EMA alpha |
| `chicory/layer1/memory_store.py` | TagSpace integration, candidate-scoped scoring, precomputed embeddings, FAISS dedup |
| `chicory/layer1/embedding_engine.py` | `store_chunks`, `embed_batch` methods |
| `chicory/layer3/semiotic_graph.py` | Tensor cache integration for all batch methods |
| `chicory/layer3/synchronicity_engine.py` | Tensor cache for `_centroid_path`, `get_glyph_association_scores`, `get_inhibition_score` |
| `chicory/layer3/centroid_subgraph.py` | Tensor cache integration |
| `chicory/layer4/directional_canopy.py` | Gutted — delegates to `directional_flow.py` |
| `chicory/layer4/bridge_optimizer.py` | Directional score config renames |
| `chicory/layer4/canopy.py` | Minor cleanup |
| `chicory/layer4/episodic_tensor.py` | Co-retrieval EMA alpha from config |
| `chicory/layer4/meta_analyzer.py` | Minor cleanup |
| `chicory/orchestrator/orchestrator.py` | Tensor cache creation, TagSpace wiring, foreground preemption, lock-free CPU work |
| `chicory/db/engine.py` | Minor addition |
| `chicory/ingest/ingestor.py` | Ignore filter integration |
| `chicory/ingest/watcher.py` | Ignore filter integration |
| `chicory/models/canopy.py` | Minor addition |
