# Chicory Changelog — Schema v25→v26

## Performance

### 1. Resonance lookup: eliminate full `memory_tags` scan
**Files**: `synchronicity_engine.py`, `memory_store.py`

The old `get_resonant_memory_ids_fast` computed partner tag scores from the tensor, then did a full `memory_tags` scan (`SELECT memory_id, tag_id FROM memory_tags WHERE tag_id IN (...)`) to resolve partner tags → memories. With 700K+ rows in `memory_tags`, this returned 12,000+ memories per retrieval — over half the database.

**New approach**: The caller (`retrieve_hybrid`) passes in the FAISS candidate set and their tag maps. The centroid path returns raw partner tag scores, and candidates are scored in-memory by checking which partner tags they carry. No `memory_tags` query at all.

- `get_resonant_memory_ids_fast()` — new params: `candidate_memory_ids`, `candidate_tag_ids_map`
- `_centroid_path()` — now returns `dict[int, float]` (tag_id → score) instead of `dict[str, float]` (memory_id → score)
- Resonant count per retrieval: **12,715 → ~374**

### 2. Minimum association strength thresholds
**Files**: `config.py`, `synchronicity_engine.py`

Two new config knobs gate what passes through the resonance pipeline:

| Config | Default | Was | Purpose |
|--------|---------|-----|---------|
| `resonance_min_tensor_score` | 0.3 | 0.0 (any nonzero) | Minimum combined tensor strength for a partner tag |
| `resonance_min_edge_strength` | 0.3 | 0.01 | Minimum co-retrieval edge strength in centroid gate + resonance reweighting |

### 3. OR → UNION ALL on tensor queries
**File**: `synchronicity_engine.py`

Tensor queries in `_centroid_path()` and `get_glyph_association_scores()` rewritten from `WHERE tag_a_id IN (...) OR tag_b_id IN (...)` to UNION ALL, allowing SQLite to use indexes on both branches.

### 4. Batched glyph metadata
**File**: `orchestrator.py`

`_get_glyph_metadata_batch(memory_ids)` replaces per-result `_get_glyph_metadata(memory_id)`. One IN-clause query instead of N.

### 5. Single embed pass in hybrid retrieval
**File**: `memory_store.py`

`retrieve_semantic()` now accepts optional `query_vec`. `retrieve_hybrid()` embeds once and passes the vector to both the main semantic search and the tag-filtered search.

### 6. Deferred heavy retrieval work
**File**: `orchestrator.py`

Forest updates, canopy observations, episodic tensor edge activation, and episode lifecycle moved from synchronous `_retrieval_bookkeeping()` into `_run_deferred_retrieval_work()` (runs in the existing background thread). Response is no longer blocked by these operations.

### 7. Removed `_filter_relevant_tags` from bookkeeping
**File**: `orchestrator.py`

The centroid-similarity gating of tags before downstream bookkeeping was removed. It was doing a redundant embed call and filtering out valid tags.

## Architecture

### Two-pathway resonance model
**File**: `synchronicity_engine.py`

`get_resonant_memory_ids_fast` now combines two independent scoring pathways:

1. **Centroid path**: tag tensor → partner tags → score FAISS candidates by tag overlap (concept-level)
2. **Episodic path**: seed memories → `memory_relational_tensor` edges → neighbor memories (instance-level)

Scores are **additive** when a memory appears in both pathways, then normalized to [0, 1]. This rewards convergent association — a memory found through both concept relationships and direct memory-to-memory edges scores higher than one found through either alone.

## Bug Fixes / Correctness

### 1. EMA replaces additive accumulation
**File**: `centroid_subgraph.py`

Both `_apply_deltas_to_tensor` and `_apply_deltas_to_resonances` were using raw additive updates:
```python
new_val = max(0.0, old + net)  # unbounded growth
```

Over hundreds of retrievals, resonance strengths inflated from 0 to 100+ (range was 106–232 on 951K rows). Replaced with exponential moving average:
```python
signal = max(0.0, add_val - sub_val)
new_val = alpha * signal + (1 - alpha) * old  # alpha = 0.15
```

**Effect**: Values stay in a natural range. Active pairs stay elevated. Inactive pairs decay toward 0 as other pairs are retrieved. No more runaway accumulation.

### 2. Resonance strength normalization (migration v26)
**File**: `db/schema.py`

One-time min-max normalization of all `resonances.resonance_strength` values to [0, 1]. Combined with the EMA update, values now stay bounded going forward.

### 3. Junction table for resonance lookups
**Files**: `db/schema.py`, `synchronicity_detector.py`, `centroid_subgraph.py`

New `sync_event_tags` table (schema v25) replaces LIKE-based JSON string scanning for finding resonances by tag. The old approach (`WHERE involved_tags LIKE '%tag_id%'`) was both slow (no index) and incorrect (could match partial numbers).

```sql
CREATE TABLE sync_event_tags (
    event_id INTEGER NOT NULL REFERENCES synchronicity_events(id),
    tag_id   INTEGER NOT NULL REFERENCES tags(id),
    PRIMARY KEY (event_id, tag_id)
);
CREATE INDEX idx_sync_event_tags_tid ON sync_event_tags(tag_id);
```

Backfilled from existing `involved_tags` JSON. Forward population in `synchronicity_detector._store_event()`.

### 4. Column name fix in meta_analyzer
**File**: `meta_analyzer.py`

`_get_tag_clusters()` was querying `centroid_edges.strength` — column is actually `edge_strength`.

## Schema Changes

**v25**: `sync_event_tags` junction table + backfill from `involved_tags` JSON
**v26**: Min-max normalize `resonances.resonance_strength` to [0, 1]

`apply_schema()` handles auto-migration. The v25 backfill may take a few seconds on large databases.

## Config Changes

```python
# New in this release
resonance_min_tensor_score: float = 0.3   # min tensor score for partner tags
resonance_min_edge_strength: float = 0.3  # min co-retrieval edge strength
```

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| `retrieve_memories` total | ~155s | ~14s |
| Search (hybrid) | ~45s | ~7s |
| Side effects (bookkeeping) | ~105s | ~7s |
| Resonant memories returned | 12,715 | ~374 |
| Resonance reweight time | 85s | 5s |

## Migration Notes

1. Schema v26 auto-migrates. The v25 step backfills `sync_event_tags` — may take a few seconds on large DBs.
2. v26 normalizes resonance strengths to [0, 1]. **One-way** — old absolute values are lost.
3. `get_resonant_memory_ids_fast` signature changed — new params `candidate_memory_ids`, `candidate_tag_ids_map`. Internal callers updated; check any external callers.
4. Timing instrumentation (`_timing` key in response, log lines) is present for debugging and should be stripped for production.
