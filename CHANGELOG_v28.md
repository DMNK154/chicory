# Chicory Changelog — Schema v27→v28

## Feature: Directional Canopy (Inflow + Outflow)

Two new canopy layers that track directional signal flow through retrieval causality, feeding back into the bridge optimizer's hub penalty.

**Direction**: Query tags are SOURCE (outflow), result memories are DESTINATION (inflow). Every retrieval event has a natural arrow: query → results.

### Inflow Canopy (memory-space)
Tracks result memory clusters that are **convergence attractors** — clusters that get pulled in by diverse, unrelated queries. A cluster reached by 30 different query contexts scores higher than one repeatedly triggered by the same query.

**Scoring**:
```
inflow_diversity = 1 - mean_pairwise_jaccard(all query_tag_sets reaching this cluster)
sample_correction = 1 - 1/sqrt(unique_query_contexts + 1)
inflow_strength = (w_diversity * diversity + w_frequency * freq_factor) * correction
```

### Outflow Canopy (tag-space)
Tracks query tag neighborhoods that are **distributors** — they trigger retrievals reaching diverse, distant result clusters. Tags that reach across many different result neighborhoods score higher.

**Scoring**:
```
outflow_diversity = 1 - mean_pairwise_jaccard(all result_memory_sets from this query context)
outflow_reach = mean cluster_distance between distinct result clusters
outflow_strength = (w_diversity * diversity + w_reach * reach) * correction
```

### Bridge Hub Penalty Modification
The bridge optimizer's `_hub_penalty` now consumes directional scores. Genuine attractors (high inflow) and genuine distributors (high outflow) get partial rescue from the degree penalty. Generic hubs with high degree but no directional signal get full penalty.

```
inflow_rescue = 1.0 + inflow_rescue_weight * max(inflow_a, inflow_b)
outflow_rescue = 1.0 + outflow_rescue_weight * max(outflow_a, outflow_b)
hub_penalty = degree_penalty * inflow_rescue * outflow_rescue
```

### Schema v28: 6 new tables

| Table | Purpose |
|-------|---------|
| `directional_retrieval_context` | Records query→result direction per retrieval (query_tag_ids, result_memory_ids, query_tag_hash) |
| `inflow_canopy_blocks` | Convergence attractor clusters with diversity/strength scores |
| `inflow_canopy_observations` | Append-only log of each inflow observation |
| `outflow_canopy_blocks` | Distributor tag neighborhoods with diversity/reach/strength scores |
| `outflow_canopy_observations` | Append-only log of each outflow observation |
| `directional_block_scores` | Per-forest-block inflow/outflow scores consumed by bridge optimizer |

### Config additions

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `canopy_directional_enabled` | `True` | Master switch |
| `canopy_directional_inflow_diversity_weight` | `0.7` | Weight of query diversity in inflow score |
| `canopy_directional_inflow_frequency_weight` | `0.3` | Weight of activation frequency in inflow score |
| `canopy_directional_outflow_diversity_weight` | `0.6` | Weight of result diversity in outflow score |
| `canopy_directional_outflow_reach_weight` | `0.4` | Weight of cluster distance in outflow score |
| `canopy_directional_inflow_rescue_weight` | `0.3` | How much inflow rescues hub penalty |
| `canopy_directional_outflow_rescue_weight` | `0.3` | How much outflow rescues hub penalty |
| `canopy_directional_query_tag_similarity_threshold` | `0.3` | Centroid cosine threshold for query-side tag detection |

### Files changed

| File | Change |
|------|--------|
| `chicory/layer4/directional_canopy.py` | **New** — `DirectionalCanopy`, `InflowObserver`, `OutflowObserver` |
| `chicory/db/schema.py` | 6 new tables, migration v27→v28 |
| `chicory/config.py` | 8 new config params |
| `chicory/models/canopy.py` | `InflowScore`, `OutflowScore` models |
| `chicory/layer4/bridge_optimizer.py` | `_hub_penalty` + `_load_directional_scores` modified |
| `chicory/orchestrator/orchestrator.py` | Query-side tag computation, directional canopy wiring |
