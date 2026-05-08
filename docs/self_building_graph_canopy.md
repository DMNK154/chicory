# Self-Building Graph Canopy

Status: design draft

This spec defines the canopy as an additional permanent layer on top of Chicory's existing database. It is not a promotion workflow, not a user-approval mechanism, and not a cross-project protocol. It is an always-on growth layer produced by the same storage, retrieval, trend, tensor, synchronicity, lattice, and meta-pattern processes that already operate on every memory.

The current clean model:

- Bridge supports a lifted relevance layer.
- Co-occurrence supports a lifted semantic-strength layer.
- Inhibition exists only between relevance and semantic-strength layers.
- A block's canopy threshold is the average inhibition across its relevance-to-semantics edges.
- Canopy growth happens where pressure exceeds that block threshold.

The central value is not "how much to suppress." It is the local amount of meaning-pressure required before this block grows upward.

## Goals

- Add a permanent graph-growth layer above the existing Chicory schema.
- Record canopy observations for all memories through the normal Chicory process.
- Treat inhibition as a canopy threshold, not as negative relevance.
- Keep bridge and co-occurrence as base substrates for lifted layers.
- Reorganize substrate indexes underneath the canopy without rewriting raw memories.
- Keep inhibition only between relevance and semantic-strength layers.
- Let the canopy recurse upward through new lifted layers and new cross-layer inhibition.
- Keep scope local to the current Chicory database for now.
- Never decay, prune, demote, or delete canopy growth because of age or inactivity.

## Non-Goals

- No special promotion step into the canopy.
- No `validated_by` or user approval path.
- No cross-project commons behavior in this design pass.
- No per-node influence weights.
- No moving, merging, or rewriting raw memories as part of substrate optimization.
- No time decay or age-based weakening in canopy tables.

## Layer Stack

Each local block `k` has two base substrates and two lifted layers.

```text
Bridge substrate        -> Relevance layer
Co-occurrence substrate -> Semantic-strength layer
```

### Bridge To Relevance

Bridge measures whether this block connects otherwise separated regions.

```text
B_k = bridge strength at block k
```

Relevance is lifted on top of bridge, with heat and recurrence as active-use signals:

```text
Rel_k =
  rB * B_k
+ rH * H_k
+ rR * R_k
```

Where:

- `B_k`: bridge strength
- `H_k`: heat
- `R_k`: recurrence

### Co-Occurrence To Semantic Strength

Co-occurrence measures repeated local togetherness.

```text
Co_k = co-occurrence strength at block k
```

Semantic strength is lifted on top of co-occurrence, with similarity as the semantic alignment signal:

```text
Sem_k =
  sCo * Co_k
+ sS  * S_k
```

Where:

- `Co_k`: co-occurrence strength
- `S_k`: similarity

### Local Pressure

The local upward pressure is the weighted combination of the lifted layers:

```text
P_k =
  wRel * Rel_k
+ wSem * Sem_k
```

Expanded form:

```text
P_k =
  wRel * (rB * B_k + rH * H_k + rR * R_k)
+ wSem * (sCo * Co_k + sS * S_k)
```

Suggested initial weights:

```text
rB = 0.50
rH = 0.25
rR = 0.25

sCo = 0.60
sS  = 0.40

wRel = 0.50
wSem = 0.50
```

Weights should be configuration, not per-node state.

## Substrate Reorganization

Chicory should not reorganize raw memories first. It should reorganize the substrate indexes underneath the canopy.

Raw terrain stays stable:

- `memories`
- `tags`
- `embeddings`
- `retrieval_events`
- `observations`

The database reorganizes by changing maps above that terrain:

- `cooccurrence_edges`
- `bridge_edges`
- `substrate_blocks`
- `block_memberships`
- `block_adjacency`
- `substrate_snapshots`

This preserves provenance. A canopy block can grow upward while Chicory can still explain exactly which memories, tags, retrievals, and events caused it.

### Co-Occurrence Optimizer

Co-occurrence asks:

```text
Which memories repeatedly activate together in the same context?
```

Use normalized association, not raw count, so common tags do not dominate.

Lift-style score:

```text
Co(a, b) = P(a, b) / (P(a) * P(b))
```

PMI-style score:

```text
PMI(a, b) = log(P(a, b) / (P(a) * P(b)))
```

The optimizer should:

1. Observe local activation scopes.
2. Update pair and small-set co-occurrence.
3. Normalize edges with lift or PMI.
4. Form local blocks from high-`Co` edges.
5. Split blocks with weak internal co-occurrence.
6. Keep stable co-occurring groups as the semantic substrate.

Block density:

```text
CoDensity(block) =
  average(Co(a, b) for pairs inside block)
```

High-density blocks become good bases for semantic strength. The co-occurrence layer asks where memory naturally clumps by lived adjacency, before semantic interpretation is applied.

### Bridge Optimizer

Bridge asks the opposite question:

```text
Which far-apart regions should remain reachable?
```

A bridge should be strong when two memories or blocks are semantically distant, cluster-distant, and repeatedly or meaningfully connected.

```text
Bridge(a, b) =
  connection_strength(a, b)
* cluster_distance(a, b)
* rarity_bonus(a, b)
```

Where:

- `connection_strength`: retrieval co-activation, shared sessions, shared tags, explicit links.
- `cluster_distance`: how far apart their co-occurrence neighborhoods are.
- `rarity_bonus`: reward for non-generic crossings.

Bridge optimization should prefer sparse, high-leverage crossings. It should not make everything connect to everything.

A useful local rule:

```text
BridgeValue(edge) =
  graph_distance_without_edge(a, b)
- graph_distance_with_edge(a, b)
```

A bridge is valuable when removing it makes two regions much farther apart.

### Compression And Expansion

The substrate optimizer balances two forces:

```text
Co-occurrence compresses local neighborhoods.
Bridge preserves global traversability.
```

Good organization:

```text
tight local neighborhoods
+ meaningful long-range crossings
- meaningless hub collapse
```

This prevents Chicory from becoming either isolated clusters or one overconnected semantic soup.

### Substrate Reorganizer

`SubstrateReorganizer` updates only `Co` and `B`. Semantic strength and recurrence read the organized substrate later; they do not organize the base.

Pseudo-flow:

```text
for each touched memory/tag/block:
    update co-occurrence counts from current activation scope
    update normalized co-occurrence edges
    update local co-occurrence blocks
    detect cluster boundaries
    update bridge candidates across separated blocks
    score bridge edges by distance, rarity, and connection evidence
    write substrate snapshot
    enqueue canopy observation
```

The mantra:

```text
Raw memories stay fixed.
Co-occurrence forms local blocks.
Bridge forms sparse crossings between blocks.
Semantic strength reads co-occurrence.
Relevance reads bridge.
Canopy grows where lifted pressure exceeds threshold.
```

## Cross-Layer Inhibition

Inhibition only goes between the relevance layer and the semantic-strength layer.

```text
Rel_i <-> Sem_j
```

These cross-layer edges define the local canopy threshold. They do not directly subtract from relevance or semantic strength.

For a cross-layer edge `e`:

```text
I_e = cross-layer inhibition on edge e
```

Simple edge seed:

```text
I_e =
  qBase
+ qGap      * abs(Rel_i - Sem_j)
+ qOpp      * local_opposition
+ qCollapse * collapse_risk
```

Where:

- `abs(Rel_i - Sem_j)` raises threshold when relevance and semantics disagree.
- `local_opposition` comes from tensor inhibition and antiparallelness.
- `collapse_risk` raises threshold when concepts are too likely to collapse prematurely.

The important rule is that `I_e` lives between the lifted layers. It is not a relevance score and not a semantic score.

## Block Threshold

A block's canopy threshold is the average of its cross-layer inhibition edges:

```text
Theta_block =
  mean(I_e for e in CrossLayerEdges(block))
```

If a block has no cross-layer inhibition edges yet, use a base threshold:

```text
Theta_block = theta_base
```

This is the value marked at the center of a block in the sketch: the canopy seed point.

It asks:

```text
Has enough relevance/semantic pressure accumulated here to grow upward?
```

## Canopy Growth

The local growth potential is:

```text
C_k = P_k - Theta_k
```

Interpretation:

```text
C_k > 0  -> canopy can grow upward here
C_k = 0  -> threshold boundary
C_k < 0  -> memory remains below canopy-growth level
```

Hard growth amount:

```text
CanopyGrowth_k = max(0, C_k)
```

Soft growth amount:

```text
CanopyGrowth_k = sigmoid(C_k / growth_temperature)
```

The canopy observer may still record an observation when `C_k <= 0`. Positive `C_k` means the block has enough accumulated pressure to sprout upward into a higher-order structure.

## Edge Channel Mixing

Cross-layer inhibition can reshape nearby edge channels. This is where the inhibition edge "speaks" to the block.

For a relevance/semantic edge `e` with inhibition `I_e`:

```text
H_edge  = clamp01(H_mix  + aH  * I_e)
S_edge  = clamp01(S_mix  + aS  * I_e)
Co_edge = clamp01(Co_mix + aCo * I_e)
R_edge  = clamp01(R_mix  + aR  * I_e)
B_edge  = clamp01(B_mix  + aB  * I_e)
```

The `a` coefficients define how relevance and semantics negotiate across that edge. They may be positive or negative.

Examples:

- `aH < 0`, `aB > 0`: inhibition lowers heat but raises bridge.
- `aR > 0`: a blocked pattern keeps returning, so recurrence rises.
- `aS < 0`: similarity is reduced so concepts do not collapse too quickly.
- `aCo > 0`: repeated blocked co-activation is preserved as a strong local pattern.

This means the edge is not just between memories. It is between canopy-growth conditions.

These mixed edge-channel scores are derived edge-local values. They must not be written back into the node, memory, tag, block, or base tensor scores that produced them.

Keep the layers distinct:

```text
Node/base scores:
  B, H, R, Co, S

Lifted block scores:
  Rel, Sem, P, Theta, C

Mixed edge-channel scores:
  H_edge, S_edge, Co_edge, R_edge, B_edge
```

Inhibition can speak across an edge without rewriting the memory field underneath it.

## Recursive Canopy

The canopy recurses upward. At every layer, bridge supports relevance, co-occurrence supports semantic strength, and cross-layer inhibition between relevance and semantics determines threshold.

Definitions:

```text
G_0 = existing Chicory memory graph
C_0 = canopy growth over G_0
C_n = canopy growth over C_(n-1), for n >= 1
```

At layer `n`, each block has:

```text
B_k_n
H_k_n
R_k_n
Co_k_n
S_k_n
Rel_k_n
Sem_k_n
Theta_k_n
```

Lifted layers:

```text
Rel_k_n =
  rB * B_k_n
+ rH * H_k_n
+ rR * R_k_n

Sem_k_n =
  sCo * Co_k_n
+ sS  * S_k_n
```

Pressure:

```text
P_k_n =
  wRel * Rel_k_n
+ wSem * Sem_k_n
```

Threshold:

```text
Theta_k_n =
  mean(I_e_n for e in CrossLayerEdges(block_k_n))
```

Growth potential:

```text
C_k_n = P_k_n - Theta_k_n
```

Positive growth at layer `n` can become a block at layer `n + 1`. That upper block receives its own bridge substrate, co-occurrence substrate, lifted relevance layer, lifted semantic-strength layer, and relevance-to-semantics inhibition edges.

### Recursive Boundaries

The recursion is conceptually unbounded but operationally bounded per pass.

Rules:

- Only recurse from canopy blocks touched by the current normal Chicory process.
- Include `layer_depth` in every recursive block key.
- Use a configurable `canopy_max_depth_per_pass` to avoid runaway loops.
- A depth cap is a compute budget, not decay, demotion, or deletion.
- Do not scan every possible canopy subset.
- Do not weaken lower layers when upper layers grow.

## Existing Signal Mapping

The canopy reads existing Chicory signals before adding new concepts:

| Signal | Existing Source | Canopy Use |
| --- | --- | --- |
| Bridge | cross-cluster links, lattice/glyph/meta resonance | `B`, relevance substrate |
| Heat | `TrendEngine`, phase coordinates | `H`, relevance lift |
| Recurrence | retrieval events, observations, sync reinforcement | `R`, relevance lift |
| Co-occurrence | memory tags, `tag_relational_tensor.cooccurrence_strength` | `Co`, semantic substrate |
| Similarity | embeddings, tag centroids, semantic tensor strength | `S`, semantic lift |
| Tensor inhibition | `tag_relational_tensor.inhibition_strength`, `parallelness` | cross-layer edge seed |
| Centroid subtraction | `CentroidSubgraph.update_on_retrieval` | cross-layer edge evidence |
| Synchronicity events | `synchronicity_events` | `R`, `B`, cross-layer edge evidence |
| Meta-patterns | `meta_patterns` | `R`, `B`, cross-layer edge evidence |

## Core Objects

### Canopy Observation

An append-only record of a memory, tag set, tensor pair, synchronicity event, resonance, meta-pattern, or upper-canopy block at the moment Chicory observed it.

Fields:

- `id`
- `observed_at`
- `source`: `store`, `retrieval`, `tensor`, `centroid`, `synchronicity`, `lattice`, `meta_pattern`, `feedback`, or `recursive_canopy`
- `source_id`: Optional id from the source table.
- `layer_depth`: `0` for base-memory observations, `1+` for canopy-over-canopy observations.
- `memory_ids`
- `tag_ids`
- `source_canopy_block_ids`: Lower-layer canopy blocks that produced this observation.
- `block_key`: Stable key for the aggregate canopy block.
- Score snapshot: `B`, `H`, `R`, `Co`, `S`, `Rel`, `Sem`, `P`, `Theta`, `C`, `CanopyGrowth`.

Observations are never updated or deleted by decay.

### Canopy Block

A permanent aggregate for an observed graph block or upper-canopy block.

Fields:

- `id`
- `block_key`: Stable hash of layer, block type, and sorted source ids.
- `block_type`: `memory_trace`, `tag_set`, `tag_pair`, `sync_shape`, `lattice_shape`, `meta_shape`, `canopy_pair`, or `canopy_shape`.
- `layer_depth`
- `tag_ids`
- `memory_ids`
- `parent_block_keys`: Lower-layer canopy block keys that support this block.
- `source_event_types`
- Peak base scores: `B`, `H`, `R`, `Co`, `S`
- Peak lifted scores: `Rel`, `Sem`
- `peak_pressure`
- `peak_threshold`
- `peak_growth_potential`
- `peak_canopy_growth`
- `evidence_count`
- `first_growth_at`: Set once when `C > 0`; never cleared.
- `created_at`
- `last_observed_at`

A canopy block is not promoted. It is created or updated by the same process that handles memories and graph events.

### Cross-Layer Inhibition Edge

A permanent edge between relevance and semantic-strength layers.

Fields:

- `relevance_block_id`
- `semantic_block_id`
- `edge_inhibition`
- `a_heat`
- `a_similarity`
- `a_cooccurrence`
- `a_recurrence`
- `a_bridge`
- `edge_heat`
- `edge_similarity`
- `edge_cooccurrence`
- `edge_recurrence`
- `edge_bridge`
- `evidence_count`
- `created_at`
- `last_observed_at`

This edge determines threshold. A block's `Theta` is the average of these edges for the block. The `edge_*` channel values are derived local scores for this edge only; they do not update `peak_*` block scores or any base memory/tensor values.

### Support Edge

A permanent relation from a canopy block to base graph objects or another canopy block.

Fields:

- `canopy_block_id`
- `target_type`: `memory`, `tag`, `sync_event`, `meta_pattern`, `lattice_position`, `resonance`, or `canopy_block`
- `target_id`
- `edge_type`: `supports`, `contains`, `bridges`, `opposes`, `stabilizes`, `derived_from`, or `canonicalizes_to`
- `strength`
- `evidence_count`
- `created_at`
- `last_observed_at`

## Growth Algorithm

### Inputs

Each canopy growth pass receives whatever the existing process just touched:

- Stored or updated memory ids.
- Activated tags and memories from retrieval.
- Tensor or centroid tag pairs updated by retrieval.
- Current phase coordinates for involved tags.
- Recent synchronicity events.
- Local meta-patterns.
- Lattice and glyph resonances.
- Lower-layer canopy blocks touched by this same process.

### Shape Generation

Generate shapes only from the existing process scope. Do not scan every possible tag subset.

Recommended shapes:

1. `memory_trace`: one shape for each stored or retrieved memory and its tags.
2. `tag_pair`: tag pairs already touched by tensor, centroid, retrieval, sync, or lattice processing.
3. `tag_set`: bounded tag sets from a memory, retrieval activation, or sync event.
4. `sync_shape`: tags and memories from one synchronicity event.
5. `lattice_shape`: tags and events from one resonance.
6. `meta_shape`: tags and sync events from one local meta-pattern.
7. `canopy_pair`: pairs of touched lower-layer canopy blocks.
8. `canopy_shape`: bounded sets of touched lower-layer canopy blocks.

Stable keys:

```text
block_key = hash(layer_depth + ":" + block_type + ":" + sorted(source_ids_or_tag_ids_or_parent_keys))
```

### Scoring Pass

For each generated shape:

1. Compute `B`, `H`, and `R`.
2. Lift bridge into relevance: `Rel`.
3. Compute `Co` and `S`.
4. Lift co-occurrence into semantic strength: `Sem`.
5. Compute local pressure `P`.
6. Upsert relevance-to-semantics inhibition edges touched by the shape.
7. Compute local threshold `Theta` as average cross-layer edge inhibition.
8. Derive edge-local channel scores with edge channel mixing.
9. Compute growth potential `C = P - Theta`.
10. Compute `CanopyGrowth = max(0, C)` or the sigmoid form.
11. Append a canopy observation.
12. Upsert the canopy block with monotonic aggregate values.
13. If `C > 0` and `first_growth_at` is null, set `first_growth_at`.
14. If `CanopyGrowth > 0`, enqueue the touched block for possible recursive growth.

### Pseudocode

```text
for shape in shapes_from_existing_process:
    bridge = compute_bridge(shape)              # B
    heat = compute_heat(shape)                  # H
    recurrence = compute_recurrence(shape)      # R
    relevance = lift_relevance(bridge, heat, recurrence)

    cooccurrence = compute_cooccurrence(shape)  # Co
    similarity = compute_similarity(shape)      # S
    semantics = lift_semantics(cooccurrence, similarity)

    pressure = wRel * relevance + wSem * semantics

    inhibition_edges = upsert_cross_layer_edges(shape, relevance, semantics)
    theta = mean(edge.edge_inhibition for edge in inhibition_edges)

    derived_edge_channels = derive_edge_channel_scores(inhibition_edges)
    store_edge_local_scores(inhibition_edges, derived_edge_channels)

    potential = pressure - theta
    growth = max(0, potential)

    observation = append_canopy_observation(
        shape,
        bridge, heat, recurrence,
        cooccurrence, similarity,
        relevance, semantics,
        pressure, theta, potential, growth,
    )

    block = upsert_canopy_block(shape.block_key)
    update_monotonic_block_aggregates(block, observation)

    if potential > 0 and block.first_growth_at is null:
        block.first_growth_at = observation.observed_at

    if growth > 0:
        enqueue_for_next_depth(block)

for depth in 1..canopy_max_depth_per_pass:
    recursive_shapes = shapes_from_touched_canopy_blocks(depth - 1)
    repeat same lifting, cross-layer inhibition, and growth process
```

The canopy grows on every pass. Positive growth adds upward structure. Negative growth potential is still informative because it records a block where pressure has not yet exceeded threshold.

## No Promotion

There is no special promotion into the canopy.

Instead:

- Every normal memory or graph event may create a canopy observation.
- Every observation maps to a stable canopy block key.
- The block aggregate is inserted if it does not exist.
- The block aggregate is updated if it already exists.
- If `C > 0`, `first_growth_at` is set.

This keeps the conceptual rule simple: the graph grows wherever lifted pressure exceeds cross-layer threshold.

## No User Approval

The canopy does not ask for approval before recording growth. User actions can still create memories, tags, retrievals, or feedback through existing Chicory mechanisms, but the canopy does not have a separate `validated_by` field and does not wait for confirmation.

## No Cross-Project Scope

This pass is local to the current Chicory database. Commons and cross-project federation can remain outside the canopy design until the local layer is stable.

## No Decay

The canopy never decays.

Rules:

- Do not time-decay canopy blocks.
- Do not time-decay cross-layer inhibition edges.
- Do not time-decay support edges.
- Do not subtract evidence because an observation is old.
- Do not remove a block because it is inactive.
- Do not clear `first_growth_at`.
- Do not lower peak scores because newer observations are cooler.
- Current heat may be computed from existing trend tables at query time, but that is a live view, not canopy decay.

If newer evidence contradicts older growth, record the contradiction as new growth. Do not erase the older structure.

## Storage Proposal

This can be implemented without replacing existing tables.

```sql
CREATE TABLE IF NOT EXISTS cooccurrence_edges (
    left_type TEXT NOT NULL,
    left_id TEXT NOT NULL,
    right_type TEXT NOT NULL,
    right_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    raw_count REAL NOT NULL DEFAULT 0.0,
    expected_count REAL NOT NULL DEFAULT 0.0,
    lift REAL NOT NULL DEFAULT 0.0,
    pmi REAL NOT NULL DEFAULT 0.0,
    co_strength REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (left_type, left_id, right_type, right_id, scope_type)
);
```

```sql
CREATE TABLE IF NOT EXISTS substrate_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_key TEXT NOT NULL UNIQUE,
    block_type TEXT NOT NULL,
    substrate_type TEXT NOT NULL, -- cooccurrence or bridge
    internal_density REAL NOT NULL DEFAULT 0.0,
    external_bridge_strength REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);
```

```sql
CREATE TABLE IF NOT EXISTS block_memberships (
    block_id INTEGER NOT NULL REFERENCES substrate_blocks(id) ON DELETE RESTRICT,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    membership_strength REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (block_id, target_type, target_id)
);
```

```sql
CREATE TABLE IF NOT EXISTS bridge_edges (
    left_block_id INTEGER NOT NULL REFERENCES substrate_blocks(id) ON DELETE RESTRICT,
    right_block_id INTEGER NOT NULL REFERENCES substrate_blocks(id) ON DELETE RESTRICT,
    connection_strength REAL NOT NULL DEFAULT 0.0,
    cluster_distance REAL NOT NULL DEFAULT 0.0,
    rarity_bonus REAL NOT NULL DEFAULT 0.0,
    bridge_strength REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (left_block_id, right_block_id),
    CHECK (left_block_id < right_block_id)
);
```

```sql
CREATE TABLE IF NOT EXISTS block_adjacency (
    left_block_id INTEGER NOT NULL REFERENCES substrate_blocks(id) ON DELETE RESTRICT,
    right_block_id INTEGER NOT NULL REFERENCES substrate_blocks(id) ON DELETE RESTRICT,
    adjacency_type TEXT NOT NULL,
    cooccurrence_weight REAL NOT NULL DEFAULT 0.0,
    bridge_weight REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (left_block_id, right_block_id, adjacency_type),
    CHECK (left_block_id < right_block_id)
);
```

```sql
CREATE TABLE IF NOT EXISTS substrate_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    trigger_type TEXT NOT NULL,
    trigger_id TEXT,
    touched_memory_ids TEXT NOT NULL DEFAULT '[]',
    touched_tag_ids TEXT NOT NULL DEFAULT '[]',
    touched_block_ids TEXT NOT NULL DEFAULT '[]',
    co_edge_count INTEGER NOT NULL DEFAULT 0,
    bridge_edge_count INTEGER NOT NULL DEFAULT 0,
    block_count INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);
```

```sql
CREATE TABLE IF NOT EXISTS canopy_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_key TEXT NOT NULL UNIQUE,
    block_type TEXT NOT NULL,
    layer_depth INTEGER NOT NULL DEFAULT 0,
    tag_ids TEXT NOT NULL DEFAULT '[]',
    memory_ids TEXT NOT NULL DEFAULT '[]',
    parent_block_keys TEXT NOT NULL DEFAULT '[]',
    source_event_types TEXT NOT NULL DEFAULT '[]',
    peak_bridge REAL NOT NULL DEFAULT 0.0,
    peak_heat REAL NOT NULL DEFAULT 0.0,
    peak_recurrence REAL NOT NULL DEFAULT 0.0,
    peak_cooccurrence REAL NOT NULL DEFAULT 0.0,
    peak_similarity REAL NOT NULL DEFAULT 0.0,
    peak_relevance REAL NOT NULL DEFAULT 0.0,
    peak_semantics REAL NOT NULL DEFAULT 0.0,
    peak_pressure REAL NOT NULL DEFAULT 0.0,
    peak_threshold REAL NOT NULL DEFAULT 0.0,
    peak_growth_potential REAL NOT NULL DEFAULT 0.0,
    peak_canopy_growth REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    first_growth_at TEXT,
    canonical_block_key TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);
```

```sql
CREATE TABLE IF NOT EXISTS canopy_cross_layer_edges (
    relevance_block_id INTEGER NOT NULL REFERENCES canopy_blocks(id) ON DELETE RESTRICT,
    semantic_block_id INTEGER NOT NULL REFERENCES canopy_blocks(id) ON DELETE RESTRICT,
    edge_inhibition REAL NOT NULL DEFAULT 0.0,
    a_heat REAL NOT NULL DEFAULT 0.0,
    a_similarity REAL NOT NULL DEFAULT 0.0,
    a_cooccurrence REAL NOT NULL DEFAULT 0.0,
    a_recurrence REAL NOT NULL DEFAULT 0.0,
    a_bridge REAL NOT NULL DEFAULT 0.0,
    edge_heat REAL NOT NULL DEFAULT 0.0,
    edge_similarity REAL NOT NULL DEFAULT 0.0,
    edge_cooccurrence REAL NOT NULL DEFAULT 0.0,
    edge_recurrence REAL NOT NULL DEFAULT 0.0,
    edge_bridge REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (relevance_block_id, semantic_block_id)
);
```

```sql
CREATE TABLE IF NOT EXISTS canopy_support_edges (
    canopy_block_id INTEGER NOT NULL REFERENCES canopy_blocks(id) ON DELETE RESTRICT,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    last_observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (canopy_block_id, target_type, target_id, edge_type)
);
```

```sql
CREATE TABLE IF NOT EXISTS canopy_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_key TEXT NOT NULL,
    observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    source TEXT NOT NULL,
    source_id TEXT,
    layer_depth INTEGER NOT NULL DEFAULT 0,
    tag_ids TEXT NOT NULL DEFAULT '[]',
    memory_ids TEXT NOT NULL DEFAULT '[]',
    source_canopy_block_ids TEXT NOT NULL DEFAULT '[]',
    bridge REAL NOT NULL,
    heat REAL NOT NULL,
    recurrence REAL NOT NULL,
    cooccurrence REAL NOT NULL,
    similarity REAL NOT NULL,
    relevance REAL NOT NULL,
    semantics REAL NOT NULL,
    pressure REAL NOT NULL,
    threshold REAL NOT NULL,
    growth_potential REAL NOT NULL,
    canopy_growth REAL NOT NULL
);
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_cooccurrence_edges_strength
    ON cooccurrence_edges(scope_type, co_strength DESC);

CREATE INDEX IF NOT EXISTS idx_cooccurrence_edges_left
    ON cooccurrence_edges(left_type, left_id);

CREATE INDEX IF NOT EXISTS idx_cooccurrence_edges_right
    ON cooccurrence_edges(right_type, right_id);

CREATE INDEX IF NOT EXISTS idx_substrate_blocks_type_density
    ON substrate_blocks(substrate_type, internal_density DESC);

CREATE INDEX IF NOT EXISTS idx_block_memberships_target
    ON block_memberships(target_type, target_id);

CREATE INDEX IF NOT EXISTS idx_bridge_edges_strength
    ON bridge_edges(bridge_strength DESC);

CREATE INDEX IF NOT EXISTS idx_block_adjacency_left
    ON block_adjacency(left_block_id);

CREATE INDEX IF NOT EXISTS idx_substrate_snapshots_time
    ON substrate_snapshots(snapshot_at DESC);

CREATE INDEX IF NOT EXISTS idx_canopy_blocks_type_growth
    ON canopy_blocks(layer_depth, block_type, peak_canopy_growth DESC);

CREATE INDEX IF NOT EXISTS idx_canopy_blocks_first_growth
    ON canopy_blocks(first_growth_at);

CREATE INDEX IF NOT EXISTS idx_canopy_blocks_last_observed
    ON canopy_blocks(last_observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_canopy_cross_layer_edges_inhibition
    ON canopy_cross_layer_edges(edge_inhibition DESC);

CREATE INDEX IF NOT EXISTS idx_canopy_support_edges_target
    ON canopy_support_edges(target_type, target_id);

CREATE INDEX IF NOT EXISTS idx_canopy_observations_key_time
    ON canopy_observations(block_key, observed_at DESC);
```

## Module Proposal

Minimal module set:

- `chicory/layer4/substrate.py`: substrate reorganizer coordinating co-occurrence and bridge optimizers.
- `chicory/layer4/cooccurrence_optimizer.py`: normalized co-occurrence edges and local block formation.
- `chicory/layer4/bridge_optimizer.py`: sparse high-leverage bridge edges between substrate blocks.
- `chicory/layer4/canopy.py`: always-on canopy observer, scorer, cross-layer inhibition mixer, and aggregate updater.
- `chicory/models/substrate.py`: Pydantic models for substrate edges, blocks, memberships, adjacency, and snapshots.
- `chicory/models/canopy.py`: Pydantic models for canopy blocks, cross-layer inhibition edges, support edges, observations, and score bundles.
- `chicory/db/schema.py`: schema migration for substrate and canopy tables.
- `chicory/orchestrator/orchestrator.py`: run substrate reorganization and canopy growth after store, retrieval, tensor/centroid updates, sync detection, meta-analysis, and bounded recursive canopy passes.
- `chicory/orchestrator/tool_handlers.py`: expose canopy inspection.
- `chicory/mcp/server.py`: optional `get_canopy` tool.

Recommended first implementation path:

1. Add models and tables.
2. Track activation scopes without changing raw memories.
3. Implement normalized co-occurrence edges.
4. Cluster high-`Co` local neighborhoods into substrate blocks.
5. Compute bridge candidates between separated substrate blocks.
6. Penalize generic hubs and store sparse high-value bridge edges.
7. Let canopy observations read only `B` and `Co` from the substrate tables.
8. Add semantic lifting from co-occurrence plus similarity.
9. Add relevance lifting from bridge plus heat/recurrence.
10. Add cross-layer inhibition edges and threshold averaging.
11. Add edge-local channel mixing with configurable coefficients.
12. Add recursive `canopy_pair` and `canopy_shape` observations over touched lower-layer blocks.
13. Add read-only canopy inspection.
14. Add retrieval contribution only after observation quality is easy to inspect.

## Safety Invariants

- The canopy is an additional layer on top of the existing database.
- The canopy does not require user approval.
- The canopy does not implement cross-project behavior in this pass.
- The canopy has no special promotion step.
- Every memory can participate through the same normal Chicory process.
- Raw memories, tags, embeddings, retrieval events, and observations stay append-only/stable.
- Substrate optimization changes maps, not terrain.
- Co-occurrence compresses local neighborhoods.
- Bridge preserves sparse global traversability.
- Generic hub collapse should be penalized.
- Semantic strength reads co-occurrence; it does not reorganize the base.
- Relevance reads bridge; it does not reorganize the base.
- Bridge is the substrate for relevance.
- Co-occurrence is the substrate for semantic strength.
- Inhibition only exists between relevance and semantic-strength layers.
- `Theta_block` is the average of cross-layer inhibition edges.
- Edge-channel mixing writes only derived `*_edge` scores to cross-layer edges.
- Edge-channel mixing must not mutate base node, memory, tag, block, or tensor scores.
- Threshold is not negative relevance.
- Canopy growth is `P - Theta`, not `P + inhibition`.
- Canopy observations are append-only.
- Block and edge aggregate evidence is monotonic.
- The canopy never decays.
- The canopy can recurse upward through new lifted layers and new cross-layer inhibition.
- A per-pass recursion depth cap is a compute bound only; it does not delete or weaken growth.
- Duplicate or equivalent blocks canonicalize by pointer; they are not deleted.

## Example

A memory arrives with tags around `inhibition`, `emergence`, `memory-network`, and `self-building-graph`.

The normal Chicory process stores the memory, assigns tags, updates embeddings and salience, and later retrieval may update tensor and centroid relationships. The canopy does not ask whether this deserves promotion. It records the observed local block.

```text
B  = 0.86
H  = 0.78
R  = 0.44
Co = 0.58
S  = 0.63

Rel = 0.50*B  + 0.25*H + 0.25*R
Rel = 0.735

Sem = 0.60*Co + 0.40*S
Sem = 0.600

P = 0.50*Rel + 0.50*Sem
P = 0.6675

Cross-layer edge inhibitions for this block:
I_1 = 0.52
I_2 = 0.61
I_3 = 0.57

Theta = mean(I_1, I_2, I_3)
Theta = 0.567

C = P - Theta
C = 0.1005

CanopyGrowth = max(0, C)
CanopyGrowth = 0.1005
```

The block crosses threshold, so Chicory sets `first_growth_at`. The relevance-to-semantics edges can then reshape local channels:

```text
H_edge  = clamp01(H_mix  + aH  * I_e)
S_edge  = clamp01(S_mix  + aS  * I_e)
Co_edge = clamp01(Co_mix + aCo * I_e)
R_edge  = clamp01(R_mix  + aR  * I_e)
B_edge  = clamp01(B_mix  + aB  * I_e)
```

If the resulting upper block has `P > Theta`, it grows into the next canopy layer.

Nothing is promoted. Nothing decays. The canopy grows where lifted relevance/semantic pressure exceeds the average cross-layer inhibition threshold.

## Open Questions

- Should co-occurrence normalization use lift, PMI, positive PMI, or a bounded hybrid?
- What activation scopes should count for co-occurrence: retrieval result sets, chat turns, sync events, ingest chunks, or sessions?
- What density threshold should split or merge substrate blocks?
- How should bridge rarity penalize generic hubs without hiding genuinely central concepts?
- Should bridge value use local betweenness approximation, removal-distance delta, or both?
- What should the default relevance lift weights be for `B`, `H`, and `R`?
- What should the default semantic lift weights be for `Co` and `S`?
- How should cross-layer edge inhibition be initialized from existing tensor inhibition?
- Should threshold use simple mean, weighted mean, geometric mean, or a robust average of cross-layer edges?
- Which blocks count as cross-layer neighbors: shared tags, shared memories, tensor adjacency, lattice proximity, or retrieval co-activation?
- Should hard growth `max(0, C)` or soft growth `sigmoid(C / tau)` be used first?
- What should the default `canopy_max_depth_per_pass` be?
