# Chicory

A self-organizing memory and associative network (MAN) for LLMs. Stores memories with semantic embeddings and tags, then discovers co-occurrence, semantic, asymmetric semiotic, and glyph-structural relationships through a Prime Ramsey Lattice projected onto a Poincaré disk. Each chicory MAN maintains a four-network tag relational tensor where each channel activates from a distinct evidence source — ingest-time co-occurrence, query-time co-retrieval, or structural resonance — with optional lateral inhibition and glyph-aware cross-referencing. Bootstraps from zero with no seed data. An episodic memory-to-memory tensor, forest block organizer, and canopy growth layer emerge from use. Association strengths are reweighted on every retrieval through centroid sub-graph dynamics — no passive time-decay, no arbitrary caps.

## Install

```bash
pip install chicory-man[mcp]
```

With additional providers and features:

```bash
# Anthropic + OpenAI LLM support
pip install chicory-man[all-llm,mcp]

# Web dashboard
pip install chicory-man[dashboard]

# Glyph system (ByT5 encoder for character-level lattice embeddings)
pip install chicory-man[glyph]

# REST API server
pip install chicory-man[api]

# Cross-project signal federation
pip install chicory-man[commons]

# Everything
pip install chicory-man[all-llm,mcp,dashboard,glyph,api,commons]
```

Or without any extras:

```bash
pip install chicory-man
```

## OpenAI / ChatGPT Setup

Chicory can use OpenAI's current flagship ChatGPT/API model through the OpenAI Responses API. Configure the OpenAI provider and model with environment variables:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:CHICORY_LLM_PROVIDER = "openai"
$env:CHICORY_LLM_MODEL = "gpt-5.5"
chicory chat
```

For a persistent `.env` setup:

```dotenv
OPENAI_API_KEY=sk-...
CHICORY_LLM_PROVIDER=openai
CHICORY_LLM_MODEL=gpt-5.5
```

## Grok / Gemini Setup

Grok and Gemini use the same `openai` Python package through each provider's OpenAI-compatible API. Pick one provider at a time with `CHICORY_LLM_PROVIDER`.

```dotenv
# xAI Grok
XAI_API_KEY=xai-...
CHICORY_LLM_PROVIDER=grok
CHICORY_GROK_MODEL=grok-4.20-reasoning
```

```dotenv
# Google Gemini
GEMINI_API_KEY=...
CHICORY_LLM_PROVIDER=gemini
CHICORY_GEMINI_MODEL=gemini-2.5-flash
```

## MCP Server Setup (Claude Code / Claude Desktop)

1. Set your API key and start the server:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
chicory-mcp
```

2. Add to your Claude Code config (`~/.claude.json`):

```json
{
  "mcpServers": {
    "chicory": {
      "type": "stdio",
      "command": "chicory-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use. The database is created at `~/.chicory/chicory.db` by default (override with `CHICORY_DB_PATH`).

### MCP Tools

The server exposes 12 tools:

| Tool | Purpose |
|------|---------|
| `store_memory` | Store a memory with tags and importance |
| `retrieve_memories` | Semantic/hybrid/tag search with tensor-boosted recall |
| `deep_retrieve` | Recursive retrieval following association chains into the deep past |
| `get_trends` | Tag trend signals: level, velocity, jerk, temperature |
| `get_phase_space` | Tag positions on the trend-temperature × retrieval-frequency plane |
| `get_synchronicities` | Detected synchronicity events with effective strength and reinforcement |
| `get_meta_patterns` | Higher-order recurring themes across synchronicity events |
| `get_lattice_resonances` | Prime Ramsey lattice: angular positions, resonances, void profile |
| `ingest_codebase` | Scan source files and store structural summaries (AST-based) |
| `ingest_documents` | Ingest full text content of files, chunked for retrieval |
| `process_signals` | Convert pending cross-project signals into commons memories |
| `get_commons_log` | Chronological history of federation activity |

## REST API & Docker

Chicory includes a FastAPI web server for browser-based demos with per-session isolated databases:

```bash
# Local
chicory-api

# Docker
docker compose up -d
```

The API server runs on port 8000 with password-based auth (set `CHICORY_PASSWORD` in `.env`).

## Quick Start (CLI)

```bash
# Interactive chat with your configured LLM + memory
chicory chat

# System status
chicory status

# Ingest a file or directory
chicory ingest path/to/docs/ --recursive

# Watch a directory for changes
chicory watch path/to/docs/

# Re-embed all memories with a new model
chicory reembed --model all-mpnet-base-v2

# Run a full model migration
chicory migrate

# Backfill single-letter composition tags
chicory backfill-letters

# Launch the web dashboard
chicory dashboard

# Commons federation commands
chicory commons --help
```

## Demo

```bash
# Run the architecture demo (no API keys, no model downloads, just numpy)
pip install chicory-man[dev]
python -m examples.ramsey_network_demo
```

The demo walks through 9 phases from blank slate to self-organizing memory in ~0.2 seconds, entirely in-memory with mock embeddings. Pass any file to demo document ingestion:

```bash
python -m examples.ramsey_network_demo path/to/any/document.txt
```

Supports `.txt`, `.md`, `.py`, `.json`, `.csv`, `.pdf`, `.docx`, and 30+ other formats.

## Architecture

```
                        store_memory()              retrieve_memories()
                             |                             |
                             v                             v
                  +---------------------+       +---------------------+
  Layer 1         |    Memory Store     |       |  Semantic / Hybrid  |
                  |  tags, embeddings,  |       |     Retrieval       |
                  |  salience scoring   |       |  (FAISS + tensor)   |
                  +---------------------+       +---------------------+
                             |                             |
                    tag assignment                  retrieval logged
                             v                             v
                  +---------------------+       +---------------------+
  Layer 2         |   Trend Engine      |       | Retrieval Tracker   |
                  | level, velocity,    |       | frequency, tag hits |
                  | jerk, temperature   |       |                     |
                  +---------------------+       +---------------------+
                             \                           /
                              \                         /
                               v                       v
                         +---------------------------+
  Layer 3                |      Phase Space          |
                         | (temperature, retrieval)  |
                         |  -> quadrant per tag      |
                         +---------------------------+
                                      |
                            statistical anomalies
                                      v
                         +---------------------------+
  Layer 3                | Synchronicity Detector    |
                         |  - dormant reactivation   |
                         |  - cross-domain bridges   |
                         |  - semantic convergence   |
                         +---------------------------+
                                      |
                           SynchronicityEvent objects
                                      v
                  +-------------------------------------------+
  Layer 3.5       |          Prime Ramsey Lattice             |
                  |  angle = PCA(tag_centroids) -> atan2      |
                  |  slot(p) = floor(angle * p / 2pi) % p    |
                  |  resonance = shared slots across primes   |
                  +-------------------------------------------+
                                      |
                              resonance pairs
                                      v
                  +-------------------------------------------+
  Layer 3.5       |      Tag Relational Tensor               |
                  |  4 independent networks per tag pair:     |
                  |    co-occurrence  (PMI, ingest+queries)   |
                  |    semantic       (Poincaré, queries only)|
                  |    semiotic       (P(B|A), ingest+queries)|
                  |    glyph          (resonance, structural) |
                  |  + lateral inhibition (parallelness gate) |
                  +-------------------------------------------+
                                      |
                           each retrieval triggers
                                      v
                  +-------------------------------------------+
  Layer 3.5       |      Centroid Sub-Graph Reweighting       |
                  |  tag centroids (EMA on store)             |
                  |  co-retrieval edges (EMA on retrieve)     |
                  |  add incoming + subtract inverted         |
                  |  parallelness-gated (orthogonal preserved)|
                  +-------------------------------------------+
                        |                           |
                        v                           v
           +------------------------+   +------------------------+
           |   Poincaré Disk        |   |   Glyph Ramsey Lattice |
           |  hyperbolic projection |   |  ByT5 character-level  |
           |  of lattice positions  |   |  structural resonance  |
           +------------------------+   +------------------------+
                                      |
                                      v
                  +-------------------------------------------+
  Layer 4         |          Episodic Tensor                   |
                  |  sparse memory-to-memory edge cache        |
                  |  gateways: semantic threshold, co-retrieval,|
                  |    same block, tag affinity threshold       |
                  |  lifecycle driven by composite strength     |
                  |  no arbitrary caps (super-hubs allowed)     |
                  +-------------------------------------------+
                                      |
                                      v
                  +-------------------------------------------+
  Layer 4         |    Forest (co-occurrence + bridge base)    |
                  |  co-occurrence optimizer: compress local    |
                  |  bridge optimizer: preserve traversability  |
                  +-------------------------------------------+
                                      |
                                      v
                  +-------------------------------------------+
  Layer 4.5       |    Canopy (emergent growth layer)          |
                  |  discovers memory clusters from episodic   |
                  |  co-activation + bridge edges               |
                  |  never decays — only grows                  |
                  +-------------------------------------------+
                                      |
                                      v
                  +-------------------------------------------+
  Layer 4         |  Meta-Analysis & Adaptive Thresholds      |
                  |  threshold EMA, burn-in, pattern clusters  |
                  +-------------------------------------------+
```

## The Four Tensor Networks

Each tag pair `(A, B)` carries four independent strength signals, each with a distinct activation story:

**Co-occurrence** (ingest + queries) — Pointwise Mutual Information computed from both memory co-occurrence and retrieval co-occurrence:
```
PMI_mem(A,B)  = log( P_mem(A,B)  / (P_mem(A)  * P_mem(B))  )   # ingest-time
PMI_ret(A,B)  = log( P_ret(A,B)  / (P_ret(A)  * P_ret(B))  )   # query-time
strength      = PMI_mem + PMI_ret
```
Memory PMI measures how *surprising* a co-occurrence is in the corpus. Retrieval PMI reinforces pairs that are actually used together in practice — tags that appear in the same query results accumulate evidence independent of how they were stored.

**Semantic** (queries only) — Poincaré geodesic distance between tag embedding centroids, gated by co-retrieval evidence. This channel starts dark and only activates for tag pairs that have been retrieved together at least once:
```
strength = exp(-hyperbolic_distance(centroid_A, centroid_B))
gate     = co_retrieval_count > 0
```
The co-retrieval gate means manual sync cannot create semantic connections that haven't been validated by actual use. This is the tensor's learning signal — it lights up as the system is used.

**Semiotic** (ingest + queries) — Directed associative strength via conditional probability from both memory and retrieval co-occurrence:
```
semiotic_forward  = P_mem(B|A) + P_ret(B|A)
semiotic_reverse  = P_mem(A|B) + P_ret(A|B)
```
These are naturally asymmetric. If "topology" appears in 1 retrieval and "mathematics" in 5, with 1 shared: P(mathematics|topology) = 1.0, but P(topology|mathematics) = 0.2. Memory co-occurrence provides the base directionality; retrieval patterns reinforce or shift the asymmetry through use.

**Glyph** (structural) — Lattice resonance strength from the glyph Ramsey lattice. Two tags resonate when they share slot positions across multiple prime scales despite being semantically distinct. Strength = `sum(log(p))` for shared primes, measuring information-theoretic surprise. This channel requires no usage evidence — it captures structural relationships invisible to embeddings.

All four networks are stored in a single `tag_relational_tensor` table with a `CHECK(tag_a_id < tag_b_id)` constraint. The semiotic layer encodes directionality as `semiotic_forward` (a→b) and `semiotic_reverse` (b→a) within this symmetric key.

### Channel Activation Summary

| Channel | Ingest (memory) | Queries (retrieval) | Structural |
|---------|:-:|:-:|:-:|
| Co-occurrence | PMI | PMI | — |
| Semantic | — | co-retrieval + Poincaré | — |
| Semiotic | P(B\|A) | P(B\|A) | — |
| Glyph | — | — | resonance |

### Lateral Inhibition

Tag pairs also carry `parallelness` and `inhibition_strength` fields. Antiparallel glyph pairs (pairs whose direction vectors in embedding space point in opposing directions) receive suppressive inhibition, preventing them from reinforcing each other. Orthogonal pairs pass through unaffected. This mirrors lateral inhibition in biological neural networks — nearby neurons suppress each other to sharpen contrast.

## Centroid Sub-Graph Reweighting

Synchronicity and resonance strengths are not passively decayed by time. Instead, every retrieval actively reweights existing scores based on what the system is actually doing:

**On store:** Tag centroids are updated via exponential moving average (EMA). Each tag's centroid is a unit-normalized running average of all memory embeddings assigned to it.

**On retrieval:** The system computes incoming association strengths for every pair of activated tags:

```
incoming(A,B) = cosine_sim(centroid_A, centroid_B) × mean_relevance × scale
```

These incoming strengths are then ranked highest to lowest and the values are inverted (reversed). Each pair receives:
- **Addition:** the raw incoming strength (boosting active associations)
- **Subtraction:** the inverted value × parallelness factor (suppressing redundant patterns)

The **parallelness factor** prevents collateral damage to independent associations. Each pair defines a direction vector in embedding space (`normalize(centroid_A - centroid_B)`). The subtraction is scaled by how geometrically aligned the pair is with any stronger pair:

```
parallelness(i) = max(|cos(direction_i, direction_j)|)  for all j ranking above i
```

- **Parallel weak pairs** (redundant with a stronger pattern): full subtraction — prevents rigidity
- **Orthogonal weak pairs** (independent patterns): near-zero subtraction — preserved
- **Dominant pair** (rank 0): no reference above it → zero subtraction → pure additive boost

The net effect: strongest incoming associations grow, weakest parallel ones shrink, and orthogonal associations pass through unharmed. Old connections that stop being actively retrieved lose strength as new retrievals erode them.

## Glyph System

The glyph system adds a parallel character-level lattice that captures structural relationships between tag *names* — relationships invisible to semantic embeddings.

**Glyph Lexicon** — A hardcoded mapping of ~80 concepts to Unicode symbols (e.g., Memory → ☍, Time → ⧖, Transformation → ⫚). Includes pair relationships (oppositions, transformations, co-occurrences) derived from GPT-GU's semiotic framework.

**Glyph Encoder** — A ByT5 (byte-level T5) encoder that produces d_model-dimensional embeddings of tag name strings at the character level. These are projected to 2D via PCA for angle computation on a separate glyph Ramsey lattice.

**Glyph Bridge** — Translates Chicory tag names to GPT-GU glyph symbols via the lexicon or ByT5 generation, then embeds the *glyph symbols* (not the English words) through ByT5. Lattice embeddings land in GPT-GU's symbolic space, where structural glyph relationships surface as Ramsey resonances.

**Glyph Analyzer** — Scans memory content for concept words and maps them to glyphs, detecting concept pairs (e.g., "X vs Y") for relational tagging. Glyph definitions are automatically injected into MCP tool responses.

When the glyph system is active, retrieval scores receive a configurable boost (`glyph_retrieval_boost_weight`) for memories whose tags share glyph associations.

## Poincaré Disk Projection

Lattice positions are optionally projected into the Poincaré disk model of hyperbolic space (H²). Angular coordinates come from PCA projection of embedding centroids; radial depth encodes hierarchical specificity derived from prime slot population density.

- Events with slots in many primes (general themes) sit near the disk center
- Events with slots in few primes (specific/niche topics) sit near the boundary
- Geodesic distance (arccosh metric) gives a natural measure of relatedness that respects the hierarchy
- Einstein midpoint computes weighted hyperbolic centroids for cluster analysis

## Three-Tier Architecture: Tag Tensor → Episodic Tensor → Canopy

The system maintains three tiers of relational structure at different granularities:

**Tag Tensor** (global, abstract) — The four-network tensor described above. Operates on tag pairs — there are relatively few tags (~thousands), so the tensor can track all resonant pairs. Answers: "How do concepts relate globally?"

**Episodic Tensor** (local, concrete) — A sparse memory-to-memory edge cache. Memories are many (~tens of thousands), so edges are materialized lazily through gateway rules:
1. **FAISS top-K**: semantic neighbors in embedding space
2. **Co-retrieval**: memories that appeared in the same query results
3. **Same block**: memories in the same forest block
4. **Tag affinity**: high projected affinity from the tag tensor

Each edge carries a composite strength = `semantic + tag_projected + co_retrieval + bridge`, where `tag_projected` is the mean of all four tensor channels minus inhibition. All thresholds are derived from the composite signal:

- **Gateway**: semantic neighbors above `similarity_threshold` (FAISS range search), plus co-retrieval, same-block, and tag affinity above `episodic_tag_affinity_threshold`
- **candidate → warm**: composite >= `episodic_tag_affinity_threshold`
- **warm → mature**: composite >= midpoint of the threshold and 1.0
- **mature → protected**: exempt from weak-edge pruning and decay

No arbitrary K or per-memory caps. Edge discovery, lifecycle promotion, and pruning are all driven by the same composite signal — the combined mathematical evidence from embeddings, tensor channels, co-retrieval history, and bridge structure. Heavily-connected memories form natural super-hubs. Edges that stop being activated decay and are eventually archived.

**Forest + Canopy** (emergent, structural) — Two layers that organize memories into blocks:

- **Forest** (base layer): The co-occurrence optimizer compresses local neighborhoods — memories that share many tags cluster into blocks. The bridge optimizer preserves global traversability by maintaining edges between blocks that would otherwise be isolated. Raw memories stay fixed; the forest reorganizes maps, not terrain.

- **Canopy** (growth layer): Discovers emergent memory clusters from episodic tensor co-activation and bridge edges. When memories are repeatedly retrieved together, the canopy observes connected components in the episodic graph, scores them by co-activation strength minus tag-overlap inhibition, and records growth. The canopy never decays, prunes, or deletes — it only grows. Tag-overlap inhibition ensures the canopy surfaces structure that tags alone don't predict.

The canopy's Ramsey adjacency filter (`canopy_ramsey_min_shared_primes`) gates tag-pair shape generation via glyph lattice prime-slot overlap, connecting the emergent graph structure back to the number-theoretic lattice.

## Cold Start

The system bootstraps from nothing:

```
t=0    All tables empty. PCA basis = random orthonormal scaffold.
       No seed data, no pre-trained weights.

t=1    First memories stored. Embeddings cached, tags created.
       PCA basis still random (< 20 embeddings for meaningful SVD).

t=N    Tensor self-seeds from ingest data:
       - co-occurrence: memory PMI (tags on same memories)
       - semiotic: memory conditional probability P(B|A)
       - glyph: structural resonance from Ramsey lattice
       Semantic channel remains dark (no retrievals yet).

t=N+k  Retrievals build co-retrieval edges and tag hit history.
       - semantic channel activates (co-retrieval gate satisfied)
       - co-occurrence and semiotic strengthened by retrieval PMI
       - query relevance filter focuses downstream analysis
       Synchronicity detection fires (rate-limited, 1/min).
       Events placed on lattice. All four channels operational.
```

Each instance starts with a unique random PCA basis — a random 2D plane through the embedding space — giving the lattice immediate angular diversity. Once 20+ real embeddings exist, SVD replaces the scaffold with a data-driven projection.

## Commons Layer

Cross-project signal federation via a shared SQLite database. Multiple Chicory instances (each a separate project) emit store/retrieve signals that are processed into commons memories with dual tags:

- **Project-namespaced:** `projectname:tagname` — tracks which project produced the signal
- **Shared:** `tagname` — enables cross-project pattern discovery

The commons layer itself is a full Chicory instance — all existing layers (trends, synchronicity detection, lattice, tensor) operate on the signal-derived data. This means cross-project synchronicities emerge naturally when unrelated projects start touching similar themes.

Optional glyph-aware federation adds heat tracking and synchronicity detection across project boundaries.

```bash
# Enable in .env
CHICORY_COMMONS_ENABLED=true
CHICORY_COMMONS_PROJECT_ID=my-project  # auto-detected from git repo name if omitted
```

## Project Structure

```
chicory/
  config.py                     Configuration (ChicoryConfig, 70+ parameters)
  exceptions.py                 Custom exception types
  api/
    app.py                      FastAPI REST server with session management
    sessions.py                 Per-user isolated database sessions
    user_db.py                  User authentication database
    static/                     Web demo frontend
  cli/
    app.py                      Typer CLI: chat, ingest, watch, status, migrate, dashboard
    chat.py                     Interactive chat session with LLM + memory
    commands.py                 Slash commands (/memories, /trends, /phase, /sync)
    commons.py                  Commons federation CLI subcommands
    conversation_log.py         Conversation history persistence
    display.py                  Rich terminal formatting
  db/
    engine.py                   SQLite with WAL mode, thread-safe RLock
    schema.py                   Schema v22, versioned migrations with idempotency
  ingest/
    parsers.py                  40+ file types (PDF, docx, code, markdown...)
    chunker.py                  Section/paragraph/sentence splitting
    ingestor.py                 File/directory ingestion with dedup (SHA256 content hash)
    document_ingestor.py        Full-text document ingestion with chunking
    code_summarizer.py          AST-based code structural summaries
    watcher.py                  Real-time directory monitoring (watchdog)
  layer1/                       Memory store, embeddings, salience, tags, FAISS
    memory_store.py             Core memory CRUD with embedding and salience
    embedding_engine.py         Sentence-transformer embeddings + FAISS index
    tag_manager.py              Tag assignment, letter decomposition, normalization
    salience.py                 Multi-tier salience scoring
    vector_index.py             FAISS index management
    glyph_encoder.py            ByT5 character-level encoder for glyph lattice
    glyph_bridge.py             GPT-GU glyph symbol translation layer
    glyph_lexicon.py            Hardcoded concept↔symbol mappings (~80 glyphs)
    glyph_analyzer.py           Content scanning for concept words → glyph mapping
  layer2/                       Trend engine, retrieval tracker, time series
    trend_engine.py             Level, velocity, jerk, temperature per tag
    retrieval_tracker.py        Retrieval frequency and tag hit logging
    time_series.py              Time series snapshot storage
  layer3/                       Phase space, synchronicity, lattice, tensor, centroid, Poincaré
    phase_space.py              Temperature × retrieval frequency quadrant placement
    synchronicity_detector.py   Dormant reactivation, cross-domain bridges, semantic convergence
    synchronicity_engine.py     Prime Ramsey lattice, tensor, resonance computation
    centroid_subgraph.py        Retrieval-driven active reweighting (EMA centroids + edges)
    chain_anisotropy.py         Directional chain analysis in the tensor graph
    cross_project_alignment.py  Cross-project signal alignment
    poincare.py                 Poincaré disk projection (exponential map, geodesics, Einstein midpoint)
    square_finder.py            Ramsey square detection in the tensor
  layer4/                       Episodic tensor, forest, canopy, meta-analysis, feedback
    episodic_tensor.py          Sparse memory-to-memory edge cache (~48 neighbors/memory)
    forest.py                   Forest reorganizer (co-occurrence + bridge optimizers)
    cooccurrence_optimizer.py   Local neighborhood compression into blocks
    bridge_optimizer.py         Global traversability preservation between blocks
    canopy.py                   Emergent memory cluster growth from co-activation
    adaptive_thresholds.py      EMA-based threshold tuning with burn-in
    meta_analyzer.py            Higher-order pattern detection across synchronicity events
    feedback.py                 User feedback integration
  llm/                          Multi-provider LLM abstraction
    base.py                     BaseLLMClient interface
    client.py                   Anthropic Claude client
    openai_client.py            OpenAI client
    null_client.py              No-op client (MCP-only / embedding-only mode)
    factory.py                  Auto-detection: Anthropic → OpenAI → Null
    prompts.py                  System prompt construction
    types.py                    Shared LLM types
  models/                       Pydantic models
    memory.py                   Memory, Embedding
    lattice.py                  LatticePosition, Resonance, PoincaréCoordinates
    synchronicity.py            SynchronicityEvent
    trend.py                    TrendSnapshot
    phase.py                    PhaseQuadrant
    meta_pattern.py             MetaPattern
    retrieval.py                RetrievalResult
    canopy.py                   CanopyShape, ScoreBundle
  orchestrator/                 Central coordinator
    orchestrator.py             Ties all layers together, background detection thread
    tool_handlers.py            Tool dispatch for MCP and CLI
  mcp/
    server.py                   FastMCP server exposing 12 tools (stdio transport)
  migration/                    Model migration utilities
  dashboard/                    Optional Dash web UI (plotly + networkx)
tests/
  conftest.py                   MockEmbeddingEngine (deterministic, no GPU)
  test_integration/             Full-cycle tests
  test_layer3/                  Lattice, tensor, semiotic layer tests
  test_llm/                     LLM client tests
  test_glyph_analyzer.py        Glyph text analysis tests
examples/
  ramsey_network_demo.py        Self-contained demo (blank slate → self-organizing)
```

## Runtime Loop

```
User sends message
  -> LLM generates tool calls
     -> store_memory(content, tags)
        -> embed, assign tags, record trend events
        -> update tag centroids (EMA)
        -> glyph analysis: scan content for concept words
        -> forest reorganization (co-occurrence + bridge)
        -> canopy observation (if block touched)
        -> invalidate PCA cache

     -> retrieve_memories(query)
        -> FAISS similarity search + tensor-boosted recall
        -> glyph association boost (if glyph system active)
        -> log retrieval
        -> query relevance filter:
           -> embed query, load centroids for result tags
           -> keep only tags above similarity_threshold
           -> filtered tags used for all downstream analysis
        -> record tag hits (filtered)
        -> centroid sub-graph reweighting (filtered):
           -> record co-retrieval edges (EMA)
           -> compute incoming association strengths (C @ C.T)
           -> rank, invert, scale by parallelness
           -> add/subtract to tensor + resonance strengths
        -> activate episodic edges for co-retrieved memory pairs
        -> forest reorganization + canopy observation
        -> [background thread]:
           -> update salience on access
           -> record trend events (filtered)
           -> reinforce synchronicity events
           -> _maybe_run_sync_detection (rate-limited)
              -> check_for_synchronicities()
                 -> detect dormant reactivation (phase space anomaly)
                 -> detect cross-domain bridges (resonance-gated)
                 -> detect semantic convergence (embedding similarity, no shared tags)
              -> place_events_batch(new_events)
                 -> compute angle (PCA -> atan2)
                 -> compute prime slots (150 primes, up to 863)
                 -> project to Poincaré disk (if enabled)
                 -> _update_synchronicity_tensor (O(n) against existing positions)
                    -> INSERT resonances
                    -> UPSERT tensor synchronicity_strength
                    -> lateral inhibition pass
```

## Configuration

Key parameters in `ChicoryConfig` (70+ total, all overridable via environment variables or `.env`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `llm_provider` | `auto` | LLM provider: `auto`, `anthropic`, `openai`, `grok`, `gemini`, `null` |
| `llm_model` | `gpt-5.5` | Generic LLM model override |
| `grok_model` | `grok-4.20-reasoning` | Default xAI Grok model |
| `gemini_model` | `gemini-2.5-flash` | Default Gemini model |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `embedding_dimension` | `384` | Embedding vector dimension |
| `lattice_primes` | `[2, 3, ..., 863]` | 150 primes defining lattice scales |
| `lattice_min_resonance_primes` | `30` | Minimum shared slots for resonance |
| `tensor_cooccurrence_weight` | `0.5` | PMI weight in recall scoring |
| `tensor_synchronicity_weight` | `0.3` | Lattice weight in recall scoring |
| `tensor_semantic_weight` | `0.2` | Cosine weight in recall scoring |
| `tensor_semiotic_weight` | `0.15` | Conditional probability weight |
| `tensor_inhibition_weight` | `0.20` | Max lateral inhibition suppression |
| `sync_detection_sigma` | `2.0` | Z-score threshold for anomaly detection |
| `centroid_ema_alpha` | `0.1` | EMA weight for tag centroid updates |
| `centroid_edge_ema_alpha` | `0.15` | EMA weight for co-retrieval edges |
| `centroid_inhibition_scale` | `0.5` | Reweighting intensity per retrieval cycle |
| `poincare_curvature` | `1.0` | Poincaré disk curvature (c > 0, sectional = -c) |
| `poincare_max_radius` | `0.95` | Clamp to open disk boundary |
| `glyph_retrieval_boost_weight` | `0.10` | Glyph association retrieval boost |
| `canopy_enabled` | `true` | Enable forest/canopy/episodic tensor |
| `episodic_tag_affinity_threshold` | `0.15` | Composite strength for candidate→warm promotion |
| `canopy_ramsey_min_shared_primes` | `7` | Glyph prime overlap for canopy shape generation |
| `commons_enabled` | `false` | Enable cross-project signal federation |

## Tests

```bash
pytest tests/ -v                                    # Full suite
pytest tests/test_layer3/ -v                        # Lattice, tensor, centroid tests
pytest tests/test_integration/ -v                   # Full-cycle integration tests
pytest tests/test_llm/ -v                           # LLM client tests
pytest tests/test_glyph_analyzer.py -v              # Glyph analysis tests
```

## Dependencies

**Runtime:** pydantic, sentence-transformers, numpy, faiss-cpu, scipy, typer, rich, python-dotenv, pymupdf, python-docx, watchdog

**Optional:** anthropic (Anthropic LLM), openai (OpenAI LLM), mcp (MCP server), fastapi + uvicorn (REST API), dash + plotly + networkx (dashboard), transformers + torch (glyph ByT5 encoder), chicory-commons-man (federation)

**Dev:** pytest, pytest-asyncio

**Demo only:** numpy (mock embeddings, no model downloads)

## Database

SQLite with WAL mode. Schema v22, versioned migrations with idempotency checks. Thread-safe via `threading.RLock` on all execute/executemany calls.

```bash
chicory status          # Show memory count, tag count, tensor state
chicory migrate         # Run model migration with burn-in period
chicory reembed         # Re-embed all memories with new model
```
