# Chicory

A self-organizing memory system for LLMs. Stores memories with semantic embeddings and tags, detects cross-domain synchronicities through a Prime Ramsey Lattice, and maintains a four-network tag relational tensor that bootstraps from zero with no seed data.

## Install

```bash
pip install chicory-man[mcp]
```

Or without MCP server support:

```bash
pip install chicory-man
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

## Quick Start (CLI)

```bash
# Interactive chat with Claude + memory
export ANTHROPIC_API_KEY=sk-ant-...
chicory chat

# System status
chicory status

# Ingest a file or directory
chicory ingest path/to/docs/ --recursive
```

## Demo

```bash
# Run the architecture demo (no API keys, no model downloads, just numpy)
pip install chicory[dev]
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
                  |    co-occurrence  (PMI, symmetric)        |
                  |    semantic       (cosine, symmetric)     |
                  |    semiotic       (P(B|A), asymmetric)    |
                  |    synchronicity  (lattice, symmetric)    |
                  +-------------------------------------------+
                                      |
                                      v
                  +-------------------------------------------+
  Layer 4         |  Meta-Analysis & Adaptive Thresholds      |
                  |  threshold EMA, burn-in, pattern clusters  |
                  +-------------------------------------------+
```

## The Four Tensor Networks

Each tag pair `(A, B)` carries four independent strength signals:

**Co-occurrence** — Pointwise Mutual Information from `memory_tags`:
```
PMI(A,B) = log( P(A,B) / (P(A) * P(B)) )
```
Measures how *surprising* a co-occurrence is. Two tags that always appear together on rare memories score higher than two tags that appear everywhere.

**Semantic** — Cosine similarity between tag embedding centroids. Each tag's centroid is the mean embedding of all memories with that tag.

**Semiotic** — Directed associative strength via conditional probability:
```
semiotic_forward  = P(B|A) = co_count / count(A)
semiotic_reverse  = P(A|B) = co_count / count(B)
```
These are naturally asymmetric. If "topology" appears on 1 memory and "mathematics" appears on 5, with 1 shared: P(mathematics|topology) = 1.0, but P(topology|mathematics) = 0.2. Topology strongly predicts mathematics, not vice versa. At query time, the system picks the right direction based on which tag is in the query context.

**Synchronicity** — Lattice resonance strength. Two events resonate when they share slot positions across multiple prime scales. Strength = `sum(log(p))` for shared primes, measuring information-theoretic surprise.

All four networks are stored in a single `tag_relational_tensor` table with a `CHECK(tag_a_id < tag_b_id)` constraint. The semiotic layer encodes directionality as `semiotic_forward` (a->b) and `semiotic_reverse` (b->a) within this symmetric key.

## Cold Start

The system bootstraps from nothing:

```
t=0    All tables empty. PCA basis = random orthonormal scaffold.
       No seed data, no pre-trained weights.

t=1    First memories stored. Embeddings cached, tags created.
       PCA basis still random (< 20 embeddings for meaningful SVD).

t=N    Tensor self-seeds: PMI, cosine, and conditional probability
       computed from memory_tags. Three of four networks populated.

t=N+k  Retrievals trigger synchronicity detection (rate-limited, 1/min).
       Events placed on lattice. Fourth network (synchronicity) activates.
       Resonances persisted. Tensor fully operational.
```

Each instance starts with a unique random PCA basis — a random 2D plane through the embedding space — giving the lattice immediate angular diversity. Once 20+ real embeddings exist, SVD replaces the scaffold with a data-driven projection.

## Project Structure

```
chicory/
  config.py                     Configuration (ChicoryConfig, 50+ parameters)
  exceptions.py                 Custom exception types
  cli/
    app.py                      Typer CLI: chat, ingest, watch, status, migrate
    chat.py                     Interactive chat session with Claude
    commands.py                 Slash commands (/memories, /trends, /phase, /sync)
  db/
    engine.py                   SQLite with WAL mode, thread-safe RLock
    schema.py                   Schema v9, 14 tables, versioned migrations
  ingest/
    parsers.py                  40+ file types (PDF, docx, code, markdown...)
    chunker.py                  Section/paragraph/sentence splitting
    ingestor.py                 File/directory ingestion with dedup
    watcher.py                  Real-time directory monitoring
  layer1/                       Memory store, embeddings, salience, tags, FAISS
  layer2/                       Trend engine, retrieval tracker, time series
  layer3/                       Phase space, synchronicity detection, lattice, tensor
  layer4/                       Adaptive thresholds, meta-analysis, feedback
  llm/                          Claude API client, tool definitions
  models/                       Pydantic models (Memory, Tag, SynchronicityEvent, ...)
  orchestrator/                 Central coordinator, tool dispatch
  dashboard/                    Optional Dash web UI
tests/
  conftest.py                   MockEmbeddingEngine (deterministic, no GPU)
  test_integration/             Full-cycle tests
  test_layer3/                  Lattice, tensor, semiotic layer tests
examples/
  ramsey_network_demo.py        Self-contained demo (blank slate -> self-organizing)
```

## Runtime Loop

```
User sends message
  -> Claude generates tool calls
     -> store_memory(content, tags)
        -> embed, assign tags, record trend events
        -> invalidate PCA cache

     -> retrieve_memories(query)
        -> FAISS similarity search + tensor-boosted recall
        -> log retrieval, record tag hits
        -> [background thread] _on_retrieval_completed_async:
           -> update salience on access
           -> record trend events
           -> reinforce synchronicity events
           -> _maybe_run_sync_detection (rate-limited)
              -> check_for_synchronicities()
                 -> detect dormant reactivation (phase space anomaly)
                 -> detect cross-domain bridges (unrelated tags co-retrieved)
                 -> detect semantic convergence (embedding similarity, no shared tags)
              -> place_events_batch(new_events)
                 -> compute angle (PCA -> atan2)
                 -> compute prime slots
                 -> _update_synchronicity_tensor (O(n) against existing positions)
                    -> INSERT resonances
                    -> UPSERT tensor synchronicity_strength
```

## Configuration

Key parameters in `ChicoryConfig`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `lattice_primes` | `[2, 3, 5, ..., 113]` | 30 primes defining lattice scales |
| `lattice_min_resonance_primes` | `4` | Minimum shared slots for resonance |
| `tensor_cooccurrence_weight` | `0.5` | PMI weight in recall scoring |
| `tensor_synchronicity_weight` | `0.3` | Lattice weight in recall scoring |
| `tensor_semantic_weight` | `0.2` | Cosine weight in recall scoring |
| `tensor_semiotic_weight` | `0.15` | Conditional probability weight |
| `sync_detection_sigma` | `2.0` | Z-score threshold for anomaly detection |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `embedding_dimension` | `384` | Embedding vector dimension |

All parameters are overridable via environment variables or `.env` file.

## Tests

```bash
pytest tests/ -v                                    # Full suite (59 tests)
pytest tests/test_layer3/test_tag_relational_tensor.py -v   # Tensor tests (21 tests)
pytest tests/test_layer3/test_synchronicity_engine.py -v    # Lattice tests
```

## Dependencies

**Runtime:** anthropic, pydantic, sentence-transformers, numpy, faiss-cpu, scipy, typer, rich, python-dotenv, pymupdf, python-docx, watchdog

**Dev:** pytest, pytest-asyncio

**Demo only:** numpy (mock embeddings, no model downloads)

## Database

SQLite with WAL mode. Schema v9, 14 tables, versioned migrations with idempotency checks. Thread-safe via `threading.RLock` on all execute/executemany calls.

```bash
chicory status          # Show memory count, tag count, tensor state
chicory migrate         # Run model migration with burn-in period
chicory reembed         # Re-embed all memories with new model
```
