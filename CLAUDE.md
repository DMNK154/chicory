# Chicory - Self-Organizing Memory System

## Codebase Exploration: Use Chicory First

This project's codebase is indexed in the Chicory MCP memory network. **Before reading files directly, query Chicory:**

1. **`retrieve_memories`** — Use `method: "hybrid"` with relevant `tags` for targeted lookup. Good for finding specific components, functions, or patterns.
2. **`deep_retrieve`** — Use when you need to follow association chains (e.g., "how does ingestion connect to embedding?"). Set `max_depth: 2-3` for broader exploration.
3. **Only then** fall back to `Glob`/`Grep`/`Read` for files Chicory identified, or for content not yet ingested.

### When to use which
- **"Where is X defined?"** → `retrieve_memories(query="X definition", method="hybrid", top_k=5)`
- **"How does X connect to Y?"** → `deep_retrieve(query="X Y relationship", max_depth=3)`
- **"What are the key files for feature Z?"** → `retrieve_memories(query="Z", tags=["z-related-tag"], method="hybrid")`
- **Exact string/symbol lookup** → Use `Grep` directly (Chicory stores summaries, not raw source)
- **File listing / structure** → Use `Glob` directly

### Tags convention
Tags in this project use lowercase with hyphens. Common tags include module names (`memory-store`, `embedding-engine`, `synchronicity-detector`, `orchestrator`, `mcp-server`, `commons`), layer names (`layer-1`, `layer-2`, `layer-3`), and concepts (`performance`, `batch`, `schema`).

## Project Structure
- `chicory/` — Main package
  - `core/` — Layer 1: memory_store, embedding_engine, tag_manager
  - `analysis/` — Layer 2: retrieval_tracker, trend_engine, time_series
  - `patterns/` — Layer 3: phase_space, synchronicity_detector, synchronicity_engine, meta_pattern_detector, lattice
  - `orchestrator/` — Ties layers together, tool handlers
  - `mcp/` — FastMCP server exposing 10 tools
  - `commons/` — Cross-project signal federation
  - `ingest/` — Codebase ingestion pipeline (3-phase)
- `tests/` — Test suite
- `chicory-commons-man/` — Separate commons manager package

## Dev Notes
- SQLite with WAL mode; schema version 10
- Embeddings via sentence-transformers (FAISS index)
- Commons uses dual tag scheme: `projectname:tagname` + `tagname`
