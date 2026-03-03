"""Chicory MCP server — exposes memory tools via Model Context Protocol."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP

from chicory.config import load_config
from chicory.orchestrator.orchestrator import Orchestrator
from chicory.orchestrator.tool_handlers import dispatch_tool_call

# File-based diagnostic log so we can debug MCP hangs
_log_path = Path.home() / ".chicory" / "mcp_debug.log"
_log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(_log_path),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────


@dataclass
class AppContext:
    """Typed lifespan context holding the Orchestrator instance."""

    orchestrator: Orchestrator
    signal_processor: object | None = None  # SignalProcessor, if commons DB


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize the Orchestrator on startup, close it on shutdown."""
    import os
    # Prevent sentence-transformers from making HuggingFace Hub HTTP requests
    # during model loading. These deadlock when run inside an asyncio event
    # loop's thread pool (httpx vs asyncio conflict). The model is cached
    # locally so network access is unnecessary.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config = load_config()
    orchestrator = Orchestrator(config)
    logger.info("Chicory orchestrator initialized (db=%s)", config.db_path)

    # Eagerly load embedding model + FAISS index in the main thread.
    # Loading lazily in FastMCP's thread pool deadlocks (torch/asyncio conflict).
    try:
        t0 = time.time()
        embedding = orchestrator._memory_store._embedding
        embedding._load_model()
        logger.info("Embedding model loaded in %.1fs", time.time() - t0)

        t1 = time.time()
        embedding.search_similar(embedding.embed("warmup"), top_k=1)
        logger.info("FAISS index built in %.1fs", time.time() - t1)
    except Exception:
        logger.exception("Eager model/index load failed (will retry lazily)")

    # Initialize signal processor if pending_signals table exists (commons mode)
    signal_processor = None
    try:
        table_check = orchestrator.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_signals'"
        ).fetchone()
        if table_check:
            from chicory_commons import SignalProcessor

            signal_processor = SignalProcessor(orchestrator)
            logger.info("Signal processor initialized (commons mode)")
    except Exception:
        logger.exception("Signal processor init failed")

    try:
        yield AppContext(
            orchestrator=orchestrator,
            signal_processor=signal_processor,
        )
    finally:
        orchestrator.close()
        logger.info("Chicory orchestrator closed")


# ── Server ──────────────────────────────────────────────────────────

mcp_server = FastMCP(
    "Chicory",
    lifespan=lifespan,
)


# ── Helper ──────────────────────────────────────────────────────────


def _call(ctx: Context, tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call and return JSON string result."""
    import threading

    logger.info(
        "TOOL CALL: %s  input_keys=%s  thread=%s",
        tool_name, list(tool_input.keys()), threading.current_thread().name,
    )
    t0 = time.time()
    app_ctx = ctx.request_context.lifespan_context
    orchestrator: Orchestrator = app_ctx.orchestrator

    # Auto-process pending signals if this is a commons instance
    if app_ctx.signal_processor:
        try:
            app_ctx.signal_processor.maybe_auto_process()
        except Exception:
            logger.exception("Signal auto-processing failed")

    try:
        result = dispatch_tool_call(orchestrator, tool_name, tool_input)
    except Exception:
        logger.exception("Tool %s failed after %.1fs", tool_name, time.time() - t0)
        result = {"error": f"Tool '{tool_name}' failed unexpectedly"}
    elapsed = time.time() - t0
    n = len(result.get("results", [])) if isinstance(result, dict) else "?"
    logger.info("TOOL DONE: %s  %.1fs  results=%s", tool_name, elapsed, n)
    return json.dumps(result, default=str)


# ── Tools ───────────────────────────────────────────────────────────


@mcp_server.tool()
def store_memory(
    ctx: Context,
    content: str,
    tags: list[str],
    importance: Optional[float] = None,
    summary: Optional[str] = None,
) -> str:
    """Store a new memory with tags and importance rating.

    Use this when the conversation contains information worth remembering.
    """
    inp: dict = {"content": content, "tags": tags}
    if importance is not None:
        inp["importance"] = importance
    if summary is not None:
        inp["summary"] = summary
    return _call(ctx, "store_memory", inp)


@mcp_server.tool()
def retrieve_memories(
    ctx: Context,
    query: str,
    tags: Optional[list[str]] = None,
    method: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """Search for relevant memories.

    Use this when you need to recall information from past conversations
    or find connections.
    """
    inp: dict = {"query": query}
    if tags is not None:
        inp["tags"] = tags
    if method is not None:
        inp["method"] = method
    if top_k is not None:
        inp["top_k"] = top_k
    return _call(ctx, "retrieve_memories", inp)


@mcp_server.tool()
def get_trends(
    ctx: Context,
    tags: Optional[list[str]] = None,
) -> str:
    """View current tag trend signals.

    Shows what topics are heating up, cooling down, or accelerating.
    """
    inp: dict = {}
    if tags is not None:
        inp["tags"] = tags
    return _call(ctx, "get_trends", inp)


@mcp_server.tool()
def get_phase_space(ctx: Context) -> str:
    """View the phase space.

    Shows where each tag sits on the trend-temperature vs
    retrieval-frequency plane.
    """
    return _call(ctx, "get_phase_space", {})


@mcp_server.tool()
def get_synchronicities(
    ctx: Context,
    limit: Optional[int] = None,
    unacknowledged_only: Optional[bool] = None,
) -> str:
    """View detected synchronicity events.

    Meaningful coincidences where retrieval patterns diverge from trend
    patterns. Includes effective_strength (decayed and reinforcement-boosted),
    reinforcement data, and overall synchronicity velocity.
    """
    inp: dict = {}
    if limit is not None:
        inp["limit"] = limit
    if unacknowledged_only is not None:
        inp["unacknowledged_only"] = unacknowledged_only
    return _call(ctx, "get_synchronicities", inp)


@mcp_server.tool()
def get_meta_patterns(ctx: Context) -> str:
    """View detected higher-order meta-patterns.

    Recurring themes across synchronicity events.
    """
    return _call(ctx, "get_meta_patterns", {})


@mcp_server.tool()
def get_lattice_resonances(ctx: Context) -> str:
    """View the prime Ramsey lattice.

    Synchronicity events organized by angular position across prime scales.
    Shows resonances (structurally entangled events) and the void profile
    (latent attractor themes).
    """
    return _call(ctx, "get_lattice_resonances", {})


@mcp_server.tool()
def deep_retrieve(
    ctx: Context,
    query: str,
    tags: Optional[list[str]] = None,
    max_depth: Optional[int] = None,
    per_level_k: Optional[int] = None,
) -> str:
    """Recursively retrieve memories by following association chains.

    Starts with a standard query, then expands through related memories.
    Deeper recursion levels progressively favor older memories, surfacing
    the deep past through semantic association.
    """
    inp: dict = {"query": query}
    if tags is not None:
        inp["tags"] = tags
    if max_depth is not None:
        inp["max_depth"] = max_depth
    if per_level_k is not None:
        inp["per_level_k"] = per_level_k
    return _call(ctx, "deep_retrieve", inp)


@mcp_server.tool()
def ingest_codebase(
    ctx: Context,
    path: str,
    file_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> str:
    """Ingest a codebase directory into the memory network.

    Scans source files and stores structural summaries (classes, functions,
    imports, signatures) as memories. Use this to build a searchable map of
    a codebase so you can retrieve summaries via retrieve_memories instead
    of re-reading files.

    Args:
        path: Directory to scan (absolute or relative).
        file_patterns: Optional glob patterns to include (e.g. ["*.py", "src/**/*.ts"]).
            Defaults to all supported code file types.
        exclude_patterns: Optional directory names to skip (in addition to
            .git, node_modules, __pycache__, etc.).
    """
    inp: dict = {"path": path}
    if file_patterns is not None:
        inp["file_patterns"] = file_patterns
    if exclude_patterns is not None:
        inp["exclude_patterns"] = exclude_patterns
    return _call(ctx, "ingest_codebase", inp)


@mcp_server.tool()
def process_signals(ctx: Context) -> str:
    """Process pending cross-project signals into commons memories.

    Reads signals from project Chicory instances, converts them to
    commons memories with project-namespaced and shared tags, and
    triggers embedding generation.
    """
    app_ctx = ctx.request_context.lifespan_context
    if not app_ctx.signal_processor:
        from chicory_commons import SignalProcessor

        app_ctx.signal_processor = SignalProcessor(app_ctx.orchestrator)
    result = app_ctx.signal_processor.process_pending()
    return json.dumps(result, default=str)


# ── Entry point ─────────────────────────────────────────────────────


def main() -> None:
    """Run the Chicory MCP server with stdio transport."""
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
