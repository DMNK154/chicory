"""Chicory MCP server — exposes memory tools via Model Context Protocol."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

# Wall clock at process start, before any heavy import.
#
# The client's connect deadline starts at SPAWN, but the first log statement
# used to come after `Orchestrator(config)` returned — so everything before
# it was dark, and "28.2s with 1s of margin" was really "28.2s plus an
# unmeasured amount, which is why it timed out".  Anchor every startup
# timing to this instead.
_T_BOOT = time.time()

# File-based diagnostic log so we can debug MCP hangs.  The pid is in the
# FORMAT rather than the filename: two servers appending to one file made
# "which process was slow" unanswerable, but a file per pid would pile up
# forever.  This keeps one file and still attributes every line.
_log_path = Path.home() / ".chicory" / "mcp_debug.log"
_log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(_log_path),
    level=logging.INFO,
    format="%(asctime)s [pid %(process)d] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("=== process start ===")

import json  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Optional  # noqa: E402

_t_imp = time.time()
from mcp.server.fastmcp import Context, FastMCP  # noqa: E402

logger.info("mcp sdk imported in %.1fs", time.time() - _t_imp)

_t_imp = time.time()
from chicory.config import load_config  # noqa: E402
from chicory.orchestrator.orchestrator import Orchestrator  # noqa: E402
from chicory.orchestrator.tool_handlers import dispatch_tool_call  # noqa: E402

logger.info(
    "chicory imported in %.1fs (%.1fs since process start)",
    time.time() - _t_imp, time.time() - _T_BOOT,
)


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

    _t = time.time()
    config = load_config()
    orchestrator = Orchestrator(config)
    logger.info(
        "Chicory orchestrator initialized in %.1fs (db=%s, %.1fs since "
        "process start)",
        time.time() - _t, config.db_path, time.time() - _T_BOOT,
    )

    # Load the embedding model HERE, before yielding, even though it costs
    # a few seconds of connect time.
    #
    # It cannot go in the background thread with the rest.  The torch /
    # sentence-transformers import does not interleave with FastMCP's
    # request handling: put it on a side thread and it stalls holding
    # _model_lock, while the first tool call that needs an embedding
    # blocks on that lock for as long as the client is willing to wait
    # (observed: "model loaded in 605.4s" for a load that takes ~6s alone,
    # the figure tracking the client's timeout exactly).  Loading it up
    # front is a few seconds; loading it beside a waiting request is
    # unbounded.
    t_model = time.time()
    try:
        orchestrator._memory_store._embedding._load_model()
        logger.info("Embedding model loaded in %.1fs", time.time() - t_model)
    except Exception:
        logger.exception("Eager model load failed (will retry lazily)")

    # faiss is the same story as torch — a native extension whose import
    # stalls for minutes if it happens on a side thread while a request is
    # in flight (observed: "loaded faiss with AVX2 support" taking 9
    # minutes).  Import it now; the module is normally pulled in lazily on
    # the first search, which is exactly the wrong moment.
    t_faiss = time.time()
    try:
        from chicory.layer1.vector_index import _import_faiss

        _import_faiss()
        logger.info("faiss imported in %.1fs", time.time() - t_faiss)
    except Exception:
        logger.exception("Eager faiss import failed (will import lazily)")

    # What remains for the background thread is pure Python/numpy/SQL —
    # building the index and warming the retrieval path — which does
    # interleave with request handling, so it stays off the connect path.
    def _background_warmup() -> None:
        try:
            embedding = orchestrator._memory_store._embedding

            t1 = time.time()
            embedding.search_similar(embedding.embed("warmup"), top_k=1)
            logger.info("FAISS index built in %.1fs", time.time() - t1)

            # Warm the whole retrieval path, not just the model and index.
            # The first hybrid retrieval also cold-loads the tag-space
            # structures and lattice caches — on a six-figure library that
            # is tens of seconds, and paying it here means the patron's
            # first question doesn't. Everything after this is sub-second.
            t2 = time.time()
            orchestrator._memory_store.retrieve_hybrid("warmup", None, 1)
            logger.info("Retrieval path warmed in %.1fs", time.time() - t2)
        except Exception:
            logger.exception("Background warmup failed (will warm lazily instead)")

        if orchestrator.glyph_encoder is not None:
            try:
                t2 = time.time()
                orchestrator.glyph_encoder._load_model()
                logger.info("Glyph encoder loaded in %.1fs", time.time() - t2)
            except Exception:
                logger.exception("Glyph encoder load failed (will use letter-count fallback)")

        # The heartbeat starts LAST, and inside the warmup thread rather
        # than beside it.  Its errands embed and retrieve, so starting it
        # before the model and index are up would have the patrol racing
        # the warmup for the same cold locks this thread exists to pay off.
        try:
            if orchestrator.start_heartbeat():
                logger.info("Heartbeat running — patrol advances while idle")
        except Exception:
            logger.exception("Heartbeat failed to start (library stays traffic-paced)")

    import threading
    threading.Thread(
        target=_background_warmup, name="chicory-warmup", daemon=True,
    ).start()

    # The number that actually matters: everything above this line is on the
    # client's connect deadline (MCP_TIMEOUT, default 30000ms).
    logger.info(
        "CONNECT PATH COMPLETE in %.1fs since process start", time.time() - _T_BOOT,
    )

    try:
        yield AppContext(
            orchestrator=orchestrator,
            signal_processor=orchestrator._signal_processor,
        )
    finally:
        orchestrator.close()
        logger.info("Chicory orchestrator closed")


# ── Server ──────────────────────────────────────────────────────────

mcp_server = FastMCP(
    "Chicory",
    lifespan=lifespan,
)


# ── Glyph legend injection ─────────────────────────────────────────

_GLYPH_LEGEND: dict[str, str] = {}  # glyph -> concept name


def _ensure_glyph_legend() -> dict[str, str]:
    """Load the glyph->concept mapping once."""
    global _GLYPH_LEGEND
    if _GLYPH_LEGEND:
        return _GLYPH_LEGEND
    try:
        from chicory.layer1.glyph_lexicon import GLYPH2DICT
        _GLYPH_LEGEND = dict(GLYPH2DICT)
    except ImportError:
        pass
    return _GLYPH_LEGEND


def _inject_glyph_definitions(response_text: str) -> str:
    """Scan a response string for glyph symbols and append a definitions block."""
    legend = _ensure_glyph_legend()
    if not legend:
        return response_text

    found: dict[str, str] = {}
    for char in response_text:
        if char in legend and char not in found:
            found[char] = legend[char]

    if not found:
        return response_text

    defs = ", ".join(f"{g}={name}" for g, name in found.items())
    return f"{response_text}\n\n[glyph definitions: {defs}]"


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

    try:
        result = dispatch_tool_call(orchestrator, tool_name, tool_input)
    except Exception:
        logger.exception("Tool %s failed after %.1fs", tool_name, time.time() - t0)
        result = {"error": f"Tool '{tool_name}' failed unexpectedly"}
    elapsed = time.time() - t0
    n = len(result.get("results", [])) if isinstance(result, dict) else "?"
    logger.info("TOOL DONE: %s  %.1fs  results=%s", tool_name, elapsed, n)
    response = json.dumps(result, default=str, ensure_ascii=False)
    return _inject_glyph_definitions(response)


# ── Tools ───────────────────────────────────────────────────────────


@mcp_server.tool()
def store_memory(
    ctx: Context,
    content: str,
    tags: list[str],
    importance: Optional[float] = None,
    summary: Optional[str] = None,
    author: Optional[str] = None,
    origin: Optional[str] = None,
) -> str:
    """Store a new memory with tags and importance rating.

    Use this when the conversation contains information worth remembering.

    Provenance: pass author (who/what wrote this, e.g. "claude",
    "librarian") and origin (where it came from, e.g. a session or
    source description) so retrieval can surface the byline.
    """
    inp: dict = {"content": content, "tags": tags}
    if importance is not None:
        inp["importance"] = importance
    if summary is not None:
        inp["summary"] = summary
    if author is not None:
        inp["author"] = author
    if origin is not None:
        inp["origin"] = origin
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
def get_lattice_resonances(ctx: Context, max_items: int = 10) -> str:
    """View the prime Ramsey lattice.

    Synchronicity events organized by angular position across prime scales.
    Shows resonances (structurally entangled events) and the void profile
    (latent attractor themes).

    Returns top items by strength. Use max_items to control how many
    positions/resonances are returned (default 10).
    """
    return _call(ctx, "get_lattice_resonances", {"max_items": max_items})


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
def get_last_trace(ctx: Context) -> str:
    """Get the activation trace from the most recent retrieval.

    Shows which tags activated and with what strength, per-memory score
    breakdowns (semantic, tag, lattice, glyph), and timing data.
    """
    return _call(ctx, "get_last_trace", {})


@mcp_server.tool()
def compare_traces(ctx: Context) -> str:
    """Compare the last two retrieval traces.

    Shows what changed between consecutive retrievals: tag activation
    deltas (sorted by largest divergence), memories that appeared or
    disappeared, per-component score deltas for shared memories, and
    rank changes.
    """
    return _call(ctx, "compare_traces", {})


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

    Returns counts plus per-file listings: `summaries` (paths ingested),
    `skipped` (files no summary could be extracted from) and
    `already_ingested` (files whose summary is already in the network).
    """
    inp: dict = {"path": path}
    if file_patterns is not None:
        inp["file_patterns"] = file_patterns
    if exclude_patterns is not None:
        inp["exclude_patterns"] = exclude_patterns
    return _call(ctx, "ingest_codebase", inp)


@mcp_server.tool()
def ingest_documents(
    ctx: Context,
    path: str,
    file_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
    critical: Optional[bool] = None,
) -> str:
    """Ingest documents into the memory network with two-tier depth.

    Each file is auto-classified based on tag overlap with the existing
    memory network:

    - **critical**: Full content stored (chunked for long files), embedded,
      and deeply integrated into the tensor/lattice/centroid/canopy network.
    - **reference**: Extractive whole-document summary stored with a
      source_path pointer to the original file. Tags assigned and summary
      embedded for search, but no deep network integration.

    Use the `critical` parameter to override auto-classification.

    Args:
        path: File or directory to ingest (absolute or relative).
        file_patterns: Optional glob patterns to include (e.g. ["*.txt", "*.md"]).
            Defaults to all text-based files.
        exclude_patterns: Optional directory names to skip.
        critical: Force all files to critical (True) or reference (False).
            When None (default), each file is auto-classified by network relevance.

    Returns counts plus per-file listings: `ingested` (path + tier),
    `skipped` (files that could not be read or were empty) and
    `already_ingested` (files whose content is already in the network).
    """
    inp: dict = {"path": path}
    if file_patterns is not None:
        inp["file_patterns"] = file_patterns
    if exclude_patterns is not None:
        inp["exclude_patterns"] = exclude_patterns
    if critical is not None:
        inp["critical"] = critical
    return _call(ctx, "ingest_documents", inp)


@mcp_server.tool()
def get_canopy(
    ctx: Context,
    block_key: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """Inspect the self-building canopy graph.

    Without arguments: returns a summary of all canopy blocks, how many
    have grown, and the top blocks by growth potential.

    With block_key: returns full detail for one canopy block including
    support edges, recent observations, and cross-layer inhibition edges.

    Args:
        block_key: Optional specific block key to inspect.
        top_k: How many top blocks to return in summary mode (default 20).
    """
    inp: dict = {}
    if block_key is not None:
        inp["block_key"] = block_key
    if top_k is not None:
        inp["top_k"] = top_k
    return _call(ctx, "get_canopy", inp)


@mcp_server.tool()
def sync_network(ctx: Context) -> str:
    """Run full network synchronization after a large ingest.

    Executes all expensive network-building passes in order: centroids,
    glyph lattice, lateral inhibition, tensor channels (co-occurrence,
    semantic, glyph, semiotic), episodic edges, canopy growth,
    synchronicity detection, and meta-pattern analysis.

    Use this after bulk ingestion (ingest_codebase / ingest_documents /
    corpus loading) to bring the network up to date. Runs against all
    memories, not just newly ingested ones.
    """
    app_ctx = ctx.request_context.lifespan_context
    orchestrator = app_ctx.orchestrator
    stats = orchestrator.run_sync()
    return _inject_glyph_definitions(json.dumps(stats, default=str, ensure_ascii=False))


@mcp_server.tool()
def process_signals(ctx: Context) -> str:
    """Process pending cross-project signals into commons memories.

    Reads signals from project Chicory instances, converts them to
    commons memories with project-namespaced and shared tags, and
    triggers embedding generation.
    """
    app_ctx = ctx.request_context.lifespan_context
    processor = app_ctx.signal_processor
    if not processor:
        return json.dumps({"error": "Commons not enabled or chicory-commons-man not installed"})
    result = processor.process_pending()
    return _inject_glyph_definitions(json.dumps(result, default=str, ensure_ascii=False))


@mcp_server.tool()
def get_commons_log(
    ctx: Context,
    last_n: Optional[int] = None,
    hours: Optional[float] = None,
) -> str:
    """View the chicory-commons activity log.

    Shows a chronological history of signals emitted, signals processed,
    glyph translations, heat map changes, and synchronicity detections.

    Args:
        last_n: Return the last N entries (default 50).
        hours: Return entries from the past N hours instead.
    """
    try:
        from chicory_commons.activity_log import read_recent, read_since

        if hours is not None:
            entries = read_since(hours)
        else:
            entries = read_recent(last_n or 50)

        return _inject_glyph_definitions(json.dumps(entries, default=str))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp_server.tool()
def librarian_status(ctx: Context) -> str:
    """Librarian overview: write-gate calibration stats, sweep patrol
    progress, quarantined memory count, and subject roster summary.

    The write gate scores every stored memory against tag centroids; in
    shadow mode it only logs verdicts (see gate.verdicts) so the
    threshold can be calibrated before enforcement is enabled.
    """
    return _call(ctx, "librarian_status", {})


@mcp_server.tool()
def librarian_review(
    ctx: Context,
    limit: Optional[int] = None,
) -> str:
    """Review the librarian's queue: quarantined memories awaiting
    judgment and open sweep findings (near-duplicates, tag drift,
    missing embeddings).

    Use librarian_resolve to act on items in this queue.

    Args:
        limit: Max items per section (default 20).
    """
    inp: dict = {}
    if limit is not None:
        inp["limit"] = limit
    return _call(ctx, "librarian_review", inp)


@mcp_server.tool()
def librarian_resolve(
    ctx: Context,
    memory_id: Optional[str] = None,
    action: Optional[str] = None,
    finding_id: Optional[int] = None,
) -> str:
    """Apply a judgment from librarian_review.

    For a quarantined memory: pass memory_id plus action —
    'release_critical' (join the network fully), 'release_reference'
    (retrievable but excluded from episode structure), or 'archive'.

    For a sweep finding: pass finding_id to mark it resolved.
    """
    inp: dict = {}
    if memory_id is not None:
        inp["memory_id"] = memory_id
    if action is not None:
        inp["action"] = action
    if finding_id is not None:
        inp["finding_id"] = finding_id
    return _call(ctx, "librarian_resolve", inp)


@mcp_server.tool()
def librarian_consult(
    ctx: Context,
    query: str,
    top_k: Optional[int] = None,
) -> str:
    """Ask the front desk. The structured consultation path:

    1. FAQ fast path — a verified cached answer if this is a known
       recurring question (with its supporting memory ids and byline).
    2. Subject dispatch — matching shelves consulted individually,
       each returning its own results.
    3. General channel — whole-library hybrid retrieval.
    Plus research_pending: open commissions near this query.

    Prefer this over retrieve_memories when answering a patron question;
    use retrieve_memories for raw lookup.
    """
    inp: dict = {"query": query}
    if top_k is not None:
        inp["top_k"] = top_k
    return _call(ctx, "librarian_consult", inp)


@mcp_server.tool()
def librarian_faq(
    ctx: Context,
    action: Optional[str] = None,
    faq_id: Optional[int] = None,
    answer: Optional[str] = None,
    author: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """FAQ desk management.

    action='list' (default): status='candidate' lists drafts awaiting
    judgment; status='active'/'stale'/'retired' lists entries.
    action='approve': activate a candidate (faq_id; optional answer
    rewrite and author byline). Approval is the judgment step — only
    approved entries are served on the fast path.
    action='retire': permanently retire an entry (faq_id).
    """
    inp: dict = {}
    for key, val in (
        ("action", action), ("faq_id", faq_id), ("answer", answer),
        ("author", author), ("status", status), ("limit", limit),
    ):
        if val is not None:
            inp[key] = val
    return _call(ctx, "librarian_faq", inp)


@mcp_server.tool()
def librarian_commissions(
    ctx: Context,
    action: Optional[str] = None,
    commission_id: Optional[int] = None,
    question: Optional[str] = None,
    report_memory_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """The commission queue — assigned research on the library's known gaps.

    action='list' (default, filter by status: open/claimed/reported/closed).
    action='file': open a ticket (question). Recurring weak-answer query
    clusters also file tickets automatically.
    action='claim': take a ticket (commission_id).
    action='report': attach a report (commission_id + report_memory_id —
    store the report first via store_memory with author and origin set).
    action='close': close a ticket (commission_id).
    """
    inp: dict = {}
    for key, val in (
        ("action", action), ("commission_id", commission_id),
        ("question", question), ("report_memory_id", report_memory_id),
        ("status", status), ("limit", limit),
    ):
        if val is not None:
            inp[key] = val
    return _call(ctx, "librarian_commissions", inp)


@mcp_server.tool()
def librarian_closure(
    ctx: Context,
    action: Optional[str] = None,
    memory_id: Optional[str] = None,
    superseding_id: Optional[str] = None,
    report_memory_id: Optional[str] = None,
    commission_id: Optional[int] = None,
    claim: Optional[str] = None,
    reason: Optional[str] = None,
    scope: Optional[str] = None,
    actor: Optional[str] = None,
    supersedes: Optional[list[dict]] = None,
    resolve_findings: Optional[list[int]] = None,
    limit: Optional[int] = None,
) -> str:
    """Specialist closure — settle what a piece of work overturned.

    When a report contradicts something the library already asserts, the
    old claim does not stop existing on its own. Close the loop:

    action='close' (report_memory_id, optional commission_id,
      supersedes=[{memory_id, claim, reason, scope}], resolve_findings=[])
      runs the whole unwind in reverse order — supersede what the report
      refuted, mark dependent FAQ answers stale, resolve findings it
      explains, close the ticket.
    action='supersede' (memory_id, superseding_id, claim, reason, scope)
      records one correction on its own.
    action='list' / 'retract'.

    Nothing is ever deleted. A superseded memory keeps its byline and its
    place, and simply arrives marked from then on. Use scope='claim' (the
    default) when only one assertion fell — a digest that got a section
    wrong is not worthless — and 'whole' only when the entire memory is
    replaced.
    """
    inp: dict = {}
    for key, val in (
        ("action", action), ("memory_id", memory_id),
        ("superseding_id", superseding_id),
        ("report_memory_id", report_memory_id),
        ("commission_id", commission_id), ("claim", claim),
        ("reason", reason), ("scope", scope), ("actor", actor),
        ("supersedes", supersedes), ("resolve_findings", resolve_findings),
        ("limit", limit),
    ):
        if val is not None:
            inp[key] = val
    return _call(ctx, "librarian_closure", inp)


@mcp_server.tool()
def librarian_desk(
    ctx: Context,
    action: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """The library's standing report — start here, before anything else.

    action='report' (default): what changed since you were last here,
    what needs your judgment (quarantine, findings, FAQ drafts, open
    commissions, flagged arrivals), what was learned (digests written,
    tickets closed), and the state of the building.
    action='ack': mark the current moment as seen, so the next report's
    "since" window starts from here.

    Reading never acknowledges — a report you did not act on comes back.
    """
    inp: dict = {}
    for key, val in (("action", action), ("limit", limit)):
        if val is not None:
            inp[key] = val
    return _call(ctx, "librarian_desk", inp)


@mcp_server.tool()
def librarian_subjects(
    ctx: Context,
    action: Optional[str] = None,
    path: Optional[str] = None,
    name: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Address the subject hierarchy — the shelves, as a tree.

    action='tree' (default): the whole active roster as an outline, with
    each node's path, label, depth, memory count and tag count.
    action='show': one shelf. Address it loosely — an exact path, a bare
    name, a given name, or just a topic ('gnosticism') resolves to the
    most specific shelf holding it. If several match you get candidates
    back, not an error; if none, you get suggestions.
    action='children': direct children of a shelf.
    action='rename' (path + name): give a shelf a human name. Names are
    authored, not derived, so they survive roster rebuilds — use this
    when a shelf's real subject is clear and its machine label isn't.

    Prefer a shelf's `label` over its derived `name` when speaking to a
    person. Routing stays coarse; teach cycles address deeper shelves.
    """
    inp: dict = {}
    for key, val in (
        ("action", action), ("path", path), ("name", name), ("limit", limit),
    ):
        if val is not None:
            inp[key] = val
    return _call(ctx, "librarian_subjects", inp)


@mcp_server.tool()
def librarian_ingress(
    ctx: Context,
    action: Optional[str] = None,
    content: Optional[str] = None,
    receipt: Optional[str] = None,
    source: Optional[str] = None,
    origin: Optional[str] = None,
    author: Optional[str] = None,
    tags: Optional[list[str]] = None,
    importance: Optional[float] = None,
    summary: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Receipted ingress dwell — hold content under custody before it is
    committed to memory, so temporal detection has a window to run in.

    action='hold': take content into custody and get a receipt. Commits
    nothing. Examines the whole dwelling window, so this arrival is
    compared against messages still held — that cross-message view is the
    point (a repetition burst is invisible one message at a time).
    action='review': list receipts by status (held/cleared/flagged/
    committed/discarded) with their findings.
    action='settle': close a dwell (receipt) or every elapsed dwell.
    action='commit': promote a settled receipt into a memory (needs tags).
    action='discard': settle without shelving; receipt and findings remain.
    action='examine' / 'stats'.

    Findings are ADVISORY — trajectory deviation fires on novelty as
    readily as on attack, and the diction screen is trivially evaded.
    Weigh them alongside provenance; never treat one as a verdict.
    """
    inp: dict = {}
    for key, val in (
        ("action", action), ("content", content), ("receipt", receipt),
        ("source", source), ("origin", origin), ("author", author),
        ("tags", tags), ("importance", importance), ("summary", summary),
        ("status", status), ("limit", limit),
    ):
        if val is not None:
            inp[key] = val
    return _call(ctx, "librarian_ingress", inp)


@mcp_server.tool()
def librarian_recall(
    ctx: Context,
    query: str,
    top_k: Optional[int] = None,
) -> str:
    """Cross-domain analogic recall: what occupies the same ROLE in other
    subjects that this query occupies in its home subject?

    Works by offset transport between subject centroids — results are
    conjectures (labeled register: analogy), not established associations.
    Use for creative bridging, cross-domain inspiration, and finding
    structural rhymes the co-occurrence machinery can't see.
    """
    inp: dict = {"query": query}
    if top_k is not None:
        inp["top_k"] = top_k
    return _call(ctx, "librarian_recall", inp)


@mcp_server.tool()
def librarian_maintain(ctx: Context, cold_rebuild: bool = False) -> str:
    """Run one librarian maintenance pass now: sweep the next batch of
    the shelf-reading patrol and rebuild the subject roster from the
    tag co-retrieval graph.

    The sweep also runs automatically in the background on an interval;
    this tool forces a batch immediately.

    The roster rebuild is WARM by default: it inherits the standing tree's
    component boundaries and split shape, so shelf paths stay stable as
    addresses across rebuilds.  Pass cold_rebuild=True to re-derive the
    tree from the geometry instead — that will move paths and retire
    shelves, orphaning addresses that digests and commissions quote, so
    it is a deliberate reset rather than routine maintenance.
    """
    return _call(ctx, "librarian_maintain", {"cold_rebuild": cold_rebuild})


# ── Entry point ─────────────────────────────────────────────────────


def main() -> None:
    """Run the Chicory MCP server with stdio transport."""
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
