"""Tool call dispatch: maps tool names to orchestrator methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chicory.orchestrator.orchestrator import Orchestrator


def dispatch_tool_call(
    orchestrator: "Orchestrator",
    tool_name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate orchestrator method."""
    handlers = {
        "store_memory": _handle_store_memory,
        "retrieve_memories": _handle_retrieve_memories,
        "get_trends": _handle_get_trends,
        "get_phase_space": _handle_get_phase_space,
        "get_synchronicities": _handle_get_synchronicities,
        "get_meta_patterns": _handle_get_meta_patterns,
        "get_lattice_resonances": _handle_get_lattice_resonances,
        "deep_retrieve": _handle_deep_retrieve,
        "ingest_codebase": _handle_ingest_codebase,
        "ingest_documents": _handle_ingest_documents,
        "get_canopy": _handle_get_canopy,
        "sync_network": _handle_sync_network,
        "get_last_trace": _handle_get_last_trace,
        "compare_traces": _handle_compare_traces,
        "librarian_status": _handle_librarian_status,
        "librarian_review": _handle_librarian_review,
        "librarian_resolve": _handle_librarian_resolve,
        "librarian_maintain": _handle_librarian_maintain,
        "librarian_consult": _handle_librarian_consult,
        "librarian_faq": _handle_librarian_faq,
        "librarian_commissions": _handle_librarian_commissions,
        "librarian_recall": _handle_librarian_recall,
        "librarian_ingress": _handle_librarian_ingress,
        "librarian_subjects": _handle_librarian_subjects,
        "librarian_desk": _handle_librarian_desk,
        "librarian_closure": _handle_librarian_closure,
    }

    handler = handlers.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}

    return handler(orchestrator, tool_input)


def _handle_store_memory(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_store_memory(
        content=inp["content"],
        tags=inp.get("tags", []),
        importance=inp.get("importance"),
        summary=inp.get("summary"),
        author=inp.get("author"),
        origin=inp.get("origin"),
    )


def _handle_retrieve_memories(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_retrieve_memories(
        query=inp["query"],
        tags=inp.get("tags"),
        method=inp.get("method", "hybrid"),
        top_k=inp.get("top_k", 10),
        actor=inp.get("actor"),
    )


def _handle_get_trends(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_get_trends(tag_names=inp.get("tags"))


def _handle_get_phase_space(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_get_phase_space()


def _handle_get_synchronicities(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_get_synchronicities(
        limit=inp.get("limit", 10),
        unacknowledged_only=inp.get("unacknowledged_only", False),
    )


def _handle_get_meta_patterns(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_get_meta_patterns()


def _handle_get_lattice_resonances(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_get_lattice_resonances(max_items=inp.get("max_items", 10))


def _handle_deep_retrieve(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_deep_retrieve(
        query=inp["query"],
        tags=inp.get("tags"),
        max_depth=inp.get("max_depth"),
        per_level_k=inp.get("per_level_k"),
        actor=inp.get("actor"),
    )


def _handle_ingest_codebase(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_ingest_codebase(
        path=inp["path"],
        file_patterns=inp.get("file_patterns"),
        exclude_patterns=inp.get("exclude_patterns"),
    )


def _handle_ingest_documents(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_ingest_documents(
        path=inp["path"],
        file_patterns=inp.get("file_patterns"),
        exclude_patterns=inp.get("exclude_patterns"),
        critical=inp.get("critical"),
    )


def _handle_get_canopy(o: "Orchestrator", inp: dict) -> dict:
    block_key = inp.get("block_key")
    if block_key:
        detail = o.canopy.get_block_detail(block_key)
        if detail is None:
            return {"error": f"Canopy block not found: {block_key}"}
        return detail
    return o.canopy.get_canopy_summary(top_k=inp.get("top_k", 20))


def _handle_sync_network(o: "Orchestrator", inp: dict) -> dict:
    return o.run_sync()


def _handle_get_last_trace(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_get_last_trace()


def _handle_compare_traces(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_compare_traces()


def _handle_librarian_status(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_status()


def _handle_librarian_review(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_review(limit=inp.get("limit", 20))


def _handle_librarian_resolve(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_resolve(
        memory_id=inp.get("memory_id"),
        action=inp.get("action"),
        finding_id=inp.get("finding_id"),
    )


def _handle_librarian_maintain(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_maintain()


def _handle_librarian_consult(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_consult(
        query=inp["query"],
        top_k=inp.get("top_k", 5),
        actor=inp.get("actor"),
    )


def _handle_librarian_faq(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_faq(
        demote=inp.get("demote", False),
        action=inp.get("action", "list"),
        faq_id=inp.get("faq_id"),
        answer=inp.get("answer"),
        author=inp.get("author"),
        status=inp.get("status", "candidate"),
        limit=inp.get("limit", 20),
    )


def _handle_librarian_recall(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_recall(
        query=inp["query"],
        top_k=inp.get("top_k", 3),
    )


def _handle_librarian_closure(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_closure(
        action=inp.get("action", "list"),
        memory_id=inp.get("memory_id"),
        superseding_id=inp.get("superseding_id"),
        report_memory_id=inp.get("report_memory_id"),
        commission_id=inp.get("commission_id"),
        claim=inp.get("claim"),
        reason=inp.get("reason"),
        scope=inp.get("scope", "claim"),
        actor=inp.get("actor"),
        supersedes=inp.get("supersedes"),
        resolve_findings=inp.get("resolve_findings"),
        limit=inp.get("limit", 20),
    )


def _handle_librarian_desk(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_desk(
        action=inp.get("action", "report"),
        limit=inp.get("limit", 5),
    )


def _handle_librarian_subjects(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_subjects(
        action=inp.get("action", "tree"),
        path=inp.get("path"),
        name=inp.get("name"),
        limit=inp.get("limit", 200),
    )


def _handle_librarian_ingress(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_ingress(
        action=inp.get("action", "review"),
        content=inp.get("content"),
        receipt=inp.get("receipt"),
        source=inp.get("source", "user"),
        origin=inp.get("origin"),
        author=inp.get("author"),
        tags=inp.get("tags"),
        importance=inp.get("importance"),
        summary=inp.get("summary"),
        status=inp.get("status", "held"),
        limit=inp.get("limit", 25),
    )


def _handle_librarian_commissions(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_librarian_commissions(
        action=inp.get("action", "list"),
        commission_id=inp.get("commission_id"),
        question=inp.get("question"),
        report_memory_id=inp.get("report_memory_id"),
        status=inp.get("status", "open"),
        limit=inp.get("limit", 20),
    )
