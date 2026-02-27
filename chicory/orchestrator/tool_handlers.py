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
    )


def _handle_retrieve_memories(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_retrieve_memories(
        query=inp["query"],
        tags=inp.get("tags"),
        method=inp.get("method", "hybrid"),
        top_k=inp.get("top_k", 10),
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
    return o.handle_get_lattice_resonances()


def _handle_deep_retrieve(o: "Orchestrator", inp: dict) -> dict:
    return o.handle_deep_retrieve(
        query=inp["query"],
        tags=inp.get("tags"),
        max_depth=inp.get("max_depth"),
        per_level_k=inp.get("per_level_k"),
    )
