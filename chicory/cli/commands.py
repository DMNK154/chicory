"""Slash command handlers for the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chicory.cli.display import (
    console,
    display_meta_patterns,
    display_phase_space,
    display_retrieval_results,
    display_status,
    display_synchronicities,
    display_trends,
)

if TYPE_CHECKING:
    from chicory.orchestrator.orchestrator import Orchestrator


def handle_slash_command(orchestrator: "Orchestrator", command: str) -> bool:
    """Handle a slash command. Returns True if handled, False otherwise."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handlers = {
        "/memories": _cmd_memories,
        "/trends": _cmd_trends,
        "/phase": _cmd_phase,
        "/sync": _cmd_sync,
        "/meta": _cmd_meta,
        "/status": _cmd_status,
        "/tags": _cmd_tags,
        "/help": _cmd_help,
    }

    handler = handlers.get(cmd)
    if handler is None:
        return False

    handler(orchestrator, args)
    return True


def _cmd_memories(o: "Orchestrator", args: str) -> None:
    if args:
        result = o.handle_retrieve_memories(query=args)
        display_retrieval_results(result["results"])
    else:
        memories = o.memory_store.list_recent(20)
        if not memories:
            console.print("[dim]No memories stored yet.[/dim]")
            return
        for m in memories:
            tags_str = ", ".join(m.tags)
            content = m.content[:80] + "..." if len(m.content) > 80 else m.content
            console.print(f"  [cyan]{tags_str}[/cyan] — {content}")


def _cmd_trends(o: "Orchestrator", args: str) -> None:
    tag_names = args.split() if args else None
    result = o.handle_get_trends(tag_names=tag_names)
    display_trends(result["trends"])


def _cmd_phase(o: "Orchestrator", args: str) -> None:
    result = o.handle_get_phase_space()
    display_phase_space(result["phase_space"])


def _cmd_sync(o: "Orchestrator", args: str) -> None:
    result = o.handle_get_synchronicities(limit=20)
    display_synchronicities(result["synchronicities"])


def _cmd_meta(o: "Orchestrator", args: str) -> None:
    result = o.handle_get_meta_patterns()
    display_meta_patterns(result["meta_patterns"])


def _cmd_status(o: "Orchestrator", args: str) -> None:
    mem_count = o.memory_store.count()
    tag_count = len(o.tag_manager.list_active())
    sync_count = len(o.sync_detector.get_recent(100))
    meta_count = len(o.meta_analyzer.get_active_patterns())

    display_status({
        "Memories": mem_count,
        "Active Tags": tag_count,
        "Synchronicity Events": sync_count,
        "Meta-Patterns": meta_count,
        "LLM Model": o._config.llm_model,
        "Embedding Model": o._config.embedding_model,
    })


def _cmd_tags(o: "Orchestrator", args: str) -> None:
    tags = o.tag_manager.list_active_names()
    if not tags:
        console.print("[dim]No tags yet.[/dim]")
        return
    console.print(f"Active tags ({len(tags)}): [cyan]{', '.join(tags)}[/cyan]")


def _cmd_help(o: "Orchestrator", args: str) -> None:
    console.print()
    console.print("[bold]Chicory Commands[/bold]")
    console.print("  /memories [query]  — List recent memories or search")
    console.print("  /trends [tags]     — Show tag trend data")
    console.print("  /phase             — Show phase space (trend vs retrieval)")
    console.print("  /sync              — Show synchronicity events")
    console.print("  /meta              — Show meta-patterns")
    console.print("  /status            — Show system status")
    console.print("  /tags              — List active tags")
    console.print("  /help              — Show this help")
    console.print("  /quit              — Exit")
    console.print()
