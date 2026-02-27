"""Rich formatting for CLI output."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def display_memory(mem: dict[str, Any]) -> None:
    """Display a single memory."""
    tags_str = ", ".join(mem.get("tags", []))
    console.print(Panel(
        f"{mem['content']}\n\n"
        f"[dim]Tags: {tags_str}[/dim]\n"
        f"[dim]Salience: {mem.get('salience', 0):.2f} | "
        f"Created: {mem.get('created_at', 'unknown')}[/dim]",
        title=f"[bold]{mem.get('summary', mem.get('memory_id', '')[:12])}[/bold]",
        border_style="blue",
    ))


def display_retrieval_results(results: list[dict[str, Any]]) -> None:
    """Display retrieval results."""
    if not results:
        console.print("[dim]No memories found.[/dim]")
        return

    table = Table(title="Retrieved Memories", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Content", max_width=60)
    table.add_column("Tags", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Salience", justify="right")

    for i, r in enumerate(results, 1):
        content = r["content"][:80] + "..." if len(r["content"]) > 80 else r["content"]
        tags = ", ".join(r.get("tags", []))
        table.add_row(
            str(i),
            content,
            tags,
            f"{r.get('relevance_score', 0):.3f}",
            f"{r.get('salience', 0):.2f}",
        )

    console.print(table)


def display_trends(trends: list[dict[str, Any]]) -> None:
    """Display tag trends."""
    if not trends:
        console.print("[dim]No trends available yet.[/dim]")
        return

    table = Table(title="Tag Trends")
    table.add_column("Tag", style="cyan")
    table.add_column("Temp", justify="right", style="red")
    table.add_column("Level", justify="right")
    table.add_column("Velocity", justify="right", style="green")
    table.add_column("Jerk", justify="right", style="yellow")
    table.add_column("Events", justify="right", style="dim")

    for t in trends:
        # Temperature bar
        bars = int(t["temperature"] * 10)
        temp_bar = "█" * bars + "░" * (10 - bars)
        table.add_row(
            t["tag"],
            f"{temp_bar} {t['temperature']:.2f}",
            f"{t['level']:.2f}",
            f"{t['velocity']:+.2f}",
            f"{t['jerk']:+.2f}",
            str(t["event_count"]),
        )

    console.print(table)


def display_phase_space(phase_data: dict[str, list[dict]]) -> None:
    """Display phase space as a quadrant table."""
    console.print()
    console.print("[bold]Phase Space: Trend Temperature vs Retrieval Frequency[/bold]")
    console.print()

    quadrant_labels = {
        "active_deep_work": ("Active Deep Work", "green"),
        "novel_exploration": ("Novel Exploration", "yellow"),
        "dormant_reactivation": ("Dormant Reactivation", "red"),
        "inactive": ("Inactive", "dim"),
    }

    for qname, tags in phase_data.items():
        label, style = quadrant_labels.get(qname, (qname, "white"))
        if not tags:
            console.print(f"  [{style}]{label}[/{style}]: [dim](empty)[/dim]")
            continue

        tag_strs = []
        for t in tags:
            tag_strs.append(
                f"{t['tag']} (T={t['temperature']:.2f}, R={t['retrieval_freq']:.2f})"
            )
        console.print(f"  [{style}]{label}[/{style}]: {', '.join(tag_strs)}")

    console.print()


def display_synchronicities(events: list[dict[str, Any]]) -> None:
    """Display synchronicity events."""
    if not events:
        console.print("[dim]No synchronicity events detected yet.[/dim]")
        return

    for e in events:
        style = "red" if e.get("strength", 0) > 3 else "yellow"
        ack = " [dim](seen)[/dim]" if e.get("acknowledged") else ""
        console.print(Panel(
            f"{e['description']}\n\n"
            f"[dim]Type: {e['type']} | Strength: {e['strength']:.1f} | "
            f"Quadrant: {e['quadrant']}[/dim]",
            title=f"[{style}]Synchronicity #{e.get('id', '?')}[/{style}]{ack}",
            border_style=style,
        ))


def display_meta_patterns(patterns: list[dict[str, Any]]) -> None:
    """Display meta-patterns."""
    if not patterns:
        console.print("[dim]No meta-patterns detected yet.[/dim]")
        return

    for p in patterns:
        console.print(Panel(
            f"{p['description']}\n\n"
            f"[dim]Type: {p['type']} | Confidence: {p['confidence']:.2f}[/dim]\n"
            f"[dim]Actions: {p.get('actions_taken', 'none')}[/dim]",
            title=f"[magenta]Meta-Pattern #{p.get('id', '?')}[/magenta]",
            border_style="magenta",
        ))


def display_status(stats: dict[str, Any]) -> None:
    """Display system status."""
    table = Table(title="Chicory Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for key, value in stats.items():
        table.add_row(key, str(value))

    console.print(table)
