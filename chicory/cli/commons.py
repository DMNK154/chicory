"""CLI sub-commands for chicory-commons (cross-project signal federation)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="commons",
    help="Cross-project signal federation (chicory-commons)",
    no_args_is_help=True,
)

console = Console()

DEFAULT_DB = str(Path.home() / ".chicory" / "commons.db")


def _open_commons_db(db_path: str) -> sqlite3.Connection:
    """Open the commons SQLite DB directly for read queries."""
    path = Path(db_path)
    if not path.exists():
        console.print(f"[red]Commons DB not found: {path}[/red]")
        console.print("Run [bold]chicory commons init[/bold] to create it.")
        raise typer.Exit(1)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _make_orchestrator(db_path: str):
    """Create an Orchestrator pointed at the commons DB."""
    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    config = load_config(db_path=Path(db_path))
    return Orchestrator(config)


@app.command()
def chat(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
    model: Optional[str] = typer.Option(None, help="Override LLM model"),
) -> None:
    """Start an interactive chat session against the commons."""
    from chicory.cli.chat import ChatSession
    from chicory.config import load_config

    overrides: dict = {"db_path": Path(db)}
    if model:
        overrides["llm_model"] = model

    config = load_config(**overrides)
    session = ChatSession(config)
    session.run()


@app.command()
def init(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
) -> None:
    """Initialize a new commons database."""
    from chicory.config import load_config
    from chicory.db.engine import DatabaseEngine
    from chicory.db.schema import apply_schema

    path = Path(db)
    existed = path.exists()
    config = load_config(db_path=path)
    engine = DatabaseEngine(config)
    engine.connect()
    apply_schema(engine)
    engine.close()

    if existed:
        console.print(f"[green]Commons DB migrated:[/green] {path}")
    else:
        console.print(f"[green]Commons DB created:[/green] {path}")


@app.command()
def status(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
) -> None:
    """Show commons system status."""
    conn = _open_commons_db(db)
    try:
        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        tag_count = conn.execute(
            "SELECT COUNT(*) FROM tags WHERE is_active = 1"
        ).fetchone()[0]

        # pending_signals table may not exist in older DBs
        try:
            pending = conn.execute(
                "SELECT COUNT(*) FROM pending_signals WHERE processed = 0"
            ).fetchone()[0]
            processed = conn.execute(
                "SELECT COUNT(*) FROM pending_signals WHERE processed = 1"
            ).fetchone()[0]
            project_count = conn.execute(
                "SELECT COUNT(DISTINCT project_id) FROM pending_signals"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            pending = processed = project_count = 0

        table = Table(title="Commons Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Database", db)
        table.add_row("Memories", str(mem_count))
        table.add_row("Active Tags", str(tag_count))
        table.add_row("Projects", str(project_count))
        table.add_row("Pending Signals", str(pending))
        table.add_row("Processed Signals", str(processed))
        console.print(table)
    finally:
        conn.close()


@app.command()
def projects(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
) -> None:
    """List projects that have emitted signals."""
    conn = _open_commons_db(db)
    try:
        rows = conn.execute("""
            SELECT project_id,
                   COUNT(*) AS total_signals,
                   SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) AS pending,
                   MIN(created_at) AS first_seen,
                   MAX(created_at) AS last_seen
            FROM pending_signals
            GROUP BY project_id
            ORDER BY last_seen DESC
        """).fetchall()

        if not rows:
            console.print("[dim]No projects have emitted signals yet.[/dim]")
            return

        table = Table(title="Commons Projects")
        table.add_column("Project", style="cyan")
        table.add_column("Total Signals", justify="right")
        table.add_column("Pending", justify="right", style="yellow")
        table.add_column("First Seen", style="dim")
        table.add_column("Last Seen")
        for r in rows:
            table.add_row(
                r["project_id"],
                str(r["total_signals"]),
                str(r["pending"]),
                r["first_seen"],
                r["last_seen"],
            )
        console.print(table)
    except sqlite3.OperationalError:
        console.print("[dim]No pending_signals table found.[/dim]")
    finally:
        conn.close()


@app.command()
def signals(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
    project: Optional[str] = typer.Option(None, help="Filter by project ID"),
    pending_only: bool = typer.Option(False, "--pending", help="Show only unprocessed signals"),
    limit: int = typer.Option(20, help="Max signals to show"),
) -> None:
    """List recent signals."""
    conn = _open_commons_db(db)
    try:
        clauses = []
        params: list = []
        if pending_only:
            clauses.append("processed = 0")
        if project:
            clauses.append("project_id = ?")
            params.append(project)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM pending_signals {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

        if not rows:
            console.print("[dim]No signals found.[/dim]")
            return

        table = Table(title="Signals")
        table.add_column("ID", style="dim", width=5)
        table.add_column("Project", style="cyan")
        table.add_column("Type")
        table.add_column("Tags", max_width=40)
        table.add_column("Strength", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Created", style="dim")

        for r in rows:
            tags = json.loads(r["tags"]) if r["tags"] else []
            tags_str = ", ".join(tags[:6])
            if len(tags) > 6:
                tags_str += f" (+{len(tags) - 6})"
            strength = f"{r['strength']:.2f}" if r["strength"] is not None else ""
            status_str = "[green]done[/green]" if r["processed"] else "[yellow]pending[/yellow]"
            table.add_row(
                str(r["id"]),
                r["project_id"],
                r["op_type"],
                tags_str,
                strength,
                status_str,
                r["created_at"],
            )
        console.print(table)
    except sqlite3.OperationalError:
        console.print("[dim]No pending_signals table found.[/dim]")
    finally:
        conn.close()


@app.command()
def process(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
) -> None:
    """Process pending signals into commons memories."""
    try:
        from chicory_commons import SignalProcessor
    except ImportError:
        console.print(
            "[red]chicory-commons-man not installed.[/red]\n"
            "Run: [bold]pip install chicory-man\\[commons][/bold]"
        )
        raise typer.Exit(1)

    o = _make_orchestrator(db)
    try:
        processor = SignalProcessor(o)
        result = processor.process_pending()
        console.print(f"[green]Processed {result.get('processed', 0)} signals.[/green]")
        if result.get("memories_created"):
            console.print(f"  Memories created: {result['memories_created']}")
        if result.get("errors"):
            console.print(f"  [yellow]Errors: {result['errors']}[/yellow]")
    finally:
        o.close()


@app.command()
def retrieve(
    query: str = typer.Argument(help="Search query"),
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tag filter"),
    method: str = typer.Option("hybrid", help="Retrieval method: semantic, tag, hybrid"),
    top_k: int = typer.Option(10, help="Max results"),
) -> None:
    """Search commons memories across all projects."""
    from chicory.cli.display import display_retrieval_results

    o = _make_orchestrator(db)
    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        result = o.handle_retrieve_memories(
            query=query, tags=tag_list, method=method, top_k=top_k,
        )
        display_retrieval_results(result["results"])
    finally:
        o.close()


@app.command()
def trends(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tag names"),
) -> None:
    """Show tag trends in the commons."""
    from chicory.cli.display import display_trends

    o = _make_orchestrator(db)
    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        result = o.handle_get_trends(tag_names=tag_list)
        display_trends(result["trends"])
    finally:
        o.close()


@app.command()
def sync(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
    limit: int = typer.Option(20, help="Max events to show"),
) -> None:
    """Show synchronicity events in the commons."""
    from chicory.cli.display import display_synchronicities

    o = _make_orchestrator(db)
    try:
        result = o.handle_get_synchronicities(limit=limit)
        display_synchronicities(result["synchronicities"])
    finally:
        o.close()


@app.command()
def meta(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
) -> None:
    """Show meta-patterns in the commons."""
    from chicory.cli.display import display_meta_patterns

    o = _make_orchestrator(db)
    try:
        result = o.handle_get_meta_patterns()
        display_meta_patterns(result["meta_patterns"])
    finally:
        o.close()


@app.command(name="tags")
def list_tags(
    db: str = typer.Option(DEFAULT_DB, help="Path to the commons database"),
    project: Optional[str] = typer.Option(None, help="Filter by project prefix"),
) -> None:
    """List active tags in the commons."""
    conn = _open_commons_db(db)
    try:
        rows = conn.execute(
            "SELECT name FROM tags WHERE is_active = 1 ORDER BY name"
        ).fetchall()
        names = [r["name"] for r in rows]
        if project:
            prefix = f"{project}:"
            names = [n for n in names if n.startswith(prefix) or not ":" in n]

        if not names:
            console.print("[dim]No tags found.[/dim]")
            return

        # Group: project-namespaced vs shared
        namespaced = [n for n in names if ":" in n]
        shared = [n for n in names if ":" not in n]

        if shared:
            console.print(f"[bold]Shared tags[/bold] ({len(shared)}):")
            console.print(f"  [cyan]{', '.join(shared)}[/cyan]")
        if namespaced:
            console.print(f"[bold]Namespaced tags[/bold] ({len(namespaced)}):")
            # Group by project
            by_project: dict[str, list[str]] = {}
            for n in namespaced:
                proj, _, tag = n.partition(":")
                by_project.setdefault(proj, []).append(tag)
            for proj, ptags in sorted(by_project.items()):
                console.print(f"  [bold]{proj}[/bold]: [cyan]{', '.join(ptags)}[/cyan]")
    finally:
        conn.close()
