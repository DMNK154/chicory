"""Typer CLI entry point for Chicory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="chicory",
    help="Dual-tracking memory architecture for LLMs",
    no_args_is_help=True,
)


@app.command()
def chat(
    db: str = typer.Option("chicory.db", help="Path to the database file"),
    model: Optional[str] = typer.Option(None, help="Override LLM model"),
) -> None:
    """Start an interactive chat session with memory."""
    from chicory.cli.chat import ChatSession
    from chicory.config import load_config

    overrides = {"db_path": Path(db)}
    if model:
        overrides["llm_model"] = model

    config = load_config(**overrides)
    session = ChatSession(config)
    session.run()


@app.command()
def status(
    db: str = typer.Option("chicory.db", help="Path to the database file"),
) -> None:
    """Show system status."""
    from chicory.cli.commands import _cmd_status
    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    config = load_config(db_path=Path(db))
    o = Orchestrator(config)
    try:
        _cmd_status(o, "")
    finally:
        o.close()


@app.command()
def reembed(
    db: str = typer.Option("chicory.db", help="Path to the database file"),
    model: Optional[str] = typer.Option(None, help="New embedding model name"),
) -> None:
    """Re-embed all memories (for model migration)."""
    from rich.console import Console

    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    console = Console()
    config = load_config(db_path=Path(db))
    o = Orchestrator(config)

    try:
        console.print("[bold]Re-embedding all memories...[/bold]")
        count = o._embedding_engine.reembed_all(new_model=model)
        console.print(f"[green]Done. Re-embedded {count} memories.[/green]")
    finally:
        o.close()


@app.command()
def migrate(
    db: str = typer.Option("chicory.db", help="Path to the database file"),
    new_model: Optional[str] = typer.Option(None, help="New LLM model name"),
    new_embedding: Optional[str] = typer.Option(None, help="New embedding model"),
) -> None:
    """Run a full model migration."""
    from chicory.config import load_config
    from chicory.migration.model_update import run_migration

    config = load_config(db_path=Path(db))
    run_migration(config, new_llm_model=new_model, new_embedding_model=new_embedding)


@app.command()
def ingest(
    path: str = typer.Argument(help="File or directory to ingest"),
    db: str = typer.Option("chicory.db", help="Path to the database file"),
    recursive: bool = typer.Option(True, help="Recurse into subdirectories"),
    chunk_size: int = typer.Option(2000, help="Max characters per chunk"),
    overlap: int = typer.Option(400, help="Character overlap between chunks"),
) -> None:
    """Ingest documents into memory. Accepts a file or directory."""
    from rich.console import Console

    from chicory.config import load_config
    from chicory.ingest.ingestor import ingest_directory, ingest_file
    from chicory.orchestrator.orchestrator import Orchestrator

    console = Console()
    config = load_config(db_path=Path(db))
    o = Orchestrator(config)

    target = Path(path)
    try:
        if target.is_file():
            console.print(f"[bold]Ingesting {target.name}...[/bold]")
            count = ingest_file(o, target, chunk_size=chunk_size, overlap=overlap)
            console.print(f"[green]Done. Created {count} memories.[/green]")
        elif target.is_dir():
            console.print(f"[bold]Ingesting directory: {target}[/bold]")
            stats = ingest_directory(
                o, target, recursive=recursive,
                chunk_size=chunk_size, overlap=overlap,
            )
            console.print()
            console.print(f"[green]Done.[/green]")
            console.print(f"  Files found:      {stats['files_found']}")
            console.print(f"  Files ingested:   {stats['files_ingested']}")
            console.print(f"  Memories created: {stats['memories_created']}")
        else:
            console.print(f"[red]Path not found: {target}[/red]")
    finally:
        o.close()


@app.command()
def watch(
    path: str = typer.Argument(help="Directory to watch"),
    db: str = typer.Option("chicory.db", help="Path to the database file"),
    chunk_size: int = typer.Option(2000, help="Max characters per chunk"),
    overlap: int = typer.Option(400, help="Character overlap between chunks"),
) -> None:
    """Watch a directory and auto-ingest new/changed files."""
    from chicory.config import load_config
    from chicory.ingest.watcher import watch_directory
    from chicory.orchestrator.orchestrator import Orchestrator

    config = load_config(db_path=Path(db))
    o = Orchestrator(config)

    target = Path(path)
    try:
        watch_directory(o, target, chunk_size=chunk_size, overlap=overlap)
    finally:
        o.close()


@app.command(name="backfill-letters")
def backfill_letters(
    db: str = typer.Option("chicory.db", help="Path to the database file"),
) -> None:
    """Backfill single-letter tags from existing word tags on all memories."""
    from rich.console import Console

    from chicory.config import load_config
    from chicory.layer1.tag_manager import TagManager
    from chicory.orchestrator.orchestrator import Orchestrator

    console = Console()
    config = load_config(db_path=Path(db))
    o = Orchestrator(config)

    try:
        # Get all memory IDs
        rows = o._db.execute("SELECT id FROM memories").fetchall()
        total = len(rows)
        console.print(f"[bold]Backfilling letter tags for {total} memories...[/bold]")

        tagged = 0
        for i, row in enumerate(rows):
            mid = row["id"]
            # Get word tags (length > 1) for this memory
            word_tags = [
                t for t in o._tag_manager.get_tags_for_memory(mid)
                if len(t) > 1
            ]
            if not word_tags:
                continue

            letter_counts = TagManager.decompose_to_letters(word_tags)
            if letter_counts:
                letter_tag_objs = o._tag_manager.assign_letter_tags(
                    mid, letter_counts
                )
                # Record tag events
                for lt in letter_tag_objs:
                    o._trend_engine.record_event(
                        tag_id=lt.id,
                        event_type="assignment",
                        memory_id=mid,
                    )
                tagged += 1

            if (i + 1) % 500 == 0:
                o._db.connection.commit()
                console.print(f"  Processed {i + 1}/{total}...")

        o._db.connection.commit()
        console.print(
            f"[green]Done. Added letter tags to {tagged} memories.[/green]"
        )
    finally:
        o.close()


@app.command()
def dashboard(
    db: str = typer.Option("chicory.db", help="Path to the database file"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8050, help="Port to bind to"),
    debug: bool = typer.Option(False, help="Enable Dash debug mode"),
) -> None:
    """Launch the web dashboard."""
    try:
        from chicory.dashboard.app import run_dashboard
    except ImportError:
        from rich.console import Console

        Console().print(
            "[red]Dashboard dependencies not installed. "
            "Run: pip install chicory\\[dashboard][/red]"
        )
        raise typer.Exit(1)

    run_dashboard(db_path=Path(db), host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
