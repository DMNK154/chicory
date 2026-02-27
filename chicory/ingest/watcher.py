"""Watch a directory for new/changed files and auto-ingest them."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from chicory.ingest.ingestor import ingest_file
from chicory.ingest.parsers import SUPPORTED_EXTENSIONS
from chicory.orchestrator.orchestrator import Orchestrator

console = Console()

# Skip these directories
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".env"}


def _should_process(path: Path) -> bool:
    """Check if a file should be ingested."""
    if not path.is_file():
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    if any(part.startswith(".") or part in SKIP_DIRS for part in path.parts):
        return False
    return True


def watch_directory(
    orchestrator: Orchestrator,
    directory: Path,
    chunk_size: int = 2000,
    overlap: int = 400,
) -> None:
    """Watch a directory and auto-ingest new/modified files.
    Blocks until interrupted."""
    from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    class IngestHandler(FileSystemEventHandler):
        def on_created(self, event):
            if isinstance(event, FileCreatedEvent):
                self._handle(Path(event.src_path))

        def on_modified(self, event):
            if isinstance(event, FileModifiedEvent):
                self._handle(Path(event.src_path))

        def _handle(self, path: Path):
            if not _should_process(path):
                return
            try:
                count = ingest_file(
                    orchestrator, path, base_dir=directory,
                    chunk_size=chunk_size, overlap=overlap,
                )
                if count > 0:
                    console.print(
                        f"  [green]Ingested {path.name} ({count} memories)[/green]"
                    )
            except Exception as e:
                console.print(f"  [yellow]Error ingesting {path.name}: {e}[/yellow]")

    observer = Observer()
    observer.schedule(IngestHandler(), str(directory), recursive=True)
    observer.start()

    console.print(f"[bold]Watching {directory} for changes...[/bold]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        console.print("[dim]Watcher stopped.[/dim]")
