"""Document ingestion pipeline — parse, chunk, store as memories."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from chicory.config import ChicoryConfig
from chicory.ingest.chunker import Chunk, chunk_document
from chicory.ingest.parsers import SUPPORTED_EXTENSIONS, parse_file
from chicory.orchestrator.orchestrator import Orchestrator

console = Console()


def _derive_tags(path: Path, base_dir: Path | None = None) -> list[str]:
    """Derive tags from file path and extension."""
    tags = []

    # Tag from file extension (file type)
    ext_tags = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".md": "markdown", ".pdf": "pdf", ".docx": "word-doc",
        ".json": "json", ".csv": "data", ".sql": "sql",
        ".html": "html", ".css": "css", ".tex": "latex",
        ".yaml": "config", ".yml": "config", ".toml": "config",
    }
    ext_tag = ext_tags.get(path.suffix.lower())
    if ext_tag:
        tags.append(ext_tag)

    # Tags from directory structure
    if base_dir:
        try:
            rel = path.relative_to(base_dir)
            for part in rel.parent.parts:
                # Clean directory name into a tag
                tag = part.lower().replace(" ", "-").replace("_", "-")
                if tag and len(tag) > 1 and tag not in (".", "src", "lib", "dist"):
                    tags.append(tag)
        except ValueError:
            pass

    # Always tag as ingested document
    tags.append("document")

    return tags


def _content_hash(text: str) -> str:
    """Hash content to detect duplicates."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ingest_file(
    orchestrator: Orchestrator,
    path: Path,
    base_dir: Path | None = None,
    chunk_size: int = 2000,
    overlap: int = 400,
) -> int:
    """Ingest a single file. Returns number of memories created."""
    text = parse_file(path)
    if not text or not text.strip():
        return 0

    tags = _derive_tags(path, base_dir)
    chunks = chunk_document(text, str(path), chunk_size, overlap)

    count = 0
    for chunk in chunks:
        # Build a summary line
        if chunk.total > 1:
            summary = f"{path.name} [{chunk.index + 1}/{chunk.total}]"
            if chunk.section_title:
                summary += f" — {chunk.section_title}"
        else:
            summary = path.name

        # Check for duplicates by content hash
        content_hash = _content_hash(chunk.text)
        existing = orchestrator.db.execute(
            "SELECT id FROM memories WHERE content LIKE ?",
            (f"%{content_hash}%",),
        ).fetchone()
        if existing:
            continue

        # Store with hash appended as metadata for dedup
        content = chunk.text + f"\n\n<!-- chicory:hash={content_hash} -->"

        orchestrator.handle_store_memory(
            content=content,
            tags=tags,
            importance=0.4,  # Documents start at moderate importance
            summary=summary,
        )
        count += 1

    return count


def ingest_directory(
    orchestrator: Orchestrator,
    directory: Path,
    recursive: bool = True,
    chunk_size: int = 2000,
    overlap: int = 400,
) -> dict[str, int]:
    """Ingest all supported files in a directory.

    Returns dict with stats: files_found, files_ingested, memories_created.
    """
    if not directory.is_dir():
        console.print(f"[red]Not a directory: {directory}[/red]")
        return {"files_found": 0, "files_ingested": 0, "memories_created": 0}

    # Collect all supported files
    if recursive:
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    # Skip hidden files and common non-content directories
    skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".env"}
    files = [
        f for f in files
        if not any(part.startswith(".") or part in skip_dirs for part in f.parts)
    ]

    files.sort()

    stats = {"files_found": len(files), "files_ingested": 0, "memories_created": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(files))

        for f in files:
            progress.update(task, description=f"Ingesting {f.name}...")
            try:
                count = ingest_file(orchestrator, f, base_dir=directory,
                                    chunk_size=chunk_size, overlap=overlap)
                if count > 0:
                    stats["files_ingested"] += 1
                    stats["memories_created"] += count
            except Exception as e:
                console.print(f"  [yellow]Skipped {f.name}: {e}[/yellow]")
            progress.advance(task)

    return stats
