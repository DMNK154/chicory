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
from chicory.ingest.ignore import is_ignored, load_ignore_patterns
from chicory.ingest.parsers import SUPPORTED_EXTENSIONS, parse_file
from chicory.llm.base import BaseLLMClient
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
    llm_client: BaseLLMClient | None = None,
) -> tuple[int, list[str]]:
    """Ingest a single file.

    Returns (count of memories created, list of new memory IDs for batch embedding).
    """
    text = parse_file(path)
    if not text or not text.strip():
        return 0, []

    base_tags = _derive_tags(path, base_dir)
    existing_tag_names = orchestrator.tag_manager.list_active_names()
    chunks = chunk_document(text, str(path), chunk_size, overlap)

    if not chunks:
        return 0, []

    # Compute all content hashes up front
    chunk_hashes = [_content_hash(c.text) for c in chunks]

    # Batch dedup: single indexed query instead of per-chunk LIKE scan
    placeholders = ",".join("?" for _ in chunk_hashes)
    existing_hashes = set()
    rows = orchestrator.db.execute(
        f"SELECT content_hash FROM memories WHERE content_hash IN ({placeholders})",
        tuple(chunk_hashes),
    ).fetchall()
    for row in rows:
        existing_hashes.add(row["content_hash"])

    # Propose tags once per file using a representative sample, not per chunk
    tags = list(base_tags)
    if llm_client is not None:
        # Use first chunk as representative content for tag proposal
        sample_text = chunks[0].text
        try:
            proposed = llm_client.propose_tags(sample_text, existing_tag_names)
            for t in proposed:
                if t not in tags:
                    tags.append(t)
            for t in proposed:
                if t not in existing_tag_names:
                    existing_tag_names.append(t)
        except Exception:
            pass

    count = 0
    new_ids: list[str] = []
    for chunk, content_hash in zip(chunks, chunk_hashes):
        if content_hash in existing_hashes:
            continue

        # Build a summary line
        if chunk.total > 1:
            summary = f"{path.name} [{chunk.index + 1}/{chunk.total}]"
            if chunk.section_title:
                summary += f" — {chunk.section_title}"
        else:
            summary = path.name

        # Store with hash in both column and content metadata for dedup
        content = chunk.text + f"\n\n<!-- chicory:hash={content_hash} -->"

        result = orchestrator.handle_store_memory(
            content=content,
            tags=tags,
            importance=0.4,
            summary=summary,
            content_hash=content_hash,
            skip_embedding=True,
            defer_side_effects=True,
        )
        new_ids.append(result["memory_id"])
        existing_hashes.add(content_hash)
        count += 1

    return count, new_ids


def ingest_directory(
    orchestrator: Orchestrator,
    directory: Path,
    recursive: bool = True,
    chunk_size: int = 2000,
    overlap: int = 400,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, int]:
    """Ingest all supported files in a directory.

    Returns dict with stats: files_found, files_ingested, memories_created.
    """
    if not directory.is_dir():
        console.print(f"[red]Not a directory: {directory}[/red]")
        return {"files_found": 0, "files_ingested": 0, "memories_created": 0}

    # Collect all supported files, respecting .chicoryignore
    patterns = load_ignore_patterns(directory)
    if recursive:
        files = [
            f for f in directory.rglob("*")
            if f.is_file()
            and f.suffix.lower() in SUPPORTED_EXTENSIONS
            and not is_ignored(f, directory, patterns)
        ]
    else:
        files = [
            f for f in directory.iterdir()
            if f.is_file()
            and f.suffix.lower() in SUPPORTED_EXTENSIONS
            and not is_ignored(f, directory, patterns)
        ]

    files.sort()

    stats = {"files_found": len(files), "files_ingested": 0, "memories_created": 0}
    all_new_ids: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(files))

        for f in files:
            progress.update(task, description=f"Ingesting {f.name}...")
            try:
                count, new_ids = ingest_file(
                    orchestrator, f, base_dir=directory,
                    chunk_size=chunk_size, overlap=overlap,
                    llm_client=llm_client,
                )
                if count > 0:
                    stats["files_ingested"] += 1
                    stats["memories_created"] += count
                    all_new_ids.extend(new_ids)
            except Exception as e:
                console.print(f"  [yellow]Skipped {f.name}: {e}[/yellow]")
            progress.advance(task)

    if all_new_ids:
        console.print(f"[dim]Batch embedding {len(all_new_ids)} memories...[/dim]")
        orchestrator._batch_embed_memories(all_new_ids)

        console.print("[dim]Updating tensor networks...[/dim]")
        orchestrator._finalize_ingested_memories(all_new_ids)

    return stats
