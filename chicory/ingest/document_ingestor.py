"""General-purpose document ingestor — stores file content as memories.

Unlike code_summarizer which extracts structural summaries, this ingestor
stores the actual text content of files, chunking long documents into
retrievable pieces. Works with any text-based file format.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

MAX_CHUNK_CHARS = 3000
OVERLAP_CHARS = 200

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".class", ".o",
    ".whl", ".egg",
    ".db", ".sqlite", ".sqlite3",
    ".pdf",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".safetensors", ".bin", ".pt", ".onnx",
}


def is_text_file(path: Path) -> bool:
    """Check if a file is likely text-based."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return False
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return False
        return True
    except Exception:
        return False


def read_document(path: Path) -> Optional[str]:
    """Read a document file and return its text content."""
    if not path.is_file():
        return None
    if not is_text_file(path):
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def chunk_document(
    text: str,
    rel_path: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[dict]:
    """Split a document into chunks for memory storage.

    Short documents become a single chunk. Long documents are split at
    paragraph boundaries with overlap for retrieval continuity.

    Returns list of {content, chunk_index, total_chunks}.
    """
    if not text.strip():
        return []

    if len(text) <= max_chars:
        return [{
            "content": f"[{rel_path}]\n\n{text.strip()}",
            "chunk_index": 0,
            "total_chunks": 1,
        }]

    paragraphs = re.split(r"\n\s*\n", text)

    chunks: list[dict] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    total = len(chunks)
    return [
        {
            "content": f"[{rel_path} (part {i + 1}/{total})]\n\n{chunk}",
            "chunk_index": i,
            "total_chunks": total,
        }
        for i, chunk in enumerate(chunks)
    ]


def derive_document_tags(filepath: Path, base_dir: Path | None = None) -> list[str]:
    """Derive tags from a document's path and type."""
    tags = ["document"]

    ext = filepath.suffix.lower()
    ext_tags = {
        ".txt": "text", ".md": "markdown", ".rst": "restructuredtext",
        ".csv": "csv", ".tsv": "tsv",
        ".log": "log", ".json": "json", ".jsonl": "jsonl",
        ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".ini": "config", ".cfg": "config",
        ".env": "config", ".properties": "config",
        ".html": "html", ".htm": "html", ".css": "css",
        ".tex": "latex", ".bib": "latex",
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".sh": "shell", ".bat": "batch", ".ps1": "powershell",
        ".sql": "sql", ".r": "r", ".jl": "julia",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".c": "c", ".cpp": "cpp", ".h": "header",
    }
    ext_tag = ext_tags.get(ext)
    if ext_tag:
        tags.append(ext_tag)

    if ext and ext != ".":
        tags.append(f"ext:{ext.lstrip('.')}")

    # Stem as a tag if meaningful
    stem = filepath.stem.lower().replace(" ", "-").replace("_", "-")
    if len(stem) > 2 and stem not in {"readme", "index", "main", "app"}:
        tags.append(stem)

    # Directory-based tags
    if base_dir:
        try:
            rel = filepath.relative_to(base_dir)
            skip = {"src", "lib", "dist", ".", "app", "docs", "doc"}
            for part in rel.parent.parts:
                tag = part.lower().replace(" ", "-").replace("_", "-")
                if tag and len(tag) > 1 and tag not in skip:
                    tags.append(tag)
        except ValueError:
            pass

    return tags
