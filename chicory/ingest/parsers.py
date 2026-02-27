"""Document parsers for various file types."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional


def parse_file(path: Path) -> Optional[str]:
    """Parse a file and return its text content, or None if unsupported."""
    suffix = path.suffix.lower()
    parsers = {
        ".txt": _parse_text,
        ".md": _parse_text,
        ".markdown": _parse_text,
        ".rst": _parse_text,
        ".py": _parse_text,
        ".js": _parse_text,
        ".ts": _parse_text,
        ".jsx": _parse_text,
        ".tsx": _parse_text,
        ".java": _parse_text,
        ".c": _parse_text,
        ".cpp": _parse_text,
        ".h": _parse_text,
        ".go": _parse_text,
        ".rs": _parse_text,
        ".rb": _parse_text,
        ".sh": _parse_text,
        ".yaml": _parse_text,
        ".yml": _parse_text,
        ".toml": _parse_text,
        ".ini": _parse_text,
        ".cfg": _parse_text,
        ".html": _parse_text,
        ".xml": _parse_text,
        ".css": _parse_text,
        ".sql": _parse_text,
        ".r": _parse_text,
        ".tex": _parse_text,
        ".log": _parse_text,
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".json": _parse_json,
        ".csv": _parse_csv,
    }

    parser = parsers.get(suffix)
    if parser is None:
        return None

    try:
        return parser(path)
    except Exception:
        return None


def _parse_text(path: Path) -> str:
    """Read plain text files."""
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _parse_docx(path: Path) -> str:
    """Extract text from Word documents."""
    import docx

    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _parse_json(path: Path) -> str:
    """Parse JSON and return formatted string."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, indent=2, ensure_ascii=False)


def _parse_csv(path: Path) -> str:
    """Parse CSV and return as readable text."""
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return ""

    # Format as a readable table
    lines = []
    header = rows[0] if rows else []
    lines.append(" | ".join(header))
    lines.append("-" * len(lines[0]))
    for row in rows[1:]:
        lines.append(" | ".join(row))

    return "\n".join(lines)


SUPPORTED_EXTENSIONS = set(
    ext for ext in [
        ".txt", ".md", ".markdown", ".rst",
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
        ".go", ".rs", ".rb", ".sh",
        ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".html", ".xml", ".css", ".sql", ".r", ".tex", ".log",
        ".pdf", ".docx", ".json", ".csv",
    ]
)
