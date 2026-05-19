"""Load .chicoryignore patterns and check paths against them."""

from __future__ import annotations

import fnmatch
from pathlib import Path

DEFAULT_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".env", ".tox", ".mypy_cache", ".pytest_cache", "dist",
    "build", ".eggs", "*.egg-info",
}


def load_ignore_patterns(directory: Path) -> set[str]:
    """Load patterns from .chicoryignore in *directory*, merged with defaults.

    Each non-blank, non-comment line is a pattern. Lines are matched against
    relative path parts (directory names) and against the full relative path
    via fnmatch for glob patterns containing ``/`` or ``*``.
    """
    patterns = set(DEFAULT_SKIP_DIRS)
    ignore_file = directory / ".chicoryignore"
    if ignore_file.is_file():
        for line in ignore_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.add(line.rstrip("/"))
    return patterns


def is_ignored(file_path: Path, base_dir: Path, patterns: set[str]) -> bool:
    """Return True if *file_path* should be skipped during ingestion."""
    try:
        rel = file_path.relative_to(base_dir)
    except ValueError:
        return False

    rel_str = str(rel)
    parts = rel.parts

    glob_patterns = [p for p in patterns if "*" in p or "?" in p]

    for part in parts[:-1]:
        if part.startswith("."):
            return True
        if part in patterns:
            return True
        if any(fnmatch.fnmatch(part, p) for p in glob_patterns):
            return True

    filename = parts[-1] if parts else ""
    if any(fnmatch.fnmatch(filename, p) for p in glob_patterns):
        return True

    for pat in patterns:
        if ("/" in pat or "\\" in pat) and fnmatch.fnmatch(rel_str, pat):
            return True

    return False
