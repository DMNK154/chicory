"""Append-only conversation logger — one Markdown file per day."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class ConversationLog:
    """Logs user/assistant exchanges to ``~/.chicory/logs/YYYY-MM-DD.md``."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir or (Path.home() / ".chicory" / "logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_today(self) -> Path:
        return self._log_dir / f"{datetime.now():%Y-%m-%d}.md"

    def _ensure_header(self, path: Path) -> None:
        """Write the date header if the file is new."""
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Chicory — {datetime.now():%Y-%m-%d}\n\n")

    def log_user(self, text: str) -> None:
        path = self._path_for_today()
        self._ensure_header(path)
        ts = datetime.now().strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## [{ts}] You\n\n{text}\n\n")

    def log_assistant(self, text: str) -> None:
        path = self._path_for_today()
        self._ensure_header(path)
        ts = datetime.now().strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## [{ts}] Chicory\n\n{text}\n\n")

    def log_tool_use(self, tool_name: str) -> None:
        path = self._path_for_today()
        self._ensure_header(path)
        ts = datetime.now().strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"*[{ts}] tool: {tool_name}*\n\n")
