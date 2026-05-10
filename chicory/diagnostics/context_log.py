"""JSONL context logger — writes activation traces to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from chicory.diagnostics.activation_trace import ActivationTrace

logger = logging.getLogger(__name__)


class ContextLogger:
    """Appends retrieval traces to JSONL files.

    Two files:
      - trace.jsonl: query, tag activations, score breakdowns, timing
      - context.jsonl: same + full assembled context (what the LLM sees)
    """

    def __init__(self, log_dir: Path, log_full: bool = False) -> None:
        self._log_dir = log_dir
        self._log_full = log_full
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self._log_dir / "trace.jsonl"
        self._context_path = self._log_dir / "context.jsonl"

    def log(self, trace: ActivationTrace) -> None:
        try:
            self._append(self._trace_path, trace.to_dict(include_context=False))
            if self._log_full and trace.context_entries is not None:
                self._append(self._context_path, trace.to_dict(include_context=True))
        except Exception:
            logger.exception("Failed to write context log")

    @staticmethod
    def _append(path: Path, data: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
