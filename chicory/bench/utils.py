"""Shared utilities for the EnterpriseRAG-Bench integration.

Handles question loading, context formatting, UUID mapping,
and thread-safe JSONL output — compatible with the benchmark's
evaluation pipeline.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


def load_questions(
    path: str | Path,
    limit: int | None = None,
    question_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load questions from the benchmark's questions.jsonl."""
    questions: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if question_ids and q["question_id"] not in question_ids:
                continue
            questions.append(q)
            if limit and len(questions) >= limit:
                break
    return questions


def load_already_processed(output_path: str | Path) -> set[str]:
    """Load question_ids already in the output file (for --resume)."""
    path = Path(output_path)
    if not path.exists():
        return set()
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "question_id" in obj:
                    seen.add(obj["question_id"])
            except json.JSONDecodeError:
                continue
    return seen


def append_result(
    path: str | Path,
    result: dict[str, Any],
    lock: threading.Lock,
) -> None:
    """Thread-safe append of a single JSON line to the output file."""
    line = json.dumps(result, ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def extract_document_content(doc: dict[str, Any]) -> tuple[str, str]:
    """Extract title and body from a benchmark document using its field labels.

    Returns (title, content) where content may join multiple fields.
    """
    title_field = doc.get("title_field_name", "")
    content_fields = doc.get("content_field_names", [])

    title = str(doc.get(title_field, "")) if title_field else ""

    if len(content_fields) == 1:
        content = str(doc.get(content_fields[0], ""))
    elif len(content_fields) > 1:
        parts = []
        for field in content_fields:
            val = doc.get(field, "")
            if val:
                parts.append(f"{field}:\n{val}")
        content = "\n\n".join(parts)
    else:
        content = ""

    return title, content


def format_context_documents(
    doc_uuids: list[str],
    doc_contents: dict[str, tuple[str, str]],
) -> str:
    """Format retrieved documents into the benchmark's context string.

    Args:
        doc_uuids: ordered list of dataset_doc_uuid strings
        doc_contents: mapping of uuid → (title, content)
    """
    parts: list[str] = []
    for i, uuid in enumerate(doc_uuids, 1):
        title, content = doc_contents.get(uuid, ("", ""))
        parts.append(
            f"--- Document {i} (ID: {uuid}) ---\n"
            f"Title: {title}\n\n"
            f"{content}"
        )
    return "\n\n".join(parts)


def build_uuid_memory_map(db) -> dict[str, list[str]]:
    """Build mapping from benchmark dataset_doc_uuid → list of Chicory memory IDs.

    Multiple memory IDs per UUID when a critical document was chunked.
    """
    rows = db.execute(
        "SELECT id, source_path FROM memories WHERE source_path IS NOT NULL"
    ).fetchall()

    mapping: dict[str, list[str]] = {}
    for row in rows:
        uuid = row["source_path"]
        mapping.setdefault(uuid, []).append(row["id"])
    return mapping


def build_memory_uuid_map(db) -> dict[str, str]:
    """Build reverse mapping: Chicory memory_id → benchmark dataset_doc_uuid."""
    rows = db.execute(
        "SELECT id, source_path FROM memories WHERE source_path IS NOT NULL"
    ).fetchall()
    return {row["id"]: row["source_path"] for row in rows}
