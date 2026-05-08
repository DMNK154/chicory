"""Tests for document ingestion behavior."""

from __future__ import annotations

from pathlib import Path

from chicory.ingest.chunker import Chunk
from chicory.ingest.ingestor import ingest_file
from chicory.layer1.tag_manager import TagManager


class RecordingOrchestrator:
    """Minimal orchestrator surface needed by ingest_file."""

    def __init__(self, db):
        self.db = db
        self.tag_manager = TagManager(db)
        self.store_count = 0

    def handle_store_memory(
        self,
        content: str,
        tags: list[str],
        importance: float | None = None,
        summary: str | None = None,
        content_hash: str | None = None,
        skip_embedding: bool = False,
        defer_side_effects: bool = False,
    ) -> dict[str, str]:
        self.store_count += 1
        memory_id = f"memory-{self.store_count}"
        self.db.execute(
            """
            INSERT INTO memories
                (id, content, summary, source_model, salience_model, content_hash)
            VALUES (?, ?, ?, 'test-model', ?, ?)
            """,
            (memory_id, content, summary, importance or 0.5, content_hash),
        )
        self.db.connection.commit()
        return {"status": "stored", "memory_id": memory_id}


def test_ingest_file_skips_duplicate_chunks_created_in_same_file(
    db,
    monkeypatch,
):
    path = Path("duplicate.txt")
    duplicate_text = "same extracted page text"

    monkeypatch.setattr(
        "chicory.ingest.ingestor.parse_file",
        lambda _: duplicate_text,
    )
    monkeypatch.setattr(
        "chicory.ingest.ingestor.chunk_document",
        lambda text, source_file, chunk_size, overlap: [
            Chunk(duplicate_text, 0, 2, source_file),
            Chunk(duplicate_text, 1, 2, source_file),
        ],
    )

    count, _new_ids = ingest_file(RecordingOrchestrator(db), path)

    rows = db.execute("SELECT content_hash FROM memories").fetchall()
    assert count == 1
    assert len(rows) == 1
