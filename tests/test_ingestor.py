"""Tests for document ingestion behavior."""

from __future__ import annotations

import pytest
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
        source_path: str | None = None,
    ) -> dict[str, str]:
        self.store_count += 1
        memory_id = f"memory-{self.store_count}"
        self.db.execute(
            """
            INSERT INTO memories
                (id, content, summary, source_model, salience_model,
                 content_hash, source_path)
            VALUES (?, ?, ?, 'test-model', ?, ?, ?)
            """,
            (memory_id, content, summary, importance or 0.5, content_hash,
             source_path),
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


class TestPerDocumentDedup:
    """A global unique hash shredded document revisions.

    57 versions of one paper each kept only the passages that differed
    from all their predecessors; the final version retained one chunk of
    twenty-seven, and nothing marked that survivor as a fragment.
    """

    def test_uniqueness_is_scoped_to_the_document(self, db):
        db.execute(
            "INSERT INTO memories (id, content, source_model, content_hash, source_path) "
            "VALUES ('a', 'shared passage', 't', 'HASH1', 'draft.docx')"
        )
        # The same passage in a DIFFERENT document must be storable.
        db.execute(
            "INSERT INTO memories (id, content, source_model, content_hash, source_path) "
            "VALUES ('b', 'shared passage', 't', 'HASH1', 'final.docx')"
        )
        n = db.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE content_hash = 'HASH1'"
        ).fetchone()["n"]
        assert n == 2

    def test_the_same_document_still_dedups(self, db):
        """Re-ingesting a file must not duplicate what it already gave."""
        import sqlite3
        db.execute(
            "INSERT INTO memories (id, content, source_model, content_hash, source_path) "
            "VALUES ('a', 'passage', 't', 'HASH2', 'draft.docx')"
        )
        with pytest.raises(Exception):
            db.execute(
                "INSERT INTO memories (id, content, source_model, content_hash, source_path) "
                "VALUES ('c', 'passage', 't', 'HASH2', 'draft.docx')"
            )

    def test_the_global_index_is_gone(self, db):
        names = {
            r["name"] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
            ).fetchall()
        }
        assert "idx_memories_content_hash" not in names
        assert "idx_memories_source_hash" in names


def test_identical_text_in_a_different_file_is_kept(db, monkeypatch):
    """The revision-shredding case, end to end.

    Two documents share a passage. Under global dedup the second file
    lost it silently; scoped to the document, each keeps its own copy.
    """
    shared = "a passage both drafts contain"
    monkeypatch.setattr("chicory.ingest.ingestor.parse_file", lambda _: shared)
    monkeypatch.setattr(
        "chicory.ingest.ingestor.chunk_document",
        lambda text, source_file, chunk_size, overlap: [
            Chunk(shared, 0, 1, source_file),
        ],
    )
    orch = RecordingOrchestrator(db)
    first, _ = ingest_file(orch, Path("draft.docx"))
    second, _ = ingest_file(orch, Path("final.docx"))
    assert (first, second) == (1, 1)
    rows = db.execute(
        "SELECT source_path FROM memories ORDER BY source_path"
    ).fetchall()
    assert [r["source_path"] for r in rows] == ["draft.docx", "final.docx"]


def test_reingesting_the_same_file_still_skips_what_it_gave(db, monkeypatch):
    text = "a passage"
    monkeypatch.setattr("chicory.ingest.ingestor.parse_file", lambda _: text)
    monkeypatch.setattr(
        "chicory.ingest.ingestor.chunk_document",
        lambda t, source_file, chunk_size, overlap: [Chunk(text, 0, 1, source_file)],
    )
    orch = RecordingOrchestrator(db)
    assert ingest_file(orch, Path("same.docx"))[0] == 1
    assert ingest_file(orch, Path("same.docx"))[0] == 0
    assert db.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 1
