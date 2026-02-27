"""Tag vocabulary management."""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher

from chicory.db.engine import DatabaseEngine
from chicory.exceptions import TagNotFoundError
from chicory.models.memory import Tag


def _normalize_tag(name: str) -> str:
    """Normalize a tag name: lowercase, strip, replace spaces with hyphens."""
    name = name.strip().lower()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^a-z0-9\-]", "", name)
    return name


class TagManager:
    """Manages the constrained tag vocabulary."""

    def __init__(self, db: DatabaseEngine) -> None:
        self._db = db

    def get_or_create(
        self,
        name: str,
        created_by: str = "system",
        description: str | None = None,
    ) -> Tag:
        """Get an existing tag or create a new one."""
        normalized = _normalize_tag(name)
        row = self._db.execute(
            "SELECT * FROM tags WHERE name = ?", (normalized,)
        ).fetchone()

        if row:
            return self._row_to_tag(row)

        self._db.execute(
            "INSERT INTO tags (name, description, created_by) VALUES (?, ?, ?)",
            (normalized, description, created_by),
        )
        self._db.connection.commit()

        row = self._db.execute(
            "SELECT * FROM tags WHERE name = ?", (normalized,)
        ).fetchone()
        return self._row_to_tag(row)

    def get_by_name(self, name: str) -> Tag | None:
        """Get a tag by name, or None."""
        normalized = _normalize_tag(name)
        row = self._db.execute(
            "SELECT * FROM tags WHERE name = ?", (normalized,)
        ).fetchone()
        return self._row_to_tag(row) if row else None

    def get_by_id(self, tag_id: int) -> Tag:
        """Get a tag by ID."""
        row = self._db.execute(
            "SELECT * FROM tags WHERE id = ?", (tag_id,)
        ).fetchone()
        if not row:
            raise TagNotFoundError(f"Tag {tag_id} not found")
        return self._row_to_tag(row)

    def validate_tags(self, tag_names: list[str]) -> list[Tag]:
        """Get or create tags for a list of names. Returns Tag objects."""
        return [self.get_or_create(name, created_by="llm") for name in tag_names]

    def merge_tags(self, source_id: int, target_id: int) -> None:
        """Merge source tag into target. Reassigns all memory_tags."""
        with self._db.transaction():
            # Reassign memory_tags, skipping duplicates
            self._db.execute(
                """
                UPDATE OR IGNORE memory_tags SET tag_id = ?
                WHERE tag_id = ?
                """,
                (target_id, source_id),
            )
            # Delete any remaining (were duplicates)
            self._db.execute(
                "DELETE FROM memory_tags WHERE tag_id = ?", (source_id,)
            )
            # Mark source as merged
            self._db.execute(
                "UPDATE tags SET is_active = 0, merged_into = ? WHERE id = ?",
                (target_id, source_id),
            )
            # Consolidate tag relational tensor entries
            self._consolidate_tensor_on_merge(source_id, target_id)

    def list_active(self) -> list[Tag]:
        """List all active tags."""
        rows = self._db.execute(
            "SELECT * FROM tags WHERE is_active = 1 ORDER BY name"
        ).fetchall()
        return [self._row_to_tag(r) for r in rows]

    def list_active_names(self) -> list[str]:
        """List all active tag names."""
        rows = self._db.execute(
            "SELECT name FROM tags WHERE is_active = 1 ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    def get_co_occurrence_count(self, tag_a_id: int, tag_b_id: int) -> int:
        """Count memories that have both tags."""
        row = self._db.execute(
            """
            SELECT COUNT(*) as cnt FROM memory_tags a
            JOIN memory_tags b ON a.memory_id = b.memory_id
            WHERE a.tag_id = ? AND b.tag_id = ?
            """,
            (tag_a_id, tag_b_id),
        ).fetchone()
        return row["cnt"] if row else 0

    def get_all_co_occurrences(
        self, min_count: int = 1
    ) -> list[tuple[int, int, int]]:
        """Return all (tag_a_id, tag_b_id, count) pairs with count >= min_count."""
        rows = self._db.execute(
            """
            SELECT a.tag_id, b.tag_id, COUNT(*) as cnt
            FROM memory_tags a
            JOIN memory_tags b ON a.memory_id = b.memory_id
              AND a.tag_id < b.tag_id
            GROUP BY a.tag_id, b.tag_id
            HAVING cnt >= ?
            """,
            (min_count,),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def find_similar_tags(self, name: str, threshold: float = 0.8) -> list[Tag]:
        """Find active tags with similar names using string similarity."""
        normalized = _normalize_tag(name)
        active = self.list_active()
        results = []
        for tag in active:
            ratio = SequenceMatcher(None, normalized, tag.name).ratio()
            if ratio >= threshold and tag.name != normalized:
                results.append(tag)
        return results

    def get_tag_count_for_memory(self, memory_id: str) -> int:
        """Count tags assigned to a memory."""
        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM memory_tags WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def get_tags_for_memory(self, memory_id: str) -> list[str]:
        """Get tag names for a memory."""
        rows = self._db.execute(
            """
            SELECT t.name FROM tags t
            JOIN memory_tags mt ON t.id = mt.tag_id
            WHERE mt.memory_id = ?
            ORDER BY t.name
            """,
            (memory_id,),
        ).fetchall()
        return [r["name"] for r in rows]

    def get_tags_for_memories(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """Get tag names for multiple memories in a single query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"""
            SELECT mt.memory_id, t.name FROM tags t
            JOIN memory_tags mt ON t.id = mt.tag_id
            WHERE mt.memory_id IN ({placeholders})
            ORDER BY t.name
            """,
            tuple(memory_ids),
        ).fetchall()
        result: dict[str, list[str]] = {mid: [] for mid in memory_ids}
        for r in rows:
            result[r["memory_id"]].append(r["name"])
        return result

    def get_tag_ids_for_memory(self, memory_id: str) -> list[int]:
        """Get tag IDs for a memory."""
        rows = self._db.execute(
            "SELECT tag_id FROM memory_tags WHERE memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [r["tag_id"] for r in rows]

    def get_tag_ids_for_memories(self, memory_ids: list[str]) -> dict[str, list[int]]:
        """Get tag IDs for multiple memories in a single query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT memory_id, tag_id FROM memory_tags WHERE memory_id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        result: dict[str, list[int]] = {mid: [] for mid in memory_ids}
        for r in rows:
            result[r["memory_id"]].append(r["tag_id"])
        return result

    @staticmethod
    def decompose_to_letters(tag_names: list[str]) -> dict[str, int]:
        """Decompose tag names into letter frequency counts.

        Counts each letter's occurrences across all tags, preserving repetitions.
        E.g. ["philosophy"] -> {p:2, h:2, i:1, l:1, o:2, s:1, y:1}
        """
        counts: dict[str, int] = {}
        for name in tag_names:
            for ch in name.lower():
                if ch.isalpha():
                    counts[ch] = counts.get(ch, 0) + 1
        return counts

    def assign_letter_tags(
        self, memory_id: str, letter_counts: dict[str, int]
    ) -> list[Tag]:
        """Assign single-letter tags with confidence = frequency count.

        Does NOT commit — caller is responsible for commit/transaction.
        """
        tags = []
        for letter, count in letter_counts.items():
            tag = self._ensure_letter_tag(letter)
            self._db.execute(
                """
                INSERT INTO memory_tags (memory_id, tag_id, assigned_by, confidence)
                VALUES (?, ?, 'system', ?)
                ON CONFLICT(memory_id, tag_id)
                DO UPDATE SET confidence = excluded.confidence
                """,
                (memory_id, tag.id, float(count)),
            )
            tags.append(tag)
        return tags

    def _ensure_letter_tag(self, letter: str) -> Tag:
        """Get or create a single-letter tag without committing."""
        row = self._db.execute(
            "SELECT * FROM tags WHERE name = ?", (letter,)
        ).fetchone()
        if row:
            return self._row_to_tag(row)
        self._db.execute(
            "INSERT OR IGNORE INTO tags (name, created_by) VALUES (?, 'system')",
            (letter,),
        )
        row = self._db.execute(
            "SELECT * FROM tags WHERE name = ?", (letter,)
        ).fetchone()
        return self._row_to_tag(row)

    def _consolidate_tensor_on_merge(
        self, source_id: int, target_id: int,
    ) -> None:
        """Consolidate tag relational tensor entries after a tag merge.

        Rewrites all tensor entries involving source_id to use target_id,
        keeping MAX strength for each network on conflict.

        The semiotic layer is directional: semiotic_forward = P(B|A) where
        A = tag_a_id, B = tag_b_id.  When re-keying causes the canonical
        ordering to flip (min/max swap), forward and reverse must be swapped
        to preserve directionality.
        """
        rows = self._db.execute(
            """
            SELECT tag_a_id, tag_b_id, cooccurrence_strength,
                   synchronicity_strength, semantic_strength,
                   semiotic_forward, semiotic_reverse, memory_ids
            FROM tag_relational_tensor
            WHERE tag_a_id = ? OR tag_b_id = ?
            """,
            (source_id, source_id),
        ).fetchall()

        if not rows:
            return

        self._db.execute(
            "DELETE FROM tag_relational_tensor WHERE tag_a_id = ? OR tag_b_id = ?",
            (source_id, source_id),
        )

        for row in rows:
            a, b = row["tag_a_id"], row["tag_b_id"]
            if a == source_id:
                a = target_id
            if b == source_id:
                b = target_id

            if a == b:
                continue  # Self-resonance after merge — drop

            new_a, new_b = min(a, b), max(a, b)

            # If canonical ordering flipped, swap semiotic directions
            ordering_flipped = (new_a, new_b) != (a, b)
            if ordering_flipped:
                sem_fwd = row["semiotic_reverse"]
                sem_rev = row["semiotic_forward"]
            else:
                sem_fwd = row["semiotic_forward"]
                sem_rev = row["semiotic_reverse"]

            self._db.execute(
                """
                INSERT INTO tag_relational_tensor
                    (tag_a_id, tag_b_id, cooccurrence_strength,
                     synchronicity_strength, semantic_strength,
                     semiotic_forward, semiotic_reverse, memory_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    cooccurrence_strength = MAX(
                        tag_relational_tensor.cooccurrence_strength,
                        excluded.cooccurrence_strength
                    ),
                    synchronicity_strength = MAX(
                        tag_relational_tensor.synchronicity_strength,
                        excluded.synchronicity_strength
                    ),
                    semantic_strength = MAX(
                        tag_relational_tensor.semantic_strength,
                        excluded.semantic_strength
                    ),
                    semiotic_forward = MAX(
                        tag_relational_tensor.semiotic_forward,
                        excluded.semiotic_forward
                    ),
                    semiotic_reverse = MAX(
                        tag_relational_tensor.semiotic_reverse,
                        excluded.semiotic_reverse
                    ),
                    memory_ids = CASE
                        WHEN length(tag_relational_tensor.memory_ids) > length(excluded.memory_ids)
                        THEN tag_relational_tensor.memory_ids
                        ELSE excluded.memory_ids
                    END,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                (
                    new_a, new_b,
                    row["cooccurrence_strength"],
                    row["synchronicity_strength"],
                    row["semantic_strength"],
                    sem_fwd, sem_rev,
                    row["memory_ids"],
                ),
            )

    @staticmethod
    def _row_to_tag(row) -> Tag:
        return Tag(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=row["created_by"],
            is_active=bool(row["is_active"]),
            parent_id=row["parent_id"],
            merged_into=row["merged_into"],
        )
