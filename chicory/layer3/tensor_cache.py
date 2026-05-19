"""In-memory cache for the tag_relational_tensor table.

The tensor is ~74K rows (~6MB) but scattered across a 10GB SQLite file.
Caching it eliminates all tensor I/O during retrieval — lookups become
dict operations (microseconds) instead of random page reads (seconds).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chicory.db.engine import DatabaseEngine

_log = logging.getLogger(__name__)

_COLS = (
    "tag_a_id, tag_b_id, cooccurrence_strength, synchronicity_strength,"
    " semantic_strength, semiotic_forward, semiotic_reverse,"
    " glyph_strength, inhibition_strength, parallelness"
)


class TensorCache:
    """Lazy-loaded in-memory mirror of tag_relational_tensor."""

    def __init__(self, db: DatabaseEngine) -> None:
        self._db = db
        self._by_a: dict[int, list[dict]] = {}
        self._by_b: dict[int, list[dict]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        rows = self._db.execute(
            f"SELECT {_COLS} FROM tag_relational_tensor"
        ).fetchall()
        for row in rows:
            d = dict(row)
            self._by_a.setdefault(d["tag_a_id"], []).append(d)
            self._by_b.setdefault(d["tag_b_id"], []).append(d)
        self._loaded = True
        total = sum(len(v) for v in self._by_a.values())
        _log.info("tensor cache loaded: %d rows", total)

    def rows_for_a(self, tag_ids: set[int] | list[int]) -> list[dict]:
        """All rows where tag_a_id is in tag_ids."""
        self._ensure_loaded()
        result: list[dict] = []
        for tid in tag_ids:
            result.extend(self._by_a.get(tid, []))
        return result

    def rows_for_b(self, tag_ids: set[int] | list[int]) -> list[dict]:
        """All rows where tag_b_id is in tag_ids."""
        self._ensure_loaded()
        result: list[dict] = []
        for tid in tag_ids:
            result.extend(self._by_b.get(tid, []))
        return result

    def rows_touching(self, tag_ids: set[int] | list[int]) -> list[dict]:
        """All rows where either side is in tag_ids, deduplicated."""
        self._ensure_loaded()
        tag_set = set(tag_ids)
        seen: set[tuple[int, int]] = set()
        result: list[dict] = []
        for tid in tag_set:
            for row in self._by_a.get(tid, []):
                key = (row["tag_a_id"], row["tag_b_id"])
                if key not in seen:
                    seen.add(key)
                    result.append(row)
            for row in self._by_b.get(tid, []):
                key = (row["tag_a_id"], row["tag_b_id"])
                if key not in seen:
                    seen.add(key)
                    result.append(row)
        return result

    def degree(self, tag_ids: set[int] | list[int]) -> dict[int, int]:
        """Total tensor degree (edges on either side) for each tag."""
        self._ensure_loaded()
        result: dict[int, int] = {}
        for tid in tag_ids:
            result[tid] = len(self._by_a.get(tid, [])) + len(self._by_b.get(tid, []))
        return result

    def invalidate(self) -> None:
        self._by_a.clear()
        self._by_b.clear()
        self._loaded = False
        _log.debug("tensor cache invalidated")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
