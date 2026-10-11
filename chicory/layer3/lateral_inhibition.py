"""Data-derived lateral inhibition between tags.

Two tags that are each common but meet on far fewer evidence units than
chance predicts are alternatives — python / javascript, postgres /
mongodb.  When the active context is about one of them, partners that
belong to the other are turned down.  The excitatory channels already use
the "together more than chance" half of the co-occurrence evidence; this
is the "together less than chance" half they throw away.

What counts as evidence:
  - one unit per source_path (otherwise per memory), so the chunks of one
    document count once;
  - only memories a tagger tagged.  File ingest (content_hash set) tags by
    path, and its extension / directory tags are mutually exclusive by
    construction — counting them would inhibit cross-module associations;
  - only the tags a tagger chose (memory_tags.assigned_by 'llm' or a
    librarian's author string), never the store's own 'system' rows: the
    split words of a compound tag, which copy its incidence exactly, and
    the temporal and letter tags;
  - never temporal, single-letter or digit tags, by name as well, since a
    tagger or librarian can still apply such a name itself.

Pairs are selected by Benjamini-Hochberg at ``inhibition_fdr`` over every
testable pair, and the strength is the FCR-adjusted Poisson lower bound on
the missing fraction of co-occurrences.  Rare tags and stores with no real
anti-association give exactly zero, so a new store starts with no
inhibition and grows it only as evidence accumulates.

Retrieval applies it divisively through :func:`apply_lateral_inhibition`
(dict operations only, no writes).  Computing is read-only and chunked;
persisting is a diff written in small transactions that can be pre-empted
and resumed.

Deliberately imports nothing from tensor_cache, tag_manager or the engine:
they import this.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaincinv, pdtr

from chicory.exceptions import DatabaseError

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine

_log = logging.getLogger(__name__)

# Same convention as layer4/wander.py and layer4/meta_shelves.py: tags that
# record when, not what.  Every memory of a month shares one, and two days
# never co-occur — exclusive by construction, so never evidence.
STRUCTURAL_TAG_PREFIXES = ("month-", "day-", "minute-", "hour-", "week-", "year-")

READ_CHUNK = 20_000          # rows per read; bounds the longest single query
WRITE_CHUNK = 1_000          # rows per write transaction; the pre-emption grain
WRITE_TOLERANCE = 0.01       # |ΔI| below this is not worth a write
CANDIDATE_BLOCK = 2_000_000  # candidate pairs tested per vectorised block
FULL_LOAD_THRESHOLD = 5_000  # more context ids than this: load the whole table
WATERMARK_KEY = "inhibition_watermark"
# Bump when what counts as evidence changes: a watermark counted under an
# older definition describes rows this code would not write, so it reads
# as absent and the next pass recounts.  2: system-assigned rows excluded.
EVIDENCE_VERSION = 2

_ID_CHUNK = 500              # ids per IN (...) list; well under SQLite's variable cap

_UNITS_SQL = (
    "SELECT rowid AS rid, COALESCE(source_path, id) AS unit FROM memories "
    "WHERE rowid > ? AND is_archived = 0 AND ingestion_tier != 'quarantine' "
    "AND content_hash IS NULL "
    "ORDER BY rowid LIMIT ?"
)

# Single-letter tags and system rows are dropped in SQL so they never reach
# Python at all; on a lattice-tagged store they are most of memory_tags.
# A split word ('borrow' from 'borrow-checker') is a system row with exactly
# its compound's incidence: counted, one anti-association would be selected
# once per word and multiplied into the noisy-OR as independent evidence.
_INCIDENCE_SQL = (
    "SELECT mt.rowid AS rid, m.rowid AS mrow, mt.tag_id FROM memory_tags mt "
    "JOIN memories m ON m.id = mt.memory_id JOIN tags t ON t.id = mt.tag_id "
    "WHERE mt.rowid > ? AND mt.rowid <= ? AND length(t.name) >= 2 "
    "AND mt.assigned_by != 'system' "
    "ORDER BY mt.rowid LIMIT ?"
)

_UPSERT_SQL = (
    "INSERT INTO tag_inhibition "
    "(tag_a_id, tag_b_id, inhibition_strength, observed, expected) "
    "VALUES (?, ?, ?, ?, ?) "
    "ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET "
    "inhibition_strength = excluded.inhibition_strength, "
    "observed = excluded.observed, "
    "expected = excluded.expected, "
    "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')"
)

_DELETE_SQL = "DELETE FROM tag_inhibition WHERE tag_a_id = ? AND tag_b_id = ?"


# ── Pure helpers ────────────────────────────────────────────────────────


def is_structural_tag(name: str) -> bool:
    """Temporal, single-character or all-digit: a tag that is never evidence."""
    return len(name) < 2 or name.isdigit() or name.startswith(STRUCTURAL_TAG_PREFIXES)


def is_lexical_variant(a: str, b: str) -> bool:
    """test / testing, postgres / postgresql: one spelling of one subject.

    Variants are rarely used together on the same item, so they look
    exactly like alternatives.  Missing one only leaves today's behaviour
    (no inhibition) in place.
    """
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 3 and long_.startswith(short)


def effective_gain(config: ChicoryConfig) -> float:
    """The shunting gain β, or 0 when inhibition is switched off.

    A negative or non-finite weight (a typo in the environment) reads as
    off: dividing by 1 + β·H would otherwise divide by zero or flip a sign.
    """
    gain = config.tensor_inhibition_weight
    if not config.inhibition_enabled or not math.isfinite(gain) or gain <= 0.0:
        return 0.0
    return gain


def poisson_upper(observed, confidence: float) -> np.ndarray:
    """Exact one-sided Poisson upper limit on the mean, given `observed` counts."""
    return gammaincinv(np.asarray(observed, dtype=np.float64) + 1.0, confidence)


def deficit_strength(observed, expected, confidence: float) -> np.ndarray:
    """Confident missing fraction: clip(1 − λ_U(O; q) / E, 0, 1).

    Zero unless the upper limit on the true co-occurrence rate still falls
    short of what independence predicts — so below E ≈ 3 it is zero even
    at O = 0, and it is zero whenever O ≥ E.
    """
    expected = np.asarray(expected, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = 1.0 - poisson_upper(observed, confidence) / expected
    strength = np.where(expected > 0.0, strength, 0.0)
    return np.clip(strength, 0.0, 1.0)


def select_anti_associations(
    observed, expected, n_tests: int, fdr: float,
) -> tuple[np.ndarray, float, int]:
    """Benjamini-Hochberg selection of co-occurrence deficits.

    `observed`/`expected` may hold only the pairs that could be selected
    (p ≤ fdr); `n_tests` is the size of the whole testable family, which
    sets the thresholds.  Returns (selected mask, q*, R) where
    q* = 1 − R·fdr/n_tests is the FCR-adjusted confidence the selected
    pairs' bounds must be computed at.
    """
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    nothing = (np.zeros(len(observed), dtype=bool), 1.0, 0)
    if n_tests <= 0 or len(observed) == 0:
        return nothing
    p = pdtr(observed, expected)
    ranked = np.sort(p)
    passing = np.flatnonzero(ranked <= np.arange(1, len(ranked) + 1) * fdr / n_tests)
    if len(passing) == 0:
        return nothing
    discoveries = int(passing[-1]) + 1
    threshold = discoveries * fdr / n_tests
    return p <= threshold, 1.0 - threshold, discoveries


def build_inhibitor_index(edges: Iterable) -> dict[int, dict[int, float]]:
    """{tag: {competitor: I}} in both directions, from (a, b, I, ...) tuples."""
    index: dict[int, dict[int, float]] = {}
    for edge in edges:
        a, b, strength = edge[0], edge[1], float(edge[2])
        index.setdefault(a, {})[b] = strength
        index.setdefault(b, {})[a] = strength
    return index


# ── Retrieval ───────────────────────────────────────────────────────────


def apply_lateral_inhibition(excitation, context_weights, index, gain):
    """Shunting lateral inhibition: R_p = E_p / (1 + gain * H_p),
    H_p = 1 - prod_s (1 - a_s * I(s, p)) over active context tags s.
    No tensor edge s-p is required.  Divisive and never a prune: H ≤ 1, so
    R_p / E_p ≥ 1 / (1 + gain) and every partner comes back — membership
    (thresholds, gates) is the excitation's business, decided before this.
    Partners with no inhibitor in the context are returned unchanged.
    Returns (scores, H) with H only for inhibited partners."""
    if gain <= 0.0 or not excitation or not context_weights or not index:
        return dict(excitation), {}
    survival: dict[int, float] = {}
    for s, a in context_weights.items():
        if a <= 0.0:
            continue
        row = index.get(s)
        if not row:
            continue
        a = 1.0 if a > 1.0 else a
        for p, w in row.items():
            if p in excitation:
                survival[p] = survival.get(p, 1.0) * (1.0 - a * w)
    if not survival:
        return dict(excitation), {}
    out, H = {}, {}
    for p, e in excitation.items():
        s = survival.get(p)
        if s is None:
            out[p] = e
            continue
        h = 1.0 - s
        H[p] = h
        out[p] = e / (1.0 + gain * h)
    return out, H


# ── Computation (read-only) ─────────────────────────────────────────────


@dataclass
class InhibitionComputation:
    """Anti-association evidence as of one memory_tags snapshot."""

    edges: list[tuple[int, int, float, int, float]]  # (a, b, I, O, E), a < b
    stats: dict
    watermark: dict


def compute_tag_inhibition(
    db: DatabaseEngine, *, fdr: float, max_pairs: int,
) -> InhibitionComputation:
    """Count unit × tag incidence and select the confident deficits.

    Read-only, through chunked db.query only, so a background caller can
    run it with the foreground lock released.  memory_tags is read up to
    the MAX(rowid) taken first; that bound is the watermark the growth
    gate compares against next time.
    """
    started = time.perf_counter()
    lam0 = -math.log(fdr)
    max_rowid = db.query_one("SELECT MAX(rowid) AS r FROM memory_tags")["r"] or 0
    stats: dict = {
        "units": 0, "frequent_tags": 0, "testable_pairs": 0, "discoveries": 0,
        "q_star": 1.0, "inhibitory_pairs": 0, "lexical_exempt": 0, "capped": 0,
        "seconds": 0.0,
    }

    def _result(edges: list) -> InhibitionComputation:
        stats["inhibitory_pairs"] = len(edges)
        stats["seconds"] = round(time.perf_counter() - started, 3)
        return InhibitionComputation(edges=edges, stats=stats, watermark=_watermark(
            max_rowid, units=stats["units"], pairs=len(edges),
        ))

    col: dict[int, int] = {}
    names: list[tuple[int, str]] = []
    for row in db.query("SELECT id, name FROM tags WHERE is_active = 1"):
        if not is_structural_tag(row["name"]):
            col[row["id"]] = len(names)
            names.append((row["id"], row["name"]))

    # Units first, so N counts units that carry no eligible tag at all —
    # they are part of the population the expectation is taken over.
    unit_idx: dict[str, int] = {}
    mrow_unit: dict[int, int] = {}
    last = 0
    while True:
        rows = db.query(_UNITS_SQL, (last, READ_CHUNK))
        if not rows:
            break
        for rid, unit in rows:
            mrow_unit[rid] = unit_idx.setdefault(unit, len(unit_idx))
        last = rows[-1]["rid"]
    n_units = len(unit_idx)
    stats["units"] = n_units

    us: list[int] = []
    cs: list[int] = []
    last = 0
    while True:
        rows = db.query(_INCIDENCE_SQL, (last, max_rowid, READ_CHUNK))
        if not rows:
            break
        for _rid, mrow, tag_id in rows:
            u = mrow_unit.get(mrow)
            c = col.get(tag_id)
            if u is not None and c is not None:
                us.append(u)
                cs.append(c)
        last = rows[-1]["rid"]
    if n_units == 0 or not us:
        return _result([])

    # float32 halves the matrices and still holds every count exactly (< 2^24 units).
    x = sp.csr_matrix(
        (np.ones(len(us), dtype=np.float32), (np.asarray(us), np.asarray(cs))),
        shape=(n_units, len(names)),
    )
    del us, cs
    # Several memories of one unit may carry the same tag: presence, not count.
    x.sum_duplicates()
    x.data[:] = 1.0
    n = np.asarray(x.sum(axis=0), dtype=np.float64).ravel()

    # Only a tag that could reach E > λ₀ with the most frequent tag can be
    # in any testable pair, since p ≥ e^(−E) means nothing below can pass.
    frequent = np.flatnonzero(n > lam0 * n_units / n.max())
    frequent = frequent[np.argsort(-n[frequent], kind="stable")]
    n_f = n[frequent]
    k = len(frequent)
    stats["frequent_tags"] = k
    if k < 2:
        return _result([])

    x_f = x[:, frequent].tocsc()
    co = sp.triu(x_f.T @ x_f, k=1).tocoo()
    keys = co.row.astype(np.int64) * k + co.col.astype(np.int64)
    order = np.argsort(keys)
    keys = keys[order]
    vals = co.data[order].astype(np.float64)
    del x, x_f, co

    # n_f is descending, so the partners j with E_ij > λ₀ are a prefix
    # [0, J_i); the candidates for row i are (i, J_i).  Row lengths fall
    # strictly until they reach zero, so the live rows are a prefix too.
    bound = np.searchsorted(-n_f, -(lam0 * n_units) / n_f, side="left")
    lengths = np.maximum(0, bound - np.arange(k) - 1)
    live = int(np.count_nonzero(lengths))

    n_tests = 0
    kept: list[tuple[np.ndarray, ...]] = []
    i = 0
    while i < live:
        # A block is whole rows up to CANDIDATE_BLOCK; one oversized row is a block.
        start, count = i, 0
        while i < live and (count == 0 or count + lengths[i] <= CANDIDATE_BLOCK):
            count += int(lengths[i])
            i += 1
        rows_i = np.arange(start, i)
        lens = lengths[start:i]
        ii = np.repeat(rows_i, lens)
        jj = ii + 1 + (np.arange(count) - np.repeat(np.cumsum(lens) - lens, lens))
        lookup = ii * k + jj
        if len(keys):
            pos = np.minimum(np.searchsorted(keys, lookup), len(keys) - 1)
            observed = np.where(keys[pos] == lookup, vals[pos], 0.0)
        else:
            observed = np.zeros(count)
        expected = n_f[ii] * n_f[jj] / n_units
        p = pdtr(observed, expected)
        n_tests += count
        # BH can only select p ≤ k·fdr/m ≤ fdr; nothing else is worth keeping.
        keep = p <= fdr
        if keep.any():
            kept.append((ii[keep], jj[keep], observed[keep], expected[keep]))
    stats["testable_pairs"] = n_tests
    if not kept:
        return _result([])

    ii = np.concatenate([b[0] for b in kept])
    jj = np.concatenate([b[1] for b in kept])
    observed = np.concatenate([b[2] for b in kept])
    expected = np.concatenate([b[3] for b in kept])
    selected, q_star, discoveries = select_anti_associations(
        observed, expected, n_tests, fdr,
    )
    stats["discoveries"] = discoveries
    stats["q_star"] = float(q_star)
    if discoveries == 0:
        return _result([])

    strength = deficit_strength(observed[selected], expected[selected], q_star)
    frequent_list = frequent.tolist()
    edges: list[tuple[int, int, float, int, float]] = []
    exempt = 0
    for x_i, y_i, w, o, e in zip(
        ii[selected].tolist(), jj[selected].tolist(), strength.tolist(),
        observed[selected].tolist(), expected[selected].tolist(),
    ):
        if w <= 0.0:
            continue  # cannot happen at q* (the bound and the test are dual)
        ta, na = names[frequent_list[x_i]]
        tb, nb = names[frequent_list[y_i]]
        # Exempt after selection: the pair stays in the family size, which
        # only makes the other thresholds stricter.
        if is_lexical_variant(na, nb):
            exempt += 1
            continue
        a, b = (ta, tb) if ta < tb else (tb, ta)
        edges.append((a, b, w, int(o), e))
    stats["lexical_exempt"] = exempt

    # A memory valve, not a tuning knob — 100k memories kept ~3.6k pairs.
    if len(edges) > max_pairs:
        edges.sort(key=lambda t: (-t[2], -t[4]))
        stats["capped"] = len(edges) - max_pairs
        del edges[max_pairs:]
        _log.warning(
            "lateral inhibition: %d pairs over inhibition_max_pairs=%d "
            "dropped (weakest first)", stats["capped"], max_pairs,
        )
    return _result(edges)


# ── Persistence ─────────────────────────────────────────────────────────


@dataclass
class InhibitionPlan:
    """The diff between the table and a computation, consumed as it is written.

    A partly written plan is a resumable backlog: the operations still
    queued are exactly the ones not yet committed.
    """

    upserts: list[tuple[int, int, float, int, float]]
    deletes: list[tuple[int, int]]
    index: dict[int, dict[int, float]]  # the table as it will be once done
    watermark: dict
    stats: dict
    written: int = 0
    deleted: int = 0

    @property
    def done(self) -> bool:
        return not self.upserts and not self.deletes


def _scan_inhibition(db: DatabaseEngine) -> Iterator[tuple[int, int, float]]:
    """Every (a, b, I) row of tag_inhibition, read in rowid chunks."""
    last = 0
    while True:
        rows = db.query(
            "SELECT rowid AS rid, tag_a_id, tag_b_id, inhibition_strength "
            "FROM tag_inhibition WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (last, READ_CHUNK),
        )
        if not rows:
            return
        for _rid, a, b, strength in rows:
            yield a, b, strength
        last = rows[-1]["rid"]


def plan_inhibition(db: DatabaseEngine, comp: InhibitionComputation) -> InhibitionPlan:
    """Diff a computation against the table.  Read-only.

    Strengths that moved less than WRITE_TOLERANCE keep their stored
    value, and the index is built from what the table will hold — so the
    published index always equals the database.
    """
    existing = {(a, b): strength for a, b, strength in _scan_inhibition(db)}
    upserts: list[tuple[int, int, float, int, float]] = []
    final: list[tuple[int, int, float]] = []
    for edge in comp.edges:
        a, b, strength = edge[0], edge[1], edge[2]
        old = existing.pop((a, b), None)
        if old is None or abs(strength - old) >= WRITE_TOLERANCE:
            upserts.append(edge)
            final.append((a, b, strength))
        else:
            final.append((a, b, old))
    return InhibitionPlan(
        upserts=upserts,
        deletes=list(existing),  # what is left was not rediscovered
        index=build_inhibitor_index(final),
        watermark=comp.watermark,
        stats=comp.stats,
    )


def persist_inhibition(
    db: DatabaseEngine,
    plan: InhibitionPlan,
    *,
    should_stop: Callable[[], bool] | None = None,
    chunk: int | None = None,
) -> InhibitionPlan:
    """Write a plan in short transactions, upserts first, then deletes.

    `should_stop` is asked after each chunk — never before the first, so
    every call makes progress — and a True answer returns the plan with
    the rest still queued.  `chunk` defaults to WRITE_CHUNK as it stands
    at call time.
    """
    size = max(1, int(WRITE_CHUNK if chunk is None else chunk))
    while not plan.done:
        if plan.upserts:
            batch = plan.upserts[:size]
            with db.transaction() as conn:
                conn.executemany(_UPSERT_SQL, batch)
            del plan.upserts[:len(batch)]
            plan.written += len(batch)
        else:
            batch = plan.deletes[:size]
            with db.transaction() as conn:
                conn.executemany(_DELETE_SQL, batch)
            del plan.deletes[:len(batch)]
            plan.deleted += len(batch)
        if should_stop is not None and should_stop():
            break
    return plan


def load_inhibitor_index(
    db: DatabaseEngine, tag_ids: Iterable[int] | None = None,
) -> dict[int, dict[int, float]]:
    """The inhibitor index for `tag_ids` (every row touching one), or all of it.

    A database without the table (foreign, or older than v48) has no
    inhibition rather than an error.
    """
    ids = None if tag_ids is None else list(tag_ids)
    try:
        if ids is None or len(ids) > FULL_LOAD_THRESHOLD:
            return build_inhibitor_index(_scan_inhibition(db))
        edges: list = []
        for start in range(0, len(ids), _ID_CHUNK):
            part = tuple(ids[start:start + _ID_CHUNK])
            marks = ",".join("?" * len(part))
            for side in ("tag_a_id", "tag_b_id"):
                edges.extend(db.query(
                    "SELECT tag_a_id, tag_b_id, inhibition_strength "
                    f"FROM tag_inhibition WHERE {side} IN ({marks})",
                    part,
                ))
        return build_inhibitor_index(edges)
    except DatabaseError:
        return {}


# ── Watermark and gating ────────────────────────────────────────────────


def _watermark(max_rowid: int, *, units: int, pairs: int) -> dict:
    """A publication of the table, compared by identity rather than content.

    Processes and queued plans decide "did anything change?" by equality,
    and two recounts in one second can agree on every count; the id makes
    each publication distinct.
    """
    return {
        "max_rowid": max_rowid,
        "units": units,
        "pairs": pairs,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence": EVIDENCE_VERSION,
        "id": uuid.uuid4().hex,
    }


def load_watermark(db: DatabaseEngine) -> dict | None:
    """The last persisted computation's snapshot, or None if absent or unreadable.

    A watermark from an older EVIDENCE_VERSION is treated as absent.
    """
    row = db.query_one(
        "SELECT value FROM librarian_state WHERE key = ?", (WATERMARK_KEY,),
    )
    if not row:
        return None
    try:
        wm = json.loads(row["value"])
    except (TypeError, ValueError):
        return None
    if not isinstance(wm, dict) or not isinstance(wm.get("max_rowid"), int):
        return None
    if wm.get("evidence") != EVIDENCE_VERSION:
        return None
    return wm


def save_watermark(db: DatabaseEngine, wm: dict) -> None:
    db.execute(
        "INSERT INTO librarian_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
        (WATERMARK_KEY, json.dumps(wm)),
    )


def inhibition_due(db: DatabaseEngine, growth: float) -> bool:
    """Whether memory_tags grew enough since the watermark to recount.

    Tag assignments only accumulate between recomputes, so the rowid
    high-water mark is a cheap proxy for new evidence.
    """
    r = db.query_one("SELECT MAX(rowid) AS r FROM memory_tags")["r"] or 0
    wm = load_watermark(db)
    if wm is None:
        return r > 0
    base = wm["max_rowid"]
    return r - base >= max(1, math.ceil(growth * base))


def forget_tag_inhibition(db: DatabaseEngine, tag_ids: Iterable[int]) -> None:
    """Drop rows touching `tag_ids` and reset the watermark, so the next pass recounts.

    For a merge: the surviving tag's statistics are a new union, so no row
    on either side still describes it.  The watermark is replaced rather
    than deleted: one at max_rowid 0 makes every process due, and as a new
    publication it is visible to a plan counted against the old one — even
    when there was no old one, where a deletion would leave None == None.
    Does not commit — it runs inside the caller's transaction.
    """
    ids = tuple(tag_ids)
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    db.execute(
        f"DELETE FROM tag_inhibition WHERE tag_a_id IN ({marks}) "
        f"OR tag_b_id IN ({marks})",
        ids + ids,
    )
    save_watermark(db, _watermark(0, units=0, pairs=0))
