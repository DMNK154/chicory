"""Tests for data-derived lateral inhibition (anti-association evidence).

Scenario A, the shared fixture: 60 units {python, asyncio, bug} and 40
units {javascript, promise, bug}, every unit also carrying a month tag and
two single letters, plus rare units {rare1, rare2} and {rare3}.  N = 102,
10 testable pairs, R = 4, q* = 0.98, and the four cross-language pairs get
I = 1 − (−ln 0.02) / 23.529 = 0.8337 at O = 0.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.special import pdtr

from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.memory_store import MemoryStore
from chicory.layer1.salience import SalienceScorer
from chicory.layer1.tag_manager import TagManager
from chicory.config import ChicoryConfig
from chicory.layer3.lateral_inhibition import (
    EVIDENCE_VERSION,
    compute_tag_inhibition,
    deficit_strength,
    effective_gain,
    inhibition_due,
    load_inhibitor_index,
    load_watermark,
    save_watermark,
    persist_inhibition,
    plan_inhibition,
    poisson_upper,
    select_anti_associations,
)
from chicory.layer3.tensor_cache import TensorCache
from tests.conftest import MockEmbeddingEngine

SUBSTITUTES = {
    frozenset(p) for p in (
        ("python", "javascript"), ("python", "promise"),
        ("asyncio", "javascript"), ("asyncio", "promise"),
    )
}
I_SCENARIO_A = 1.0 - (-math.log(0.02)) / (60 * 40 / 102)  # 0.8337


@pytest.fixture
def make_db(config):
    """Fresh in-memory databases, for tests that compare two stores."""
    made: list[DatabaseEngine] = []

    def _make() -> DatabaseEngine:
        db = DatabaseEngine(config)
        db.connect()
        apply_schema(db)
        made.append(db)
        return db

    yield _make
    for db in made:
        db.close()


def _tag(db, name: str) -> int:
    row = db.query_one("SELECT id FROM tags WHERE name = ?", (name,))
    if row:
        return row["id"]
    return db.execute("INSERT INTO tags (name) VALUES (?)", (name,)).lastrowid


def _mem(db, mid: str, names: list[str], *, source_path=None, content_hash=None,
         archived: int = 0, tier: str = "critical") -> None:
    db.execute(
        "INSERT INTO memories (id, content, source_model, source_path, "
        "content_hash, is_archived, ingestion_tier) VALUES (?, ?, 'test', ?, ?, ?, ?)",
        (mid, f"content of {mid}", source_path, content_hash, archived, tier),
    )
    for name in names:
        db.execute(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
            (mid, _tag(db, name)),
        )


def _scenario_a(db) -> None:
    for i in range(60):
        _mem(db, f"py{i}", ["python", "asyncio", "bug", "month-2026-10", "p", "y"])
    for i in range(40):
        _mem(db, f"js{i}", ["javascript", "promise", "bug", "month-2026-10", "j", "a"])
    _mem(db, "rare-a", ["rare1", "rare2"])
    _mem(db, "rare-b", ["rare3"])


def _compute(db, *, fdr: float = 0.05, max_pairs: int = 200_000):
    return compute_tag_inhibition(db, fdr=fdr, max_pairs=max_pairs)


def _by_names(db, edges) -> dict[frozenset, tuple]:
    names = {r["id"]: r["name"] for r in db.query("SELECT id, name FROM tags")}
    return {frozenset((names[e[0]], names[e[1]])): e for e in edges}


def _table(db) -> dict[tuple[int, int], tuple]:
    return {
        (r["tag_a_id"], r["tag_b_id"]): (r["inhibition_strength"], r["observed"],
                                         r["expected"], r["updated_at"])
        for r in db.query("SELECT * FROM tag_inhibition")
    }


# ── Statistics ──────────────────────────────────────────────────────────


def test_poisson_upper_matches_exact_limits():
    got = poisson_upper([0, 1, 2, 3, 10, 80], 0.95)
    assert got == pytest.approx([2.996, 4.744, 6.296, 7.754, 16.962, 96.35], abs=1e-3)
    assert float(poisson_upper(0, 0.95)) == pytest.approx(-math.log(0.05))


def test_deficit_zero_without_testable_evidence():
    # Even a total absence is not evidence until about 3 meetings were due.
    for expected in (0.5, 0.99, 2.99):
        assert float(deficit_strength(0, expected, 0.95)) == 0.0
    observed = np.array([0, 1, 5, 20, 100])
    assert np.all(deficit_strength(observed, observed, 0.95) == 0.0)
    assert np.all(deficit_strength(observed + 3, observed, 0.95) == 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert float(deficit_strength(0, 0.0, 0.95)) == 0.0
        assert np.all(deficit_strength([0, 4], [0.0, 0.0], 0.95) == 0.0)


def test_deficit_monotone_and_bounded():
    expected = np.linspace(3.5, 500.0, 200)
    at_zero = deficit_strength(0, expected, 0.95)
    assert np.all(np.diff(at_zero) > 0)

    observed = np.arange(0, 60)
    at_fixed_e = deficit_strength(observed, 60.0, 0.95)
    assert np.all(np.diff(at_fixed_e) <= 0)
    positive = at_fixed_e[at_fixed_e > 0]
    assert len(positive) > 10 and np.all(np.diff(positive) < 0)

    grid = deficit_strength(observed[:, None], expected[None, :], 0.95)
    assert grid.min() >= 0.0 and grid.max() < 1.0


def test_bound_and_test_are_dual():
    rng = np.random.default_rng(20261011)
    observed = rng.integers(0, 120, 10_000)
    expected = rng.uniform(0.05, 200.0, 10_000)
    bound_positive = (1.0 - poisson_upper(observed, 0.95) / expected) > 0
    assert np.array_equal(bound_positive, pdtr(observed, expected) < 0.05)


def test_bh_selects_nothing_under_independence(db):
    rng = np.random.default_rng(0)
    n_units, n_tags = 3000, 300
    rate = 1.0 / np.arange(1, n_tags + 1) ** 0.9
    rate = np.minimum(rate / rate.sum() * 8, 0.5)  # ~8 tags per unit, Zipf
    present = rng.random((n_units, n_tags)) < rate
    names = [f"t{i:03d}" for i in range(n_tags)]  # equal length: no lexical variants
    db.executemany("INSERT INTO tags (name) VALUES (?)", [(n,) for n in names])
    ids = {r["name"]: r["id"] for r in db.query("SELECT id, name FROM tags")}
    db.executemany(
        "INSERT INTO memories (id, content, source_model) VALUES (?, 'c', 'test')",
        [(f"m{u}",) for u in range(n_units)],
    )
    units, tags = np.nonzero(present)
    db.executemany(
        "INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
        [(f"m{u}", ids[names[t]]) for u, t in zip(units.tolist(), tags.tolist())],
    )

    comp = _compute(db)

    assert comp.stats["testable_pairs"] > 1000  # the family really was tested
    assert comp.stats["discoveries"] == 0
    assert comp.edges == []


def test_bh_fcr_values():
    n, n_units = {"bug": 100, "python": 60, "asyncio": 60, "javascript": 40, "promise": 40}, 102
    observed_pairs = {
        ("bug", "python"): 60, ("bug", "asyncio"): 60, ("bug", "javascript"): 40,
        ("bug", "promise"): 40, ("python", "asyncio"): 60, ("javascript", "promise"): 40,
        ("python", "javascript"): 0, ("python", "promise"): 0,
        ("asyncio", "javascript"): 0, ("asyncio", "promise"): 0,
    }
    pairs = list(observed_pairs)
    observed = np.array([observed_pairs[p] for p in pairs], dtype=float)
    expected = np.array([n[a] * n[b] / n_units for a, b in pairs])

    selected, q_star, discoveries = select_anti_associations(observed, expected, 10, 0.05)

    assert discoveries == 4
    assert q_star == pytest.approx(0.98)
    assert {frozenset(p) for p, s in zip(pairs, selected) if s} == SUBSTITUTES
    strength = deficit_strength(observed[selected], expected[selected], q_star)
    assert strength == pytest.approx([0.8337] * 4, abs=1e-4)


# ── Computing from data ─────────────────────────────────────────────────


def test_compute_finds_substitutes_only(db):
    _scenario_a(db)

    comp = _compute(db)

    by_name = _by_names(db, comp.edges)
    assert set(by_name) == SUBSTITUTES
    for a, b, strength, observed, expected in comp.edges:
        assert a < b
        assert observed == 0
        assert expected == pytest.approx(23.529, abs=1e-3)
        assert strength == pytest.approx(I_SCENARIO_A, abs=1e-12)
        assert strength == pytest.approx(0.8337, abs=1e-4)
    involved = {name for pair in by_name for name in pair}
    assert not involved & {"bug", "rare1", "rare2", "rare3", "month-2026-10", "p", "y", "j", "a"}
    assert comp.stats["units"] == 102
    assert comp.stats["testable_pairs"] == 10
    assert comp.stats["discoveries"] == 4
    assert comp.stats["q_star"] == pytest.approx(0.98)
    assert comp.watermark["max_rowid"] == db.query_one(
        "SELECT MAX(rowid) AS r FROM memory_tags")["r"]
    assert comp.watermark["units"] == 102 and comp.watermark["pairs"] == 4


def test_unit_with_only_structural_tags_still_counts(db):
    # N is the population the expectation is taken over: a unit that
    # carries no eligible tag is still a unit, so E = 60·40/103.
    _scenario_a(db)
    _mem(db, "bare", ["month-2026-10", "q"])

    comp = _compute(db)

    assert comp.stats["units"] == 103
    assert set(_by_names(db, comp.edges)) == SUBSTITUTES
    expected = 60 * 40 / 103
    for _a, _b, strength, observed, e in comp.edges:
        assert e == pytest.approx(expected, abs=1e-12)
        assert strength == pytest.approx(1.0 - (-math.log(0.02)) / expected, abs=1e-12)


def test_structural_tags_never_inhibit(make_db):
    def build(day: str, year: str):
        db = make_db()
        # 30 mutually exclusive day tags, frequent enough to be testable,
        # and a year tag on days 0-14 only: exclusive with days 15-29.
        for u in range(3000):
            names = [f"{day}{u % 30:02d}"] + ([year] if u % 30 < 15 else [])
            _mem(db, f"m{u}", names)
        comp = _compute(db)
        return comp, [pair for pair in _by_names(db, comp.edges) if year in pair]

    days, _ = build("day-2026-10-", "2026")
    assert days.edges == []
    # Ordinary exclusive names inhibit one another, yet the all-digit year
    # is still never evidence...
    shifts, year_edges = build("shift-", "2026")
    assert shifts.edges and year_edges == []
    # ...while the same year under a non-digit name is strong evidence
    # against shifts 15-29, so the absence above is the name rule.
    _, year_edges = build("shift-", "y2026")
    assert len(year_edges) == 15


def test_file_ingested_memories_are_not_evidence(make_db):
    def build(hashed: bool):
        db = make_db()
        for i in range(40):
            _mem(db, f"c{i}", ["code-summary", "python", "layer3"],
                 source_path=f"/src/m{i}.py", content_hash=f"h{i}" if hashed else None)
        for i in range(40):
            _mem(db, f"o{i}", ["code-summary", "markdown", "orchestrator"],
                 source_path=f"/docs/d{i}.md", content_hash=f"k{i}" if hashed else None)
        return _compute(db)

    comp = build(hashed=True)
    assert comp.edges == []
    assert comp.stats["units"] == 0
    # Layout tags are exclusive by construction: as tagger evidence they would inhibit.
    assert len(build(hashed=False).edges) > 0


def test_split_words_are_not_evidence(make_db, config):
    # Through the real store: a compound's split words are 'system' rows
    # with exactly its incidence, so counting them would select the one
    # python / borrow-checker deficit once per word.
    def build(rust_tags: list[str]):
        db = make_db()
        tags = TagManager(db)
        store = MemoryStore(config, db, MockEmbeddingEngine(32), tags,
                            SalienceScorer(config, db))
        for i in range(60):
            store.store(f"rust note {i}", rust_tags, skip_embedding=True)
        for i in range(40):
            store.store(f"python note {i}", ["python"], skip_embedding=True)
        roles = {(r["name"], r["assigned_by"]) for r in db.query(
            "SELECT DISTINCT t.name, mt.assigned_by FROM memory_tags mt "
            "JOIN tags t ON t.id = mt.tag_id")}
        comp = _compute(db)
        return roles, comp, set(_by_names(db, comp.edges))

    roles, comp, pairs = build(["borrow-checker"])
    assert {("borrow-checker", "llm"), ("borrow", "system"), ("checker", "system"),
            ("python", "llm")} <= roles
    assert pairs == {frozenset(("python", "borrow-checker"))}
    # One test, so q* = 0.95, and E = 60·40/100.
    assert comp.edges[0][2] == pytest.approx(1.0 - (-math.log(0.05)) / 24.0, abs=1e-12)

    # A word the tagger chose itself is its own evidence.
    roles, _, pairs = build(["borrow-checker", "borrow"])
    assert ("borrow", "llm") in roles and ("borrow", "system") not in roles
    assert pairs == {frozenset(("python", "borrow-checker")),
                     frozenset(("python", "borrow"))}


def test_chunks_of_one_source_count_once(make_db):
    def build(source_path):
        db = make_db()
        for i in range(50):
            _mem(db, f"d{i}", ["alpha"], source_path=source_path)
        for i in range(30):
            _mem(db, f"b{i}", ["beta"])
        return _compute(db)

    one_document = build("/doc.pdf")
    assert one_document.edges == []
    assert one_document.stats["units"] == 31

    separate = build(None)
    assert len(separate.edges) == 1
    _, _, strength, observed, expected = separate.edges[0]
    assert strength == pytest.approx(0.8402, abs=1e-4)
    assert observed == 0
    assert expected == pytest.approx(18.75)
    assert separate.stats["q_star"] == pytest.approx(0.95)


def test_archived_and_quarantined_ignored(db):
    _scenario_a(db)
    js_ids = [f"js{i}" for i in range(40)]
    db.executemany("UPDATE memories SET is_archived = 1 WHERE id = ?",
                   [(m,) for m in js_ids[:20]])
    db.executemany("UPDATE memories SET ingestion_tier = 'quarantine' WHERE id = ?",
                   [(m,) for m in js_ids[20:]])

    comp = _compute(db)

    assert comp.edges == []
    assert comp.stats["units"] == 62


def test_lexical_variants_exempt(db):
    for i in range(40):
        _mem(db, f"t{i}", ["test", "x1"])
    for i in range(40):
        _mem(db, f"u{i}", ["testing", "x2"])

    comp = _compute(db)

    by_name = _by_names(db, comp.edges)
    assert frozenset(("test", "testing")) not in by_name
    assert frozenset(("x1", "x2")) in by_name  # the selection itself still ran
    assert comp.stats["lexical_exempt"] >= 1


def test_bridge_region_is_zero(db):
    for i in range(20):
        _mem(db, f"f{i}", ["filler"])
    _mem(db, "b1", ["bridge-left"])
    _mem(db, "b2", ["bridge-left"])
    _mem(db, "b3", ["bridge-right"])
    _mem(db, "b4", ["bridge-right"])

    comp = _compute(db)

    assert comp.edges == []
    assert comp.stats["testable_pairs"] == 0


def test_safety_valve_keeps_strongest(db):
    # Four discoveries with four different E, hence four different I.
    for i in range(60):
        _mem(db, f"py{i}", ["python"] + (["asyncio"] if i < 50 else []))
    for i in range(40):
        _mem(db, f"js{i}", ["javascript"] + (["promise"] if i < 30 else []))

    full = _compute(db)
    capped = _compute(db, max_pairs=2)

    assert full.stats["discoveries"] == 4 and len(full.edges) == 4
    strongest = sorted(full.edges, key=lambda e: -e[2])[:2]
    assert sorted(capped.edges) == sorted(strongest)
    assert capped.stats["capped"] == 2
    assert capped.stats["inhibitory_pairs"] == 2


def test_empty_and_tiny_stores(make_db):
    empty = make_db()
    tiny = make_db()
    _mem(tiny, "a", ["python", "bug"])
    _mem(tiny, "b", ["javascript"])
    _mem(tiny, "c", ["python"])

    for db in (empty, tiny):
        comp = _compute(db)
        assert comp.edges == []
        plan = plan_inhibition(db, comp)
        assert plan.done
        persist_inhibition(db, plan)
        assert plan.written == 0 and plan.deleted == 0
        assert plan.index == {}
        assert _table(db) == {}


# ── Persistence ─────────────────────────────────────────────────────────


def test_persist_is_diff_based(db):
    _scenario_a(db)
    first = plan_inhibition(db, _compute(db))
    persist_inhibition(db, first)
    assert first.written == 4 and first.done
    assert load_inhibitor_index(db) == first.index
    before = _table(db)

    again = plan_inhibition(db, _compute(db))
    assert again.upserts == [] and again.deletes == []
    assert again.index == first.index

    # A few units tagged with both languages: the python/javascript deficit
    # shrinks sharply, while every other pair moves by less than the write
    # tolerance (O stays 0; only N and the margins shift slightly).
    for i in range(4):
        _mem(db, f"both{i}", ["python", "javascript"])
    python, javascript = _tag(db, "python"), _tag(db, "javascript")
    changed = (min(python, javascript), max(python, javascript))

    plan = plan_inhibition(db, _compute(db))
    assert [(a, b) for a, b, *_ in plan.upserts] == [changed]
    assert plan.deletes == []
    persist_inhibition(db, plan)
    after = _table(db)
    assert after[changed][0] < before[changed][0] - 0.1
    assert after[changed][1] == 4
    for key in before:
        if key != changed:
            assert after[key] == before[key]  # untouched, timestamp included
    assert load_inhibitor_index(db) == plan.index

    # Retire the javascript side altogether: every row goes, by delete.
    db.execute("UPDATE memories SET is_archived = 1 WHERE id LIKE 'js%' OR id LIKE 'both%'")
    gone = plan_inhibition(db, _compute(db))
    assert gone.upserts == [] and len(gone.deletes) == 4
    persist_inhibition(db, gone)
    assert gone.deleted == 4
    assert _table(db) == {} and gone.index == {}


def test_persist_yields_and_resumes(db):
    _scenario_a(db)
    comp = _compute(db)
    plan = plan_inhibition(db, comp)
    assert len(plan.upserts) == 4

    calls = 0
    while not plan.done:
        before = plan.written + plan.deleted
        persist_inhibition(db, plan, should_stop=lambda: True, chunk=1)
        calls += 1
        assert plan.written + plan.deleted == before + 1
        assert len(_table(db)) == plan.written
        if calls < 4:
            assert not plan.done
    assert calls == 4

    table = _table(db)
    assert {k: v[:3] for k, v in table.items()} == {
        (a, b): (strength, observed, expected)
        for a, b, strength, observed, expected in comp.edges
    }
    assert load_inhibitor_index(db) == plan.index


# ── TensorCache ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("weight, enabled, gain", [
    (2.0, True, 2.0),
    (2.0, False, 0.0),
    (0.0, True, 0.0),
    (-2.0, True, 0.0),
    (float("nan"), True, 0.0),
    (float("inf"), True, 0.0),
])
def test_effective_gain_reads_bad_weights_as_off(weight, enabled, gain):
    cfg = ChicoryConfig(tensor_inhibition_weight=weight, inhibition_enabled=enabled)
    assert effective_gain(cfg) == gain


def test_watermarks_are_distinct_publications(db):
    # Peers and queued plans compare watermarks by equality; two recounts
    # in one second agree on every count, so each carries its own id.
    _scenario_a(db)
    first, second = _compute(db).watermark, _compute(db).watermark
    assert first["evidence"] == second["evidence"] == EVIDENCE_VERSION
    assert {k: v for k, v in first.items() if k not in ("id", "computed_at")} == (
        {k: v for k, v in second.items() if k not in ("id", "computed_at")}
    )
    assert first != second


def test_watermark_from_older_evidence_reads_as_absent(db):
    # Counted before system rows were excluded: those rows are not what this
    # code would write, so the next pass must recount however little grew.
    _scenario_a(db)
    current = _compute(db).watermark
    save_watermark(db, current)
    assert load_watermark(db) == current
    assert not inhibition_due(db, 0.02)

    save_watermark(db, {k: v for k, v in current.items() if k != "evidence"})
    assert load_watermark(db) is None
    assert inhibition_due(db, 0.02)


def test_tensor_cache_inhibitor_lifecycle(db):
    _scenario_a(db)
    bug, javascript = _tag(db, "bug"), _tag(db, "javascript")
    db.execute(
        "INSERT INTO tag_relational_tensor (tag_a_id, tag_b_id, glyph_strength, "
        "inhibition_strength, parallelness) VALUES (?, ?, 14.0, 0.7, 0.9)",
        (min(bug, javascript), max(bug, javascript)),
    )
    cache = TensorCache(db)

    rows = cache.rows_touching([bug])
    assert len(rows) == 1 and rows[0]["glyph_strength"] == 14.0
    assert "inhibition_strength" not in rows[0]
    assert "parallelness" not in rows[0]

    # Loaded from the (still empty) table, then kept across a tensor invalidate.
    index = cache.inhibitor_index()
    assert index == {}
    cache.invalidate()
    assert cache.is_loaded is False
    assert cache.inhibitor_index() is index

    swapped = {1: {2: 0.5}, 2: {1: 0.5}}
    cache.set_inhibitor_index(swapped)
    assert cache.inhibitor_index() is swapped

    plan = plan_inhibition(db, _compute(db))
    persist_inhibition(db, plan)
    assert cache.inhibitor_index() is swapped  # persisting alone publishes nothing
    cache.invalidate_inhibition()
    reloaded = cache.inhibitor_index()
    assert reloaded == plan.index == load_inhibitor_index(db)
    python = _tag(db, "python")
    assert reloaded[python][javascript] == pytest.approx(0.8337, abs=1e-4)
    assert reloaded[javascript][python] == reloaded[python][javascript]
