"""Retrieval-side lateral inhibition: centroid path, glyph association, TagSpace.

Scenario A (as in test_lateral_inhibition.py): 60 units {python, asyncio,
bug} and 40 units {javascript, promise, bug}, each also carrying a month
tag and two single letters, plus rare units.  Two glyph tensor rows,
(bug, javascript) 14.0 and (bug, asyncio) 11.0, give bug's partners the
excitation js 2.80 and asyncio 2.20 (glyph weight 0.2).  python has no
tensor rows at all — it only inhibits.

The evidence is computed, planned and persisted from the memories, never
inserted by hand: the four cross-language pairs get I = 0.8337, so with
python active js falls to 2.80 / (1 + 2·0.8337) = 1.0497 and drops below
asyncio.  No centroid_edges exist, so the co-retrieval gate is skipped.
"""

from __future__ import annotations

import math

import pytest

from chicory.layer1.tag_manager import TagManager
from chicory.layer3.lateral_inhibition import (
    apply_lateral_inhibition,
    compute_tag_inhibition,
    persist_inhibition,
    plan_inhibition,
)
from chicory.layer3.semiotic_graph import DiscoveredTag
from chicory.layer3.synchronicity_engine import SynchronicityEngine
from chicory.layer3.tag_space import TagSpace
from chicory.layer3.tensor_cache import TensorCache
from tests.conftest import MockEmbeddingEngine

I_SCENARIO_A = 1.0 - (-math.log(0.02)) / (60 * 40 / 102)  # 0.8337
BETA = 2.0  # default tensor_inhibition_weight
JS_GLYPH, ASYNCIO_GLYPH = 14.0, 11.0


def _tag(db, name: str) -> int:
    row = db.query_one("SELECT id FROM tags WHERE name = ?", (name,))
    if row:
        return row["id"]
    return db.execute("INSERT INTO tags (name) VALUES (?)", (name,)).lastrowid


def _mem(db, mid: str, names: list[str]) -> None:
    db.execute(
        "INSERT INTO memories (id, content, source_model) VALUES (?, ?, 'test')",
        (mid, f"content of {mid}"),
    )
    for name in names:
        db.execute(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
            (mid, _tag(db, name)),
        )


def _glyph_row(db, a: int, b: int, glyph: float) -> None:
    db.execute(
        "INSERT INTO tag_relational_tensor (tag_a_id, tag_b_id, glyph_strength) "
        "VALUES (?, ?, ?)",
        (min(a, b), max(a, b), glyph),
    )


@pytest.fixture(params=["cache", "sql"])
def world(request, config, db):
    """Scenario A with glyph rows; engine with a TensorCache or the SQL fallback."""
    for i in range(60):
        _mem(db, f"py{i}", ["python", "asyncio", "bug", "month-2026-10", "p", "y"])
    for i in range(40):
        _mem(db, f"js{i}", ["javascript", "promise", "bug", "month-2026-10", "j", "a"])
    _mem(db, "rare-a", ["rare1", "rare2"])
    _mem(db, "rare-b", ["rare3"])
    ids = {name: _tag(db, name) for name in ("python", "asyncio", "bug", "javascript")}
    _glyph_row(db, ids["bug"], ids["javascript"], JS_GLYPH)
    _glyph_row(db, ids["bug"], ids["asyncio"], ASYNCIO_GLYPH)
    db.connection.commit()

    cache = TensorCache(db) if request.param == "cache" else None
    tags = TagManager(db)

    def engine_for(cfg):
        emb = MockEmbeddingEngine(dimension=32)
        emb.set_db(db)
        return SynchronicityEngine(cfg, db, emb, tags, tensor_cache=cache)

    return {
        "config": config,
        "db": db,
        "cache": cache,
        "tags": tags,
        "engine": engine_for(config),
        "engine_for": engine_for,
        "py": ids["python"],
        "asyncio": ids["asyncio"],
        "bug": ids["bug"],
        "js": ids["javascript"],
    }


def _recompute(w):
    """Compute → plan → persist, and publish to the cache as the orchestrator does."""
    cfg, db = w["config"], w["db"]
    comp = compute_tag_inhibition(db, fdr=cfg.inhibition_fdr,
                                  max_pairs=cfg.inhibition_max_pairs)
    plan = plan_inhibition(db, comp)
    persist_inhibition(db, plan)
    if w["cache"] is not None:
        w["cache"].set_inhibitor_index(plan.index)
    return plan


def _excitation(cfg, glyph: float) -> float:
    """The centroid-path excitation of a glyph-only row, computed as the engine does."""
    return (cfg.tensor_cooccurrence_weight * 0.0 + cfg.tensor_synchronicity_weight * 0.0
            + cfg.tensor_semantic_weight * 0.0 + cfg.tensor_glyph_weight * glyph)


# ── Centroid path ───────────────────────────────────────────────────────


def test_centroid_path_flips_after_recompute(world):
    w = world
    engine, py, bug, js, asyncio = w["engine"], w["py"], w["bug"], w["js"], w["asyncio"]

    before = engine._centroid_path([py, bug])
    assert before == {js: pytest.approx(2.80), asyncio: pytest.approx(2.20)}
    assert before[js] > before[asyncio]
    assert engine._last_partner_inhibition == {}

    plan = _recompute(w)
    strength = plan.index[py][js]
    assert strength == pytest.approx(I_SCENARIO_A, abs=1e-12)

    after = engine._centroid_path([py, bug])
    assert after[asyncio] == pytest.approx(2.20)
    assert after[js] == pytest.approx(2.80 / (1 + BETA * strength), abs=1e-6)
    assert after[js] == pytest.approx(1.0497, abs=1e-4)
    assert after[asyncio] > after[js]
    assert engine._last_partner_inhibition == {js: pytest.approx(0.8337, abs=1e-4)}


def test_resonant_memory_order_flips(world):
    w = world
    engine, py, bug = w["engine"], w["py"], w["bug"]
    candidates = {"js0": [w["js"]], "py0": [w["asyncio"]]}

    def rank():
        return engine.get_resonant_memory_ids_fast(
            [py, bug], candidate_memory_ids=["js0", "py0"],
            candidate_tag_ids_map=candidates,
        )

    before = rank()
    assert [mid for mid, _ in before] == ["js0", "py0"]
    assert dict(before) == {"js0": pytest.approx(1.0), "py0": pytest.approx(0.786, abs=1e-3)}

    _recompute(w)

    after = rank()
    assert [mid for mid, _ in after] == ["py0", "js0"]
    assert dict(after) == {"py0": pytest.approx(1.0), "js0": pytest.approx(0.477, abs=1e-3)}


def test_inhibition_comes_from_context_not_edges(world):
    w = world
    engine, py, bug, js, asyncio = w["engine"], w["py"], w["bug"], w["js"], w["asyncio"]
    plan = _recompute(w)

    assert js not in plan.index.get(bug, {})  # bug and js co-occur: no evidence
    py_rows = w["db"].query_one(
        "SELECT COUNT(*) AS n FROM tag_relational_tensor WHERE tag_a_id = ? OR tag_b_id = ?",
        (py, py),
    )["n"]
    assert py_rows == 0

    alone = engine._centroid_path([bug])
    assert alone[js] == pytest.approx(2.80)
    assert alone[asyncio] == pytest.approx(2.20)

    # python excites nothing, yet as an active context tag it divides js.
    with_python = engine._centroid_path([bug], context_weights={bug: 1.0, py: 1.0})
    assert with_python[js] == pytest.approx(1.0497, abs=1e-4)
    assert with_python[asyncio] == pytest.approx(2.20)


def test_activation_weight_scales_inhibition(world):
    w = world
    engine, py, bug, js, asyncio = w["engine"], w["py"], w["bug"], w["js"], w["asyncio"]
    strength = _recompute(w).index[py][js]

    faint = engine._centroid_path([py, bug], context_weights={bug: 1.0, py: 0.1})
    assert faint[js] == pytest.approx(2.80 / (1 + BETA * 0.1 * strength), abs=1e-9)
    assert faint[js] == pytest.approx(2.3999, abs=1e-4)
    assert faint[js] > faint[asyncio]  # one stray python result does not flip it

    full = engine._centroid_path([py, bug], context_weights={bug: 1.0, py: 1.0})
    assert full[js] < full[asyncio]


def test_directed_discovery_cannot_rescue(world, monkeypatch):
    w = world
    engine, py, bug, js = w["engine"], w["py"], w["bug"], w["js"]
    # score 10 × semiotic_directed_weight 0.3 = 3.0 > js's tensor excitation 2.80.
    discovery = DiscoveredTag(tag_id=js, max_strength=10.0, paths=[(bug, 10.0)])
    assert discovery.score * w["config"].semiotic_directed_weight == pytest.approx(3.0)
    monkeypatch.setattr(engine._semiotic_graph, "expand_in_through_out",
                        lambda tag_ids: {js: discovery})

    assert engine._centroid_path([py, bug])[js] == pytest.approx(3.0)

    strength = _recompute(w).index[py][js]

    after = engine._centroid_path([py, bug])
    assert after[js] == pytest.approx(3.0 / (1 + BETA * strength), abs=1e-9)
    assert after[js] == pytest.approx(3.0 / 2.6674, abs=1e-4)


# ── Glyph association ───────────────────────────────────────────────────


def test_glyph_association_flips(world):
    w = world
    engine, py, bug = w["engine"], w["py"], w["bug"]
    candidates = {"js0": [w["js"]], "py0": [w["asyncio"]]}

    def rank():
        return engine.get_glyph_association_scores(
            [py, bug], candidate_memory_ids=["js0", "py0"],
            candidate_tag_ids_map=candidates,
        )

    before = rank()
    assert [mid for mid, _ in before] == ["js0", "py0"]
    assert dict(before)["py0"] == pytest.approx(ASYNCIO_GLYPH / JS_GLYPH)

    strength = _recompute(w).index[py][w["js"]]

    after = rank()
    assert [mid for mid, _ in after] == ["py0", "js0"]
    # Normalised by the top partner (asyncio, 11.0, uninhibited).
    js_partner = dict(after)["js0"] * ASYNCIO_GLYPH
    assert js_partner == pytest.approx(JS_GLYPH / (1 + BETA * strength), abs=1e-9)
    assert js_partner == pytest.approx(5.2486, abs=1e-3)  # 14.0 / 2.6674


# ── TagSpace ────────────────────────────────────────────────────────────


def test_tag_space_lexical_inhibition(world):
    w = world
    space = TagSpace(w["config"], w["db"], w["tags"], tensor_cache=w["cache"])
    py, bug, js, asyncio = w["py"], w["bug"], w["js"], w["asyncio"]
    _recompute(w)

    scores, seeds = space._lexical_tag_scores("python bug")
    assert seeds == {py, bug}
    # Each partner is reached from one of two seeds: inward ratio 0.5.
    assert scores[asyncio] == pytest.approx(1.1)
    assert scores[js] == pytest.approx(0.525, abs=1e-3)
    assert scores[asyncio] > scores[js]
    assert scores[py] == scores[bug] == 1.0  # seeds are never inhibited

    scores, seeds = space._lexical_tag_scores("bug")
    assert seeds == {bug}
    assert scores[js] > scores[asyncio]

    memories, seeds = space.score_lexical("python bug")
    assert seeds == {py, bug} and memories
    assert space.score_lexical("a b") == ({}, set())  # no usable token


# ── Exact identity without evidence ─────────────────────────────────────


def test_no_evidence_is_exact_identity(world):
    w = world
    cfg, py, bug, js, asyncio = w["config"], w["py"], w["bug"], w["js"], w["asyncio"]
    candidates = {"js0": [js], "py0": [asyncio]}
    js_exc, asyncio_exc = _excitation(cfg, JS_GLYPH), _excitation(cfg, ASYNCIO_GLYPH)
    expected_centroid = {js: js_exc, asyncio: asyncio_exc}
    expected_glyph = [("js0", JS_GLYPH / JS_GLYPH), ("py0", ASYNCIO_GLYPH / JS_GLYPH)]
    # Seeds at 1.0; each partner reached from one of the two seeds, all its edges inward.
    expected_lexical = {py: 1.0, bug: 1.0, js: js_exc * 0.5, asyncio: asyncio_exc * 0.5}

    def outputs(engine, cfg_):
        space = TagSpace(cfg_, w["db"], w["tags"], tensor_cache=w["cache"])
        return (
            engine._centroid_path([py, bug]),
            engine.get_glyph_association_scores(
                [py, bug], candidate_memory_ids=["js0", "py0"],
                candidate_tag_ids_map=candidates,
            ),
            space._lexical_tag_scores("python bug")[0],
        )

    # The table is empty: the helper must hand back the excitation untouched.
    assert w["db"].query_one("SELECT COUNT(*) AS n FROM tag_inhibition")["n"] == 0
    assert outputs(w["engine"], cfg) == (expected_centroid, expected_glyph, expected_lexical)
    assert w["engine"]._last_partner_inhibition == {}

    # Evidence present, but the gain is zero or inhibition is switched off.
    assert _recompute(w).index
    for update in ({"tensor_inhibition_weight": 0.0}, {"inhibition_enabled": False}):
        off = cfg.model_copy(update=update)
        engine = w["engine_for"](off)
        assert outputs(engine, off) == (expected_centroid, expected_glyph, expected_lexical)
        assert engine._last_partner_inhibition == {}


# ── The helper itself ───────────────────────────────────────────────────


def test_apply_helper_properties():
    excitation = {10: 2.0, 11: 3.0, 12: 0.5}

    # No index, no context, no gain: the identity (as a copy).
    for index, weights, gain in (({}, {1: 1.0}, 2.0), ({1: {10: 0.5}}, {}, 2.0),
                                 ({1: {10: 0.5}}, {1: 1.0}, 0.0)):
        out, h = apply_lateral_inhibition(excitation, weights, index, gain)
        assert out == excitation and out is not excitation and h == {}

    # Two inhibitors of one partner combine by noisy-OR.
    i2, i3 = 0.6, 0.3
    index = {2: {10: i2}, 3: {10: i3}, 10: {2: i2, 3: i3}}
    out, h = apply_lateral_inhibition(excitation, {2: 1.0, 3: 1.0}, index, BETA)
    combined = 1 - (1 - i2) * (1 - i3)
    assert out[10] == pytest.approx(2.0 / (1 + BETA * combined), abs=1e-12)
    assert h == {10: pytest.approx(combined)}
    assert out[11] == 3.0 and out[12] == 0.5  # no inhibitor in the context: unchanged

    # Activations scale I; above 1 they are clipped, at or below 0 inert.
    half, _ = apply_lateral_inhibition(excitation, {2: 0.5}, index, BETA)
    assert half[10] == pytest.approx(2.0 / (1 + BETA * 0.5 * i2))
    over, _ = apply_lateral_inhibition(excitation, {2: 5.0}, index, BETA)
    assert over[10] == pytest.approx(2.0 / (1 + BETA * i2))
    assert apply_lateral_inhibition(excitation, {2: 0.0, 3: -1.0}, index, BETA) == (
        excitation, {})

    # However many confident inhibitors, a partner keeps at least 1/(1 + β).
    many = {s: {10: 0.999, 11: 0.999} for s in range(100, 140)}
    out, _ = apply_lateral_inhibition(excitation, {s: 1.0 for s in many}, many, BETA)
    for p in (10, 11):
        assert out[p] / excitation[p] >= 1 / (1 + BETA)

    # The floor drops only inhibited partners: 12 is below it but untouched.
    floor_index = {2: {10: 0.9, 11: 0.1}}
    out, _ = apply_lateral_inhibition(excitation, {2: 1.0}, floor_index, BETA, floor=1.0)
    assert 10 not in out  # 2.0 / 2.8 < 1.0
    assert out[11] == pytest.approx(3.0 / 1.2)
    assert out[12] == 0.5

    # Divisive: scale-free.
    scaled, _ = apply_lateral_inhibition(
        {p: 10 * e for p, e in excitation.items()}, {2: 1.0, 3: 1.0}, index, BETA)
    base, _ = apply_lateral_inhibition(excitation, {2: 1.0, 3: 1.0}, index, BETA)
    assert scaled == {p: pytest.approx(10 * r) for p, r in base.items()}
