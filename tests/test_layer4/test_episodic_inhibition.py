"""Layer-4 readers of lateral inhibition.

The episodic tensor shunts its tag projection by the anti-association
between the two memories' tags (from tag_inhibition) — through bootstrap
and update_on_store alike, pooled as a noisy-OR — and the meta analyzer's
connectivity no longer subtracts the retired tensor column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.tag_manager import TagManager
from chicory.layer4.adaptive_thresholds import AdaptiveThresholds
from chicory.layer4.episodic_tensor import EpisodicTensor
from chicory.layer4.meta_analyzer import MetaAnalyzer

# The column order _batch_insert writes an eager-field tuple in.
_EDGE_COLUMNS = (
    "memory_a_id", "memory_b_id",
    "semantic_strength", "tag_projected_strength", "co_retrieval_strength",
    "temporal_proximity", "source_proximity",
    "tag_semantic_projected", "tag_sync_projected",
    "tag_cooccurrence_projected", "tag_inhibition_projected",
    "tag_glyph_projected",
    "retrieval_reinforcement", "narrative_continuity",
    "supersession_strength", "supersession_direction",
    "contradiction_strength", "bridge_strength",
    "activation_count", "edge_status",
)


@pytest.fixture
def stack():
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        embedding_model="mock",
        embedding_dimension=8,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    yield {"config": config, "db": db, "tags": TagManager(db)}
    db.close()


def _tensor_row(db, a: int, b: int, **strengths: float) -> None:
    cols = ", ".join(strengths)
    marks = ", ".join("?" * len(strengths))
    db.execute(
        f"INSERT INTO tag_relational_tensor (tag_a_id, tag_b_id, {cols}) "
        f"VALUES (?, ?, {marks})",
        (min(a, b), max(a, b), *strengths.values()),
    )


def _inhibition_row(db, a: int, b: int, strength: float) -> None:
    db.execute(
        "INSERT INTO tag_inhibition "
        "(tag_a_id, tag_b_id, inhibition_strength, observed, expected) "
        "VALUES (?, ?, ?, 0, 20.0)",
        (min(a, b), max(a, b), strength),
    )


def _eager(et: EpisodicTensor, mem_tags: dict[str, list[int]], *, inhibition: bool):
    """Score the ma–mb edge from the tables, as bootstrap/update_on_store do."""
    vectors = np.eye(2, 8, dtype=np.float32)  # orthogonal: semantic = 0
    edge = et._compute_eager_fields(
        "ma", "mb", {"ma": 0, "mb": 1}, vectors, mem_tags,
        et._load_tag_tensor_cache(mem_tags), {}, {},
        tag_inhibition=et._load_tag_inhibition_index(mem_tags) if inhibition else None,
    )
    assert edge is not None
    return dict(zip(_EDGE_COLUMNS, edge))


def test_eager_fields_shunt_tag_projection(stack):
    db, config, tags = stack["db"], stack["config"], stack["tags"]
    ids = {
        name: tags.get_or_create(name).id
        for name in ("python", "javascript", "bug", "rust", "p", "month-2026-10")
    }
    py, js, bug = ids["python"], ids["javascript"], ids["bug"]
    _tensor_row(db, py, bug, cooccurrence_strength=0.2, glyph_strength=0.4)
    _tensor_row(db, bug, js, semantic_strength=0.1, glyph_strength=0.6)
    _inhibition_row(db, py, js, 0.5)
    _inhibition_row(db, js, ids["rust"], 0.9)  # rust is on neither memory
    db.connection.commit()

    et = EpisodicTensor(db, config)
    mem_tags = {"ma": [py, bug], "mb": [js, bug]}

    # Only pairs with both ends in scope are loaded.
    assert et._load_tag_inhibition_index(mem_tags) == {py: {js: 0.5}, js: {py: 0.5}}

    plain = _eager(et, mem_tags, inhibition=False)
    # Mean over the two cross pairs with tensor rows: 0.05 + 0.1 + 0.5.
    excitatory_sum = 0.65
    assert plain["tag_projected_strength"] == pytest.approx(excitatory_sum, rel=1e-12)
    assert plain["tag_inhibition_projected"] == 0.0

    shunted = _eager(et, mem_tags, inhibition=True)
    assert shunted["tag_inhibition_projected"] == pytest.approx(0.5, rel=1e-12)
    assert shunted["tag_projected_strength"] == pytest.approx(
        excitatory_sum / (1 + 2 * 0.5), rel=1e-12,
    )
    # Every other field is untouched by inhibition.
    for col in _EDGE_COLUMNS:
        if col not in ("tag_projected_strength", "tag_inhibition_projected"):
            assert shunted[col] == plain[col], col

    # Letter and temporal tags carry no inhibition, so they cannot dilute it.
    padded = {m: t + [ids["p"], ids["month-2026-10"]] for m, t in mem_tags.items()}
    diluted = _eager(et, padded, inhibition=True)
    assert diluted["tag_inhibition_projected"] == shunted["tag_inhibition_projected"]
    assert diluted["tag_projected_strength"] == shunted["tag_projected_strength"]

    # Shared tags reach the same unordered pair from both sides; it counts once.
    both = {"ma": [py, js, bug], "mb": [py, js, bug]}
    assert _eager(et, both, inhibition=True)["tag_inhibition_projected"] == (
        pytest.approx(0.5, rel=1e-12)
    )


@pytest.mark.parametrize("weight", [-2.0, -3.0, float("nan"), float("inf")])
def test_bad_gain_reads_as_off(stack, weight):
    # The weight is settable from the environment; a typo must not divide
    # by zero, flip the projection's sign, or turn it into nan.
    db, config, tags = stack["db"], stack["config"], stack["tags"]
    py, js, bug = (tags.get_or_create(n).id for n in ("python", "javascript", "bug"))
    _tensor_row(db, py, bug, cooccurrence_strength=0.2, glyph_strength=0.4)
    _tensor_row(db, bug, js, semantic_strength=0.1, glyph_strength=0.6)
    _inhibition_row(db, py, js, 0.5)
    db.connection.commit()
    config.tensor_inhibition_weight = weight

    edge = _eager(EpisodicTensor(db, config), {"ma": [py, bug], "mb": [js, bug]},
                  inhibition=True)

    assert edge["tag_projected_strength"] == pytest.approx(0.65, rel=1e-12)


def _stored(db, mid: str, tag_ids: list[int], episode: int | None) -> None:
    """A memory with tags and a chunk-0 embedding.

    Every memory gets the same vector, so each pair is a candidate edge
    with semantic = 1.
    """
    db.execute(
        "INSERT INTO memories (id, content, source_model) VALUES (?, ?, 'test')",
        (mid, f"content of {mid}"),
    )
    db.execute(
        "INSERT INTO embeddings (memory_id, chunk_index, embedding, model_name, dimension) "
        "VALUES (?, 0, ?, 'mock', 8)",
        (mid, np.ones(8, dtype=np.float32).tobytes()),
    )
    for tag_id in tag_ids:
        db.execute(
            "INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)", (mid, tag_id),
        )
    if episode is not None:
        db.execute(
            "INSERT INTO memory_episode_assignments (memory_id, episode_id) "
            "VALUES (?, ?)", (mid, episode),
        )


def _edge_through(path: str, stack, mem_tags: dict[str, list[int]]) -> dict:
    """Store the memories, build their edge the way `path` does, read it back."""
    db, config = stack["db"], stack["config"]
    episode = None
    if path == "update_on_store":
        episode = db.execute(
            "INSERT INTO temporal_episodes (centroid) VALUES (?)",
            (np.zeros(8, dtype=np.float32).tobytes(),),
        ).lastrowid
    for mid, tag_ids in mem_tags.items():
        _stored(db, mid, tag_ids, episode)
    db.connection.commit()

    et = EpisodicTensor(db, config)
    if path == "bootstrap":
        et.bootstrap()
    else:
        assert et.update_on_store("mb") == 1
    row = db.query_one(
        "SELECT tag_projected_strength, tag_inhibition_projected "
        "FROM memory_relational_tensor WHERE memory_a_id = 'ma' AND memory_b_id = 'mb'",
    )
    assert row is not None
    return dict(row)


@pytest.mark.parametrize("path", ["bootstrap", "update_on_store"])
def test_stored_edges_are_shunted(stack, path):
    db, tags = stack["db"], stack["tags"]
    py, js, bug = (tags.get_or_create(n).id for n in ("python", "javascript", "bug"))
    _tensor_row(db, py, bug, cooccurrence_strength=0.2, glyph_strength=0.4)
    _tensor_row(db, bug, js, semantic_strength=0.1, glyph_strength=0.6)
    _inhibition_row(db, py, js, 0.5)

    edge = _edge_through(path, stack, {"ma": [py, bug], "mb": [js, bug]})

    assert edge["tag_inhibition_projected"] == pytest.approx(0.5, rel=1e-12)
    assert edge["tag_projected_strength"] == pytest.approx(
        0.65 / (1 + 2 * 0.5), rel=1e-12,
    )


@pytest.mark.parametrize("path", ["bootstrap", "update_on_store"])
def test_inhibitory_pairs_pool_as_noisy_or(stack, path):
    # Two anti-associated cross pairs: a sum would give H = 0.9 and keep
    # growing past 1 with more pairs; the noisy-OR gives 1 − 0.5·0.6.
    db, tags = stack["db"], stack["tags"]
    py, go, js, bug = (
        tags.get_or_create(n).id for n in ("python", "golang", "javascript", "bug")
    )
    _tensor_row(db, py, bug, glyph_strength=0.4)
    _inhibition_row(db, py, js, 0.5)
    _inhibition_row(db, go, js, 0.4)

    edge = _edge_through(path, stack, {"ma": [py, go], "mb": [js, bug]})

    pooled = 1 - (1 - 0.5) * (1 - 0.4)
    assert edge["tag_inhibition_projected"] == pytest.approx(pooled, rel=1e-12)
    assert edge["tag_projected_strength"] == pytest.approx(
        0.4 / (1 + 2 * pooled), rel=1e-12,
    )


def test_meta_subgraphs_ignore_retired_column(stack):
    db, config, tags = stack["db"], stack["config"], stack["tags"]
    a, b, c = (tags.get_or_create(n).id for n in ("alpha", "beta", "gamma"))
    # The legacy column used to be subtracted: 0.1 − 1.0 < 0.05 split the pair.
    _tensor_row(db, a, b, glyph_strength=0.1, inhibition_strength=1.0)
    db.connection.commit()

    analyzer = MetaAnalyzer(config, db, AdaptiveThresholds(config, db))
    components = analyzer._find_tensor_subgraphs([a, b, c])

    assert sorted(sorted(group) for group in components) == sorted([sorted([a, b]), [c]])
