"""Tests for square motif discovery over the tag relational tensor."""

from __future__ import annotations

from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.tag_manager import TagManager
from chicory.layer3.square_finder import find_square_motifs
from chicory.layer3.square_visualizer import (
    render_chain_anisotropy_report_html,
    render_cross_project_alignment_html,
    render_hidden_bridges_html,
    render_square_motifs_html,
    render_tensor_vertical_square_motifs_html,
    render_vertical_square_motifs_html,
    render_zigzag_report_html,
)
from chicory.layer3.chain_anisotropy import analyze_chain_anisotropy
from chicory.layer3.cross_hidden_bridges import (
    find_hidden_bridges,
    format_hidden_bridge_context,
)
from chicory.layer3.cross_project_alignment import analyze_cross_project_alignment
from chicory.layer3.cross_project_middle import (
    build_middle_layer_document,
    load_middle_layer,
    query_middle_layer,
    write_middle_layer,
)
from chicory.layer3.vertical_square_finder import (
    analyze_cell_layer_orientation,
    find_tensor_vertical_square_motifs,
    find_vertical_square_motifs,
)
from chicory.layer3.zigzag_analyzer import (
    analyze_square_zigzags,
    canonical_side_signature,
    classify_side_layers,
)


@pytest.fixture
def stack():
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test",
        llm_model="test-model",
        embedding_model="mock",
        embedding_dimension=16,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    tags = TagManager(db)

    yield {"db": db, "tags": tags}

    db.close()


def _tag_ids(tags: TagManager, names: list[str]) -> dict[str, int]:
    return {name: tags.get_or_create(name).id for name in names}


def _insert_edge(
    db: DatabaseEngine,
    a: int,
    b: int,
    *,
    cooccurrence: float = 0.0,
    synchronicity: float = 0.0,
    semantic: float = 0.0,
    semiotic: float = 0.0,
    glyph: float = 0.0,
    inhibition: float = 0.0,
) -> None:
    tag_a, tag_b = min(a, b), max(a, b)
    db.execute(
        """
        INSERT INTO tag_relational_tensor
            (tag_a_id, tag_b_id, cooccurrence_strength,
             synchronicity_strength, semantic_strength,
             semiotic_forward, semiotic_reverse, glyph_strength,
             inhibition_strength, memory_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]')
        """,
        (
            tag_a,
            tag_b,
            cooccurrence,
            synchronicity,
            semantic,
            semiotic,
            semiotic,
            glyph,
            inhibition,
        ),
    )
    db.connection.commit()


def _insert_memory(
    db: DatabaseEngine,
    memory_id: str,
    summary: str,
    tag_ids: list[int],
) -> None:
    db.execute(
        "INSERT INTO memories (id, content, summary, source_model) VALUES (?, ?, ?, 'test')",
        (memory_id, f"content for {summary}", summary),
    )
    for tag_id in tag_ids:
        db.execute(
            "INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
            (memory_id, tag_id),
        )
    db.connection.commit()


def _make_stack() -> tuple[DatabaseEngine, TagManager]:
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test",
        llm_model="test-model",
        embedding_model="mock",
        embedding_dimension=16,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    return db, TagManager(db)


def test_finds_alternating_square_with_void_diagonals(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["alpha", "beta", "gamma", "delta"])

    _insert_edge(db, ids["alpha"], ids["beta"], cooccurrence=2.0)
    _insert_edge(db, ids["beta"], ids["gamma"], semantic=2.0)
    _insert_edge(db, ids["gamma"], ids["delta"], cooccurrence=2.0)
    _insert_edge(db, ids["delta"], ids["alpha"], semantic=2.0)

    motifs = find_square_motifs(
        db,
        min_edge_score=0.1,
        min_colors=2,
        require_distinct_incident_layers=True,
    )

    assert motifs
    motif = motifs[0]
    assert set(motif.tags) == {"alpha", "beta", "gamma", "delta"}
    assert motif.color_diversity == 2
    assert motif.repeated_color_vertices == ()
    assert motif.center_score == pytest.approx(1.0)
    assert motif.ac_diagonal.status == "void"
    assert motif.bd_diagonal.status == "void"
    assert motif.void_score > 0


def test_distinct_incident_layer_rule_filters_repeated_vertex_color(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["alpha", "beta", "gamma", "delta"])

    _insert_edge(db, ids["alpha"], ids["beta"], cooccurrence=2.0)
    _insert_edge(db, ids["beta"], ids["gamma"], cooccurrence=2.0)
    _insert_edge(db, ids["gamma"], ids["delta"], semantic=2.0)
    _insert_edge(db, ids["delta"], ids["alpha"], semantic=2.0)

    strict = find_square_motifs(
        db,
        min_edge_score=0.1,
        min_colors=2,
        require_distinct_incident_layers=True,
    )
    relaxed = find_square_motifs(
        db,
        min_edge_score=0.1,
        min_colors=2,
        require_distinct_incident_layers=False,
    )

    assert strict == []
    assert relaxed
    assert set(relaxed[0].repeated_color_vertices) == {"beta", "delta"}


def test_visualizer_renders_svg_with_void_diagonal(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["alpha", "beta", "gamma", "delta"])

    _insert_edge(db, ids["alpha"], ids["beta"], cooccurrence=2.0)
    _insert_edge(db, ids["beta"], ids["gamma"], semantic=2.0)
    _insert_edge(db, ids["gamma"], ids["delta"], cooccurrence=2.0)
    _insert_edge(db, ids["delta"], ids["alpha"], semantic=2.0)

    motifs = find_square_motifs(db, min_edge_score=0.1)
    html = render_square_motifs_html(motifs)

    assert "<svg" in html
    assert "diag-void" in html
    assert "cooccurrence" in html
    assert "semantic" in html


def test_zigzag_analyzer_counts_rotation_collapsed_pattern(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["alpha", "beta", "gamma", "delta"])

    _insert_edge(db, ids["alpha"], ids["beta"], cooccurrence=2.0)
    _insert_edge(db, ids["beta"], ids["gamma"], semantic=2.0)
    _insert_edge(db, ids["gamma"], ids["delta"], cooccurrence=2.0)
    _insert_edge(db, ids["delta"], ids["alpha"], semantic=2.0)

    report = analyze_square_zigzags(
        db,
        min_edge_score=0.1,
        max_edges=10,
        per_layer_edges=0,
        shuffle_iterations=0,
    )

    assert report.motif_count == 1
    assert report.zigzag_count == 1
    assert report.class_counts[0] == ("zigzag", 1)
    assert report.zigzag_pair_counts[0] == ("cooccurrence|semantic", 1)
    assert report.vector_zigzag_mean > 0


def test_side_signature_is_rotation_invariant():
    assert classify_side_layers(
        ("semantic", "cooccurrence", "semantic", "cooccurrence")
    ) == "zigzag"
    assert canonical_side_signature(
        ("semantic", "cooccurrence", "semantic", "cooccurrence")
    ) == canonical_side_signature(
        ("cooccurrence", "semantic", "cooccurrence", "semantic")
    )


def test_chain_anisotropy_detects_long_axis_vs_short_sides(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(
        tags,
        [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "side-one",
            "side-two",
            "side-three",
        ],
    )

    _insert_edge(db, ids["alpha"], ids["beta"], glyph=2.0)
    _insert_edge(db, ids["beta"], ids["gamma"], glyph=2.0)
    _insert_edge(db, ids["gamma"], ids["delta"], glyph=2.0)
    _insert_edge(db, ids["delta"], ids["epsilon"], glyph=2.0)
    _insert_edge(db, ids["alpha"], ids["side-one"], semantic=2.0)
    _insert_edge(db, ids["beta"], ids["side-two"], semantic=2.0)
    _insert_edge(db, ids["gamma"], ids["side-three"], semantic=2.0)

    report = analyze_chain_anisotropy(
        db,
        edge_limit=20,
        per_layer_edges=0,
        min_edge_score=0.1,
        min_layer_edges=1,
        max_depth=6,
        sample_edges_per_layer=10,
        shuffle_iterations=0,
    )

    glyph_report = next(item for item in report.layer_reports if item.layer == "glyph")
    assert glyph_report.axis_mean > glyph_report.side_mean
    assert glyph_report.contrast > 1.0
    assert glyph_report.top_probes


def test_vertical_square_finder_finds_source_tag_rectangles(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["w", "sequence", "other"])

    _insert_memory(db, "a1", "Game A.xml [1/2]", [ids["w"], ids["sequence"]])
    _insert_memory(db, "a2", "Game A.xml [2/2]", [ids["w"], ids["sequence"]])
    _insert_memory(db, "b1", "Game B.xml [1/1]", [ids["w"], ids["sequence"]])
    _insert_memory(db, "c1", "Other.txt [1/1]", [ids["w"], ids["other"]])

    motifs = find_vertical_square_motifs(db, limit=5)

    assert motifs
    motif = motifs[0]
    assert motif.sources == ("Game A.xml", "Game B.xml")
    assert motif.tags == ("w", "sequence")
    assert motif.counts == (2, 2, 1, 1)
    assert motif.relation_status == "source-repeated"


def test_tensor_vertical_square_finder_uses_source_file_tags_as_rows(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["game-a", "game-b", "w", "sequence"])

    _insert_memory(db, "a1", "Game A.xml [1/1]", [ids["game-a"]])
    _insert_memory(db, "b1", "Game B.xml [1/1]", [ids["game-b"]])
    _insert_edge(db, ids["game-a"], ids["w"], semantic=2.0)
    _insert_edge(db, ids["game-a"], ids["sequence"], glyph=2.0)
    _insert_edge(db, ids["game-b"], ids["w"], semantic=2.0)
    _insert_edge(db, ids["game-b"], ids["sequence"], glyph=2.0)

    motifs = find_tensor_vertical_square_motifs(db, limit=5)

    assert motifs
    motif = motifs[0]
    assert motif.sources == ("Game A.xml", "Game B.xml")
    assert motif.row_tags == ("game-a", "game-b")
    assert motif.column_tags == ("sequence", "w")
    assert set(motif.cell_layers) == {"semantic", "glyph"}
    assert motif.orientation_class == "column-banded"
    assert motif.diagonal_bias == "none"
    assert motif.symmetry_variants == 4
    assert motif.as_dict()["color_signature"] == "glyph|glyph/semantic|semantic"


def test_tensor_vertical_orientation_signature_detects_diagonal_bias():
    signature = analyze_cell_layer_orientation(
        ("glyph", "semantic", "semantic", "glyph")
    )

    assert signature.orientation_class == "checkerboard-diagonal"
    assert signature.diagonal_bias == "both"
    assert signature.symmetry_variants == 2


def test_vertical_square_source_filter_excludes_local_rectangles(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["w", "sequence"])

    _insert_memory(db, "a1", "Game A.xml [1/1]", [ids["w"], ids["sequence"]])
    _insert_memory(db, "b1", "Game B.xml [1/1]", [ids["w"], ids["sequence"]])

    motifs = find_vertical_square_motifs(db, exclude_source="game")

    assert motifs == []


def test_vertical_visualizer_renders_matrix_cells(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["w", "sequence"])

    _insert_memory(db, "a1", "Game A.xml [1/1]", [ids["w"], ids["sequence"]])
    _insert_memory(db, "b1", "Game B.xml [1/1]", [ids["w"], ids["sequence"]])

    motifs = find_vertical_square_motifs(db)
    html = render_vertical_square_motifs_html(motifs)

    assert "<svg" in html
    assert "Game A.xml" in html
    assert "sequence" in html


def test_tensor_vertical_visualizer_renders_orientation(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["game-a", "game-b", "w", "sequence"])

    _insert_memory(db, "a1", "Game A.xml [1/1]", [ids["game-a"]])
    _insert_memory(db, "b1", "Game B.xml [1/1]", [ids["game-b"]])
    _insert_edge(db, ids["game-a"], ids["w"], semantic=2.0)
    _insert_edge(db, ids["game-a"], ids["sequence"], glyph=2.0)
    _insert_edge(db, ids["game-b"], ids["w"], semantic=2.0)
    _insert_edge(db, ids["game-b"], ids["sequence"], glyph=2.0)

    motifs = find_tensor_vertical_square_motifs(db)
    html = render_tensor_vertical_square_motifs_html(motifs)

    assert "column-banded" in html
    assert "diagonal=none" in html


def test_zigzag_report_visualizer_renders_metric(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["alpha", "beta", "gamma", "delta"])

    _insert_edge(db, ids["alpha"], ids["beta"], cooccurrence=2.0)
    _insert_edge(db, ids["beta"], ids["gamma"], semantic=2.0)
    _insert_edge(db, ids["gamma"], ids["delta"], cooccurrence=2.0)
    _insert_edge(db, ids["delta"], ids["alpha"], semantic=2.0)

    report = analyze_square_zigzags(
        db,
        min_edge_score=0.1,
        max_edges=10,
        per_layer_edges=0,
        shuffle_iterations=0,
    )
    html = render_zigzag_report_html(report)

    assert "Zigzag Orientation Report" in html
    assert "cooccurrence|semantic" in html


def test_chain_anisotropy_visualizer_renders_metric(stack):
    db, tags = stack["db"], stack["tags"]
    ids = _tag_ids(tags, ["alpha", "beta", "gamma", "side-one"])

    _insert_edge(db, ids["alpha"], ids["beta"], glyph=2.0)
    _insert_edge(db, ids["beta"], ids["gamma"], glyph=2.0)
    _insert_edge(db, ids["alpha"], ids["side-one"], semantic=2.0)

    report = analyze_chain_anisotropy(
        db,
        edge_limit=10,
        per_layer_edges=0,
        min_edge_score=0.1,
        min_layer_edges=1,
        max_depth=4,
        shuffle_iterations=0,
    )
    html = render_chain_anisotropy_report_html(report)

    assert "Chain Anisotropy Report" in html
    assert "glyph" in html


def test_cross_project_alignment_finds_shared_anchor_cells():
    db_a, tags_a = _make_stack()
    db_b, tags_b = _make_stack()
    try:
        ids_a = _tag_ids(tags_a, ["shared", "glyph", "sequence", "a-only"])
        ids_b = _tag_ids(tags_b, ["shared", "glyph", "sequence", "b-only"])

        _insert_edge(db_a, ids_a["shared"], ids_a["glyph"], glyph=2.0)
        _insert_edge(db_a, ids_a["shared"], ids_a["sequence"], semantic=2.0)
        _insert_edge(db_a, ids_a["shared"], ids_a["a-only"], cooccurrence=1.0)
        _insert_edge(db_b, ids_b["shared"], ids_b["glyph"], glyph=3.0)
        _insert_edge(db_b, ids_b["shared"], ids_b["sequence"], cooccurrence=2.0)
        _insert_edge(db_b, ids_b["shared"], ids_b["b-only"], synchronicity=1.0)

        report = analyze_cross_project_alignment(
            db_a,
            db_b,
            project_a="a",
            project_b="b",
            edge_limit=20,
            per_layer_edges=0,
            min_edge_score=0.1,
        )

        assert report.shared_tag_count >= 3
        assert report.exact_pair_count == 2
        assert report.strongest_neighborhood_pair is not None
        assert any(
            cell.layer_a == "glyph" and cell.layer_b == "glyph"
            for cell in report.top_exact_cells
        )
    finally:
        db_a.close()
        db_b.close()


def test_cross_project_middle_layer_queries_materialized_cells():
    output_path = Path.cwd() / "chicory-middle-test.json"
    db_a, tags_a = _make_stack()
    db_b, tags_b = _make_stack()
    try:
        ids_a = _tag_ids(tags_a, ["shared", "glyph", "sequence"])
        ids_b = _tag_ids(tags_b, ["shared", "glyph", "sequence"])

        _insert_edge(db_a, ids_a["shared"], ids_a["glyph"], glyph=2.0)
        _insert_edge(db_a, ids_a["shared"], ids_a["sequence"], semantic=1.0)
        _insert_edge(db_b, ids_b["shared"], ids_b["glyph"], glyph=3.0)
        _insert_edge(db_b, ids_b["shared"], ids_b["sequence"], cooccurrence=1.0)

        report = analyze_cross_project_alignment(
            db_a,
            db_b,
            project_a="a",
            project_b="b",
            edge_limit=20,
            per_layer_edges=0,
            min_edge_score=0.1,
            top_cells=10,
        )
        output_path.unlink(missing_ok=True)
        document = write_middle_layer(report, output_path)

        assert output_path.exists()
        assert document["cells"]
        loaded = load_middle_layer(output_path)
        results = query_middle_layer(loaded, "shared glyph", limit=5)
        filtered = query_middle_layer(
            loaded,
            layer_a="glyph",
            layer_b="glyph",
            cell_type="exact",
            limit=5,
        )

        assert results
        assert all("search_text" in cell for cell in loaded["cells"])
        assert any(cell["layer_a"] == "glyph" for cell in results)
        assert any(cell["cell_type"] == "exact-edge-pair" for cell in filtered)
    finally:
        db_a.close()
        db_b.close()
        output_path.unlink(missing_ok=True)


def test_hidden_bridge_report_finds_raw_only_anchor():
    db_a, tags_a = _make_stack()
    db_b, tags_b = _make_stack()
    try:
        ids_a = _tag_ids(tags_a, ["shared", "glyph", "faint"])
        ids_b = _tag_ids(tags_b, ["shared", "glyph", "faint"])

        _insert_edge(db_a, ids_a["shared"], ids_a["glyph"], glyph=2.0)
        _insert_edge(db_b, ids_b["shared"], ids_b["glyph"], glyph=2.0)
        _insert_edge(db_a, ids_a["faint"], ids_a["glyph"], glyph=0.2)
        _insert_edge(db_b, ids_b["faint"], ids_b["glyph"], glyph=0.2)

        visible = analyze_cross_project_alignment(
            db_a,
            db_b,
            project_a="a",
            project_b="b",
            edge_limit=20,
            per_layer_edges=0,
            min_edge_score=0.1,
            top_cells=20,
        )
        raw = analyze_cross_project_alignment(
            db_a,
            db_b,
            project_a="a",
            project_b="b",
            edge_limit=20,
            per_layer_edges=0,
            min_edge_score=0.01,
            top_cells=20,
        )

        report = find_hidden_bridges(
            build_middle_layer_document(visible),
            build_middle_layer_document(raw),
            limit=10,
        )
        html = render_hidden_bridges_html(report)
        context = format_hidden_bridge_context(report, max_cells=3)

        assert any(cell["anchor"] == "faint" for cell in report.hidden_cells)
        assert any(
            cell["hidden_reason"] == "raw-only-anchor"
            for cell in report.hidden_cells
        )
        assert "Hidden Bridge Report" in html
        assert "faint" in html
        assert "Hidden bridge scan" in context
        assert "anchor=faint" in context
    finally:
        db_a.close()
        db_b.close()


def test_cross_project_alignment_visualizer_renders_matrix():
    db_a, tags_a = _make_stack()
    db_b, tags_b = _make_stack()
    try:
        ids_a = _tag_ids(tags_a, ["shared", "glyph"])
        ids_b = _tag_ids(tags_b, ["shared", "glyph"])
        _insert_edge(db_a, ids_a["shared"], ids_a["glyph"], glyph=2.0)
        _insert_edge(db_b, ids_b["shared"], ids_b["glyph"], glyph=3.0)

        report = analyze_cross_project_alignment(
            db_a,
            db_b,
            project_a="a",
            project_b="b",
            edge_limit=20,
            per_layer_edges=0,
            min_edge_score=0.1,
        )
        html = render_cross_project_alignment_html(report)

        assert "Cross-Project Alignment" in html
        assert "glyph" in html
        assert "shared anchors" in html
    finally:
        db_a.close()
        db_b.close()
