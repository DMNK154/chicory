"""Typer CLI entry point for Chicory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

_DEFAULT_DB = str(Path.home() / ".chicory" / "chicory.db")

app = typer.Typer(
    name="chicory",
    help="Dual-tracking memory architecture for LLMs",
    no_args_is_help=True,
)

# Register sub-apps
from chicory.cli.commons import app as commons_app

app.add_typer(commons_app)


@app.command()
def chat(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    model: Optional[str] = typer.Option(None, help="Override LLM model"),
) -> None:
    """Start an interactive chat session with memory."""
    from chicory.cli.chat import ChatSession
    from chicory.config import load_config

    overrides = {"db_path": Path(db)}
    if model:
        overrides["llm_model"] = model

    config = load_config(**overrides)
    session = ChatSession(config)
    session.run()


@app.command()
def status(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
) -> None:
    """Show system status."""
    from chicory.cli.commands import _cmd_status
    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    config = load_config(db_path=Path(db))
    o = Orchestrator(config)
    try:
        _cmd_status(o, "")
    finally:
        o.close()


@app.command()
def reembed(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    model: Optional[str] = typer.Option(None, help="New embedding model name"),
) -> None:
    """Re-embed all memories (for model migration)."""
    from rich.console import Console

    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    console = Console()
    config = load_config(db_path=Path(db))
    o = Orchestrator(config)

    try:
        console.print("[bold]Re-embedding all memories...[/bold]")
        count = o._embedding_engine.reembed_all(new_model=model)
        console.print(f"[green]Done. Re-embedded {count} memories.[/green]")
    finally:
        o.close()


@app.command()
def migrate(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    new_model: Optional[str] = typer.Option(None, help="New LLM model name"),
    new_embedding: Optional[str] = typer.Option(None, help="New embedding model"),
) -> None:
    """Run a full model migration."""
    from chicory.config import load_config
    from chicory.migration.model_update import run_migration

    config = load_config(db_path=Path(db))
    run_migration(config, new_llm_model=new_model, new_embedding_model=new_embedding)


@app.command()
def ingest(
    path: str = typer.Argument(help="File or directory to ingest"),
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    recursive: bool = typer.Option(True, help="Recurse into subdirectories"),
    chunk_size: int = typer.Option(2000, help="Max characters per chunk"),
    overlap: int = typer.Option(400, help="Character overlap between chunks"),
) -> None:
    """Ingest documents into memory. Accepts a file or directory."""
    from rich.console import Console

    from chicory.config import load_config
    from chicory.ingest.ingestor import ingest_directory, ingest_file
    from chicory.llm.factory import create_llm_client
    from chicory.orchestrator.orchestrator import Orchestrator

    console = Console()
    config = load_config(db_path=Path(db))
    o = Orchestrator(config)
    llm = create_llm_client(config)

    target = Path(path)
    try:
        if target.is_file():
            console.print(f"[bold]Ingesting {target.name}...[/bold]")
            count, new_ids = ingest_file(o, target, chunk_size=chunk_size, overlap=overlap,
                                         llm_client=llm)
            if new_ids:
                console.print(f"[dim]Batch embedding {len(new_ids)} memories...[/dim]")
                o._batch_embed_memories(new_ids)
                console.print("[dim]Updating tensor networks...[/dim]")
                o._finalize_ingested_memories(new_ids)
            console.print(f"[green]Done. Created {count} memories.[/green]")
        elif target.is_dir():
            console.print(f"[bold]Ingesting directory: {target}[/bold]")
            stats = ingest_directory(
                o, target, recursive=recursive,
                chunk_size=chunk_size, overlap=overlap,
                llm_client=llm,
            )
            console.print()
            console.print(f"[green]Done.[/green]")
            console.print(f"  Files found:      {stats['files_found']}")
            console.print(f"  Files ingested:   {stats['files_ingested']}")
            console.print(f"  Memories created: {stats['memories_created']}")
        else:
            console.print(f"[red]Path not found: {target}[/red]")
    finally:
        o.close()


@app.command()
def watch(
    path: str = typer.Argument(help="Directory to watch"),
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    chunk_size: int = typer.Option(2000, help="Max characters per chunk"),
    overlap: int = typer.Option(400, help="Character overlap between chunks"),
) -> None:
    """Watch a directory and auto-ingest new/changed files."""
    from chicory.config import load_config
    from chicory.ingest.watcher import watch_directory
    from chicory.orchestrator.orchestrator import Orchestrator

    config = load_config(db_path=Path(db))
    o = Orchestrator(config)

    target = Path(path)
    try:
        watch_directory(o, target, chunk_size=chunk_size, overlap=overlap)
    finally:
        o.close()


@app.command()
def sync(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
) -> None:
    """Run full network synchronization after a large ingest.

    Executes all expensive network-building passes: centroids, glyph lattice,
    tensor channels, episodic edges, canopy growth, sync detection, and
    meta-pattern analysis.
    """
    import time

    from rich.console import Console

    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    console = Console()
    config = load_config(db_path=Path(db))

    mem_count = None
    tag_count = None
    try:
        from chicory.db.engine import DatabaseEngine
        tmp_db = DatabaseEngine(config)
        tmp_db.connect()
        mem_count = tmp_db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        tag_count = tmp_db.execute("SELECT COUNT(*) as c FROM tags WHERE is_active=1").fetchone()["c"]
        tmp_db.close()
    except Exception:
        pass

    console.print("[bold]Chicory Network Sync[/bold]")
    if mem_count is not None:
        console.print(f"  Memories: {mem_count:,}  |  Active tags: {tag_count:,}")
    console.print()

    o = Orchestrator(config)
    try:
        def on_step(step: str, detail: str) -> None:
            if step == "done":
                console.print(f"  [green]{detail}[/green]")
            else:
                console.print(f"  [dim][{step}][/dim] {detail}")

        stats = o.run_sync(on_step=on_step)

        console.print()
        console.print("[bold green]Sync complete.[/bold green]")
        console.print(f"  Centroids rebuilt:     {stats.get('centroids_rebuilt', 0):,}")
        console.print(f"  Glyph positions:       {stats.get('glyph_positions', 0):,}")
        console.print(f"  Parallelness updated:  {stats.get('parallelness_updated', 0):,}")
        console.print(f"  Co-occurrence pairs:   {stats.get('cooccurrence_pairs', 0):,}")
        console.print(f"  Semantic pairs:        {stats.get('semantic_pairs', 0):,}")
        console.print(f"  Glyph pairs:           {stats.get('glyph_pairs', 0):,}")
        console.print(f"  Semiotic pairs:        {stats.get('semiotic_pairs', 0):,}")
        console.print(f"  Episodic edges:        {stats.get('episodic_edges', 0):,}")
        console.print(f"  Canopy blocks grown:   {stats.get('canopy_grown_blocks', 0):,}")
        console.print(f"  Sync events detected:  {stats.get('sync_events_detected', 0):,}")
        console.print(f"  Meta patterns:         {stats.get('meta_patterns', 0):,}")
        console.print(f"  Total time:            {stats.get('total_seconds', 0):.1f}s")
    finally:
        o.close()


@app.command(name="backfill-letters")
def backfill_letters(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
) -> None:
    """Backfill single-letter tags from existing word tags on all memories."""
    from rich.console import Console

    from chicory.config import load_config
    from chicory.layer1.tag_manager import TagManager
    from chicory.orchestrator.orchestrator import Orchestrator

    console = Console()
    config = load_config(db_path=Path(db))
    o = Orchestrator(config)

    try:
        # Get all memory IDs
        rows = o._db.execute("SELECT id FROM memories").fetchall()
        total = len(rows)
        console.print(f"[bold]Backfilling letter tags for {total} memories...[/bold]")

        tagged = 0
        for i, row in enumerate(rows):
            mid = row["id"]
            # Get word tags (length > 1) for this memory
            word_tags = [
                t for t in o._tag_manager.get_tags_for_memory(mid)
                if len(t) > 1
            ]
            if not word_tags:
                continue

            letter_counts = TagManager.decompose_to_letters(word_tags)
            if letter_counts:
                letter_tag_objs = o._tag_manager.assign_letter_tags(
                    mid, letter_counts
                )
                # Record tag events
                for lt in letter_tag_objs:
                    o._trend_engine.record_event(
                        tag_id=lt.id,
                        event_type="assignment",
                        memory_id=mid,
                    )
                tagged += 1

            if (i + 1) % 500 == 0:
                o._db.connection.commit()
                console.print(f"  Processed {i + 1}/{total}...")

        o._db.connection.commit()
        console.print(
            f"[green]Done. Added letter tags to {tagged} memories.[/green]"
        )
    finally:
        o.close()


@app.command()
def squares(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    tag: Optional[str] = typer.Option(None, help="Restrict motifs to one tag"),
    limit: int = typer.Option(10, help="Maximum motifs to show"),
    max_edges: int = typer.Option(
        500,
        help="Top tensor edges to use when enumerating candidate cycles",
    ),
    per_layer_edges: int = typer.Option(
        100,
        help="Extra top edges to include from each dominant relation layer",
    ),
    min_edge_score: float = typer.Option(
        0.05,
        help="Minimum weighted tensor score for square sides",
    ),
    min_colors: int = typer.Option(
        2,
        help="Minimum number of distinct side relation layers",
    ),
    min_tag_length: int = typer.Option(
        2,
        help="Ignore shorter tags when building square sides",
    ),
    require_void: bool = typer.Option(
        False,
        "--require-void",
        help="Only show squares with at least one implied missing diagonal",
    ),
    include_text: Optional[str] = typer.Option(
        None,
        "--include-text",
        help="Only show motifs whose tags or source summaries include this text",
    ),
    exclude_text: Optional[str] = typer.Option(
        None,
        "--exclude-text",
        help="Hide motifs whose tags or source summaries include this text",
    ),
    allow_repeated_colors: bool = typer.Option(
        False,
        "--allow-repeated-colors",
        help="Allow the two side edges touching a vertex to share a layer",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw motif JSON",
    ),
    html_output: Optional[Path] = typer.Option(
        None,
        "--html",
        help="Write a standalone HTML/SVG visualization",
    ),
) -> None:
    """Find Ramsey-style square motifs in the tag tensor."""
    import json

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.db.engine import DatabaseEngine
    from chicory.layer3.square_finder import find_square_motifs

    console = Console()
    config = load_config(db_path=Path(db))
    db_engine = DatabaseEngine(config)
    db_engine.connect()

    try:
        layer_weights = {
            "cooccurrence": config.tensor_cooccurrence_weight,
            "synchronicity": config.tensor_synchronicity_weight,
            "semantic": config.tensor_semantic_weight,
            "semiotic": config.tensor_semiotic_weight,
            "glyph": config.tensor_glyph_weight,
            "inhibition": config.tensor_inhibition_weight,
        }
        motifs = find_square_motifs(
            db_engine,
            tag=tag,
            limit=limit,
            max_edges=max_edges,
            per_layer_edges=per_layer_edges,
            min_edge_score=min_edge_score,
            min_colors=min_colors,
            min_tag_length=min_tag_length,
            require_void=require_void,
            include_text=include_text,
            exclude_text=exclude_text,
            require_distinct_incident_layers=not allow_repeated_colors,
            layer_weights=layer_weights,
        )
    finally:
        db_engine.close()

    if json_output:
        console.print_json(json.dumps([m.as_dict() for m in motifs]))
        return

    if html_output is not None:
        from chicory.layer3.square_visualizer import render_square_motifs_html

        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(
            render_square_motifs_html(motifs),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote visualization:[/green] {html_output}")

    if not motifs:
        console.print("[dim]No square motifs found.[/dim]")
        return

    table = Table(title="Square Motifs")
    table.add_column("#", justify="right")
    table.add_column("Cycle")
    table.add_column("Layers")
    table.add_column("AC")
    table.add_column("BD")
    table.add_column("Center", justify="right")
    table.add_column("Void", justify="right")
    table.add_column("Sources")
    table.add_column("Score", justify="right")

    def diag_text(signal) -> str:
        endpoints = "-".join(signal.endpoints)
        return (
            f"{endpoints} {signal.status} "
            f"support={signal.support:.2f} gap={signal.gap:.2f}"
        )

    for idx, motif in enumerate(motifs, start=1):
        cycle = " -> ".join([*motif.tags, motif.tags[0]])
        sources = "; ".join(
            f"{summary} ({count})"
            for summary, count in motif.source_summaries[:2]
        ) or "none"
        table.add_row(
            str(idx),
            cycle,
            " / ".join(motif.side_layers),
            diag_text(motif.ac_diagonal),
            diag_text(motif.bd_diagonal),
            f"{motif.center_score:.2f}",
            f"{motif.void_score:.2f}",
            sources,
            f"{motif.interestingness:.2f}",
        )

    console.print(table)


@app.command(name="cross-align")
def cross_align(
    other_db: Path = typer.Argument(help="Path to the other Chicory-style DB"),
    db: str = typer.Option("chicory.db", help="Path to this database file"),
    project_a: str = typer.Option("chicory", help="Display name for --db"),
    project_b: str = typer.Option("other", help="Display name for other_db"),
    edge_limit: int = typer.Option(
        1500,
        "--edge-limit",
        help="Top tensor edges to include before per-layer balancing",
    ),
    per_layer_edges: int = typer.Option(
        200,
        help="Extra top edges to include from each dominant relation layer",
    ),
    min_edge_score: float = typer.Option(
        0.05,
        help="Minimum weighted tensor score for alignment edges",
    ),
    min_tag_length: int = typer.Option(
        1,
        help="Ignore shorter canonical tags when aligning",
    ),
    anchor_text: Optional[str] = typer.Option(
        None,
        "--anchor-text",
        help="Only use shared anchors containing these comma-separated terms",
    ),
    top_cells: int = typer.Option(
        24,
        help="Number of strongest middle cells to keep",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw report JSON",
    ),
    html_output: Optional[Path] = typer.Option(
        None,
        "--html",
        help="Write a standalone HTML report",
    ),
) -> None:
    """Align two Chicory-style tensor DBs through shared tag anchors."""
    import json
    import sqlite3

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.layer3.cross_project_alignment import (
        analyze_cross_project_alignment,
    )

    console = Console()
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]DB not found:[/red] {db_path}")
        raise typer.Exit(1)
    if not other_db.exists():
        console.print(f"[red]Other DB not found:[/red] {other_db}")
        raise typer.Exit(1)

    config = load_config(db_path=db_path)
    layer_weights = {
        "cooccurrence": config.tensor_cooccurrence_weight,
        "synchronicity": config.tensor_synchronicity_weight,
        "semantic": config.tensor_semantic_weight,
        "semiotic": config.tensor_semiotic_weight,
        "glyph": config.tensor_glyph_weight,
        "inhibition": config.tensor_inhibition_weight,
    }

    def open_readonly(path: Path) -> sqlite3.Connection:
        uri_path = path.resolve().as_posix()
        conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    conn_a = open_readonly(db_path)
    conn_b = open_readonly(other_db)
    try:
        report = analyze_cross_project_alignment(
            conn_a,
            conn_b,
            project_a=project_a,
            project_b=project_b,
            edge_limit=edge_limit,
            per_layer_edges=per_layer_edges,
            min_edge_score=min_edge_score,
            min_tag_length=min_tag_length,
            anchor_text=anchor_text,
            top_cells=top_cells,
            layer_weights=layer_weights,
        )
    finally:
        conn_a.close()
        conn_b.close()

    if json_output:
        console.print_json(json.dumps(report.as_dict()))
        return

    if html_output is not None:
        from chicory.layer3.square_visualizer import (
            render_cross_project_alignment_html,
        )

        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(
            render_cross_project_alignment_html(report),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote report:[/green] {html_output}")

    summary = Table(title="Cross-Project Alignment")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("project A", project_a)
    summary.add_row("project B", project_b)
    summary.add_row("A edges", str(report.edge_count_a))
    summary.add_row("B edges", str(report.edge_count_b))
    summary.add_row("shared anchors", str(report.shared_tag_count))
    summary.add_row("exact edge pairs", str(report.exact_pair_count))
    if report.strongest_neighborhood_pair:
        a_layer, b_layer, score = report.strongest_neighborhood_pair
        summary.add_row("strongest neighborhood", f"{a_layer}->{b_layer} {score:.2f}")
    if report.strongest_exact_pair:
        a_layer, b_layer, score = report.strongest_exact_pair
        summary.add_row("strongest exact", f"{a_layer}->{b_layer} {score:.2f}")
    console.print(summary)

    matrix = Table(title="Top Neighborhood Layer Pairs")
    matrix.add_column(f"{project_a} layer")
    matrix.add_column(f"{project_b} layer")
    matrix.add_column("Score", justify="right")
    for a_layer, b_layer, score in report.neighborhood_matrix[:10]:
        matrix.add_row(a_layer, b_layer, f"{score:.2f}")
    console.print(matrix)

    cells = Table(title="Top Middle Cells")
    cells.add_column("Anchor")
    cells.add_column("Layers")
    cells.add_column("Score", justify="right")
    cells.add_column(f"{project_a} detail")
    cells.add_column(f"{project_b} detail")
    for cell in report.top_cells[:10]:
        cells.add_row(
            cell.anchor,
            f"{cell.layer_a}->{cell.layer_b}",
            f"{cell.score:.2f}",
            "; ".join(cell.detail_a[:2]),
            "; ".join(cell.detail_b[:2]),
        )
    console.print(cells)


@app.command(name="cross-middle-build")
def cross_middle_build(
    other_db: Path = typer.Argument(help="Path to the other Chicory-style DB"),
    db: str = typer.Option("chicory.db", help="Path to this database file"),
    project_a: str = typer.Option("chicory", help="Display name for --db"),
    project_b: str = typer.Option("other", help="Display name for other_db"),
    out: Path = typer.Option(
        Path("cross_middle_layer.json"),
        "--out",
        help="Path for the materialized middle-layer JSON",
    ),
    edge_limit: int = typer.Option(
        1500,
        "--edge-limit",
        help="Top tensor edges to include before per-layer balancing",
    ),
    per_layer_edges: int = typer.Option(
        200,
        help="Extra top edges to include from each dominant relation layer",
    ),
    min_edge_score: float = typer.Option(
        0.05,
        help="Minimum weighted tensor score for alignment edges",
    ),
    min_tag_length: int = typer.Option(
        1,
        help="Ignore shorter canonical tags when aligning",
    ),
    anchor_text: Optional[str] = typer.Option(
        None,
        "--anchor-text",
        help="Only use shared anchors containing these comma-separated terms",
    ),
    cell_limit: int = typer.Option(
        1000,
        "--cell-limit",
        help="Number of middle cells to materialize from each alignment mode",
    ),
    neighbor_limit: int = typer.Option(
        5,
        "--neighbor-limit",
        help="Neighbor details retained per anchor/layer cell",
    ),
) -> None:
    """Materialize the cross-project in-between layer as standalone JSON."""
    import sqlite3

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.layer3.cross_project_alignment import (
        analyze_cross_project_alignment,
    )
    from chicory.layer3.cross_project_middle import write_middle_layer

    console = Console()
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]DB not found:[/red] {db_path}")
        raise typer.Exit(1)
    if not other_db.exists():
        console.print(f"[red]Other DB not found:[/red] {other_db}")
        raise typer.Exit(1)

    config = load_config(db_path=db_path)
    layer_weights = {
        "cooccurrence": config.tensor_cooccurrence_weight,
        "synchronicity": config.tensor_synchronicity_weight,
        "semantic": config.tensor_semantic_weight,
        "semiotic": config.tensor_semiotic_weight,
        "glyph": config.tensor_glyph_weight,
        "inhibition": config.tensor_inhibition_weight,
    }

    def open_readonly(path: Path) -> sqlite3.Connection:
        uri_path = path.resolve().as_posix()
        conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    conn_a = open_readonly(db_path)
    conn_b = open_readonly(other_db)
    try:
        report = analyze_cross_project_alignment(
            conn_a,
            conn_b,
            project_a=project_a,
            project_b=project_b,
            edge_limit=edge_limit,
            per_layer_edges=per_layer_edges,
            min_edge_score=min_edge_score,
            min_tag_length=min_tag_length,
            anchor_text=anchor_text,
            top_cells=cell_limit,
            neighbor_limit=neighbor_limit,
            layer_weights=layer_weights,
        )
    finally:
        conn_a.close()
        conn_b.close()

    document = write_middle_layer(
        report,
        out,
        source_db_a=db_path,
        source_db_b=other_db,
    )

    summary = Table(title="Cross-Project Middle Layer")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("project A", project_a)
    summary.add_row("project B", project_b)
    summary.add_row("shared anchors", str(report.shared_tag_count))
    summary.add_row("exact edge pairs", str(report.exact_pair_count))
    summary.add_row("materialized cells", str(len(document["cells"])))
    summary.add_row("artifact", str(out))
    console.print(summary)


@app.command(name="cross-middle-query")
def cross_middle_query(
    middle_file: Path = typer.Argument(
        help="Path to a materialized cross-middle JSON file"
    ),
    query: Optional[str] = typer.Argument(
        None,
        help="Optional lexical query over anchors, layers, and cell details",
    ),
    limit: int = typer.Option(10, help="Maximum middle cells to show"),
    layer_a: Optional[str] = typer.Option(
        None,
        "--layer-a",
        help="Filter to one relation layer in project A",
    ),
    layer_b: Optional[str] = typer.Option(
        None,
        "--layer-b",
        help="Filter to one relation layer in project B",
    ),
    cell_type: Optional[str] = typer.Option(
        None,
        "--cell-type",
        help="Filter to neighborhood/exact or tag-neighborhood/exact-edge-pair",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw query result JSON",
    ),
) -> None:
    """Query only the materialized cross-project in-between layer."""
    import json

    from rich.console import Console
    from rich.table import Table

    from chicory.layer3.cross_project_middle import (
        load_middle_layer,
        query_middle_layer,
    )

    console = Console()
    if not middle_file.exists():
        console.print(f"[red]Middle-layer file not found:[/red] {middle_file}")
        raise typer.Exit(1)

    try:
        document = load_middle_layer(middle_file)
    except (json.JSONDecodeError, ValueError) as exc:
        console.print(f"[red]Could not read middle-layer file:[/red] {exc}")
        raise typer.Exit(1) from exc

    results = query_middle_layer(
        document,
        query,
        limit=limit,
        layer_a=layer_a,
        layer_b=layer_b,
        cell_type=cell_type,
    )

    if json_output:
        console.print_json(json.dumps(results))
        return

    if not results:
        console.print("[dim]No middle-layer cells matched.[/dim]")
        return

    def detail_text(values: object) -> str:
        if isinstance(values, list):
            return "; ".join(str(value) for value in values[:3])
        return str(values or "")

    table = Table(
        title=(
            f"Middle-Layer Query: {document['project_a']} <-> "
            f"{document['project_b']}"
        )
    )
    table.add_column("Anchor")
    table.add_column("Layers")
    table.add_column("Type")
    table.add_column("Score", justify="right")
    table.add_column(f"{document['project_a']} detail")
    table.add_column(f"{document['project_b']} detail")

    for cell in results:
        table.add_row(
            str(cell.get("anchor", "")),
            f"{cell.get('layer_a', '')}->{cell.get('layer_b', '')}",
            str(cell.get("scope", cell.get("cell_type", ""))),
            f"{float(cell.get('score', 0.0)):.2f}",
            detail_text(cell.get("detail_a")),
            detail_text(cell.get("detail_b")),
        )
    console.print(table)


@app.command(name="cross-raw-query")
def cross_raw_query(
    other_db: Path = typer.Argument(help="Path to the other Chicory-style DB"),
    query: Optional[str] = typer.Argument(
        None,
        help="Optional lexical query over raw-derived anchors and edge details",
    ),
    db: str = typer.Option("chicory.db", help="Path to this database file"),
    project_a: str = typer.Option("chicory", help="Display name for --db"),
    project_b: str = typer.Option("other", help="Display name for other_db"),
    edge_limit: int = typer.Option(
        10000,
        "--edge-limit",
        help="Top raw edges to include before per-layer balancing",
    ),
    per_layer_edges: int = typer.Option(
        2500,
        help="Extra raw edges to include from each dominant relation layer",
    ),
    min_edge_score: float = typer.Option(
        0.0,
        help="Minimum weighted tensor score for raw alignment edges",
    ),
    min_tag_length: int = typer.Option(
        1,
        help="Ignore shorter canonical tags when aligning",
    ),
    cell_scan_limit: int = typer.Option(
        10000,
        "--cell-scan-limit",
        help="Middle cells to scan from each raw-derived alignment mode",
    ),
    neighbor_limit: int = typer.Option(
        25,
        "--neighbor-limit",
        help="Raw neighbor details retained per anchor/layer cell",
    ),
    limit: int = typer.Option(20, help="Maximum matching cells to show"),
    layer_a: Optional[str] = typer.Option(
        None,
        "--layer-a",
        help="Filter to one relation layer in project A",
    ),
    layer_b: Optional[str] = typer.Option(
        None,
        "--layer-b",
        help="Filter to one relation layer in project B",
    ),
    cell_type: Optional[str] = typer.Option(
        None,
        "--cell-type",
        help="Filter to neighborhood/exact or tag-neighborhood/exact-edge-pair",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw query result JSON",
    ),
) -> None:
    """Query a cross-project bridge directly from the raw DBs, read-only."""
    import json
    import sqlite3

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.layer3.cross_project_alignment import (
        analyze_cross_project_alignment,
    )
    from chicory.layer3.cross_project_middle import (
        build_middle_layer_document,
        query_middle_layer,
    )

    console = Console()
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]DB not found:[/red] {db_path}")
        raise typer.Exit(1)
    if not other_db.exists():
        console.print(f"[red]Other DB not found:[/red] {other_db}")
        raise typer.Exit(1)

    config = load_config(db_path=db_path)
    layer_weights = {
        "cooccurrence": config.tensor_cooccurrence_weight,
        "synchronicity": config.tensor_synchronicity_weight,
        "semantic": config.tensor_semantic_weight,
        "semiotic": config.tensor_semiotic_weight,
        "glyph": config.tensor_glyph_weight,
        "inhibition": config.tensor_inhibition_weight,
    }

    def open_readonly(path: Path) -> sqlite3.Connection:
        uri_path = path.resolve().as_posix()
        conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    conn_a = open_readonly(db_path)
    conn_b = open_readonly(other_db)
    try:
        report = analyze_cross_project_alignment(
            conn_a,
            conn_b,
            project_a=project_a,
            project_b=project_b,
            edge_limit=edge_limit,
            per_layer_edges=per_layer_edges,
            min_edge_score=min_edge_score,
            min_tag_length=min_tag_length,
            top_cells=cell_scan_limit,
            neighbor_limit=neighbor_limit,
            layer_weights=layer_weights,
        )
    finally:
        conn_a.close()
        conn_b.close()

    document = build_middle_layer_document(
        report,
        source_db_a=db_path,
        source_db_b=other_db,
    )
    results = query_middle_layer(
        document,
        query,
        limit=limit,
        layer_a=layer_a,
        layer_b=layer_b,
        cell_type=cell_type,
    )

    if json_output:
        console.print_json(
            json.dumps(
                {
                    "summary": document["summary"],
                    "parameters": document["parameters"],
                    "scanned_cells": len(document["cells"]),
                    "matched_cells": len(results),
                    "results": results,
                }
            )
        )
        return

    summary = Table(title="Raw Cross-Project Query")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("project A", project_a)
    summary.add_row("project B", project_b)
    summary.add_row("A raw-derived edges", str(report.edge_count_a))
    summary.add_row("B raw-derived edges", str(report.edge_count_b))
    summary.add_row("shared anchors", str(report.shared_tag_count))
    summary.add_row("exact edge pairs", str(report.exact_pair_count))
    summary.add_row("scanned cells", str(len(document["cells"])))
    summary.add_row("matched cells", str(len(results)))
    console.print(summary)

    if not results:
        console.print("[dim]No raw-derived cells matched.[/dim]")
        return

    def detail_text(values: object) -> str:
        if isinstance(values, list):
            return "; ".join(str(value) for value in values[:4])
        return str(values or "")

    table = Table(title="Raw-Derived Middle Cells")
    table.add_column("Anchor")
    table.add_column("Layers")
    table.add_column("Type")
    table.add_column("Score", justify="right")
    table.add_column(f"{project_a} detail")
    table.add_column(f"{project_b} detail")

    for cell in results:
        table.add_row(
            str(cell.get("anchor", "")),
            f"{cell.get('layer_a', '')}->{cell.get('layer_b', '')}",
            str(cell.get("scope", cell.get("cell_type", ""))),
            f"{float(cell.get('score', 0.0)):.2f}",
            detail_text(cell.get("detail_a")),
            detail_text(cell.get("detail_b")),
        )
    console.print(table)


@app.command(name="cross-hidden-bridges")
def cross_hidden_bridges(
    other_db: Path = typer.Argument(help="Path to the other Chicory-style DB"),
    query: Optional[str] = typer.Argument(
        None,
        help="Optional lexical query over hidden anchors and edge details",
    ),
    db: str = typer.Option("chicory.db", help="Path to this database file"),
    project_a: str = typer.Option("chicory", help="Display name for --db"),
    project_b: str = typer.Option("other", help="Display name for other_db"),
    visible_edge_limit: int = typer.Option(
        1500,
        "--visible-edge-limit",
        help="Visible-view top edges before per-layer balancing",
    ),
    visible_per_layer_edges: int = typer.Option(
        200,
        "--visible-per-layer-edges",
        help="Visible-view extra top edges from each relation layer",
    ),
    visible_min_edge_score: float = typer.Option(
        0.05,
        "--visible-min-edge-score",
        help="Visible-view minimum weighted tensor score",
    ),
    raw_edge_limit: int = typer.Option(
        10000,
        "--raw-edge-limit",
        help="Raw-view top edges before per-layer balancing",
    ),
    raw_per_layer_edges: int = typer.Option(
        2500,
        "--raw-per-layer-edges",
        help="Raw-view extra top edges from each relation layer",
    ),
    raw_min_edge_score: float = typer.Option(
        0.0,
        "--raw-min-edge-score",
        help="Raw-view minimum weighted tensor score",
    ),
    min_tag_length: int = typer.Option(
        1,
        help="Ignore shorter canonical tags when aligning",
    ),
    cell_scan_limit: int = typer.Option(
        10000,
        "--cell-scan-limit",
        help="Cells to scan from each alignment mode in each view",
    ),
    visible_neighbor_limit: int = typer.Option(
        5,
        "--visible-neighbor-limit",
        help="Neighbor details retained per visible anchor/layer cell",
    ),
    raw_neighbor_limit: int = typer.Option(
        25,
        "--raw-neighbor-limit",
        help="Neighbor details retained per raw anchor/layer cell",
    ),
    limit: int = typer.Option(30, help="Maximum hidden bridge cells to show"),
    layer_a: Optional[str] = typer.Option(
        None,
        "--layer-a",
        help="Filter to one relation layer in project A",
    ),
    layer_b: Optional[str] = typer.Option(
        None,
        "--layer-b",
        help="Filter to one relation layer in project B",
    ),
    cell_type: Optional[str] = typer.Option(
        None,
        "--cell-type",
        help="Filter to neighborhood/exact or tag-neighborhood/exact-edge-pair",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw hidden bridge JSON",
    ),
    html_output: Optional[Path] = typer.Option(
        None,
        "--html",
        help="Write a standalone hidden bridge HTML report",
    ),
) -> None:
    """Find raw-only bridge cells hidden by the stricter alignment view."""
    import json
    import sqlite3

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.layer3.cross_hidden_bridges import find_hidden_bridges
    from chicory.layer3.cross_project_alignment import (
        analyze_cross_project_alignment,
    )
    from chicory.layer3.cross_project_middle import build_middle_layer_document

    console = Console()
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]DB not found:[/red] {db_path}")
        raise typer.Exit(1)
    if not other_db.exists():
        console.print(f"[red]Other DB not found:[/red] {other_db}")
        raise typer.Exit(1)

    config = load_config(db_path=db_path)
    layer_weights = {
        "cooccurrence": config.tensor_cooccurrence_weight,
        "synchronicity": config.tensor_synchronicity_weight,
        "semantic": config.tensor_semantic_weight,
        "semiotic": config.tensor_semiotic_weight,
        "glyph": config.tensor_glyph_weight,
        "inhibition": config.tensor_inhibition_weight,
    }

    def open_readonly(path: Path) -> sqlite3.Connection:
        uri_path = path.resolve().as_posix()
        conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    conn_a = open_readonly(db_path)
    conn_b = open_readonly(other_db)
    try:
        visible_report = analyze_cross_project_alignment(
            conn_a,
            conn_b,
            project_a=project_a,
            project_b=project_b,
            edge_limit=visible_edge_limit,
            per_layer_edges=visible_per_layer_edges,
            min_edge_score=visible_min_edge_score,
            min_tag_length=min_tag_length,
            top_cells=cell_scan_limit,
            neighbor_limit=visible_neighbor_limit,
            layer_weights=layer_weights,
        )
        raw_report = analyze_cross_project_alignment(
            conn_a,
            conn_b,
            project_a=project_a,
            project_b=project_b,
            edge_limit=raw_edge_limit,
            per_layer_edges=raw_per_layer_edges,
            min_edge_score=raw_min_edge_score,
            min_tag_length=min_tag_length,
            top_cells=cell_scan_limit,
            neighbor_limit=raw_neighbor_limit,
            layer_weights=layer_weights,
        )
    finally:
        conn_a.close()
        conn_b.close()

    visible_document = build_middle_layer_document(
        visible_report,
        source_db_a=db_path,
        source_db_b=other_db,
    )
    raw_document = build_middle_layer_document(
        raw_report,
        source_db_a=db_path,
        source_db_b=other_db,
    )
    report = find_hidden_bridges(
        visible_document,
        raw_document,
        query=query,
        limit=limit,
        layer_a=layer_a,
        layer_b=layer_b,
        cell_type=cell_type,
    )

    if json_output:
        console.print_json(json.dumps(report.as_dict()))
        return

    if html_output is not None:
        from chicory.layer3.square_visualizer import render_hidden_bridges_html

        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(
            render_hidden_bridges_html(report),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote hidden bridge report:[/green] {html_output}")

    summary = Table(title="Hidden Bridge Scan")
    summary.add_column("Metric")
    summary.add_column("Visible", justify="right")
    summary.add_column("Raw", justify="right")
    summary.add_row(
        "edges",
        str(visible_report.edge_count_a + visible_report.edge_count_b),
        str(raw_report.edge_count_a + raw_report.edge_count_b),
    )
    summary.add_row(
        "shared anchors",
        str(visible_report.shared_tag_count),
        str(raw_report.shared_tag_count),
    )
    summary.add_row(
        "exact edge pairs",
        str(visible_report.exact_pair_count),
        str(raw_report.exact_pair_count),
    )
    summary.add_row(
        "cells",
        str(len(visible_document["cells"])),
        str(len(raw_document["cells"])),
    )
    summary.add_row(
        "hidden candidates",
        "",
        str(report.parameters["hidden_candidate_count"]),
    )
    console.print(summary)

    if not report.hidden_cells:
        console.print("[dim]No hidden bridge cells matched.[/dim]")
        return

    def detail_text(values: object) -> str:
        if isinstance(values, list):
            return "; ".join(str(value) for value in values[:4])
        return str(values or "")

    table = Table(title="Hidden Bridge Cells")
    table.add_column("Anchor")
    table.add_column("Layers")
    table.add_column("Type")
    table.add_column("Reason")
    table.add_column("Score", justify="right")
    table.add_column("Hidden", justify="right")
    table.add_column(f"{project_a} detail")
    table.add_column(f"{project_b} detail")

    for cell in report.hidden_cells:
        table.add_row(
            str(cell.get("anchor", "")),
            f"{cell.get('layer_a', '')}->{cell.get('layer_b', '')}",
            str(cell.get("scope", cell.get("cell_type", ""))),
            str(cell.get("hidden_reason", "")),
            f"{float(cell.get('score', 0.0)):.2f}",
            f"{float(cell.get('hidden_score', 0.0)):.2f}",
            detail_text(cell.get("detail_a")),
            detail_text(cell.get("detail_b")),
        )
    console.print(table)


@app.command(name="cross-hidden-ask")
def cross_hidden_ask(
    other_db: Path = typer.Argument(help="Path to the other Chicory-style DB"),
    question: str = typer.Argument(
        "What useful hidden bridge structure is present here?",
        help="Question to ask the configured LLM about hidden bridge evidence",
    ),
    db: str = typer.Option("chicory.db", help="Path to this database file"),
    project_a: str = typer.Option("chicory", help="Display name for --db"),
    project_b: str = typer.Option("other", help="Display name for other_db"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Override LLM provider for this ask, e.g. openai, anthropic, null",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override LLM model for this ask",
    ),
    visible_edge_limit: int = typer.Option(
        1500,
        "--visible-edge-limit",
        help="Visible-view top edges before per-layer balancing",
    ),
    visible_per_layer_edges: int = typer.Option(
        200,
        "--visible-per-layer-edges",
        help="Visible-view extra top edges from each relation layer",
    ),
    visible_min_edge_score: float = typer.Option(
        0.05,
        "--visible-min-edge-score",
        help="Visible-view minimum weighted tensor score",
    ),
    raw_edge_limit: int = typer.Option(
        10000,
        "--raw-edge-limit",
        help="Raw-view top edges before per-layer balancing",
    ),
    raw_per_layer_edges: int = typer.Option(
        2500,
        "--raw-per-layer-edges",
        help="Raw-view extra top edges from each relation layer",
    ),
    raw_min_edge_score: float = typer.Option(
        0.0,
        "--raw-min-edge-score",
        help="Raw-view minimum weighted tensor score",
    ),
    min_tag_length: int = typer.Option(
        1,
        help="Ignore shorter canonical tags when aligning",
    ),
    cell_scan_limit: int = typer.Option(
        10000,
        "--cell-scan-limit",
        help="Cells to scan from each alignment mode in each view",
    ),
    visible_neighbor_limit: int = typer.Option(
        5,
        "--visible-neighbor-limit",
        help="Neighbor details retained per visible anchor/layer cell",
    ),
    raw_neighbor_limit: int = typer.Option(
        25,
        "--raw-neighbor-limit",
        help="Neighbor details retained per raw anchor/layer cell",
    ),
    bridge_limit: int = typer.Option(
        40,
        "--bridge-limit",
        help="Hidden bridge cells to rank before building LLM context",
    ),
    context_cells: int = typer.Option(
        24,
        "--context-cells",
        help="Hidden bridge cells included in the LLM context",
    ),
    detail_limit: int = typer.Option(
        4,
        "--detail-limit",
        help="Neighbor details included per hidden bridge cell",
    ),
    layer_a: Optional[str] = typer.Option(
        None,
        "--layer-a",
        help="Filter to one relation layer in project A",
    ),
    layer_b: Optional[str] = typer.Option(
        None,
        "--layer-b",
        help="Filter to one relation layer in project B",
    ),
    cell_type: Optional[str] = typer.Option(
        None,
        "--cell-type",
        help="Filter to neighborhood/exact or tag-neighborhood/exact-edge-pair",
    ),
    use_tools: bool = typer.Option(
        True,
        "--tools/--no-tools",
        help="Allow read-only Chicory tools during the LLM answer loop",
    ),
    max_tool_rounds: int = typer.Option(
        4,
        "--max-tool-rounds",
        help="Maximum read-only tool-use rounds allowed for the LLM",
    ),
) -> None:
    """Ask the configured LLM to interpret hidden bridge evidence."""
    import json
    import sqlite3

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.table import Table

    from chicory.config import load_config
    from chicory.layer3.cross_hidden_bridges import (
        HIDDEN_BRIDGE_SYSTEM_PROMPT,
        build_hidden_bridge_prompt,
        find_hidden_bridges,
    )
    from chicory.layer3.cross_project_alignment import (
        analyze_cross_project_alignment,
    )
    from chicory.layer3.cross_project_middle import build_middle_layer_document
    from chicory.llm.factory import create_llm_client
    from chicory.orchestrator.orchestrator import Orchestrator
    from chicory.orchestrator.tool_handlers import dispatch_tool_call

    console = Console()
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]DB not found:[/red] {db_path}")
        raise typer.Exit(1)
    if not other_db.exists():
        console.print(f"[red]Other DB not found:[/red] {other_db}")
        raise typer.Exit(1)

    overrides = {"db_path": db_path}
    if provider:
        overrides["llm_provider"] = provider
    if model:
        overrides["llm_model"] = model
    config = load_config(**overrides)
    layer_weights = {
        "cooccurrence": config.tensor_cooccurrence_weight,
        "synchronicity": config.tensor_synchronicity_weight,
        "semantic": config.tensor_semantic_weight,
        "semiotic": config.tensor_semiotic_weight,
        "glyph": config.tensor_glyph_weight,
        "inhibition": config.tensor_inhibition_weight,
    }

    def open_readonly(path: Path) -> sqlite3.Connection:
        uri_path = path.resolve().as_posix()
        conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    conn_a = open_readonly(db_path)
    conn_b = open_readonly(other_db)
    try:
        visible_report = analyze_cross_project_alignment(
            conn_a,
            conn_b,
            project_a=project_a,
            project_b=project_b,
            edge_limit=visible_edge_limit,
            per_layer_edges=visible_per_layer_edges,
            min_edge_score=visible_min_edge_score,
            min_tag_length=min_tag_length,
            top_cells=cell_scan_limit,
            neighbor_limit=visible_neighbor_limit,
            layer_weights=layer_weights,
        )
        raw_report = analyze_cross_project_alignment(
            conn_a,
            conn_b,
            project_a=project_a,
            project_b=project_b,
            edge_limit=raw_edge_limit,
            per_layer_edges=raw_per_layer_edges,
            min_edge_score=raw_min_edge_score,
            min_tag_length=min_tag_length,
            top_cells=cell_scan_limit,
            neighbor_limit=raw_neighbor_limit,
            layer_weights=layer_weights,
        )
    finally:
        conn_a.close()
        conn_b.close()

    report = find_hidden_bridges(
        build_middle_layer_document(
            visible_report,
            source_db_a=db_path,
            source_db_b=other_db,
        ),
        build_middle_layer_document(
            raw_report,
            source_db_a=db_path,
            source_db_b=other_db,
        ),
        query=None,
        limit=max(bridge_limit, context_cells),
        layer_a=layer_a,
        layer_b=layer_b,
        cell_type=cell_type,
    )

    summary = Table(title="Hidden Bridge LLM Context")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("provider", config.llm_provider)
    summary.add_row("model", config.llm_model)
    summary.add_row("visible shared anchors", str(visible_report.shared_tag_count))
    summary.add_row("raw shared anchors", str(raw_report.shared_tag_count))
    summary.add_row("hidden candidates", str(report.parameters["hidden_candidate_count"]))
    summary.add_row("context cells", str(min(context_cells, len(report.hidden_cells))))
    console.print(summary)

    prompt = build_hidden_bridge_prompt(
        report,
        question,
        max_cells=context_cells,
        detail_limit=detail_limit,
    )
    system = HIDDEN_BRIDGE_SYSTEM_PROMPT
    if not use_tools:
        system += "\nDo not call tools for this answer."

    client = create_llm_client(config)
    orchestrator = Orchestrator(config) if use_tools else None
    if orchestrator is not None:
        active_tags = orchestrator.get_relevant_tags(question)
        if active_tags:
            system += "\nCurrent active Chicory tags: " + ", ".join(active_tags)
        client.update_active_tags(active_tags)
    messages = [{"role": "user", "content": prompt}]
    readonly_tools = {
        "retrieve_memories",
        "deep_retrieve",
        "get_trends",
        "get_phase_space",
        "get_synchronicities",
        "get_meta_patterns",
        "get_lattice_resonances",
    }
    try:
        for _round in range(max(1, max_tool_rounds)):
            response = client.chat(messages, system=system)
            text_parts = [
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text.strip()
            ]
            tool_blocks = [
                block
                for block in response.content
                if getattr(block, "type", "") == "tool_use"
            ]
            if text_parts and (not tool_blocks or not use_tools):
                console.print()
                console.print(Markdown("\n\n".join(text_parts)))
                console.print()
                return
            if not tool_blocks or not use_tools or orchestrator is None:
                console.print("[dim]No LLM text response received.[/dim]")
                return

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_blocks:
                console.print(f"  [dim]Using {block.name}...[/dim]")
                if block.name not in readonly_tools:
                    result = {
                        "error": (
                            f"Tool {block.name} is not allowed in "
                            "cross-hidden-ask read-only mode."
                        )
                    }
                else:
                    try:
                        result = dispatch_tool_call(orchestrator, block.name, block.input)
                    except Exception as exc:
                        result = {"error": str(exc)}
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        console.print("[yellow]Stopped after max tool rounds without a final answer.[/yellow]")
    finally:
        if orchestrator is not None:
            orchestrator.close()
        client.close()


@app.command(name="chain-anisotropy")
def chain_anisotropy(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    edge_limit: int = typer.Option(
        1500,
        "--edge-limit",
        help="Top tensor edges to include before per-layer balancing",
    ),
    per_layer_edges: int = typer.Option(
        200,
        help="Extra top edges to include from each dominant relation layer",
    ),
    min_edge_score: float = typer.Option(
        0.05,
        help="Minimum weighted tensor score for chain edges",
    ),
    min_tag_length: int = typer.Option(
        2,
        help="Ignore shorter tags when building chains",
    ),
    min_layer_edges: int = typer.Option(
        10,
        help="Minimum eligible edges required before reporting a layer axis",
    ),
    max_depth: int = typer.Option(
        6,
        help="Maximum chain depth to follow",
    ),
    sample_edges_per_layer: int = typer.Option(
        100,
        help="Seed edges sampled per relation layer",
    ),
    max_branching: int = typer.Option(
        5,
        help="Maximum strongest same-layer branches considered per node",
    ),
    side_branching: int = typer.Option(
        2,
        help="Maximum strongest perpendicular branches considered per node/layer",
    ),
    shuffle_iterations: int = typer.Option(
        80,
        help="Random shuffles for the color-frequency baseline",
    ),
    random_seed: int = typer.Option(
        23,
        help="Seed for shuffled baseline reproducibility",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw report JSON",
    ),
    html_output: Optional[Path] = typer.Option(
        None,
        "--html",
        help="Write a standalone HTML report",
    ),
) -> None:
    """Measure long same-layer chains vs short perpendicular side branches."""
    import json

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.db.engine import DatabaseEngine
    from chicory.layer3.chain_anisotropy import analyze_chain_anisotropy

    console = Console()
    config = load_config(db_path=Path(db))
    db_engine = DatabaseEngine(config)
    db_engine.connect()

    try:
        layer_weights = {
            "cooccurrence": config.tensor_cooccurrence_weight,
            "synchronicity": config.tensor_synchronicity_weight,
            "semantic": config.tensor_semantic_weight,
            "semiotic": config.tensor_semiotic_weight,
            "glyph": config.tensor_glyph_weight,
            "inhibition": config.tensor_inhibition_weight,
        }
        report = analyze_chain_anisotropy(
            db_engine,
            edge_limit=edge_limit,
            per_layer_edges=per_layer_edges,
            min_edge_score=min_edge_score,
            min_tag_length=min_tag_length,
            min_layer_edges=min_layer_edges,
            max_depth=max_depth,
            sample_edges_per_layer=sample_edges_per_layer,
            max_branching=max_branching,
            side_branching=side_branching,
            shuffle_iterations=shuffle_iterations,
            random_seed=random_seed,
            layer_weights=layer_weights,
        )
    finally:
        db_engine.close()

    if json_output:
        console.print_json(json.dumps(report.as_dict()))
        return

    if html_output is not None:
        from chicory.layer3.square_visualizer import (
            render_chain_anisotropy_report_html,
        )

        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(
            render_chain_anisotropy_report_html(report),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote report:[/green] {html_output}")

    table = Table(title="Chain Anisotropy")
    table.add_column("Layer")
    table.add_column("Edges", justify="right")
    table.add_column("Probes", justify="right")
    table.add_column("Axis", justify="right")
    table.add_column("Side", justify="right")
    table.add_column("Contrast", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Z", justify="right")
    table.add_column("p", justify="right")

    for layer_report in report.layer_reports:
        baseline = (
            f"{layer_report.baseline_contrast_mean:.2f} +/- "
            f"{layer_report.baseline_contrast_std:.2f}"
        )
        table.add_row(
            layer_report.layer,
            str(layer_report.edge_count),
            str(layer_report.probe_count),
            f"{layer_report.axis_mean:.2f}",
            f"{layer_report.side_mean:.2f}",
            f"{layer_report.contrast:.2f}",
            f"{layer_report.ratio:.2f}",
            baseline,
            f"{layer_report.z_score:.2f}",
            f"{layer_report.p_value:.3f}",
        )

    console.print(table)


@app.command(name="zigzag")
def zigzag(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    motif_limit: int = typer.Option(
        2000,
        "--limit",
        help="Maximum square motifs to include in the metric",
    ),
    max_edges: int = typer.Option(
        500,
        help="Top tensor edges to use when enumerating candidate cycles",
    ),
    per_layer_edges: int = typer.Option(
        100,
        help="Extra top edges to include from each dominant relation layer",
    ),
    min_edge_score: float = typer.Option(
        0.05,
        help="Minimum weighted tensor score for square sides",
    ),
    min_colors: int = typer.Option(
        1,
        help="Minimum number of distinct side relation layers",
    ),
    min_tag_length: int = typer.Option(
        2,
        help="Ignore shorter tags when building square sides",
    ),
    require_void: bool = typer.Option(
        False,
        "--require-void",
        help="Only include squares with at least one implied missing diagonal",
    ),
    include_text: Optional[str] = typer.Option(
        None,
        "--include-text",
        help="Only include motifs whose tags or source summaries include this text",
    ),
    exclude_text: Optional[str] = typer.Option(
        None,
        "--exclude-text",
        help="Hide motifs whose tags or source summaries include this text",
    ),
    require_distinct_incident_layers: bool = typer.Option(
        False,
        "--strict-vertices",
        help="Require the two side colors touching each vertex to be distinct",
    ),
    shuffle_iterations: int = typer.Option(
        200,
        help="Random shuffles for the color-frequency baseline",
    ),
    random_seed: int = typer.Option(
        13,
        help="Seed for shuffled baseline reproducibility",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw report JSON",
    ),
    html_output: Optional[Path] = typer.Option(
        None,
        "--html",
        help="Write a standalone HTML report",
    ),
) -> None:
    """Measure rotation-invariant zigzag color bias in square motifs."""
    import json

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.db.engine import DatabaseEngine
    from chicory.layer3.zigzag_analyzer import analyze_square_zigzags

    console = Console()
    config = load_config(db_path=Path(db))
    db_engine = DatabaseEngine(config)
    db_engine.connect()

    try:
        layer_weights = {
            "cooccurrence": config.tensor_cooccurrence_weight,
            "synchronicity": config.tensor_synchronicity_weight,
            "semantic": config.tensor_semantic_weight,
            "semiotic": config.tensor_semiotic_weight,
            "glyph": config.tensor_glyph_weight,
            "inhibition": config.tensor_inhibition_weight,
        }
        report = analyze_square_zigzags(
            db_engine,
            motif_limit=motif_limit,
            max_edges=max_edges,
            per_layer_edges=per_layer_edges,
            min_edge_score=min_edge_score,
            min_colors=min_colors,
            min_tag_length=min_tag_length,
            require_void=require_void,
            include_text=include_text,
            exclude_text=exclude_text,
            require_distinct_incident_layers=require_distinct_incident_layers,
            shuffle_iterations=shuffle_iterations,
            random_seed=random_seed,
            layer_weights=layer_weights,
        )
    finally:
        db_engine.close()

    if json_output:
        console.print_json(json.dumps(report.as_dict()))
        return

    if html_output is not None:
        from chicory.layer3.square_visualizer import render_zigzag_report_html

        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(
            render_zigzag_report_html(report),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote report:[/green] {html_output}")

    summary = Table(title="Zigzag Orientation Metric")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("motifs", str(report.motif_count))
    summary.add_row("zigzags", f"{report.zigzag_count} ({report.zigzag_rate:.1%})")
    summary.add_row(
        "baseline",
        f"{report.baseline_zigzag_mean:.1f} +/- {report.baseline_zigzag_std:.1f}",
    )
    summary.add_row("z-score", f"{report.zigzag_z_score:.2f}")
    summary.add_row("p-value", f"{report.zigzag_p_value:.3f}")
    summary.add_row("vector mean", f"{report.vector_zigzag_mean:.3f}")
    summary.add_row(
        "positive vectors",
        f"{report.vector_zigzag_positive_rate:.1%}",
    )
    if report.dominant_zigzag_pair:
        pair, count = report.dominant_zigzag_pair
        summary.add_row("dominant zigzag pair", f"{pair} ({count})")
    console.print(summary)

    classes = Table(title="Orientation Classes")
    classes.add_column("Class")
    classes.add_column("Count", justify="right")
    classes.add_column("Share", justify="right")
    for label, count in report.class_counts:
        share = count / report.motif_count if report.motif_count else 0.0
        classes.add_row(label, str(count), f"{share:.1%}")
    console.print(classes)

    signatures = Table(title="Top Rotation-Collapsed Signatures")
    signatures.add_column("Signature")
    signatures.add_column("Count", justify="right")
    signatures.add_column("Share", justify="right")
    for label, count in report.signature_counts[:8]:
        share = count / report.motif_count if report.motif_count else 0.0
        signatures.add_row(label, str(count), f"{share:.1%}")
    console.print(signatures)


@app.command(name="vertical-squares")
def vertical_squares(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    mode: str = typer.Option(
        "tensor",
        help="Scan mode: tensor or incidence",
    ),
    limit: int = typer.Option(12, help="Maximum motifs to show"),
    min_count: int = typer.Option(
        1,
        help="Minimum memory chunks for each source/tag incidence",
    ),
    min_tag_length: int = typer.Option(
        1,
        help="Ignore shorter tags when building motif columns",
    ),
    require_letter: bool = typer.Option(
        True,
        help="Require one column to be a single-letter tag",
    ),
    tag_text: Optional[str] = typer.Option(
        "sequence",
        "--tag-text",
        help="Require the other column to include this text",
    ),
    exact_tag_text: bool = typer.Option(
        True,
        help="Match --tag-text as an exact tag instead of a substring",
    ),
    include_source: Optional[str] = typer.Option(
        None,
        "--include-source",
        help="Only use sources whose summary includes this text",
    ),
    exclude_source: Optional[str] = typer.Option(
        None,
        "--exclude-source",
        help="Skip sources whose summary includes this text",
    ),
    include_text: Optional[str] = typer.Option(
        None,
        "--include-text",
        help="Only show motifs whose sources or tags include this text",
    ),
    exclude_text: Optional[str] = typer.Option(
        None,
        "--exclude-text",
        help="Hide motifs whose sources or tags include this text",
    ),
    min_relation_score: float = typer.Option(
        0.0,
        help="Minimum tensor score between the two tag columns",
    ),
    max_tags_per_source: int = typer.Option(
        80,
        help="Maximum candidate tags retained per source",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw motif JSON",
    ),
    html_output: Optional[Path] = typer.Option(
        None,
        "--html",
        help="Write a standalone HTML/SVG visualization",
    ),
) -> None:
    """Find source-by-tag vertical square motifs."""
    import json

    from rich.console import Console
    from rich.table import Table

    from chicory.config import load_config
    from chicory.db.engine import DatabaseEngine
    from chicory.layer3.vertical_square_finder import (
        find_tensor_vertical_square_motifs,
        find_vertical_square_motifs,
    )

    console = Console()
    config = load_config(db_path=Path(db))
    db_engine = DatabaseEngine(config)
    db_engine.connect()

    try:
        layer_weights = {
            "cooccurrence": config.tensor_cooccurrence_weight,
            "synchronicity": config.tensor_synchronicity_weight,
            "semantic": config.tensor_semantic_weight,
            "semiotic": config.tensor_semiotic_weight,
            "glyph": config.tensor_glyph_weight,
            "inhibition": config.tensor_inhibition_weight,
        }
        if mode == "tensor":
            motifs = find_tensor_vertical_square_motifs(
                db_engine,
                limit=limit,
                min_edge_score=min_relation_score,
                require_letter=require_letter,
                tag_text=tag_text,
                exact_tag_text=exact_tag_text,
                include_source=include_source,
                exclude_source=exclude_source,
                include_text=include_text,
                exclude_text=exclude_text,
                layer_weights=layer_weights,
            )
        elif mode == "incidence":
            motifs = find_vertical_square_motifs(
                db_engine,
                limit=limit,
                min_count=min_count,
                min_tag_length=min_tag_length,
                require_letter=require_letter,
                tag_text=tag_text,
                exact_tag_text=exact_tag_text,
                include_source=include_source,
                exclude_source=exclude_source,
                include_text=include_text,
                exclude_text=exclude_text,
                min_relation_score=min_relation_score,
                max_tags_per_source=max_tags_per_source,
                layer_weights=layer_weights,
            )
        else:
            console.print("[red]Mode must be 'tensor' or 'incidence'.[/red]")
            raise typer.Exit(1)
    finally:
        db_engine.close()

    if json_output:
        console.print_json(json.dumps([m.as_dict() for m in motifs]))
        return

    if html_output is not None:
        from chicory.layer3.square_visualizer import (
            render_tensor_vertical_square_motifs_html,
            render_vertical_square_motifs_html,
        )

        html_output.parent.mkdir(parents=True, exist_ok=True)
        html = (
            render_tensor_vertical_square_motifs_html(motifs)
            if mode == "tensor"
            else render_vertical_square_motifs_html(motifs)
        )
        html_output.write_text(
            html,
            encoding="utf-8",
        )
        console.print(f"[green]Wrote visualization:[/green] {html_output}")

    if not motifs:
        console.print("[dim]No vertical square motifs found.[/dim]")
        return

    if mode == "tensor":
        table = Table(title="Tensor-Vertical Square Motifs")
        table.add_column("#", justify="right")
        table.add_column("Source A")
        table.add_column("Source B")
        table.add_column("Columns")
        table.add_column("Cell Layers")
        table.add_column("Orient")
        table.add_column("Diag")
        table.add_column("Sym", justify="right")
        table.add_column("Cell Scores")
        table.add_column("Column Relation")
        table.add_column("Score", justify="right")

        for idx, motif in enumerate(motifs, start=1):
            relation = (
                f"{motif.column_relation_layer or 'none'} "
                f"({motif.column_relation_status}, "
                f"{motif.column_relation_score:.2f})"
            )
            table.add_row(
                str(idx),
                motif.sources[0],
                motif.sources[1],
                " / ".join(motif.column_tags),
                " / ".join(motif.cell_layers),
                motif.orientation_class,
                motif.diagonal_bias,
                str(motif.symmetry_variants),
                ", ".join(f"{score:.2f}" for score in motif.cell_scores),
                relation,
                f"{motif.interestingness:.2f}",
            )

        console.print(table)
        return

    table = Table(title="Vertical Square Motifs")
    table.add_column("#", justify="right")
    table.add_column("Source A")
    table.add_column("Source B")
    table.add_column("Columns")
    table.add_column("Counts")
    table.add_column("Relation")
    table.add_column("Balance", justify="right")
    table.add_column("Score", justify="right")

    for idx, motif in enumerate(motifs, start=1):
        relation = (
            f"{motif.relation_layer or 'none'} "
            f"({motif.relation_status}, {motif.relation_score:.2f})"
        )
        table.add_row(
            str(idx),
            motif.sources[0],
            motif.sources[1],
            " / ".join(motif.tags),
            ", ".join(str(c) for c in motif.counts),
            relation,
            f"{motif.source_balance:.2f}",
            f"{motif.interestingness:.2f}",
        )

    console.print(table)


@app.command()
def dashboard(
    db: str = typer.Option(_DEFAULT_DB, help="Path to the database file"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8050, help="Port to bind to"),
    debug: bool = typer.Option(False, help="Enable Dash debug mode"),
) -> None:
    """Launch the web dashboard."""
    try:
        from chicory.dashboard.app import run_dashboard
    except ImportError:
        from rich.console import Console

        Console().print(
            "[red]Dashboard dependencies not installed. "
            "Run: pip install chicory\\[dashboard][/red]"
        )
        raise typer.Exit(1)

    run_dashboard(db_path=Path(db), host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()
