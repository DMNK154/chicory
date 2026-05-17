"""Ingest the EnterpriseRAG-Bench document corpus into a Chicory instance.

Reads document JSONs from the benchmark's sources/ directory, extracts
content using each document's field labels, and stores them in Chicory
with source_path=dataset_doc_uuid for evaluation mapping.

Usage:
    python -m chicory.bench.corpus_loader \
        --sources-dir /path/to/EnterpriseRAG-Bench/sources \
        --db-path ~/.chicory/bench.db \
        --limit 1000          # optional: cap for testing
        --source-types github slack  # optional: filter by source
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from chicory.bench.utils import extract_document_content

logger = logging.getLogger(__name__)


def discover_documents(
    sources_dir: Path,
    source_types: list[str] | None = None,
    limit: int | None = None,
) -> list[tuple[Path, str]]:
    """Walk sources/ and yield (file_path, source_type) pairs."""
    results: list[tuple[Path, str]] = []

    if not sources_dir.is_dir():
        logger.error("Sources directory not found: %s", sources_dir)
        return results

    for source_dir in sorted(sources_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source_type = source_dir.name
        if source_types and source_type not in source_types:
            continue

        for json_file in sorted(source_dir.rglob("*.json")):
            results.append((json_file, source_type))
            if limit and len(results) >= limit:
                return results

    return results


def load_document(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Skipping %s: %s", path, e)
        return None


def ingest_corpus(
    sources_dir: str | Path,
    db_path: str | Path | None = None,
    source_types: list[str] | None = None,
    limit: int | None = None,
    critical: bool | None = None,
    batch_size: int = 200,
    skip_finalize: bool = False,
) -> dict[str, Any]:
    """Ingest benchmark documents into a fresh Chicory instance.

    Args:
        sources_dir: path to EnterpriseRAG-Bench/sources/
        db_path: Chicory DB path (default: ~/.chicory/bench.db)
        source_types: filter to specific source types (e.g. ["github", "slack"])
        limit: max documents to ingest (None = all)
        critical: force tier (None = auto-classify, True = all critical, False = all reference)
        batch_size: how many docs to batch before embedding

    Returns:
        stats dict with counts and timing
    """
    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    sources_path = Path(sources_dir).resolve()
    db = Path(db_path).expanduser() if db_path else Path.home() / ".chicory" / "bench.db"

    config = load_config(db_path=db)
    orchestrator = Orchestrator(config)

    docs = discover_documents(sources_path, source_types, limit)
    logger.info("Found %d documents to ingest", len(docs))

    stats: dict[str, Any] = {
        "total_discovered": len(docs),
        "ingested": 0,
        "skipped_parse": 0,
        "skipped_empty": 0,
        "skipped_duplicate": 0,
        "critical": 0,
        "reference": 0,
        "source_type_counts": {},
        "elapsed_seconds": 0.0,
    }

    # Load all already-ingested source_paths upfront (single query)
    existing_source_paths: set[str] = set()
    rows = orchestrator._db.execute(
        "SELECT source_path FROM memories WHERE source_path IS NOT NULL"
    ).fetchall()
    for row in rows:
        existing_source_paths.add(row["source_path"])
    logger.info("Found %d already-ingested source paths", len(existing_source_paths))

    t0 = time.time()
    pending_batch: list[dict[str, Any]] = []

    for i, (file_path, source_type) in enumerate(docs):
        doc = load_document(file_path)
        if doc is None:
            stats["skipped_parse"] += 1
            continue

        dataset_doc_uuid = doc.get("dataset_doc_uuid", "")
        if not dataset_doc_uuid:
            stats["skipped_parse"] += 1
            continue

        if dataset_doc_uuid in existing_source_paths:
            stats["skipped_duplicate"] += 1
            continue

        title, content = extract_document_content(doc)
        if not content.strip():
            stats["skipped_empty"] += 1
            continue

        full_text = f"{title}\n\n{content}" if title else content

        tags = [
            "bench-document",
            f"bench-source:{source_type}",
            source_type,
        ]

        source_metadata = doc.get("source_metadata", {})
        if isinstance(source_metadata, dict):
            for key in ("channel", "project", "repo", "label", "category"):
                val = source_metadata.get(key)
                if val and isinstance(val, str) and len(val) < 60:
                    tags.append(val.lower().replace(" ", "-"))

        result = orchestrator.handle_store_memory(
            content=full_text,
            tags=tags,
            importance=0.5,
            summary=f"[{source_type}] {title[:120]}" if title else f"[{source_type}] {dataset_doc_uuid}",
            skip_embedding=True,
            defer_side_effects=True,
            source_path=dataset_doc_uuid,
            ingestion_tier="reference" if critical is False else "critical" if critical is True else "auto",
        )

        pending_batch.append(result)
        stats["ingested"] += 1
        stats["source_type_counts"][source_type] = stats["source_type_counts"].get(source_type, 0) + 1

        if len(pending_batch) >= batch_size:
            _flush_batch(orchestrator, pending_batch, stats, critical, skip_finalize)
            pending_batch.clear()
            logger.info(
                "Progress: %d/%d ingested (%d skipped)",
                stats["ingested"], len(docs),
                stats["skipped_parse"] + stats["skipped_empty"] + stats["skipped_duplicate"],
            )

    if pending_batch:
        _flush_batch(orchestrator, pending_batch, stats, critical, skip_finalize)

    stats["elapsed_seconds"] = round(time.time() - t0, 1)
    logger.info(
        "Ingestion complete: %d docs in %.1fs (%d critical, %d reference)",
        stats["ingested"], stats["elapsed_seconds"],
        stats["critical"], stats["reference"],
    )
    return stats


def _flush_batch(
    orchestrator,
    batch: list[dict[str, Any]],
    stats: dict[str, Any],
    critical: bool | None = None,
    skip_finalize: bool = False,
) -> None:
    """Batch-embed accumulated memories and optionally finalize."""
    memory_ids = [r["memory_id"] for r in batch]

    orchestrator._batch_embed_memories(memory_ids)

    if critical is False or skip_finalize:
        stats["reference"] += len(memory_ids)
    else:
        orchestrator._finalize_ingested_memories(memory_ids)
        stats["critical"] += len(memory_ids)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingest EnterpriseRAG-Bench corpus into Chicory"
    )
    parser.add_argument(
        "--sources-dir", required=True,
        help="Path to EnterpriseRAG-Bench/sources/ directory",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Chicory DB path (default: ~/.chicory/bench.db)",
    )
    parser.add_argument(
        "--source-types", nargs="*", default=None,
        help="Filter to specific source types (e.g. github slack)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max documents to ingest (for testing)",
    )
    parser.add_argument(
        "--critical", type=str, default=None, choices=["true", "false", "auto"],
        help="Force ingestion tier (default: auto-classify)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Batch size for embedding (default: 200)",
    )
    parser.add_argument(
        "--skip-finalize", action="store_true",
        help="Skip tensor/lattice finalization (use 'chicory sync' afterward)",
    )
    args = parser.parse_args()

    critical_flag: bool | None = None
    if args.critical == "true":
        critical_flag = True
    elif args.critical == "false":
        critical_flag = False

    result = ingest_corpus(
        sources_dir=args.sources_dir,
        db_path=args.db_path,
        source_types=args.source_types,
        limit=args.limit,
        critical=critical_flag,
        batch_size=args.batch_size,
        skip_finalize=args.skip_finalize,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
