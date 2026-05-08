"""Bootstrap the forest + canopy from existing Chicory data.

Reads existing memories, tags, retrievals, sync events, and tensor data
to seed co-occurrence edges, forest blocks, bridge edges, and canopy
observations in bulk. Designed for one-time seeding of an existing DB.

Usage:
    python scripts/bootstrap_canopy.py [--db PATH] [--batch-size N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from itertools import combinations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap forest + canopy from existing Chicory data")
    parser.add_argument("--db", help="Path to chicory.db (default: from config)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    parser.add_argument("--skip-bridges", action="store_true", help="Skip bridge edge computation")
    parser.add_argument("--phase", type=int, help="Run only this phase (1-8)")
    args = parser.parse_args()

    from chicory.config import load_config
    from chicory.db.engine import DatabaseEngine
    from chicory.db.schema import apply_schema

    overrides = {}
    if args.db:
        overrides["db_path"] = args.db
    cfg = load_config(**overrides)

    db = DatabaseEngine(cfg)
    db.connect()
    apply_schema(db)

    log.info("DB: %s", cfg.db_path)

    def should_run(phase: int) -> bool:
        return args.phase is None or args.phase == phase

    t0 = time.time()

    co_edges: dict = {}
    tag_mem_counts: dict = {}
    total_memories = 0

    if should_run(1):
        existing = db.execute("SELECT COUNT(*) as c FROM cooccurrence_edges").fetchone()["c"]
        if existing > 0 and args.phase is None:
            log.info("Phase 1: Skipping — %d co-occurrence edges exist", existing)
        else:
            log.info("Phase 1: Building co-occurrence from memory_tags...")
            co_edges, tag_mem_counts, total_memories = build_cooccurrence_from_memories(db)
            log.info("  %d tag pairs with positive PPMI from %d memories", len(co_edges), total_memories)

    if should_run(2) and co_edges:
        log.info("Phase 2: Adding retrieval co-occurrence...")
        ret_edges, total_retrievals = build_cooccurrence_from_retrievals(db)
        log.info("  %d tag pairs from %d retrievals", len(ret_edges), total_retrievals)

        for pair, vals in ret_edges.items():
            if pair in co_edges:
                co_edges[pair] = max(co_edges[pair], vals)
            else:
                co_edges[pair] = vals

    if args.dry_run:
        log.info("DRY RUN — would write %d co-occurrence edges", len(co_edges))
        log.info("Elapsed: %.1fs", time.time() - t0)
        return

    if should_run(3) and co_edges:
        log.info("Phase 3: Writing %d co-occurrence edges...", len(co_edges))
        write_cooccurrence_edges(db, co_edges, tag_mem_counts, total_memories, args.batch_size)

    if should_run(4):
        existing = db.execute("SELECT COUNT(*) as c FROM forest_blocks").fetchone()["c"]
        if existing > 0 and args.phase is None:
            log.info("Phase 4: Skipping — %d forest blocks exist", existing)
        else:
            log.info("Phase 4: Building forest blocks...")
            block_keys = build_forest_blocks(db, args.batch_size)
            log.info("  %d forest blocks created", len(block_keys))

    if should_run(5) and not args.skip_bridges:
        existing = db.execute("SELECT COUNT(*) as c FROM bridge_edges").fetchone()["c"]
        if existing > 0 and args.phase is None:
            log.info("Phase 5: Skipping — %d bridge edges exist", existing)
        else:
            log.info("Phase 5: Building bridge edges...")
            from chicory.layer4.bridge_optimizer import BridgeOptimizer
            bridge = BridgeOptimizer(db, cfg)
            block_keys_for_bridges = [
                r["block_key"] for r in
                db.execute("SELECT block_key FROM forest_blocks").fetchall()
            ]
            bridge_count = bridge.update_bridges(block_keys_for_bridges)
            db.connection.commit()
            log.info("  %d bridge edges created", bridge_count)
    elif args.skip_bridges:
        log.info("Phase 5: Skipped (--skip-bridges)")

    # Phase 6: Build episodic relational tensor
    if should_run(6):
        existing = db.execute("SELECT COUNT(*) as c FROM memory_relational_tensor").fetchone()["c"]
        if existing > 0 and args.phase is None:
            log.info("Phase 6: Skipping — %d episodic edges exist", existing)
        else:
            log.info("Phase 6: Building episodic relational tensor...")
            from chicory.layer4.episodic_tensor import EpisodicTensor
            episodic = EpisodicTensor(db, cfg)
            edge_count = episodic.bootstrap(batch_size=args.batch_size)
            log.info("  %d episodic edges created", edge_count)

    # Phase 7: Run canopy observations using episodic neighborhoods
    if should_run(7):
        existing = db.execute("SELECT COUNT(*) as c FROM canopy_blocks").fetchone()["c"]
        if existing > 0 and args.phase is None:
            log.info("Phase 7: Skipping — %d canopy blocks exist", existing)
        else:
            log.info("Phase 7: Running canopy observations...")
            from chicory.layer4.forest import ForestReorganizer
            from chicory.layer4.canopy import CanopyObserver
            forest = ForestReorganizer(db, cfg)
            canopy = CanopyObserver(db, cfg, forest)

            grown = observe_all_memories(db, cfg, canopy, args.batch_size)
            db.connection.commit()
            log.info("  Canopy observations complete")

    # Phase 8: Observe sync events
    if should_run(8):
        log.info("Phase 8: Observing sync events (top 1000 by strength)...")
        from chicory.layer4.forest import ForestReorganizer
        from chicory.layer4.canopy import CanopyObserver
        forest = ForestReorganizer(db, cfg)
        canopy = CanopyObserver(db, cfg, forest)
        observe_sync_events(db, canopy, limit=1000)
        db.connection.commit()

    # Final stats
    stats = {}
    for table in ["cooccurrence_edges", "forest_blocks", "bridge_edges",
                   "memory_relational_tensor",
                   "canopy_blocks", "canopy_observations", "canopy_cross_layer_edges",
                   "canopy_support_edges"]:
        stats[table] = db.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]

    grown_count = db.execute(
        "SELECT COUNT(*) as c FROM canopy_blocks WHERE first_growth_at IS NOT NULL"
    ).fetchone()["c"]

    elapsed = time.time() - t0
    log.info("Bootstrap complete in %.1fs", elapsed)
    log.info("Results:")
    for k, v in stats.items():
        log.info("  %-35s %d", k, v)
    log.info("  %-35s %d", "grown_blocks", grown_count)


def build_cooccurrence_from_memories(db):
    """Compute tag pair PPMI from memory_tags co-occurrence."""
    rows = db.execute(
        "SELECT memory_id, tag_id FROM memory_tags ORDER BY memory_id"
    ).fetchall()

    mem_tags: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        mem_tags[r["memory_id"]].append(r["tag_id"])

    total_memories = len(mem_tags)

    tag_counts: dict[int, int] = defaultdict(int)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for mid, tags in mem_tags.items():
        for t in tags:
            tag_counts[t] += 1
        for a, b in combinations(sorted(set(tags)), 2):
            pair_counts[(a, b)] += 1

    co_edges: dict[tuple[int, int], float] = {}
    n = max(total_memories, 1)
    for (a, b), count in pair_counts.items():
        p_ab = count / n
        p_a = tag_counts[a] / n
        p_b = tag_counts[b] / n
        denom = p_a * p_b
        if denom > 0:
            pmi = math.log(p_ab / denom)
            ppmi = max(0.0, pmi)
            if ppmi > 0:
                co_edges[(a, b)] = ppmi

    return co_edges, tag_counts, total_memories


def build_cooccurrence_from_retrievals(db):
    """Compute tag pair PPMI from retrieval_tag_hits co-occurrence."""
    rows = db.execute(
        "SELECT retrieval_id, tag_id FROM retrieval_tag_hits ORDER BY retrieval_id"
    ).fetchall()

    ret_tags: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        ret_tags[r["retrieval_id"]].append(r["tag_id"])

    total = len(ret_tags)
    if total == 0:
        return {}, 0

    tag_counts: dict[int, int] = defaultdict(int)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for rid, tags in ret_tags.items():
        for t in tags:
            tag_counts[t] += 1
        for a, b in combinations(sorted(set(tags)), 2):
            pair_counts[(a, b)] += 1

    co_edges: dict[tuple[int, int], float] = {}
    n = max(total, 1)
    for (a, b), count in pair_counts.items():
        p_ab = count / n
        p_a = tag_counts[a] / n
        p_b = tag_counts[b] / n
        denom = p_a * p_b
        if denom > 0:
            pmi = math.log(p_ab / denom)
            ppmi = max(0.0, pmi)
            if ppmi > 0:
                co_edges[(a, b)] = ppmi

    return co_edges, total


def write_cooccurrence_edges(db, co_edges, tag_counts, total_memories, batch_size):
    """Bulk-insert co-occurrence edges."""
    n = max(total_memories, 1)
    rows = []
    for (a, b), ppmi in co_edges.items():
        p_a = tag_counts.get(a, 1) / n
        p_b = tag_counts.get(b, 1) / n
        denom = p_a * p_b
        lift = (ppmi / math.log(max(denom, 1e-10))) if denom > 0 else 0.0

        rows.append((
            "tag", str(a), "tag", str(b), "memory",
            1.0,  # raw_count placeholder
            denom * n,
            lift,
            ppmi if ppmi > 0 else 0.0,
            ppmi,
            1,
        ))

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        db.connection.executemany(
            """INSERT INTO cooccurrence_edges
               (left_type, left_id, right_type, right_id, scope_type,
                raw_count, expected_count, lift, pmi, co_strength, evidence_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(left_type, left_id, right_type, right_id, scope_type)
               DO UPDATE SET co_strength = MAX(co_strength, excluded.co_strength),
                             evidence_count = evidence_count + 1""",
            batch,
        )

    db.connection.commit()
    log.info("  Wrote %d co-occurrence edges", len(rows))


def build_forest_blocks(db, batch_size):
    """Build forest blocks from distinct tag sets per memory."""
    rows = db.execute(
        "SELECT memory_id, tag_id FROM memory_tags ORDER BY memory_id"
    ).fetchall()

    mem_tags: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        mem_tags[r["memory_id"]].append(r["tag_id"])

    import hashlib

    seen_keys: set[str] = set()
    block_rows = []
    membership_rows = []

    for mid, tags in mem_tags.items():
        if len(tags) < 2:
            continue
        sorted_strs = sorted(str(t) for t in tags)
        raw = f"cooccurrence:tag_set:{','.join(sorted_strs)}"
        block_key = hashlib.sha256(raw.encode()).hexdigest()[:32]

        if block_key in seen_keys:
            continue
        seen_keys.add(block_key)

        # Compute internal density from co-occurrence edges
        pair_strengths = []
        for i in range(len(sorted_strs)):
            for j in range(i + 1, len(sorted_strs)):
                row = db.execute(
                    """SELECT co_strength FROM cooccurrence_edges
                       WHERE left_type='tag' AND left_id=? AND right_type='tag' AND right_id=?
                       AND scope_type='memory'""",
                    (sorted_strs[i], sorted_strs[j]),
                ).fetchone()
                pair_strengths.append(row["co_strength"] if row else 0.0)

        density = sum(pair_strengths) / len(pair_strengths) if pair_strengths else 0.0

        block_rows.append((block_key, "tag_set", "cooccurrence", density, 1))

        for tid in tags:
            membership_rows.append((block_key, "tag", str(tid), density, 1))
        membership_rows.append((block_key, "memory", mid, 1.0, 1))

    # Bulk insert blocks
    for i in range(0, len(block_rows), batch_size):
        batch = block_rows[i:i + batch_size]
        db.connection.executemany(
            """INSERT INTO forest_blocks (block_key, block_type, forest_type, internal_density, evidence_count)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(block_key) DO UPDATE SET
                   internal_density = MAX(internal_density, excluded.internal_density),
                   evidence_count = evidence_count + 1""",
            batch,
        )

    db.connection.commit()

    # Build block_id lookup
    block_id_map: dict[str, int] = {}
    for key in seen_keys:
        row = db.execute("SELECT id FROM forest_blocks WHERE block_key=?", (key,)).fetchone()
        if row:
            block_id_map[key] = row["id"]

    # Bulk insert memberships
    resolved_memberships = []
    for block_key, target_type, target_id, strength, evidence in membership_rows:
        bid = block_id_map.get(block_key)
        if bid:
            resolved_memberships.append((bid, target_type, target_id, strength, evidence))

    for i in range(0, len(resolved_memberships), batch_size):
        batch = resolved_memberships[i:i + batch_size]
        db.connection.executemany(
            """INSERT INTO block_memberships (block_id, target_type, target_id, membership_strength, evidence_count)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(block_id, target_type, target_id) DO UPDATE SET
                   membership_strength = MAX(membership_strength, excluded.membership_strength),
                   evidence_count = evidence_count + 1""",
            batch,
        )

    db.connection.commit()
    log.info("  %d blocks, %d memberships", len(block_rows), len(resolved_memberships))

    return list(seen_keys)


def observe_all_memories(db, cfg, canopy, batch_size):
    """Run canopy observations from episodic tensor neighborhoods.

    For each memory, pull its episodic neighbors and observe the
    cluster.  The canopy discovers co-retrieval structure that
    tag-level associations don't predict.
    """
    mem_rows = db.execute("SELECT id FROM memories ORDER BY id").fetchall()
    total = len(mem_rows)
    grown_total = 0
    processed = 0

    for row in mem_rows:
        mid = row["id"]
        grown = canopy.observe_single_memory(
            source="bootstrap",
            source_id=mid,
            memory_id=mid,
        )
        grown_total += len(grown)
        processed += 1

        if processed % batch_size == 0:
            db.connection.commit()
            log.info("  Observed %d/%d memories (%d grown blocks so far)", processed, total, grown_total)

    db.connection.commit()
    log.info("  Observed %d memories total, %d grown blocks", processed, grown_total)
    return grown_total


def observe_sync_events(db, canopy, limit=1000):
    """Run canopy observations for the strongest sync events."""
    rows = db.execute(
        """SELECT id, involved_tags, involved_memories
           FROM synchronicity_events
           ORDER BY strength DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    grown_total = 0
    for i, row in enumerate(rows):
        mem_rows = db.execute(
            "SELECT memory_id FROM sync_event_memories WHERE event_id=?",
            (row["id"],),
        ).fetchall()
        mem_ids = [r["memory_id"] for r in mem_rows]

        if len(mem_ids) < 2:
            continue

        grown = canopy.observe(
            source="synchronicity",
            source_id=str(row["id"]),
            memory_ids=mem_ids[:20],
        )
        grown_total += len(grown)

        if (i + 1) % 100 == 0:
            db.connection.commit()
            log.info("  Observed %d/%d sync events (%d grown)", i + 1, len(rows), grown_total)

    log.info("  Observed %d sync events, %d grown blocks", len(rows), grown_total)


if __name__ == "__main__":
    main()
