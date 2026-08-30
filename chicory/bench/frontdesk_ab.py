"""Front-desk A/B bench: librarian_consult (dispatch) vs flat hybrid retrieval.

Runs a stratified query set through both retrieval paths of the SAME
Chicory instance and reports per-class metrics.  This is the experiment
the routing literature says to run: routing helps synthesis/temporal
query classes and should be ~neutral on factoid; the only way to know
for a specific corpus is to measure it.

Query file: JSONL, one object per line:
    {"query": "...", "class": "factoid|synthesis|temporal|unanswerable",
     "relevant_ids": ["memory-uuid", ...]}   # relevant_ids optional

Metrics per class:
  - hit@k / recall@k against relevant_ids (when provided), for each arm
  - answered_rate: fraction of queries returning ANY results
      (for the 'unanswerable' class, LOWER is better — measures abstention)
  - faq_hit_rate, subjects_consulted (consult arm)
  - mean latency per arm

The bench disables the demand/penumbra ledgers and the sweep by default
(--live re-enables them) so bench traffic cannot promote FAQs, file
commissions, or skew burial statistics.  Retrieval side effects (salience,
reinforcement, centroid reweighting) still fire — benchmark a COPY of the
database when the store must stay pristine.

Usage:
    python -m chicory.bench.frontdesk_ab queries.jsonl \
        --db-path ~/.chicory/bench_copy.db --top-k 5 --output ab_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def _load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            q.setdefault("class", "unclassified")
            q.setdefault("relevant_ids", [])
            queries.append(q)
    return queries


def _consult_result_ids(consult: dict) -> list[str]:
    """Memory ids the consult response surfaced, channels interleaved.

    Round-robin across shelves and the general channel (FAQ supporting
    ids first) so that a top_k slice represents every channel — a strict
    shelf-first concatenation would let one full shelf crowd general-
    channel hits out of the judged window.
    """
    ids: list[str] = []
    if consult.get("faq"):
        ids.extend(consult["faq"].get("supporting_memory_ids", []))
    channels = [
        [r["memory_id"] for r in shelf.get("results", [])]
        for shelf in consult.get("subjects", [])
    ]
    channels.append([r["memory_id"] for r in consult.get("general", [])])
    rank = 0
    while any(rank < len(c) for c in channels):
        for c in channels:
            if rank < len(c):
                ids.append(c[rank])
        rank += 1
    seen: set[str] = set()
    ordered = []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)
    return ordered


def run_ab(orchestrator, queries: list[dict], top_k: int = 5) -> dict:
    """Run both arms over the query set. Returns per-class + overall stats."""
    per_class: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "flat_hits": 0, "flat_judged": 0, "flat_recall_sum": 0.0,
        "consult_hits": 0, "consult_judged": 0, "consult_recall_sum": 0.0,
        "flat_answered": 0, "consult_answered": 0,
        "faq_hits": 0, "subjects_consulted_sum": 0,
        "flat_latency_sum": 0.0, "consult_latency_sum": 0.0,
    })
    rows = []

    for i, q in enumerate(queries):
        cls = q["class"]
        stats = per_class[cls]
        stats["n"] += 1
        relevant = set(q["relevant_ids"])

        # Alternate arm order per query: each arm's side effects (salience,
        # reinforcement) slightly reweight the store, so a fixed order
        # would systematically advantage the second arm.
        def _run_flat():
            t0 = time.perf_counter()
            r = orchestrator.handle_retrieve_memories(
                query=q["query"], method="hybrid", top_k=top_k,
            )
            return r, time.perf_counter() - t0

        def _run_consult():
            t0 = time.perf_counter()
            r = orchestrator.handle_librarian_consult(
                query=q["query"], top_k=top_k,
            )
            return r, time.perf_counter() - t0

        if i % 2 == 0:
            flat, flat_latency = _run_flat()
            consult, consult_latency = _run_consult()
        else:
            consult, consult_latency = _run_consult()
            flat, flat_latency = _run_flat()
        flat_ids = [r["memory_id"] for r in flat.get("results", [])]
        consult_ids = _consult_result_ids(consult)

        stats["flat_latency_sum"] += flat_latency
        stats["consult_latency_sum"] += consult_latency
        stats["flat_answered"] += 1 if flat_ids else 0
        stats["consult_answered"] += 1 if consult_ids else 0
        stats["faq_hits"] += 1 if consult.get("faq") else 0
        stats["subjects_consulted_sum"] += len(consult.get("subjects", []))

        row = {
            "query": q["query"], "class": cls,
            "flat_ids": flat_ids[:top_k], "consult_ids": consult_ids[:top_k],
            "faq_hit": bool(consult.get("faq")),
            "flat_latency": round(flat_latency, 4),
            "consult_latency": round(consult_latency, 4),
        }
        if relevant:
            stats["flat_judged"] += 1
            stats["consult_judged"] += 1
            flat_found = relevant & set(flat_ids[:top_k])
            consult_found = relevant & set(consult_ids[:top_k])
            stats["flat_hits"] += 1 if flat_found else 0
            stats["consult_hits"] += 1 if consult_found else 0
            stats["flat_recall_sum"] += len(flat_found) / len(relevant)
            stats["consult_recall_sum"] += len(consult_found) / len(relevant)
            row["flat_recall"] = round(len(flat_found) / len(relevant), 3)
            row["consult_recall"] = round(len(consult_found) / len(relevant), 3)
        rows.append(row)

    summary: dict[str, dict] = {}
    for cls, s in per_class.items():
        n = s["n"]
        entry = {
            "queries": n,
            "flat_answered_rate": round(s["flat_answered"] / n, 3),
            "consult_answered_rate": round(s["consult_answered"] / n, 3),
            "faq_hit_rate": round(s["faq_hits"] / n, 3),
            "mean_subjects_consulted": round(s["subjects_consulted_sum"] / n, 2),
            "flat_mean_latency_s": round(s["flat_latency_sum"] / n, 4),
            "consult_mean_latency_s": round(s["consult_latency_sum"] / n, 4),
        }
        if s["flat_judged"]:
            entry["flat_hit_rate"] = round(s["flat_hits"] / s["flat_judged"], 3)
            entry["consult_hit_rate"] = round(s["consult_hits"] / s["consult_judged"], 3)
            entry["flat_mean_recall"] = round(s["flat_recall_sum"] / s["flat_judged"], 3)
            entry["consult_mean_recall"] = round(s["consult_recall_sum"] / s["consult_judged"], 3)
        summary[cls] = entry

    return {"summary": summary, "rows": rows}


def _print_summary(summary: dict) -> None:
    print()
    print("Front-desk A/B — consult (dispatch) vs flat hybrid")
    print("=" * 66)
    for cls, s in sorted(summary.items()):
        print(f"\n[{cls}]  ({s['queries']} queries)")
        if "flat_hit_rate" in s:
            print(f"  hit@k       flat {s['flat_hit_rate']:.3f}   consult {s['consult_hit_rate']:.3f}")
            print(f"  recall@k    flat {s['flat_mean_recall']:.3f}   consult {s['consult_mean_recall']:.3f}")
        answered_note = "  (lower is better)" if cls == "unanswerable" else ""
        print(f"  answered    flat {s['flat_answered_rate']:.3f}   consult {s['consult_answered_rate']:.3f}{answered_note}")
        print(f"  faq hits    {s['faq_hit_rate']:.3f}    shelves/query {s['mean_subjects_consulted']}")
        print(f"  latency     flat {s['flat_mean_latency_s']}s   consult {s['consult_mean_latency_s']}s")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path, help="Stratified query JSONL")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("frontdesk_ab_results.json"))
    parser.add_argument(
        "--live", action="store_true",
        help="Keep the demand/penumbra ledgers and sweep ON during the bench. "
             "By default they are disabled so bench queries cannot pollute "
             "query clusters, promote themselves into FAQs, file commissions, "
             "or skew the near-miss ledger of the instance being measured.",
    )
    args = parser.parse_args()

    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    overrides = {}
    if args.db_path:
        overrides["db_path"] = args.db_path
    if not args.live:
        # Ledger isolation: measurement must not become curriculum.
        # (Retrieval still fires salience/reinforcement side effects —
        # run against a COPY of the DB if the store must stay pristine.)
        overrides.update(
            librarian_demand_enabled=False,
            penumbra_enabled=False,
            librarian_sweep_enabled=False,
        )
    config = load_config(**overrides)
    orchestrator = Orchestrator(config)
    try:
        queries = _load_queries(args.queries)
        print(f"Loaded {len(queries)} queries; running both arms...")
        results = run_ab(orchestrator, queries, top_k=args.top_k)
        _print_summary(results["summary"])
        args.output.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8",
        )
        print(f"Full results written to {args.output}")
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
