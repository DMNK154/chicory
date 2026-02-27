#!/usr/bin/env python3
"""Chicory Ramsey Network — from blank slate to self-organizing memory.

This demo shows the full lifecycle of the Prime Ramsey Lattice and
Tag Relational Tensor, bootstrapping from an empty database with zero
external seed data.

    python -m examples.ramsey_network_demo

No API keys, no model downloads, no external dependencies beyond numpy.
Runs entirely in-memory with deterministic mock embeddings.

Architecture overview:

    Layer 1 — Memory Store     Tags, embeddings, salience scoring
    Layer 2 — Trend Engine     Time-series tag activity, retrieval tracking
    Layer 3 — Phase Space      Temperature/retrieval frequency quadrants
    Layer 3 — Synchronicity    Event detection + Prime Ramsey Lattice
    Layer 3.5 — Tensor         Four-network tag relational tensor
    Layer 4 — Meta-Patterns    Adaptive thresholds, cross-domain analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.salience import SalienceScorer
from chicory.layer1.tag_manager import TagManager
from chicory.layer2.retrieval_tracker import RetrievalTracker
from chicory.layer2.trend_engine import TrendEngine
from chicory.layer3.phase_space import PhaseSpace
from chicory.layer3.synchronicity_detector import SynchronicityDetector
from chicory.layer3.synchronicity_engine import SynchronicityEngine
from chicory.layer1.memory_store import MemoryStore
from chicory.layer4.adaptive_thresholds import AdaptiveThresholds
from chicory.layer4.meta_analyzer import MetaAnalyzer
from chicory.models.synchronicity import SynchronicityEvent
from tests.conftest import MockEmbeddingEngine


# ── Formatting helpers ───────────────────────────────────────────────

def banner(title: str) -> None:
    width = 64
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def kv(key: str, value, indent: int = 2) -> None:
    print(f"{' ' * indent}{key}: {value}")


# ── Build the stack ──────────────────────────────────────────────────

def build_stack() -> dict:
    """Assemble the full Chicory stack with in-memory DB and mock embeddings."""
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="demo",
        llm_model="demo-model",
        embedding_model="mock",
        embedding_dimension=64,
        similarity_threshold=0.0,
        # Smaller prime set for readable output
        lattice_primes=[2, 3, 5, 7, 11, 13, 17, 19, 23],
        lattice_min_resonance_primes=3,
    )

    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)

    emb = MockEmbeddingEngine(dimension=64)
    emb.set_db(db)

    tags = TagManager(db)
    salience = SalienceScorer(config, db)
    mem_store = MemoryStore(config, db, emb, tags, salience)
    trend = TrendEngine(config, db)
    retrieval = RetrievalTracker(config, db)
    phase = PhaseSpace(config, db, trend, retrieval)
    sync_detector = SynchronicityDetector(
        config, db, phase, trend, retrieval, tags, emb,
    )
    sync_engine = SynchronicityEngine(config, db, emb, tags)
    adaptive = AdaptiveThresholds(config, db)
    meta = MetaAnalyzer(config, db, adaptive)

    return {
        "config": config, "db": db, "emb": emb, "tags": tags,
        "salience": salience, "mem_store": mem_store, "trend": trend,
        "retrieval": retrieval, "phase": phase,
        "sync_detector": sync_detector, "sync_engine": sync_engine,
        "adaptive": adaptive, "meta": meta,
    }


# ── Demo scenarios ───────────────────────────────────────────────────

MEMORIES = [
    # Domain: Mathematics
    ("Topology studies continuous deformations of shapes and spaces",
     ["mathematics", "topology", "geometry"]),
    ("Group theory captures symmetry through algebraic structures",
     ["mathematics", "algebra", "symmetry"]),
    ("The Riemann hypothesis connects prime distribution to complex analysis",
     ["mathematics", "primes", "analysis"]),

    # Domain: Cognitive Science
    ("Neural binding problem: how distributed processing creates unified experience",
     ["neuroscience", "consciousness", "binding"]),
    ("Embodied cognition argues that thought is shaped by the body",
     ["cognition", "embodiment", "philosophy"]),

    # Domain: Music Theory
    ("Harmonic series creates overtone relationships between frequencies",
     ["music", "harmonics", "frequency"]),
    ("Polyrhythm layers incommensurate time signatures",
     ["music", "rhythm", "pattern"]),

    # Cross-domain bridges (these should trigger synchronicity later)
    ("Symmetry groups describe both crystal structures and musical intervals",
     ["mathematics", "symmetry", "music", "crystallography"]),
    ("Neural oscillations synchronize at harmonic frequency ratios",
     ["neuroscience", "harmonics", "frequency", "synchronization"]),
    ("Prime number spirals reveal geometric structure in number theory",
     ["mathematics", "primes", "geometry", "pattern"]),
]


def demo_blank_slate(s: dict) -> None:
    """Show that everything starts empty."""
    banner("PHASE 0: Blank Slate")

    tables = s["db"].execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    print(f"\n  Schema created: {len(tables)} tables")
    for t in tables:
        count = s["db"].execute(
            f"SELECT COUNT(*) as cnt FROM [{t['name']}]"
        ).fetchone()["cnt"]
        if t["name"] in ("memories", "tags", "embeddings", "synchronicity_events",
                         "lattice_positions", "resonances", "tag_relational_tensor"):
            kv(t["name"], f"{count} rows")

    print(f"\n  PCA basis: None (will be random-scaffolded on first use)")
    print(f"  Lattice primes: {s['config'].lattice_primes}")
    print(f"  Tensor weights: co={s['config'].tensor_cooccurrence_weight}, "
          f"sync={s['config'].tensor_synchronicity_weight}, "
          f"sem={s['config'].tensor_semantic_weight}, "
          f"semio={s['config'].tensor_semiotic_weight}")
    print(f"\n  The network is pure configuration. No data, no seed, no model.")
    print(f"  Everything self-assembles through usage.")


def demo_store_memories(s: dict) -> list:
    """Store memories and show Layer 1 populating."""
    banner("PHASE 1: Store Memories (Layer 1)")

    stored = []
    for content, tag_names in MEMORIES:
        mem = s["mem_store"].store(
            content=content,
            tags=tag_names,
            salience_model=0.6,
        )
        stored.append(mem)

        # Record tag assignment events for trend tracking
        for tn in tag_names:
            tag = s["tags"].get_by_name(tn)
            if tag:
                s["trend"].record_event(tag.id, "assignment", memory_id=mem.id)

    section("Stored")
    for mem in stored:
        print(f"  [{mem.id[:8]}] {mem.content[:50]}...")
        print(f"           tags: {mem.tags}")

    section("Database state")
    for table in ("memories", "tags", "embeddings", "memory_tags"):
        count = s["db"].execute(
            f"SELECT COUNT(*) as cnt FROM {table}"
        ).fetchone()["cnt"]
        kv(table, f"{count} rows")

    active_tags = s["tags"].list_active_names()
    section(f"Active tags ({len(active_tags)})")
    # Show word tags only (skip single-letter tags)
    word_tags = [t for t in active_tags if len(t) > 1]
    print(f"  {', '.join(word_tags)}")

    return stored


def demo_seed_tensor(s: dict) -> None:
    """Seed the tensor — the bootstrap moment."""
    banner("PHASE 2: Seed Tag Relational Tensor (Layer 3.5)")

    print("\n  Calling seed_tensor_from_associations()...")
    print("  This populates three of four tensor networks from existing data:\n")

    n_co = s["sync_engine"].update_cooccurrence_tensor()
    print(f"  1. Co-occurrence (PMI):           {n_co} pairs")

    n_sem = s["sync_engine"].update_semantic_tensor()
    print(f"  2. Semantic (cosine similarity):   {n_sem} pairs")

    n_semio = s["sync_engine"].update_semiotic_tensor()
    print(f"  3. Semiotic (conditional prob):    {n_semio} pairs")

    print(f"  4. Synchronicity (lattice):        0 pairs (no events yet)")

    section("Tensor sample (top 5 by co-occurrence, word tags only)")
    rows = s["db"].execute(
        """SELECT t.tag_a_id, t.tag_b_id, t.cooccurrence_strength,
                  t.semantic_strength, t.semiotic_forward, t.semiotic_reverse
           FROM tag_relational_tensor t
           JOIN tags ta ON ta.id = t.tag_a_id
           JOIN tags tb ON tb.id = t.tag_b_id
           WHERE t.cooccurrence_strength > 0
             AND length(ta.name) > 1 AND length(tb.name) > 1
           ORDER BY t.cooccurrence_strength DESC LIMIT 5"""
    ).fetchall()

    for r in rows:
        tag_a = s["tags"].get_by_id(r["tag_a_id"]).name
        tag_b = s["tags"].get_by_id(r["tag_b_id"]).name
        print(f"  {tag_a:20s} <-> {tag_b:20s}  "
              f"PMI={r['cooccurrence_strength']:.2f}  "
              f"sem={r['semantic_strength']:.2f}")

    section("Semiotic asymmetry examples (word tags only)")
    rows = s["db"].execute(
        """SELECT t.tag_a_id, t.tag_b_id, t.semiotic_forward, t.semiotic_reverse
           FROM tag_relational_tensor t
           JOIN tags ta ON ta.id = t.tag_a_id
           JOIN tags tb ON tb.id = t.tag_b_id
           WHERE t.semiotic_forward > 0 AND t.semiotic_reverse > 0
             AND abs(t.semiotic_forward - t.semiotic_reverse) > 0.1
             AND length(ta.name) > 1 AND length(tb.name) > 1
           ORDER BY abs(t.semiotic_forward - t.semiotic_reverse) DESC LIMIT 5"""
    ).fetchall()

    for r in rows:
        tag_a = s["tags"].get_by_id(r["tag_a_id"]).name
        tag_b = s["tags"].get_by_id(r["tag_b_id"]).name
        print(f"  P({tag_b}|{tag_a}) = {r['semiotic_forward']:.2f}"
              f"    P({tag_a}|{tag_b}) = {r['semiotic_reverse']:.2f}"
              f"    {'<-- asymmetric' if abs(r['semiotic_forward'] - r['semiotic_reverse']) > 0.2 else ''}")


def demo_simulate_retrieval(s: dict, stored: list) -> None:
    """Simulate retrieval to trigger trend and sync detection."""
    banner("PHASE 3: Simulate Retrievals (Layer 2)")

    queries = [
        ("mathematical symmetry and group structures", ["mathematics", "symmetry"]),
        ("neural synchronization and binding", ["neuroscience", "synchronization"]),
        ("harmonic relationships in nature", ["harmonics", "frequency"]),
        ("patterns across mathematics and music", ["mathematics", "music", "pattern"]),
    ]

    for query_text, query_tags in queries:
        results = s["mem_store"].retrieve_by_tags(query_tags)
        if not results:
            continue

        rid = s["retrieval"].log_retrieval(
            query_text=query_text,
            method="tag",
            results=[(m.id, i + 1, 0.9) for i, m in enumerate(results)],
            model_version="demo",
        )

        # Record tag hits and trend events
        for mem in results:
            tag_ids = s["tags"].get_tag_ids_for_memory(mem.id)
            s["retrieval"].log_tag_hits(rid, [(tid, "direct_match") for tid in tag_ids])
            for tid in tag_ids:
                s["trend"].record_event(tid, "retrieval", memory_id=mem.id)

        section(f'Query: "{query_text}"')
        print(f"  Retrieved {len(results)} memories, logged {len(results)} trend events")

    section("Trend snapshot (top tags by activity)")
    active_tags = s["tags"].list_active()
    # Filter to word tags only
    word_tags = [t for t in active_tags if len(t.name) > 1]
    trends = []
    for tag in word_tags:
        trend = s["trend"].compute_trend(tag.id)
        if trend.event_count > 0:
            trends.append((tag, trend))

    trends.sort(key=lambda x: x[1].temperature, reverse=True)
    for tag, trend in trends[:10]:
        bar = "#" * int(trend.temperature * 20)
        print(f"  {tag.name:20s}  events={trend.event_count:2d}  "
              f"temp={trend.temperature:.2f} {bar}")


def demo_synchronicity_detection(s: dict) -> list[SynchronicityEvent]:
    """Create curated synchronicity events to demonstrate the lattice."""
    banner("PHASE 4: Synchronicity Events (Layer 3)")

    print("\n  In production, three detection modes fire after retrievals:")
    print("    1. Dormant reactivation  (low trend + high retrieval anomaly)")
    print("    2. Cross-domain bridges  (unrelated tags co-retrieved)")
    print("    3. Semantic convergence  (embedding similarity, no shared tags)")
    print()
    print("  For this demo, we inject curated events that represent the kind")
    print("  of cross-domain connections the detector surfaces:\n")

    events = _create_synthetic_events(s)
    return events


def _create_synthetic_events(s: dict) -> list[SynchronicityEvent]:
    """Create synthetic synchronicity events to demonstrate the lattice."""
    synth_events = [
        {
            "event_type": "cross_domain_bridge",
            "description": "Mathematics-symmetry and music-harmonics retrieved together "
                           "despite no prior co-occurrence: algebraic structures mirror "
                           "harmonic interval relationships",
            "strength": 3.2,
            "quadrant": "cross_domain",
            "tag_names": ["mathematics", "symmetry", "music", "harmonics"],
        },
        {
            "event_type": "unexpected_semantic_cluster",
            "description": "Topology-consciousness convergence: continuous deformation "
                           "concepts cluster with neural binding problem in embedding space",
            "strength": 0.82,
            "quadrant": "semantic_convergence",
            "tag_names": ["topology", "consciousness", "binding"],
        },
        {
            "event_type": "cross_domain_bridge",
            "description": "Prime-frequency bridge: Riemann hypothesis memory retrieved "
                           "alongside harmonic series memory. Number-theoretic structure "
                           "maps onto acoustic structure.",
            "strength": 4.1,
            "quadrant": "cross_domain",
            "tag_names": ["primes", "frequency", "pattern", "mathematics"],
        },
        {
            "event_type": "low_trend_high_retrieval",
            "description": "Crystallography tag dormant for extended period suddenly "
                           "retrieved in context of symmetry group discussion",
            "strength": 2.5,
            "quadrant": "dormant_reactivation",
            "tag_names": ["crystallography", "symmetry"],
        },
        {
            "event_type": "unexpected_semantic_cluster",
            "description": "Polyrhythm-algebra convergence: layered incommensurate "
                           "time signatures share embedding structure with group theory",
            "strength": 0.75,
            "quadrant": "semantic_convergence",
            "tag_names": ["rhythm", "algebra", "pattern"],
        },
    ]

    events = []
    for spec in synth_events:
        tag_ids = []
        for tn in spec["tag_names"]:
            tag = s["tags"].get_by_name(tn)
            if tag:
                tag_ids.append(tag.id)

        # Get associated memory IDs
        memory_ids = set()
        for tid in tag_ids:
            rows = s["db"].execute(
                "SELECT memory_id FROM memory_tags WHERE tag_id = ?", (tid,)
            ).fetchall()
            for r in rows:
                memory_ids.add(r["memory_id"])

        s["db"].execute(
            """INSERT INTO synchronicity_events
               (event_type, description, strength, quadrant,
                involved_tags, involved_memories, last_reinforced)
               VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%f', 'now'))""",
            (spec["event_type"], spec["description"], spec["strength"],
             spec["quadrant"], json.dumps(tag_ids),
             json.dumps(sorted(memory_ids))),
        )
        s["db"].connection.commit()
        event_id = s["db"].execute("SELECT last_insert_rowid()").fetchone()[0]

        ev = SynchronicityEvent(
            id=event_id,
            event_type=spec["event_type"],
            description=spec["description"],
            strength=spec["strength"],
            quadrant=spec["quadrant"],
            involved_tags=json.dumps(tag_ids),
            involved_memories=json.dumps(sorted(memory_ids)),
        )
        events.append(ev)

        print(f"  [{ev.event_type}] strength={ev.strength:.2f}")
        print(f"    tags: {', '.join(spec['tag_names'])}")

    return events


def demo_lattice_placement(s: dict, events: list[SynchronicityEvent]) -> None:
    """Place events on the Prime Ramsey Lattice and detect resonances."""
    banner("PHASE 5: Prime Ramsey Lattice (Layer 3.5)")

    print("\n  Placing synchronicity events on the lattice...")
    print(f"  Primes: {s['config'].lattice_primes}")
    print(f"  Min shared primes for resonance: {s['config'].lattice_min_resonance_primes}")
    print(f"  PCA basis: random scaffold (< 20 embeddings)\n")

    positions = s["sync_engine"].place_events_batch(events)

    section(f"Placed {len(positions)} events")
    for pos in positions:
        tag_ids = []
        ev_row = s["db"].execute(
            "SELECT involved_tags, event_type FROM synchronicity_events WHERE id = ?",
            (pos.sync_event_id,),
        ).fetchone()
        if ev_row:
            tag_ids = json.loads(ev_row["involved_tags"])

        tag_names = []
        for tid in tag_ids:
            try:
                tag_names.append(s["tags"].get_by_id(tid).name)
            except Exception:
                pass

        slots = json.loads(pos.prime_slots)
        slot_str = " ".join(f"p{p}:{v}" for p, v in sorted(slots.items(), key=lambda x: x[0]))
        print(f"  Event {pos.sync_event_id} | angle={pos.angle:.3f} rad | {ev_row['event_type']}")
        print(f"    tags: {', '.join(tag_names)}")
        print(f"    slots: {slot_str}")

    # Show resonances
    resonance_rows = s["db"].execute(
        """SELECT event_a_id, event_b_id, shared_primes, resonance_strength, description
           FROM resonances ORDER BY resonance_strength DESC"""
    ).fetchall()

    if resonance_rows:
        section(f"Resonances found: {len(resonance_rows)}")
        for r in resonance_rows:
            shared = json.loads(r["shared_primes"])
            print(f"  Events ({r['event_a_id']}, {r['event_b_id']}) "
                  f"share {len(shared)} prime scales {shared}")
            print(f"    strength={r['resonance_strength']:.2f} nats "
                  f"(chance: {2.718 ** -r['resonance_strength']:.2e})")
    else:
        section("No resonances (events landed in different prime slots)")
        print("  This is expected with diverse content. Resonance = structural")
        print("  entanglement, not semantic similarity.")


def demo_tensor_state(s: dict) -> None:
    """Show the full tensor state with all four networks."""
    banner("PHASE 6: Four-Network Tensor State")

    total = s["db"].execute(
        "SELECT COUNT(*) as cnt FROM tag_relational_tensor"
    ).fetchone()["cnt"]
    print(f"\n  Total tensor entries: {total}")

    section("Network coverage")
    for col, label in [
        ("cooccurrence_strength", "Co-occurrence (PMI)"),
        ("semantic_strength", "Semantic (cosine)"),
        ("semiotic_forward", "Semiotic (directional)"),
        ("synchronicity_strength", "Synchronicity (lattice)"),
    ]:
        count = s["db"].execute(
            f"SELECT COUNT(*) as cnt FROM tag_relational_tensor WHERE {col} > 0"
        ).fetchone()["cnt"]
        kv(label, f"{count} pairs with non-zero strength")

    section("Strongest cross-network connections (top 5, word tags only)")
    rows = s["db"].execute(
        """SELECT t.tag_a_id, t.tag_b_id,
                  t.cooccurrence_strength as co, t.synchronicity_strength as sync,
                  t.semantic_strength as sem, t.semiotic_forward as sf,
                  t.semiotic_reverse as sr
           FROM tag_relational_tensor t
           JOIN tags ta ON ta.id = t.tag_a_id
           JOIN tags tb ON tb.id = t.tag_b_id
           WHERE length(ta.name) > 1 AND length(tb.name) > 1
           ORDER BY (t.cooccurrence_strength + t.synchronicity_strength
                     + t.semantic_strength + t.semiotic_forward) DESC
           LIMIT 5"""
    ).fetchall()

    for r in rows:
        ta = s["tags"].get_by_id(r["tag_a_id"]).name
        tb = s["tags"].get_by_id(r["tag_b_id"]).name
        networks = []
        if r["co"] > 0:
            networks.append(f"co={r['co']:.2f}")
        if r["sem"] > 0:
            networks.append(f"sem={r['sem']:.2f}")
        if r["sf"] > 0 or r["sr"] > 0:
            networks.append(f"semio={r['sf']:.2f}/{r['sr']:.2f}")
        if r["sync"] > 0:
            networks.append(f"sync={r['sync']:.2f}")
        print(f"  {ta:20s} <-> {tb:20s}  {', '.join(networks)}")


def demo_fast_recall(s: dict) -> None:
    """Show tensor-based O(k) recall."""
    banner("PHASE 7: Tensor-Powered Recall")

    print("\n  get_resonant_memory_ids_fast() uses the four-network tensor")
    print("  for O(k) lookup instead of O(n^2) pairwise comparison.\n")

    query_tags = ["mathematics", "symmetry"]
    tag_ids = []
    for tn in query_tags:
        tag = s["tags"].get_by_name(tn)
        if tag:
            tag_ids.append(tag.id)

    results = s["sync_engine"].get_resonant_memory_ids_fast(tag_ids)

    section(f'Query tags: {query_tags}')
    if results:
        print(f"  Retrieved {len(results)} resonant memories:\n")
        for mid, score in results[:8]:
            row = s["db"].execute(
                "SELECT content FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            content = row["content"][:60] if row else "?"
            tags = s["tags"].get_tags_for_memory(mid)
            word_tags = [t for t in tags if len(t) > 1]
            print(f"  score={score:.3f}  {content}...")
            print(f"           tags: {', '.join(word_tags)}")
    else:
        print("  No resonant memories found for these tags.")


def demo_phase_space(s: dict) -> None:
    """Show phase space coordinates."""
    banner("PHASE 8: Phase Space (Layer 3)")

    coords = s["phase"].compute_all_coordinates()

    section(f"Tag phase coordinates ({len(coords)} tags)")
    entries = []
    for tag_id, coord in coords.items():
        try:
            tag = s["tags"].get_by_id(tag_id)
            if len(tag.name) > 1:  # Skip letter tags
                entries.append((tag, coord))
        except Exception:
            pass

    entries.sort(key=lambda x: x[1].temperature, reverse=True)
    for tag, coord in entries[:12]:
        print(f"  {tag.name:20s}  temp={coord.temperature:.2f}  "
              f"retrieval={coord.retrieval_freq:.2f}  "
              f"quadrant={coord.quadrant.value}")


def demo_void_profile(s: dict) -> None:
    """Show the lattice void — what all synchronicities orbit."""
    banner("PHASE 9: Lattice Void Profile")

    void = s["sync_engine"].compute_void_profile()
    if void:
        edge_tags = json.loads(void.edge_tags)
        print(f"\n  {void.description}")
        print(f"\n  Edge tags (the themes that circle but never name the void):")
        for t in edge_tags[:8]:
            print(f"    - {t}")
    else:
        print("\n  Not enough lattice positions for void profile (need >= 3)")


def demo_modularity(s: dict) -> None:
    """Summarize the modular architecture."""
    banner("ARCHITECTURE: Modular Layer Summary")

    print("""
  Each layer is an independent module with clean interfaces:

  Layer 1 — Memory Store
    Input:  content + tags
    Output: Memory objects with embeddings and salience scores
    Deps:   DatabaseEngine, EmbeddingEngine, TagManager, SalienceScorer

  Layer 2 — Trends & Retrieval
    Input:  tag events, retrieval logs
    Output: trend snapshots (level, velocity, temperature)
    Deps:   DatabaseEngine, config

  Layer 3 — Phase Space
    Input:  trend data, retrieval frequency
    Output: (temperature, retrieval_freq) coordinates per tag
    Deps:   TrendEngine, RetrievalTracker

  Layer 3 — Synchronicity Detection
    Input:  phase coordinates, retrieval history, embeddings
    Output: SynchronicityEvent objects
    Deps:   PhaseSpace, TrendEngine, RetrievalTracker, TagManager, EmbeddingEngine

  Layer 3.5 — Prime Ramsey Lattice
    Input:  SynchronicityEvent objects
    Output: lattice positions, resonances
    Deps:   EmbeddingEngine (for angle computation), TagManager

  Layer 3.5 — Tag Relational Tensor (4 networks)
    Input:  memory_tags, embeddings, lattice positions
    Output: weighted (memory_id, score) pairs for recall
    Networks:
      - Co-occurrence:  PMI (symmetric)
      - Semantic:       cosine similarity (symmetric)
      - Semiotic:       P(B|A), P(A|B) (asymmetric)
      - Synchronicity:  lattice resonance strength (symmetric)

  Layer 4 — Meta-Patterns
    Input:  synchronicity events, adaptive thresholds
    Output: meta-pattern detections, threshold adjustments
    Deps:   DatabaseEngine, AdaptiveThresholds

  Cold Start Behavior:
    - t=0: All tables empty. PCA basis = random scaffold.
    - Tensor self-seeds from first batch of memories.
    - Synchronicity detection triggers on retrievals.
    - Lattice grows organically. No external training data needed.
    - Each instance has a unique random geometric fingerprint.
""")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + " " * 8 + "CHICORY RAMSEY NETWORK DEMO")
    print(" " * 8 + "From blank slate to self-organizing memory")

    s = build_stack()

    try:
        # Phase 0: Show empty state
        demo_blank_slate(s)

        # Phase 1: Store memories (Layer 1)
        stored = demo_store_memories(s)

        # Phase 2: Seed the tensor (Layer 3.5)
        demo_seed_tensor(s)

        # Phase 3: Simulate retrievals (Layer 2)
        demo_simulate_retrieval(s, stored)

        # Phase 4: Synchronicity detection (Layer 3)
        events = demo_synchronicity_detection(s)

        # Phase 5: Lattice placement (Layer 3.5)
        demo_lattice_placement(s, events)

        # Phase 6: Full tensor state
        demo_tensor_state(s)

        # Phase 7: Fast recall through tensor
        demo_fast_recall(s)

        # Phase 8: Phase space
        demo_phase_space(s)

        # Phase 9: Void profile
        demo_void_profile(s)

        # Architecture summary
        demo_modularity(s)

    finally:
        s["db"].close()


if __name__ == "__main__":
    main()
