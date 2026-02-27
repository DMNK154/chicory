"""Data access layer for the Chicory dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chicory.config import load_config
from chicory.orchestrator.orchestrator import Orchestrator

_orchestrator: Orchestrator | None = None
_current_db_path: str | None = None


def _get_orchestrator(db_path: Path) -> Orchestrator:
    """Return a cached Orchestrator, creating one if needed."""
    global _orchestrator, _current_db_path
    db_str = str(db_path)
    if _orchestrator is None or _current_db_path != db_str:
        if _orchestrator is not None:
            _orchestrator.close()
        config = load_config(db_path=db_path)
        _orchestrator = Orchestrator(config)
        _current_db_path = db_str
    return _orchestrator


def get_overview(db_path: Path) -> dict[str, Any]:
    """Summary stats and current tag trends."""
    o = _get_orchestrator(db_path)
    trends = o.handle_get_trends()["trends"]
    return {
        "memory_count": o._memory_store.count(),
        "tag_names": o._tag_manager.list_active_names(),
        "tag_count": len(o._tag_manager.list_active()),
        "sync_count": len(o._sync_detector.get_recent(100)),
        "meta_count": len(o._meta_analyzer.get_active_patterns()),
        "trends": trends,
    }


def get_phase_space(db_path: Path) -> dict[str, Any]:
    """Phase space quadrant populations."""
    return _get_orchestrator(db_path).handle_get_phase_space()


def get_trend_history(
    db_path: Path, tag_name: str, periods: int = 50
) -> list[dict[str, Any]]:
    """Historical trend snapshots for a single tag."""
    o = _get_orchestrator(db_path)
    tag = o._tag_manager.get_by_name(tag_name)
    if not tag:
        return []
    snapshots = o._trend_engine.get_trend_history(tag.id, periods=periods)
    return [
        {
            "computed_at": s.computed_at.isoformat(),
            "temperature": s.temperature,
            "level": s.level,
            "velocity": s.velocity,
            "jerk": s.jerk,
            "event_count": s.event_count,
        }
        for s in reversed(snapshots)
    ]


def get_synchronicities(
    db_path: Path, limit: int = 100
) -> dict[str, Any]:
    """Synchronicity events with velocity."""
    return _get_orchestrator(db_path).handle_get_synchronicities(limit=limit)


def get_lattice(db_path: Path) -> dict[str, Any]:
    """Lattice positions, resonances, and void profile."""
    return _get_orchestrator(db_path).handle_get_lattice_resonances()


def get_meta_patterns(db_path: Path) -> dict[str, Any]:
    """Active meta-patterns."""
    return _get_orchestrator(db_path).handle_get_meta_patterns()


def get_network_data(db_path: Path) -> dict[str, Any]:
    """Tag co-occurrence network: nodes with metrics, edges with weights."""
    o = _get_orchestrator(db_path)

    # Node data from phase space (temperature, retrieval_freq, quadrant)
    ps = o.handle_get_phase_space()["phase_space"]
    # Trend data for event counts
    trends = {t["tag"]: t for t in o.handle_get_trends()["trends"]}

    tag_lookup: dict[str, dict[str, Any]] = {}  # tag_name -> node dict
    id_to_name: dict[int, str] = {}  # tag_id -> tag_name

    for quadrant, items in ps.items():
        for item in items:
            name = item["tag"]
            trend = trends.get(name, {})
            tag_obj = o._tag_manager.get_by_name(name)
            tag_id = tag_obj.id if tag_obj else -1
            tag_lookup[name] = {
                "tag": name,
                "tag_id": tag_id,
                "temperature": item["temperature"],
                "retrieval_freq": item["retrieval_freq"],
                "quadrant": quadrant,
                "event_count": trend.get("event_count", 0),
            }
            if tag_id >= 0:
                id_to_name[tag_id] = name

    # Edge data from co-occurrence
    co_occurrences = o._tag_manager.get_all_co_occurrences(min_count=1)
    edges = []
    for tag_a_id, tag_b_id, count in co_occurrences:
        if tag_a_id in id_to_name and tag_b_id in id_to_name:
            edges.append({
                "source": id_to_name[tag_a_id],
                "target": id_to_name[tag_b_id],
                "weight": count,
            })

    return {"nodes": list(tag_lookup.values()), "edges": edges}


def close() -> None:
    """Close the cached orchestrator."""
    global _orchestrator, _current_db_path
    if _orchestrator is not None:
        _orchestrator.close()
        _orchestrator = None
        _current_db_path = None
