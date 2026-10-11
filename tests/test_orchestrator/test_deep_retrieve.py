"""deep_retrieve end to end, through a real Orchestrator.

It used to raise NameError the moment level 0 found anything: the actor
was resolved in handle_deep_retrieve but never handed to the locked body
that logs the retrievals.
"""

from __future__ import annotations


def _store_corpus(orch):
    for i in range(6):
        orch.handle_store_memory(
            content=f"python async bug number {i} in the event loop",
            tags=["python", "async", "bug"],
        )


def test_deep_retrieve_returns_results(orchestrator):
    _store_corpus(orchestrator)

    out = orchestrator.handle_deep_retrieve("python async bug")

    assert out["method"] == "deep_retrieve"
    assert out["count"] > 0
    assert all("depth" in r for r in out["results"])


def test_deep_retrieve_logs_the_resolved_actor(orchestrator):
    _store_corpus(orchestrator)

    orchestrator.handle_deep_retrieve("python async bug", actor="research")
    orchestrator.handle_deep_retrieve("python async bug")

    rows = orchestrator._db.execute(
        "SELECT actor FROM retrieval_events WHERE method = 'deep_retrieve' "
        "ORDER BY id"
    ).fetchall()
    default = orchestrator._config.retrieval_default_actor
    assert [r["actor"] for r in rows] == ["research", default]
