"""Tests for the front-desk A/B bench helpers."""

from __future__ import annotations

from chicory.bench.frontdesk_ab import _consult_result_ids


def test_consult_ids_interleave_channels():
    consult = {
        "faq": None,
        "subjects": [
            {"results": [{"memory_id": "s1a"}, {"memory_id": "s1b"}]},
            {"results": [{"memory_id": "s2a"}]},
        ],
        "general": [{"memory_id": "g1"}, {"memory_id": "g2"}],
    }
    ids = _consult_result_ids(consult)
    # Round-robin: rank 0 of each channel first, so a top-k slice
    # represents every channel instead of shelf-1 crowding out general
    assert ids[:3] == ["s1a", "s2a", "g1"]
    assert set(ids) == {"s1a", "s1b", "s2a", "g1", "g2"}


def test_consult_ids_faq_first_and_dedupe():
    consult = {
        "faq": {"supporting_memory_ids": ["f1", "g1"]},
        "subjects": [],
        "general": [{"memory_id": "g1"}, {"memory_id": "g2"}],
    }
    ids = _consult_result_ids(consult)
    assert ids[0] == "f1"
    assert ids.count("g1") == 1
