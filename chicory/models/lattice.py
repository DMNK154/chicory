"""Lattice position and resonance models for the prime Ramsey synchronicity engine."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class LatticePosition(BaseModel):
    """Angular position of a synchronicity event on the prime Ramsey lattice."""

    id: Optional[int] = None
    sync_event_id: int
    angle: float  # theta in [0, 2*pi)
    prime_slots: str  # JSON dict mapping prime (int) -> slot index (int)
    placed_at: Optional[datetime] = None


class Resonance(BaseModel):
    """A group of events sharing slots across multiple primes."""

    id: Optional[int] = None
    event_ids: str  # JSON array of synchronicity_event IDs
    shared_primes: str  # JSON array of prime numbers where slots match
    resonance_strength: float  # -log(product(1/p for shared primes))
    description: str
    detected_at: Optional[datetime] = None


class VoidProfile(BaseModel):
    """Characterization of the central void in the prime lattice."""

    edge_tags: str  # JSON array of tag name strings
    edge_angles: str  # JSON array of float angles
    void_radius: float  # Angular radius of the inner ring
    description: str


class TagResonanceEntry(BaseModel):
    """A single entry in the tag relational tensor R(tag_a, tag_b).

    Three overlapping Ramsey networks on the same tag vertex set,
    each capturing a different type of relational strength.
    """

    tag_a_id: int
    tag_b_id: int
    cooccurrence_strength: float = 0.0   # PMI from memory_tags
    synchronicity_strength: float = 0.0  # Lattice resonance + sync event co-involvement
    semantic_strength: float = 0.0       # Embedding cosine similarity between tag centroids
    memory_ids: str = "[]"               # JSON array of reachable memory IDs
    updated_at: Optional[datetime] = None
