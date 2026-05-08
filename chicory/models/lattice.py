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
    poincare_x: Optional[float] = None  # Poincaré disk x coordinate
    poincare_y: Optional[float] = None  # Poincaré disk y coordinate
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


class GlyphPosition(BaseModel):
    """Position of a word tag on the glyph Ramsey lattice."""

    tag_id: int
    tag_name: str = ""
    angle: float  # theta in [0, 2*pi)
    prime_slots: str  # JSON dict mapping prime (int) -> slot index (int)
    glyph_vector: bytes  # float32 blob — 26-dim letter freq or 1472-dim ByT5
    glyph_dimension: int = 26  # vector dimensionality
    placed_at: Optional[datetime] = None


class TagResonanceEntry(BaseModel):
    """A single entry in the tag relational tensor R(tag_a, tag_b).

    Seven overlapping Ramsey networks on the same tag vertex set,
    each capturing a different type of relational strength.
    """

    tag_a_id: int
    tag_b_id: int
    cooccurrence_strength: float = 0.0   # PMI from memory_tags
    synchronicity_strength: float = 0.0  # Lattice resonance + sync event co-involvement
    semantic_strength: float = 0.0       # Embedding cosine similarity between tag centroids
    semiotic_forward: float = 0.0        # Directional glyph transformation P(B|A)
    semiotic_reverse: float = 0.0        # Directional glyph transformation P(A|B)
    glyph_strength: float = 0.0          # Glyph Ramsey lattice resonance
    inhibition_strength: float = 0.0     # Opposition/contrast suppression strength
    parallelness: float = 0.0            # cos(angular_diff): +1 parallel, -1 antiparallel
    memory_ids: str = "[]"               # JSON array of reachable memory IDs
    updated_at: Optional[datetime] = None
