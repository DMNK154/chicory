"""Canopy layer models: blocks, cross-layer inhibition edges, support edges, observations, score bundles."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ScoreBundle(BaseModel):
    """Base/lifted scores for a canopy shape or block."""

    bridge: float = 0.0
    heat: float = 0.0
    recurrence: float = 0.0
    cooccurrence: float = 0.0
    similarity: float = 0.0
    relevance: float = 0.0
    semantics: float = 0.0
    pressure: float = 0.0
    threshold: float = 0.0
    growth_potential: float = 0.0
    canopy_growth: float = 0.0


class CanopyBlock(BaseModel):
    id: Optional[int] = None
    block_key: str
    block_type: str  # memory_trace, tag_set, tag_pair, sync_shape, lattice_shape, meta_shape, canopy_pair, canopy_shape
    layer_depth: int = 0
    tag_ids: str = "[]"
    memory_ids: str = "[]"
    parent_block_keys: str = "[]"
    source_event_types: str = "[]"
    peak_bridge: float = 0.0
    peak_heat: float = 0.0
    peak_recurrence: float = 0.0
    peak_cooccurrence: float = 0.0
    peak_similarity: float = 0.0
    peak_relevance: float = 0.0
    peak_semantics: float = 0.0
    peak_pressure: float = 0.0
    peak_threshold: float = 0.0
    peak_growth_potential: float = 0.0
    peak_canopy_growth: float = 0.0
    evidence_count: int = 0
    first_growth_at: Optional[datetime] = None
    canonical_block_key: Optional[str] = None
    created_at: Optional[datetime] = None
    last_observed_at: Optional[datetime] = None


class CrossLayerEdge(BaseModel):
    relevance_block_id: int
    semantic_block_id: int
    edge_inhibition: float = 0.0
    a_heat: float = 0.0
    a_similarity: float = 0.0
    a_cooccurrence: float = 0.0
    a_recurrence: float = 0.0
    a_bridge: float = 0.0
    edge_heat: float = 0.0
    edge_similarity: float = 0.0
    edge_cooccurrence: float = 0.0
    edge_recurrence: float = 0.0
    edge_bridge: float = 0.0
    evidence_count: int = 0
    created_at: Optional[datetime] = None
    last_observed_at: Optional[datetime] = None


class SupportEdge(BaseModel):
    canopy_block_id: int
    target_type: str  # memory, tag, sync_event, meta_pattern, lattice_position, resonance, canopy_block
    target_id: str
    edge_type: str  # supports, contains, bridges, opposes, stabilizes, derived_from, canonicalizes_to
    strength: float = 0.0
    evidence_count: int = 0
    created_at: Optional[datetime] = None
    last_observed_at: Optional[datetime] = None


class CanopyObservation(BaseModel):
    id: Optional[int] = None
    block_key: str
    observed_at: Optional[datetime] = None
    source: str  # store, retrieval, tensor, centroid, synchronicity, lattice, meta_pattern, feedback, recursive_canopy
    source_id: Optional[str] = None
    layer_depth: int = 0
    tag_ids: str = "[]"
    memory_ids: str = "[]"
    source_canopy_block_ids: str = "[]"
    bridge: float = 0.0
    heat: float = 0.0
    recurrence: float = 0.0
    cooccurrence: float = 0.0
    similarity: float = 0.0
    relevance: float = 0.0
    semantics: float = 0.0
    pressure: float = 0.0
    threshold: float = 0.0
    growth_potential: float = 0.0
    canopy_growth: float = 0.0


class CanopyShape(BaseModel):
    """A generated shape from the current process scope, ready for scoring."""

    block_key: str
    block_type: str
    layer_depth: int = 0
    tag_ids: list[int]
    memory_ids: list[str]
    parent_block_keys: list[str]
    source: str
    source_id: Optional[str] = None
    source_event_types: list[str]


class InflowScore(BaseModel):
    """Convergence attractor score for an inflow canopy block."""

    block_key: str = ""
    inflow_diversity: float = 0.0
    unique_query_contexts: int = 0
    total_activations: int = 0
    inflow_strength: float = 0.0


class OutflowScore(BaseModel):
    """Distributor score for an outflow canopy block."""

    outflow_diversity: float = 0.0
    outflow_reach: float = 0.0
    unique_result_clusters: int = 0
    total_activations: int = 0
    outflow_strength: float = 0.0
