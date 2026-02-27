"""Synchronicity event models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SynchronicityEvent(BaseModel):
    id: Optional[int] = None
    detected_at: Optional[datetime] = None
    event_type: str  # 'low_trend_high_retrieval', 'cross_domain_bridge', 'unexpected_semantic_cluster'
    description: str
    strength: float  # How anomalous (z-score or log-surprise)
    quadrant: str
    involved_tags: str  # JSON array of tag IDs
    involved_memories: Optional[str] = None  # JSON array of memory IDs
    trigger_retrieval_id: Optional[int] = None
    acknowledged: bool = False
    last_reinforced: Optional[datetime] = None
    reinforcement_count: int = 0
