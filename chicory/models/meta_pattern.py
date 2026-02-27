"""Meta-pattern models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MetaPattern(BaseModel):
    id: Optional[int] = None
    detected_at: Optional[datetime] = None
    description: str
    pattern_type: str  # 'recurring_sync', 'cross_domain_theme', 'emergent_category'
    confidence: float  # [0, 1]
    involved_sync_ids: str  # JSON array of synchronicity_event IDs
    involved_tag_clusters: str  # JSON array of arrays of tag IDs
    actions_taken: Optional[str] = None  # JSON describing feedback actions
    is_active: bool = True
    validated_by: Optional[str] = None  # 'system', 'llm', 'user'
