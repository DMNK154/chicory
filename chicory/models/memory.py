"""Memory, Tag, and TagEvent models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Memory(BaseModel):
    id: str
    content: str
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    source_model: str
    salience_model: float = 0.5
    salience_usage: float = 0.0
    salience_composite: float = 0.5
    retrieval_success_count: int = 0
    retrieval_total_count: int = 0
    is_archived: bool = False
    tags: list[str] = Field(default_factory=list)


class Tag(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    created_by: str = "system"
    is_active: bool = True
    parent_id: Optional[int] = None
    merged_into: Optional[int] = None


class TagEvent(BaseModel):
    id: int
    tag_id: int
    event_type: str  # 'assignment', 'retrieval', 'decay', 'promotion', 'demotion'
    occurred_at: datetime
    memory_id: Optional[str] = None
    weight: float = 1.0
    metadata: Optional[str] = None
