"""Retrieval event models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class RetrievalEvent(BaseModel):
    id: int
    query_text: str
    context_summary: Optional[str] = None
    method: str  # 'semantic', 'tag', 'hybrid', 'direct'
    occurred_at: datetime
    result_count: int = 0
    model_version: str


class RetrievalResult(BaseModel):
    retrieval_id: int
    memory_id: str
    rank: int
    relevance_score: float
    was_useful: Optional[bool] = None
