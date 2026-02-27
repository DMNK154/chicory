"""Trend computation models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class TrendVector(BaseModel):
    """Computed trend for a single tag at a point in time."""

    tag_id: int
    level: float  # Zeroth derivative — absolute activity
    velocity: float  # First derivative — acceleration
    jerk: float  # Second derivative — change in acceleration
    temperature: float  # Composite [0, 1]
    event_count: int


class TrendSnapshot(BaseModel):
    """Persisted trend snapshot."""

    id: int
    tag_id: int
    computed_at: datetime
    window_hours: float
    level: float
    velocity: float
    jerk: float
    temperature: float
    event_count: int
