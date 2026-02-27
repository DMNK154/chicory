"""Phase space models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Quadrant(str, Enum):
    ACTIVE_DEEP_WORK = "active_deep_work"  # High temp, high retrieval
    NOVEL_EXPLORATION = "novel_exploration"  # High temp, low retrieval
    DORMANT_REACTIVATION = "dormant_reactivation"  # Low temp, high retrieval
    INACTIVE = "inactive"  # Low temp, low retrieval


class PhaseCoordinate(BaseModel):
    """A tag's position in the trend-temperature vs retrieval-frequency plane."""

    tag_id: int
    tag_name: str
    temperature: float  # x-axis [0, 1]
    retrieval_freq: float  # y-axis [0, 1] (normalized)
    quadrant: Quadrant
    off_diagonal_distance: float  # |r - t| / sqrt(2), signed
