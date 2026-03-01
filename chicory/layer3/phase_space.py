"""Phase space computation and quadrant classification."""

from __future__ import annotations

import math

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer2.retrieval_tracker import RetrievalTracker
from chicory.layer2.trend_engine import TrendEngine
from chicory.models.phase import PhaseCoordinate, Quadrant


class PhaseSpace:
    """Computes and classifies tag positions in the trend-retrieval phase space."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        trend_engine: TrendEngine,
        retrieval_tracker: RetrievalTracker,
    ) -> None:
        self._config = config
        self._db = db
        self._trends = trend_engine
        self._retrieval = retrieval_tracker

    def compute_coordinate(self, tag_id: int) -> PhaseCoordinate:
        """Compute a single tag's phase space coordinate."""
        trend = self._trends.compute_trend(tag_id)
        retrieval_freq = self._retrieval.get_normalized_frequency(tag_id)

        tag_row = self._db.execute(
            "SELECT name FROM tags WHERE id = ?", (tag_id,)
        ).fetchone()
        tag_name = tag_row["name"] if tag_row else f"tag-{tag_id}"

        temperature = trend.temperature
        quadrant = self._classify(temperature, retrieval_freq)

        # Off-diagonal distance: signed, positive when r > t (dormant reactivation side)
        off_diag = (retrieval_freq - temperature) / math.sqrt(2)

        return PhaseCoordinate(
            tag_id=tag_id,
            tag_name=tag_name,
            temperature=temperature,
            retrieval_freq=retrieval_freq,
            quadrant=quadrant,
            off_diagonal_distance=off_diag,
        )

    def compute_all_coordinates(self) -> dict[int, PhaseCoordinate]:
        """Compute phase coordinates for all active tags.

        Batches trend computation (single query) and retrieval frequencies
        (single base rate + single grouped frequency query) instead of
        O(3t) per-tag queries.
        """
        # Batch trends: single query via compute_all_trends
        all_trends = self._trends.compute_all_trends()
        if not all_trends:
            return {}

        # Batch retrieval frequencies: one base rate + one grouped query
        all_freqs = self._retrieval.get_all_normalized_frequencies(
            list(all_trends.keys()),
        )

        # Batch tag names
        tag_ids = list(all_trends.keys())
        placeholders = ",".join("?" * len(tag_ids))
        name_rows = self._db.execute(
            f"SELECT id, name FROM tags WHERE id IN ({placeholders})",
            tuple(tag_ids),
        ).fetchall()
        tag_names = {r["id"]: r["name"] for r in name_rows}

        result: dict[int, PhaseCoordinate] = {}
        for tag_id, trend in all_trends.items():
            temperature = trend.temperature
            retrieval_freq = all_freqs.get(tag_id, 0.0)
            quadrant = self._classify(temperature, retrieval_freq)
            off_diag = (retrieval_freq - temperature) / math.sqrt(2)
            tag_name = tag_names.get(tag_id, f"tag-{tag_id}")

            result[tag_id] = PhaseCoordinate(
                tag_id=tag_id,
                tag_name=tag_name,
                temperature=temperature,
                retrieval_freq=retrieval_freq,
                quadrant=quadrant,
                off_diagonal_distance=off_diag,
            )

        return result

    def get_quadrant_populations(self) -> dict[Quadrant, list[PhaseCoordinate]]:
        """Group all tags by their quadrant."""
        coords = self.compute_all_coordinates()
        populations: dict[Quadrant, list[PhaseCoordinate]] = {
            q: [] for q in Quadrant
        }
        for coord in coords.values():
            populations[coord.quadrant].append(coord)
        return populations

    def get_off_diagonal_tags(
        self, min_distance: float = 0.1
    ) -> list[PhaseCoordinate]:
        """Get tags with significant off-diagonal distance."""
        coords = self.compute_all_coordinates()
        return [
            c for c in coords.values()
            if abs(c.off_diagonal_distance) >= min_distance
        ]

    def _classify(self, temperature: float, retrieval_freq: float) -> Quadrant:
        """Classify a (temperature, retrieval_freq) point into a quadrant."""
        t_thresh = self._config.phase_temperature_threshold
        r_thresh = self._config.phase_retrieval_threshold

        if temperature >= t_thresh:
            if retrieval_freq >= r_thresh:
                return Quadrant.ACTIVE_DEEP_WORK
            return Quadrant.NOVEL_EXPLORATION
        else:
            if retrieval_freq >= r_thresh:
                return Quadrant.DORMANT_REACTIVATION
            return Quadrant.INACTIVE
