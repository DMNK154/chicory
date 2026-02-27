"""Adaptive thresholds with EMA updates and burn-in mode."""

from __future__ import annotations

from datetime import datetime, timedelta

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine

_DEFAULTS = {
    "sync_detection_sigma": 2.0,
    "cross_domain_surprise": 3.0,
    "semantic_convergence": 0.7,
    "meta_base_rate_multiplier": 3.0,
}


class AdaptiveThresholds:
    """Manages per-metric adaptive thresholds."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    def get_threshold(self, metric_name: str) -> float:
        """Get current threshold, respecting burn-in mode."""
        row = self._db.execute(
            "SELECT current_value, baseline_value, burn_in_until FROM adaptive_thresholds WHERE metric_name = ?",
            (metric_name,),
        ).fetchone()

        if not row:
            return _DEFAULTS.get(metric_name, 2.0)

        current = row["current_value"]
        baseline = row["baseline_value"]
        burn_in_until = row["burn_in_until"]

        if burn_in_until:
            burn_in_end = datetime.fromisoformat(burn_in_until)
            if datetime.utcnow() < burn_in_end:
                return max(current, baseline) * self._config.burn_in_threshold_multiplier

        return current

    def update_threshold(self, metric_name: str, observed_value: float) -> None:
        """Update threshold using exponential moving average."""
        row = self._db.execute(
            "SELECT current_value, sample_count FROM adaptive_thresholds WHERE metric_name = ?",
            (metric_name,),
        ).fetchone()

        now = datetime.utcnow().isoformat()

        if not row:
            self._db.execute(
                """
                INSERT INTO adaptive_thresholds
                    (metric_name, current_value, baseline_value, sample_count, model_version)
                VALUES (?, ?, ?, 1, ?)
                """,
                (metric_name, observed_value, observed_value, self._config.llm_model),
            )
            self._db.connection.commit()
            return

        current = row["current_value"]
        count = row["sample_count"]
        alpha = 0.1  # EMA smoothing factor
        new_value = alpha * observed_value + (1 - alpha) * current

        self._db.execute(
            """
            UPDATE adaptive_thresholds
            SET current_value = ?, sample_count = ?, last_updated = ?
            WHERE metric_name = ?
            """,
            (new_value, count + 1, now, metric_name),
        )
        self._db.connection.commit()

    def enter_burn_in(self, model_version: str) -> None:
        """Put all thresholds into burn-in mode after a model swap."""
        burn_until = (
            datetime.utcnow() + timedelta(hours=self._config.burn_in_hours)
        ).isoformat()

        # Save current values as baselines, set burn-in deadline
        self._db.execute(
            """
            UPDATE adaptive_thresholds
            SET baseline_value = current_value,
                burn_in_until = ?,
                model_version = ?
            """,
            (burn_until, model_version),
        )
        self._db.connection.commit()

    def is_in_burn_in(self, metric_name: str) -> bool:
        """Check if a metric is in burn-in mode."""
        row = self._db.execute(
            "SELECT burn_in_until FROM adaptive_thresholds WHERE metric_name = ?",
            (metric_name,),
        ).fetchone()

        if not row or not row["burn_in_until"]:
            return False

        return datetime.utcnow() < datetime.fromisoformat(row["burn_in_until"])

    def reset_baselines(self) -> None:
        """Clear burn-in and reset baselines to current values."""
        self._db.execute(
            """
            UPDATE adaptive_thresholds
            SET baseline_value = current_value, burn_in_until = NULL
            """
        )
        self._db.connection.commit()
