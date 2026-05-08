"""Poincaré disk projection for the prime Ramsey lattice.

Maps Ramsey lattice positions into the Poincaré disk model of hyperbolic
space (H²).  Angular coordinates come from PCA projection of embedding
centroids; radial depth encodes hierarchical specificity derived from
prime slot population density.

Key operations:
  - Exponential map: Euclidean tangent vector → Poincaré disk point
  - Geodesic distance: arccosh-based metric (single and batch)
  - Einstein midpoint: weighted hyperbolic centroid
  - Ramsey depth adjustment: radial shift based on prime slot density
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class PoincareProjection:
    """Projects points into the Poincaré disk and computes hyperbolic geometry.

    Curvature is stored as a positive value c; the actual sectional curvature
    is -c.  All formulas use the convention from Nickel & Kiela (2017) and
    Ganea et al. (2018).
    """

    def __init__(self, curvature: float = 1.0, max_radius: float = 0.95) -> None:
        self._c = abs(curvature)
        self._max_radius = max_radius

    @property
    def curvature(self) -> float:
        return self._c

    # ── Exponential / logarithmic maps ──────────────────────────────

    def exp_map_origin(self, v: np.ndarray) -> np.ndarray:
        """Exponential map at the origin: tangent vector → Poincaré disk.

        exp_0(v) = tanh(√c · ‖v‖) · v / (√c · ‖v‖)

        Uses the Ganea et al. (2018) convention where exp and log at the
        origin are exact inverses without scaling factors.
        """
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            return np.zeros_like(v, dtype=np.float32)

        sqrt_c = math.sqrt(self._c)
        coeff = math.tanh(sqrt_c * norm) / (sqrt_c * norm)
        result = coeff * v
        return self._clamp(result)

    def log_map_origin(self, y: np.ndarray) -> np.ndarray:
        """Logarithmic map at the origin: Poincaré disk → tangent vector.

        log_0(y) = arctanh(√c · ‖y‖) · y / (√c · ‖y‖)
        """
        norm = float(np.linalg.norm(y))
        if norm < 1e-12:
            return np.zeros_like(y, dtype=np.float32)

        sqrt_c = math.sqrt(self._c)
        coeff = math.atanh(min(sqrt_c * norm, 1.0 - 1e-7)) / (sqrt_c * norm)
        return (coeff * y).astype(np.float32)

    # ── Distance ────────────────────────────────────────────────────

    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Geodesic distance between two points in the Poincaré disk.

        d(u,v) = (2/√c) · arctanh(√c · ‖-u ⊕ v‖)

        Equivalent to the arccosh form:
        d(u,v) = (1/√c) · arccosh(1 + 2c·‖u-v‖² / ((1-c‖u‖²)(1-c‖v‖²)))
        """
        c = self._c
        diff_sq = float(np.sum((u - v) ** 2))
        u_sq = float(np.sum(u ** 2))
        v_sq = float(np.sum(v ** 2))

        denom = (1.0 - c * u_sq) * (1.0 - c * v_sq)
        if denom <= 1e-12:
            return float("inf")

        arg = 1.0 + 2.0 * c * diff_sq / denom
        arg = max(arg, 1.0)
        return math.acosh(arg) / math.sqrt(c)

    def distance_batch(self, u: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """Geodesic distances from u to each row of vs.

        u:  shape (2,)
        vs: shape (n, 2)
        Returns: shape (n,)
        """
        c = self._c
        diff = vs - u[np.newaxis, :]
        diff_sq = np.sum(diff ** 2, axis=1)
        u_sq = float(np.sum(u ** 2))
        vs_sq = np.sum(vs ** 2, axis=1)

        denom = (1.0 - c * u_sq) * (1.0 - c * vs_sq)
        denom = np.maximum(denom, 1e-12)

        arg = 1.0 + 2.0 * c * diff_sq / denom
        arg = np.maximum(arg, 1.0)
        return np.arccosh(arg) / math.sqrt(c)

    def distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Pairwise geodesic distance matrix.

        points: shape (n, 2)
        Returns: shape (n, n) symmetric matrix with zeros on diagonal
        """
        n = len(points)
        c = self._c
        norms_sq = np.sum(points ** 2, axis=1)  # (n,)

        # (n, n) pairwise squared Euclidean distances
        dots = points @ points.T
        diff_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * dots

        conformal_u = 1.0 - c * norms_sq  # (n,)
        denom = conformal_u[:, None] * conformal_u[None, :]
        denom = np.maximum(denom, 1e-12)

        arg = 1.0 + 2.0 * c * diff_sq / denom
        arg = np.maximum(arg, 1.0)
        return np.arccosh(arg) / math.sqrt(c)

    # ── Möbius operations ───────────────────────────────────────────

    def mobius_add(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Möbius addition u ⊕ v in the Poincaré disk.

        u ⊕ v = ((1+2c⟨u,v⟩+c‖v‖²)u + (1-c‖u‖²)v) /
                  (1+2c⟨u,v⟩+c²‖u‖²‖v‖²)
        """
        c = self._c
        uv = float(np.dot(u, v))
        u_sq = float(np.sum(u ** 2))
        v_sq = float(np.sum(v ** 2))

        num = (1.0 + 2.0 * c * uv + c * v_sq) * u + (1.0 - c * u_sq) * v
        denom = 1.0 + 2.0 * c * uv + c * c * u_sq * v_sq
        result = num / max(denom, 1e-12)
        return self._clamp(result)

    # ── Centroids ───────────────────────────────────────────────────

    def einstein_midpoint(
        self, points: np.ndarray, weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Weighted Einstein midpoint (hyperbolic centroid).

        centroid = Σ(w_i·γ_i·x_i) / Σ(w_i·γ_i)
        where γ_i = 1/√(1 - c·‖x_i‖²) is the Lorentz factor.

        points:  shape (n, 2)
        weights: shape (n,), default uniform
        Returns: shape (2,)
        """
        if len(points) == 0:
            return np.zeros(2, dtype=np.float32)
        if len(points) == 1:
            return points[0].astype(np.float32)

        c = self._c
        if weights is None:
            weights = np.ones(len(points), dtype=np.float32)

        norms_sq = np.sum(points ** 2, axis=1)
        gamma = 1.0 / np.sqrt(np.maximum(1.0 - c * norms_sq, 1e-12))

        wg = weights * gamma
        centroid = np.sum(wg[:, None] * points, axis=0) / max(float(np.sum(wg)), 1e-12)
        return self._clamp(centroid)

    # ── Ramsey depth mapping ────────────────────────────────────────

    def project_with_ramsey_depth(
        self,
        projected_2d: np.ndarray,
        prime_slots: dict[int, int],
        slot_populations: dict[int, dict[int, int]],
        total_events: int,
    ) -> np.ndarray:
        """Map a 2D PCA projection into the Poincaré disk with Ramsey-derived depth.

        1. Apply exponential map to get base Poincaré coordinates
        2. Adjust radial position using prime slot population density:
           - Dense slots (many events share this slot) → general → closer to center
           - Sparse slots (few events here) → specific → closer to boundary

        Returns: shape (2,) point in the Poincaré disk
        """
        base = self.exp_map_origin(projected_2d)

        if total_events < 2 or not prime_slots:
            return base

        return self._adjust_depth(base, prime_slots, slot_populations, total_events)

    def recompute_depth(
        self,
        poincare_point: np.ndarray,
        prime_slots: dict[int, int],
        slot_populations: dict[int, dict[int, int]],
        total_events: int,
    ) -> np.ndarray:
        """Re-adjust an existing Poincaré point's radial depth.

        Useful when slot populations change (new events added) and existing
        points need their depth recalibrated without recomputing the angle.
        """
        if total_events < 2 or not prime_slots:
            return poincare_point

        return self._adjust_depth(poincare_point, prime_slots, slot_populations, total_events)

    def _adjust_depth(
        self,
        point: np.ndarray,
        prime_slots: dict[int, int],
        slot_populations: dict[int, dict[int, int]],
        total_events: int,
    ) -> np.ndarray:
        norm = float(np.linalg.norm(point))
        if norm < 1e-12:
            return point.astype(np.float32)

        # For each prime p, compute how populated this event's slot is
        # relative to the uniform expectation (total_events / p).
        log_ratios = []
        for p, slot in prime_slots.items():
            pop = slot_populations.get(p, {}).get(slot, 1)
            expected = total_events / p
            if expected > 0:
                log_ratios.append(math.log(max(pop, 1) / max(expected, 1e-6)))

        if not log_ratios:
            return point.astype(np.float32)

        # mean_log_ratio > 0 → denser than expected → general → pull inward
        # mean_log_ratio < 0 → sparser than expected → specific → push outward
        mean_log_ratio = float(np.mean(log_ratios))

        # Sigmoid-like adjustment centered at 0: maps R → (0.5, 1.5)
        # density_factor < 1 shrinks radius (general), > 1 grows it (specific)
        density_factor = 1.0 / (1.0 + math.exp(mean_log_ratio))  # logistic
        # Rescale from (0, 1) to (0.5, 1.5) so adjustment is moderate
        density_factor = 0.5 + density_factor

        new_norm = min(norm * density_factor, self._max_radius)
        direction = point / norm
        return (new_norm * direction).astype(np.float32)

    # ── Coordinate utilities ────────────────────────────────────────

    def to_polar(self, point: np.ndarray) -> tuple[float, float]:
        """Convert Poincaré disk point to (radius, angle)."""
        r = float(np.linalg.norm(point))
        theta = math.atan2(float(point[1]), float(point[0]))
        if theta < 0:
            theta += 2.0 * math.pi
        return r, theta

    def from_polar(self, r: float, theta: float) -> np.ndarray:
        """Convert (radius, angle) to Poincaré disk point."""
        r = min(r, self._max_radius)
        return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)

    def conformal_factor(self, point: np.ndarray) -> float:
        """Conformal factor λ(x) = 2 / (1 - c·‖x‖²).

        The local scale distortion at this point — how much larger
        infinitesimal Euclidean distances appear in hyperbolic space.
        Near center ≈ 2, grows to ∞ at boundary.
        """
        c = self._c
        norm_sq = float(np.sum(point ** 2))
        return 2.0 / max(1.0 - c * norm_sq, 1e-12)

    def _clamp(self, point: np.ndarray) -> np.ndarray:
        """Clamp point to stay within the open disk."""
        norm = float(np.linalg.norm(point))
        if norm >= self._max_radius:
            point = point * (self._max_radius / norm)
        return point.astype(np.float32)
