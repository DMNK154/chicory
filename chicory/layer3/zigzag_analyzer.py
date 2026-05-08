"""Rotation-invariant zigzag metrics over square motifs."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from chicory.layer3.square_finder import SquareMotif, find_square_motifs


@dataclass(frozen=True)
class ZigZagSample:
    """One representative square for a zigzag report."""

    tags: tuple[str, str, str, str]
    side_layers: tuple[str, str, str, str]
    orientation_class: str
    signature: str
    vector_score: float
    center_score: float
    void_score: float
    interestingness: float
    ac_status: str
    bd_status: str
    source_summaries: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tags": list(self.tags),
            "side_layers": list(self.side_layers),
            "orientation_class": self.orientation_class,
            "signature": self.signature,
            "vector_score": round(self.vector_score, 6),
            "center_score": round(self.center_score, 6),
            "void_score": round(self.void_score, 6),
            "interestingness": round(self.interestingness, 6),
            "ac_status": self.ac_status,
            "bd_status": self.bd_status,
            "source_summaries": [
                {"summary": summary, "count": count}
                for summary, count in self.source_summaries
            ],
        }


@dataclass(frozen=True)
class ZigZagReport:
    """Summary of color-pattern anisotropy over many square motifs."""

    motif_count: int
    class_counts: tuple[tuple[str, int], ...]
    signature_counts: tuple[tuple[str, int], ...]
    zigzag_pair_counts: tuple[tuple[str, int], ...]
    zigzag_count: int
    zigzag_rate: float
    baseline_iterations: int
    baseline_zigzag_mean: float
    baseline_zigzag_std: float
    zigzag_z_score: float
    zigzag_p_value: float
    vector_zigzag_mean: float
    vector_zigzag_positive_rate: float
    samples: tuple[ZigZagSample, ...]
    parameters: dict[str, Any]

    @property
    def dominant_class(self) -> tuple[str, int] | None:
        return self.class_counts[0] if self.class_counts else None

    @property
    def dominant_signature(self) -> tuple[str, int] | None:
        return self.signature_counts[0] if self.signature_counts else None

    @property
    def dominant_zigzag_pair(self) -> tuple[str, int] | None:
        return self.zigzag_pair_counts[0] if self.zigzag_pair_counts else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "motif_count": self.motif_count,
            "class_counts": dict(self.class_counts),
            "signature_counts": dict(self.signature_counts),
            "zigzag_pair_counts": dict(self.zigzag_pair_counts),
            "zigzag_count": self.zigzag_count,
            "zigzag_rate": round(self.zigzag_rate, 6),
            "baseline_iterations": self.baseline_iterations,
            "baseline_zigzag_mean": round(self.baseline_zigzag_mean, 6),
            "baseline_zigzag_std": round(self.baseline_zigzag_std, 6),
            "zigzag_z_score": round(self.zigzag_z_score, 6),
            "zigzag_p_value": round(self.zigzag_p_value, 6),
            "vector_zigzag_mean": round(self.vector_zigzag_mean, 6),
            "vector_zigzag_positive_rate": round(
                self.vector_zigzag_positive_rate, 6
            ),
            "samples": [sample.as_dict() for sample in self.samples],
            "parameters": self.parameters,
        }


def analyze_square_zigzags(
    db: Any,
    *,
    motif_limit: int = 2000,
    max_edges: int = 500,
    per_layer_edges: int = 100,
    min_edge_score: float = 0.05,
    min_colors: int = 1,
    min_tag_length: int = 2,
    require_void: bool = False,
    include_text: str | None = None,
    exclude_text: str | None = None,
    require_distinct_incident_layers: bool = False,
    shuffle_iterations: int = 200,
    random_seed: int = 13,
    sample_limit: int = 12,
    layer_weights: dict[str, float] | None = None,
) -> ZigZagReport:
    """Analyze square side-color patterns against a shuffled-color baseline."""
    motifs = find_square_motifs(
        db,
        limit=motif_limit,
        max_edges=max_edges,
        per_layer_edges=per_layer_edges,
        min_edge_score=min_edge_score,
        min_colors=min_colors,
        min_tag_length=min_tag_length,
        require_void=require_void,
        include_text=include_text,
        exclude_text=exclude_text,
        require_distinct_incident_layers=require_distinct_incident_layers,
        layer_weights=layer_weights,
    )

    class_counter: Counter[str] = Counter()
    signature_counter: Counter[str] = Counter()
    zigzag_pair_counter: Counter[str] = Counter()
    vector_scores: list[float] = []
    samples: list[ZigZagSample] = []

    for motif in motifs:
        layers = motif.side_layers
        orientation_class = classify_side_layers(layers)
        signature = canonical_side_signature(layers)
        vector_score = vector_zigzag_score(motif)

        class_counter[orientation_class] += 1
        signature_counter[signature] += 1
        vector_scores.append(vector_score)

        if orientation_class == "zigzag":
            zigzag_pair_counter[_zigzag_pair(layers)] += 1

        if len(samples) < sample_limit:
            samples.append(
                ZigZagSample(
                    tags=motif.tags,
                    side_layers=layers,
                    orientation_class=orientation_class,
                    signature=signature,
                    vector_score=vector_score,
                    center_score=motif.center_score,
                    void_score=motif.void_score,
                    interestingness=motif.interestingness,
                    ac_status=motif.ac_diagonal.status,
                    bd_status=motif.bd_diagonal.status,
                    source_summaries=motif.source_summaries[:3],
                )
            )

    zigzag_count = class_counter["zigzag"]
    motif_count = len(motifs)
    zigzag_rate = zigzag_count / motif_count if motif_count else 0.0
    baseline_counts = _shuffle_zigzag_counts(
        [layer for motif in motifs for layer in motif.side_layers],
        motif_count=motif_count,
        iterations=shuffle_iterations,
        random_seed=random_seed,
    )
    baseline_mean = _mean(baseline_counts)
    baseline_std = _stddev(baseline_counts, baseline_mean)
    z_score = (
        (zigzag_count - baseline_mean) / baseline_std
        if baseline_std > 1e-12
        else 0.0
    )
    p_value = _two_sided_p_value(zigzag_count, baseline_counts, baseline_mean)
    positive_vectors = sum(1 for score in vector_scores if score > 0)
    vector_positive_rate = (
        positive_vectors / len(vector_scores) if vector_scores else 0.0
    )

    return ZigZagReport(
        motif_count=motif_count,
        class_counts=tuple(class_counter.most_common()),
        signature_counts=tuple(signature_counter.most_common(12)),
        zigzag_pair_counts=tuple(zigzag_pair_counter.most_common(12)),
        zigzag_count=zigzag_count,
        zigzag_rate=zigzag_rate,
        baseline_iterations=shuffle_iterations,
        baseline_zigzag_mean=baseline_mean,
        baseline_zigzag_std=baseline_std,
        zigzag_z_score=z_score,
        zigzag_p_value=p_value,
        vector_zigzag_mean=_mean(vector_scores),
        vector_zigzag_positive_rate=vector_positive_rate,
        samples=tuple(samples),
        parameters={
            "motif_limit": motif_limit,
            "max_edges": max_edges,
            "per_layer_edges": per_layer_edges,
            "min_edge_score": min_edge_score,
            "min_colors": min_colors,
            "min_tag_length": min_tag_length,
            "require_void": require_void,
            "include_text": include_text,
            "exclude_text": exclude_text,
            "require_distinct_incident_layers": require_distinct_incident_layers,
            "shuffle_iterations": shuffle_iterations,
            "random_seed": random_seed,
        },
    )


def classify_side_layers(layers: tuple[str, str, str, str]) -> str:
    """Classify side colors, ignoring square rotations/reflections."""
    a, b, c, d = layers
    if len({a, b, c, d}) == 1:
        return "uniform"
    if a == c and b == d and a != b:
        return "zigzag"
    if (a == b and c == d and a != c) or (b == c and d == a and b != d):
        return "banded"
    if max(Counter(layers).values()) == 3:
        return "three-one"
    return "mixed"


def canonical_side_signature(layers: tuple[str, str, str, str]) -> str:
    """Return the canonical D4 signature for a square side-coloring."""
    return min(_format_side_layers(transformed) for transformed in _side_d4(layers))


def vector_zigzag_score(motif: SquareMotif) -> float:
    """Positive when opposite side vectors match more than adjacent sides."""
    vectors = [edge.vector for edge in motif.side_edges]
    opposite = (
        _cosine(vectors[0], vectors[2]) + _cosine(vectors[1], vectors[3])
    ) / 2.0
    adjacent = (
        _cosine(vectors[0], vectors[1])
        + _cosine(vectors[1], vectors[2])
        + _cosine(vectors[2], vectors[3])
        + _cosine(vectors[3], vectors[0])
    ) / 4.0
    return opposite - adjacent


def _side_d4(
    layers: tuple[str, str, str, str],
) -> tuple[tuple[str, str, str, str], ...]:
    rotated = (
        layers,
        (layers[3], layers[0], layers[1], layers[2]),
        (layers[2], layers[3], layers[0], layers[1]),
        (layers[1], layers[2], layers[3], layers[0]),
    )
    reflected = tuple((item[0], item[3], item[2], item[1]) for item in rotated)
    return rotated + reflected


def _format_side_layers(layers: tuple[str, str, str, str]) -> str:
    return "|".join(layers)


def _zigzag_pair(layers: tuple[str, str, str, str]) -> str:
    return "|".join(sorted({layers[0], layers[1]}))


def _shuffle_zigzag_counts(
    layers: list[str],
    *,
    motif_count: int,
    iterations: int,
    random_seed: int,
) -> list[int]:
    if motif_count <= 0 or iterations <= 0:
        return []

    rng = random.Random(random_seed)
    counts: list[int] = []
    shuffled = list(layers)
    for _ in range(iterations):
        rng.shuffle(shuffled)
        count = 0
        for idx in range(motif_count):
            start = idx * 4
            motif_layers = (
                shuffled[start],
                shuffled[start + 1],
                shuffled[start + 2],
                shuffled[start + 3],
            )
            if classify_side_layers(motif_layers) == "zigzag":
                count += 1
        counts.append(count)
    return counts


def _two_sided_p_value(
    observed: int,
    baseline_counts: list[int],
    baseline_mean: float,
) -> float:
    if not baseline_counts:
        return 1.0
    observed_delta = abs(observed - baseline_mean)
    exceedances = sum(
        1 for count in baseline_counts
        if abs(count - baseline_mean) >= observed_delta
    )
    return (exceedances + 1) / (len(baseline_counts) + 1)


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: list[int], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b)) / denom))
