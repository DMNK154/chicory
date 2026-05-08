"""Chain anisotropy metrics for Chicory's edge-colored tensor graph."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from chicory.layer3.square_finder import LAYER_ORDER, TensorEdge, load_tensor_edges


@dataclass(frozen=True)
class ChainEdge:
    """Lightweight tensor edge used for repeated shuffled scans."""

    tag_a_id: int
    tag_b_id: int
    tag_a: str
    tag_b: str
    dominant_layer: str
    score: float


@dataclass(frozen=True)
class ChainProbe:
    """One sampled axis edge and its same-layer/side-branch persistence."""

    axis_layer: str
    start_tag: str
    next_tag: str
    axis_length: int
    side_length: float
    contrast: float
    side_layers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis_layer": self.axis_layer,
            "start_tag": self.start_tag,
            "next_tag": self.next_tag,
            "axis_length": self.axis_length,
            "side_length": round(self.side_length, 6),
            "contrast": round(self.contrast, 6),
            "side_layers": list(self.side_layers),
        }


@dataclass(frozen=True)
class LayerAnisotropy:
    """Axis-vs-side chain persistence for one relation layer."""

    layer: str
    edge_count: int
    probe_count: int
    axis_mean: float
    side_mean: float
    contrast: float
    ratio: float
    baseline_contrast_mean: float
    baseline_contrast_std: float
    z_score: float
    p_value: float
    top_probes: tuple[ChainProbe, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "edge_count": self.edge_count,
            "probe_count": self.probe_count,
            "axis_mean": round(self.axis_mean, 6),
            "side_mean": round(self.side_mean, 6),
            "contrast": round(self.contrast, 6),
            "ratio": round(self.ratio, 6),
            "baseline_contrast_mean": round(self.baseline_contrast_mean, 6),
            "baseline_contrast_std": round(self.baseline_contrast_std, 6),
            "z_score": round(self.z_score, 6),
            "p_value": round(self.p_value, 6),
            "top_probes": [probe.as_dict() for probe in self.top_probes],
        }


@dataclass(frozen=True)
class ChainAnisotropyReport:
    """Full chain anisotropy report across relation layers."""

    layer_reports: tuple[LayerAnisotropy, ...]
    total_edges: int
    parameters: dict[str, Any]

    @property
    def strongest_layer(self) -> LayerAnisotropy | None:
        if not self.layer_reports:
            return None
        return max(self.layer_reports, key=lambda report: report.z_score)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_edges": self.total_edges,
            "strongest_layer": (
                self.strongest_layer.layer if self.strongest_layer else None
            ),
            "layer_reports": [report.as_dict() for report in self.layer_reports],
            "parameters": self.parameters,
        }


Adjacency = dict[str, dict[int, list[tuple[int, ChainEdge]]]]


def analyze_chain_anisotropy(
    db: Any,
    *,
    edge_limit: int = 1500,
    per_layer_edges: int = 200,
    min_edge_score: float = 0.05,
    min_tag_length: int = 2,
    min_layer_edges: int = 10,
    max_depth: int = 6,
    sample_edges_per_layer: int = 100,
    max_branching: int = 5,
    side_branching: int = 2,
    shuffle_iterations: int = 80,
    random_seed: int = 23,
    layer_weights: dict[str, float] | None = None,
) -> ChainAnisotropyReport:
    """Measure whether same-layer chains persist beyond side branches."""
    tensor_edges = load_tensor_edges(db, layer_weights=layer_weights)
    edges = _select_edges(
        tensor_edges,
        edge_limit=edge_limit,
        per_layer_edges=per_layer_edges,
        min_edge_score=min_edge_score,
        min_tag_length=min_tag_length,
    )
    adjacency = _build_adjacency(edges, max_branching=max_branching)
    observed = _measure_layers(
        edges,
        adjacency,
        max_depth=max_depth,
        min_layer_edges=min_layer_edges,
        sample_edges_per_layer=sample_edges_per_layer,
        max_branching=max_branching,
        side_branching=side_branching,
        include_probes=True,
    )

    baseline = _shuffle_baseline(
        edges,
        iterations=shuffle_iterations,
        random_seed=random_seed,
        max_depth=max_depth,
        sample_edges_per_layer=sample_edges_per_layer,
        max_branching=max_branching,
        side_branching=side_branching,
    )

    reports: list[LayerAnisotropy] = []
    for layer in LAYER_ORDER:
        measured = observed.get(layer)
        if measured is None:
            continue
        baseline_values = baseline.get(layer, [])
        baseline_mean = _mean(baseline_values)
        baseline_std = _stddev(baseline_values, baseline_mean)
        z_score = (
            (measured["contrast"] - baseline_mean) / baseline_std
            if baseline_std > 1e-12
            else 0.0
        )
        p_value = _two_sided_p_value(
            measured["contrast"],
            baseline_values,
            baseline_mean,
        )
        reports.append(
            LayerAnisotropy(
                layer=layer,
                edge_count=measured["edge_count"],
                probe_count=measured["probe_count"],
                axis_mean=measured["axis_mean"],
                side_mean=measured["side_mean"],
                contrast=measured["contrast"],
                ratio=measured["ratio"],
                baseline_contrast_mean=baseline_mean,
                baseline_contrast_std=baseline_std,
                z_score=z_score,
                p_value=p_value,
                top_probes=tuple(measured["top_probes"]),
            )
        )

    reports.sort(key=lambda report: report.z_score, reverse=True)
    return ChainAnisotropyReport(
        layer_reports=tuple(reports),
        total_edges=len(edges),
        parameters={
            "edge_limit": edge_limit,
            "per_layer_edges": per_layer_edges,
            "min_edge_score": min_edge_score,
            "min_tag_length": min_tag_length,
            "min_layer_edges": min_layer_edges,
            "max_depth": max_depth,
            "sample_edges_per_layer": sample_edges_per_layer,
            "max_branching": max_branching,
            "side_branching": side_branching,
            "shuffle_iterations": shuffle_iterations,
            "random_seed": random_seed,
        },
    )


def _select_edges(
    edges: list[TensorEdge],
    *,
    edge_limit: int,
    per_layer_edges: int,
    min_edge_score: float,
    min_tag_length: int,
) -> list[ChainEdge]:
    selected: dict[tuple[int, int], TensorEdge] = {}
    eligible = [
        edge for edge in edges
        if edge.score >= min_edge_score
        and len(edge.tag_a) >= min_tag_length
        and len(edge.tag_b) >= min_tag_length
    ]
    for edge in eligible[:edge_limit]:
        selected[edge.key] = edge
    if per_layer_edges > 0:
        for layer in LAYER_ORDER:
            count = 0
            for edge in eligible:
                if edge.dominant_layer != layer:
                    continue
                selected[edge.key] = edge
                count += 1
                if count >= per_layer_edges:
                    break

    return [
        ChainEdge(
            tag_a_id=edge.tag_a_id,
            tag_b_id=edge.tag_b_id,
            tag_a=edge.tag_a,
            tag_b=edge.tag_b,
            dominant_layer=edge.dominant_layer,
            score=edge.score,
        )
        for edge in sorted(selected.values(), key=lambda item: item.score, reverse=True)
    ]


def _build_adjacency(
    edges: list[ChainEdge],
    *,
    max_branching: int,
) -> Adjacency:
    adjacency: Adjacency = {
        layer: defaultdict(list) for layer in LAYER_ORDER
    }
    for edge in edges:
        adjacency[edge.dominant_layer][edge.tag_a_id].append((edge.tag_b_id, edge))
        adjacency[edge.dominant_layer][edge.tag_b_id].append((edge.tag_a_id, edge))

    for layer in LAYER_ORDER:
        for node, neighbors in adjacency[layer].items():
            neighbors.sort(key=lambda item: item[1].score, reverse=True)
            if max_branching > 0:
                adjacency[layer][node] = neighbors[:max_branching]
    return adjacency


def _measure_layers(
    edges: list[ChainEdge],
    adjacency: Adjacency,
    *,
    max_depth: int,
    min_layer_edges: int,
    sample_edges_per_layer: int,
    max_branching: int,
    side_branching: int,
    include_probes: bool,
) -> dict[str, dict[str, Any]]:
    edges_by_layer: dict[str, list[ChainEdge]] = defaultdict(list)
    for edge in edges:
        edges_by_layer[edge.dominant_layer].append(edge)

    measured: dict[str, dict[str, Any]] = {}
    for layer in LAYER_ORDER:
        layer_edges = edges_by_layer.get(layer, [])
        if len(layer_edges) < min_layer_edges:
            continue
        probes: list[ChainProbe] = []
        axis_lengths: list[float] = []
        side_lengths: list[float] = []

        for edge in layer_edges[:sample_edges_per_layer]:
            forward = _longest_layer_chain(
                adjacency[layer],
                previous=edge.tag_a_id,
                current=edge.tag_b_id,
                max_depth=max_depth,
                max_branching=max_branching,
            )
            backward = _longest_layer_chain(
                adjacency[layer],
                previous=edge.tag_b_id,
                current=edge.tag_a_id,
                max_depth=max_depth,
                max_branching=max_branching,
            )
            axis_length = max(forward, backward)
            side_length, side_layers = _side_branch_length(
                adjacency,
                axis_layer=layer,
                nodes=(edge.tag_a_id, edge.tag_b_id),
                max_depth=max_depth,
                max_branching=max_branching,
                side_branching=side_branching,
            )
            axis_lengths.append(float(axis_length))
            side_lengths.append(side_length)
            if include_probes:
                probes.append(
                    ChainProbe(
                        axis_layer=layer,
                        start_tag=edge.tag_a,
                        next_tag=edge.tag_b,
                        axis_length=axis_length,
                        side_length=side_length,
                        contrast=axis_length - side_length,
                        side_layers=side_layers,
                    )
                )

        if not axis_lengths:
            continue
        axis_mean = _mean(axis_lengths)
        side_mean = _mean(side_lengths)
        contrast = axis_mean - side_mean
        ratio = axis_mean / side_mean if side_mean > 1e-12 else axis_mean
        probes.sort(key=lambda probe: probe.contrast, reverse=True)
        measured[layer] = {
            "edge_count": len(layer_edges),
            "probe_count": len(axis_lengths),
            "axis_mean": axis_mean,
            "side_mean": side_mean,
            "contrast": contrast,
            "ratio": ratio,
            "top_probes": probes[:8],
        }
    return measured


def _longest_layer_chain(
    adjacency: dict[int, list[tuple[int, ChainEdge]]],
    *,
    previous: int,
    current: int,
    max_depth: int,
    max_branching: int,
) -> int:
    visited = {previous, current}
    return 1 + _longest_continuation(
        adjacency,
        current=current,
        visited=visited,
        remaining=max(0, max_depth - 1),
        max_branching=max_branching,
    )


def _longest_continuation(
    adjacency: dict[int, list[tuple[int, ChainEdge]]],
    *,
    current: int,
    visited: set[int],
    remaining: int,
    max_branching: int,
) -> int:
    if remaining <= 0:
        return 0
    best = 0
    neighbors = adjacency.get(current, [])
    if max_branching > 0:
        neighbors = neighbors[:max_branching]
    for neighbor, _edge in neighbors:
        if neighbor in visited:
            continue
        visited.add(neighbor)
        best = max(
            best,
            1
            + _longest_continuation(
                adjacency,
                current=neighbor,
                visited=visited,
                remaining=remaining - 1,
                max_branching=max_branching,
            ),
        )
        visited.discard(neighbor)
    return best


def _side_branch_length(
    adjacency: Adjacency,
    *,
    axis_layer: str,
    nodes: tuple[int, int],
    max_depth: int,
    max_branching: int,
    side_branching: int,
) -> tuple[float, tuple[str, ...]]:
    lengths: list[float] = []
    side_layers: list[str] = []
    axis_nodes = set(nodes)
    for node in nodes:
        for layer in LAYER_ORDER:
            if layer == axis_layer:
                continue
            starts = [
                (neighbor, edge)
                for neighbor, edge in adjacency[layer].get(node, [])
                if neighbor not in axis_nodes
            ][:side_branching]
            for neighbor, _edge in starts:
                length = 1 + _longest_continuation(
                    adjacency[layer],
                    current=neighbor,
                    visited={node, neighbor},
                    remaining=max(0, max_depth - 1),
                    max_branching=max_branching,
                )
                lengths.append(float(length))
                side_layers.append(layer)

    if not lengths:
        return 0.0, ()
    return _mean(lengths), tuple(sorted(set(side_layers)))


def _shuffle_baseline(
    edges: list[ChainEdge],
    *,
    iterations: int,
    random_seed: int,
    max_depth: int,
    sample_edges_per_layer: int,
    max_branching: int,
    side_branching: int,
) -> dict[str, list[float]]:
    baseline: dict[str, list[float]] = {layer: [] for layer in LAYER_ORDER}
    if iterations <= 0 or not edges:
        return baseline

    rng = random.Random(random_seed)
    layers = [edge.dominant_layer for edge in edges]
    for _ in range(iterations):
        shuffled_layers = list(layers)
        rng.shuffle(shuffled_layers)
        shuffled = [
            ChainEdge(
                tag_a_id=edge.tag_a_id,
                tag_b_id=edge.tag_b_id,
                tag_a=edge.tag_a,
                tag_b=edge.tag_b,
                dominant_layer=layer,
                score=edge.score,
            )
            for edge, layer in zip(edges, shuffled_layers)
        ]
        adjacency = _build_adjacency(shuffled, max_branching=max_branching)
        measured = _measure_layers(
            shuffled,
            adjacency,
            max_depth=max_depth,
            min_layer_edges=1,
            sample_edges_per_layer=sample_edges_per_layer,
            max_branching=max_branching,
            side_branching=side_branching,
            include_probes=False,
        )
        for layer, result in measured.items():
            baseline[layer].append(result["contrast"])
    return baseline


def _two_sided_p_value(
    observed: float,
    baseline_values: list[float],
    baseline_mean: float,
) -> float:
    if not baseline_values:
        return 1.0
    observed_delta = abs(observed - baseline_mean)
    exceedances = sum(
        1 for value in baseline_values
        if abs(value - baseline_mean) >= observed_delta
    )
    return (exceedances + 1) / (len(baseline_values) + 1)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
