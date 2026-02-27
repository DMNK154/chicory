"""Time series utility functions for exponential decay and derivative estimation."""

from __future__ import annotations

import math


def exponential_decay(age_hours: float, halflife_hours: float) -> float:
    """Compute exponential decay factor given age and half-life."""
    if halflife_hours <= 0:
        return 0.0
    decay_lambda = math.log(2) / halflife_hours
    return math.exp(-decay_lambda * age_hours)


def multi_tier_decay(
    age_hours: float,
    tiers: list[tuple[float, float]],
) -> float:
    """Sum-of-exponentials decay across multiple timescale tiers.

    Each tier is (weight, halflife_hours).  Returns the weighted sum of
    exponential decays, giving a fast-decaying "active" component and a
    slow-decaying "long-term" component.
    """
    total = 0.0
    for weight, halflife in tiers:
        if halflife <= 0:
            continue
        decay_lambda = math.log(2) / halflife
        total += weight * math.exp(-decay_lambda * age_hours)
    return total


def sigmoid(x: float) -> float:
    """Standard sigmoid function, clamped to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def weighted_sum_with_decay(
    events: list[tuple[float, float]],
    halflife_hours: float,
) -> float:
    """Sum of (weight * decay) for events as (age_hours, weight) pairs."""
    decay_lambda = math.log(2) / halflife_hours if halflife_hours > 0 else 0
    total = 0.0
    for age, weight in events:
        total += weight * math.exp(-decay_lambda * age)
    return total


def split_window_derivative(
    events: list[tuple[float, float]],
    window_hours: float,
    halflife_hours: float,
) -> float:
    """Estimate first derivative by comparing recent half vs older half of window."""
    midpoint = window_hours / 2
    decay_lambda = math.log(2) / halflife_hours if halflife_hours > 0 else 0

    recent = 0.0
    older = 0.0
    for age, weight in events:
        decayed = weight * math.exp(-decay_lambda * age)
        if age <= midpoint:
            recent += decayed
        else:
            older += decayed

    return recent - older


def three_part_jerk(
    events: list[tuple[float, float]],
    window_hours: float,
    halflife_hours: float,
) -> float:
    """Estimate second derivative by splitting window into thirds."""
    third = window_hours / 3
    decay_lambda = math.log(2) / halflife_hours if halflife_hours > 0 else 0

    t1 = 0.0  # oldest third
    t2 = 0.0  # middle third
    t3 = 0.0  # newest third

    for age, weight in events:
        decayed = weight * math.exp(-decay_lambda * age)
        if age <= third:
            t3 += decayed  # newest
        elif age <= 2 * third:
            t2 += decayed
        else:
            t1 += decayed  # oldest

    return t3 - 2 * t2 + t1
