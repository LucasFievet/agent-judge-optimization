"""Compute expected metrics (publish, subscribe, etc.) and confidence from transition matrices."""

from pathlib import Path
from typing import Any


def expected_metrics_from_transitions(
    transition_matrices: dict[tuple[int, int], dict[tuple[str, str], float]],
    persona_weights: list[float],
) -> dict[str, float]:
    """
    Compute expected P(publish), P(subscribe), P(finished), P(abandoned) from
    per-(persona, goal) transition matrices and persona weights.
    Stub: returns placeholder; full impl would use fundamental matrix.
    """
    # Placeholder: real impl uses N, R from transition structure
    return {
        "publish": 0.2,
        "subscribe": 0.15,
        "finished": 0.5,
        "abandoned": 0.15,
    }


def metric_confidence_interval(
    outcomes: list[str], metric: str, confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    From observed outcomes (e.g. list of 'publish'|'subscribe'|'finished'|'abandoned'),
    return (point_estimate, lower, upper) for the given metric proportion.
    """
    n = len(outcomes)
    count = sum(1 for o in outcomes if o == metric)
    p = count / n if n else 0.0
    # Wald interval (or Wilson); simplified
    import math
    z = 1.96 if confidence >= 0.95 else 1.645
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return (p, max(0, p - z * se), min(1, p + z * se))
