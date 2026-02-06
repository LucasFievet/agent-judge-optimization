"""Compute transition matrices per (persona, goal) from inferred state sequences."""

from pathlib import Path
from typing import Any

from collections import defaultdict


def compute_nested_transition_counts(
    trajectory_paths: list[Path],
    persona_assignments: list[int],
    goal_phase_labels: list[list[str]],
    goal_cluster_labels: list[list[int]],
) -> dict[tuple[int, int], dict[tuple[str, str], int]]:
    """
    Count nested transitions (continue->succeeded, etc.) per (persona_id, goal_cluster).
    Returns (persona_k, goal_g) -> {(from_state, to_state): count}.
    """
    counts: dict[tuple[int, int], dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for path, persona_k, phases, clusters in zip(
        trajectory_paths, persona_assignments, goal_phase_labels, goal_cluster_labels
    ):
        for i in range(len(phases) - 1):
            g = clusters[i] if i < len(clusters) else 0
            key = (persona_k, g)
            counts[key][(phases[i], phases[i + 1])] += 1
    return {k: dict(v) for k, v in counts.items()}


def compute_top_level_transition_counts(
    trajectory_paths: list[Path],
    outcomes: list[str],
) -> dict[tuple[str, str], int]:
    """Count top-level transitions from trajectory outcomes (e.g. goal -> finished)."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for path, outcome in zip(trajectory_paths, outcomes):
        # From trajectory we have sequence of top-level states; last -> outcome
        counts[("goal", outcome)] += 1  # simplified
    return dict(counts)


def normalize_transition_counts(
    counts: dict[tuple[str, str], int]
) -> dict[str, dict[str, float]]:
    """Convert counts to transition probability matrix (from_state -> {to_state: prob})."""
    from collections import defaultdict

    row_sums: dict[str, float] = defaultdict(float)
    for (a, b), c in counts.items():
        row_sums[a] += c
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), c in counts.items():
        matrix[a][b] = c / row_sums[a] if row_sums[a] else 0.0
    return dict(matrix)
