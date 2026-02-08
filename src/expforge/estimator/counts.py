"""Compute transition counts from labelled trajectories (hard or soft counts for EM)."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from expforge.trajectory.io import load_trajectory


def _get_label_sequences(trajectory_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """
    Load trajectory and return (persona_id, [(goal_id, phase), ...]) from label or steps.
    goal_id is empty when not in a goal; phase is succeeded|continue|failed.
    """
    traj = load_trajectory(trajectory_path)
    if hasattr(traj, "label") and traj.label:
        pid = traj.label.get("persona_id", traj.persona_id)
        gp = traj.label.get("goal_phase", [])
        return pid, [(g.get("goal_id", ""), g.get("phase", "continue")) for g in gp]
    # Fallback: infer from steps (legacy)
    pid = traj.persona_id
    seq = []
    for s in traj.steps:
        goal_id = s.top_level_state if s.top_level_state not in ("start", "publish", "subscribe", "finished", "abandoned") else ""
        phase = s.nested_state or "continue"
        seq.append((goal_id, phase))
    return pid, seq


def compute_nested_transition_counts_from_labels(
    trajectory_paths: list[Path],
    persona_assignments: list[int | str],
    goal_phase_labels: list[list[tuple[str, str]]],
    goal_cluster_labels: list[list[int]] | None = None,
) -> dict[tuple[Any, Any], dict[tuple[str, str], float]]:
    """
    Count nested transitions (continue->succeeded, etc.) per (persona_id, goal_cluster).
    Returns (persona_k, goal_g) -> {(from_state, to_state): count}.
    persona_assignments can be int indices or persona_id strings; goal_phase_labels is
    list of [(goal_id, phase), ...] per trajectory.
    If goal_cluster_labels is None, goal_id is used directly as cluster (string).
    Weights are float to support soft counts from EM.
    """
    counts: dict[tuple[Any, Any], dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    for path, persona_k, phases, clusters in zip(
        trajectory_paths,
        persona_assignments,
        goal_phase_labels,
        goal_cluster_labels if goal_cluster_labels is not None else [None] * len(trajectory_paths),
    ):
        for i in range(len(phases) - 1):
            g_id, ph_from = phases[i]
            g_id_next, ph_to = phases[i + 1]
            g = (clusters[i] if i < len(clusters) else g_id) if clusters is not None else g_id
            key = (persona_k, g)
            counts[key][(ph_from, ph_to)] += 1.0
    return {k: dict(v) for k, v in counts.items()}


def compute_nested_transition_counts(
    trajectory_paths: list[Path],
    persona_assignments: list[int],
    goal_phase_labels: list[list[str]],
    goal_cluster_labels: list[list[int]],
) -> dict[tuple[int, int], dict[tuple[str, str], int]]:
    """
    Legacy: count nested transitions per (persona_id, goal_cluster) with string phases.
    goal_phase_labels is list of [phase_str, ...]; goal_cluster_labels list of [cluster_int, ...].
    """
    gp_as_tuples = [
        [(str(clusters[i]) if i < len(clusters) else "0", p) for i, p in enumerate(phases)]
        for phases, clusters in zip(goal_phase_labels, goal_cluster_labels)
    ]
    raw = compute_nested_transition_counts_from_labels(
        trajectory_paths,
        persona_assignments,
        gp_as_tuples,
        goal_cluster_labels,
    )
    out: dict[tuple[int, int], dict[tuple[str, str], int]] = {}
    for (pk, g), cdict in raw.items():
        k = (
            int(pk) if isinstance(pk, (int, float)) else hash(pk) % (2**31),
            int(g) if isinstance(g, (int, float)) else hash(g) % (2**31),
        )
        out[k] = {k2: int(v) for k2, v in cdict.items()}
    return out


def normalize_transition_counts(
    counts: dict[tuple[str, str], float],
) -> dict[str, dict[str, float]]:
    """Convert counts to transition probability matrix (from_state -> {to_state: prob})."""
    row_sums: dict[str, float] = defaultdict(float)
    for (a, b), c in counts.items():
        row_sums[a] += c
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), c in counts.items():
        matrix[a][b] = c / row_sums[a] if row_sums[a] else 0.0
    return dict(matrix)


def compute_top_level_transition_counts(
    trajectory_paths: list[Path],
    outcomes: list[str],
) -> dict[tuple[str, str], int]:
    """Count top-level transitions from trajectory outcomes (e.g. goal -> finished)."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for path, outcome in zip(trajectory_paths, outcomes):
        counts[("goal", outcome)] += 1
    return dict(counts)


def get_label_sequences_batch(
    trajectory_paths: list[Path],
    persona_id_to_index: dict[str, int],
    goal_id_to_index: dict[str, int],
) -> tuple[list[int], list[list[tuple[int, str]]]]:
    """
    Load labels for all trajectories. Returns (persona_indices, goal_phase_per_traj)
    where goal_phase_per_traj[t][i] = (goal_index, phase_str).
    """
    persona_indices = []
    goal_phase_per_traj = []
    for path in trajectory_paths:
        pid, gp = _get_label_sequences(path)
        persona_indices.append(persona_id_to_index.get(pid, 0))
        row = []
        for g_id, ph in gp:
            g_idx = goal_id_to_index.get(g_id, 0)
            row.append((g_idx, ph))
        goal_phase_per_traj.append(row)
    return persona_indices, goal_phase_per_traj
