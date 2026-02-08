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
    debug_cell: tuple[Any, Any] | None = None,
) -> dict[tuple[Any, Any], dict[tuple[str, str], float]]:
    """
    Count nested transitions (continue->succeeded, etc.) per (persona_id, goal_cluster).
    Returns (persona_k, goal_g) -> {(from_state, to_state): count}.
    persona_assignments can be int indices or persona_id strings; goal_phase_labels is
    list of [(goal_id, phase), ...] per trajectory.
    If goal_cluster_labels is None, goal_id is used directly as cluster (string).
    Weights are float to support soft counts from EM.

    IMPORTANT: This function now applies sub-segmentation to split goal segments at terminal
    states (succeeded/failed), ensuring we only count transitions within single attempts where
    succeeded/failed are absorbing states. This prevents counting invalid transitions like
    (succeeded, continue) or (failed, failed) that occur when users retry the same goal.

    debug_cell: if provided, log detailed transitions for this (persona_k, goal_g) cell.
    """
    # Import sub-segmentation function
    from expforge.estimator.em import _segment_by_goal, _sub_segment_by_attempt
    import logging
    logger = logging.getLogger(__name__)

    counts: dict[tuple[Any, Any], dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    debug_transitions = []  # Store transitions for debug cell

    for path_idx, (path, persona_k, phases, clusters) in enumerate(zip(
        trajectory_paths,
        persona_assignments,
        goal_phase_labels,
        goal_cluster_labels if goal_cluster_labels is not None else [None] * len(trajectory_paths),
    )):
        # First segment by goal (consecutive steps with same goal_id)
        segments = _segment_by_goal(phases)
        # Then sub-segment to split at each terminal state (succeeded/failed)
        sub_segments = _sub_segment_by_attempt(segments)

        # Count transitions within each sub-segment
        # IMPORTANT: Only count explicit transitions from length-2+ segments.
        # Length-1 segments represent direct outcomes from the initial state but don't
        # provide observations of state transitions (we only see the final state, not
        # the transition). Including them would bias the transition probability estimates.
        for goal_id, sub_phases in sub_segments:
            if not goal_id or len(sub_phases) < 2:
                continue

            # Determine the cluster/goal index for this segment
            if clusters is not None:
                # Find first occurrence of this goal in original phases to get cluster
                g_cluster = None
                for orig_idx, (g, _) in enumerate(phases):
                    if g == goal_id and orig_idx < len(clusters):
                        g_cluster = clusters[orig_idx]
                        break
                g = g_cluster if g_cluster is not None else goal_id
            else:
                g = goal_id

            key = (persona_k, g)

            # Count consecutive pairs (explicit transitions only)
            for i in range(len(sub_phases) - 1):
                ph_from = sub_phases[i]
                ph_to = sub_phases[i + 1]
                counts[key][(ph_from, ph_to)] += 1.0

                # Debug logging
                if debug_cell and key == debug_cell:
                    debug_transitions.append({
                        "traj_idx": path_idx,
                        "traj": path.name,
                        "segment": sub_phases,
                        "transition": f"{ph_from} -> {ph_to}",
                    })

    if debug_cell and debug_transitions:
        logger.info(f"[DEBUG] Transitions for cell {debug_cell}: {len(debug_transitions)} total")
        # Show first 10 and summary stats
        for i, t in enumerate(debug_transitions[:10]):
            logger.info(f"  [{i+1}] traj={t['traj']}: segment={t['segment']} | {t['transition']}")
        trans_counts = {}
        for t in debug_transitions:
            trans_counts[t['transition']] = trans_counts.get(t['transition'], 0) + 1
        logger.info(f"  Summary: {trans_counts}")

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
            # Only map non-empty goal IDs; empty goals should remain unmapped (use -1 as sentinel)
            if g_id:
                g_idx = goal_id_to_index.get(g_id, 0)
            else:
                g_idx = -1  # Sentinel for non-goal steps
            row.append((g_idx, ph))
        goal_phase_per_traj.append(row)
    return persona_indices, goal_phase_per_traj
