"""
EM-type algorithm to estimate transition probabilities from labelled trajectories
with unknown label error rate (paper §3.8). Optional confusion model for phase labels.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from expforge.trajectory.io import load_trajectory

from expforge.estimator.counts import (
    _get_label_sequences,
    compute_nested_transition_counts_from_labels,
    normalize_transition_counts,
)

PHASES = ("succeeded", "continue", "failed")


def _default_confusion(error_rate: float) -> dict[str, dict[str, float]]:
    """Uniform off-diagonal confusion: P(obs=j | true=i) = 1 - err on diagonal, err/(K-1) off."""
    K = len(PHASES)
    c = 1.0 - error_rate
    e = error_rate / (K - 1) if K > 1 else 0.0
    return {p: {q: c if p == q else e for q in PHASES} for p in PHASES}


def _forward_backward(
    observed_phases: list[str],
    transition_row: dict[str, dict[str, float]],
    confusion: dict[str, dict[str, float]],
) -> tuple[list[dict[str, float]], list[dict[tuple[str, str], float]]]:
    """
    Forward-backward for a single segment (one goal, one persona). States = PHASES.
    transition_row[from_ph][to_ph], confusion[true_ph][obs_ph].
    Returns gamma_t(state) and xi_t((s,s')) normalized.
    """
    T = len(observed_phases)
    if T == 0:
        return [], []

    alpha = [{} for _ in range(T)]
    beta = [{} for _ in range(T)]
    for s in PHASES:
        obs = observed_phases[0]
        alpha[0][s] = (1.0 / len(PHASES)) * confusion.get(s, {}).get(obs, 1.0 / len(PHASES))
    norm = sum(alpha[0].values())
    for s in PHASES:
        alpha[0][s] /= norm

    for t in range(1, T):
        obs = observed_phases[t]
        for s in PHASES:
            alpha[t][s] = sum(
                alpha[t - 1][s_prev] * transition_row.get(s_prev, {}).get(s, 0.0)
                for s_prev in PHASES
            ) * confusion.get(s, {}).get(obs, 1.0 / len(PHASES))
        norm = sum(alpha[t].values())
        for s in PHASES:
            alpha[t][s] /= norm

    beta[T - 1] = {s: 1.0 for s in PHASES}
    for t in range(T - 2, -1, -1):
        obs_next = observed_phases[t + 1]
        for s in PHASES:
            beta[t][s] = sum(
                transition_row.get(s, {}).get(s_next, 0.0)
                * confusion.get(s_next, {}).get(obs_next, 1.0 / len(PHASES))
                * beta[t + 1][s_next]
                for s_next in PHASES
            )
        norm = sum(beta[t].values())
        for s in PHASES:
            beta[t][s] /= norm

    gamma = []
    for t in range(T):
        tot = sum(alpha[t][s] * beta[t][s] for s in PHASES)
        gamma.append({s: (alpha[t][s] * beta[t][s] / tot) if tot else (1.0 / len(PHASES)) for s in PHASES})

    xi = []
    for t in range(T - 1):
        obs_next = observed_phases[t + 1]
        denom = 0.0
        for s in PHASES:
            for s_next in PHASES:
                denom += alpha[t][s] * transition_row.get(s, {}).get(s_next, 0.0) * confusion.get(s_next, {}).get(obs_next, 1.0 / len(PHASES)) * beta[t + 1][s_next]
        xi_t = defaultdict(float)
        for s in PHASES:
            for s_next in PHASES:
                xi_t[(s, s_next)] = (alpha[t][s] * transition_row.get(s, {}).get(s_next, 0.0) * confusion.get(s_next, {}).get(obs_next, 1.0 / len(PHASES)) * beta[t + 1][s_next]) / denom if denom else 0.0
        xi.append(dict(xi_t))
    return gamma, xi


def _segment_by_goal(goal_phase: list[tuple[Any, str]]) -> list[tuple[Any, list[str]]]:
    """Split goal_phase into segments of same goal; each segment is (goal_id, [phase, ...]). Skip non-goal steps (empty goal_id)."""
    if not goal_phase:
        return []
    segments = []
    cur_goal: Any = None
    cur_phases: list[str] = []
    for g, ph in goal_phase:
        if g:
            if g == cur_goal:
                cur_phases.append(ph)
            else:
                if cur_goal and cur_phases:
                    segments.append((cur_goal, cur_phases))
                cur_goal = g
                cur_phases = [ph]
        else:
            if cur_goal and cur_phases:
                segments.append((cur_goal, cur_phases))
            cur_goal = None
            cur_phases = []
    if cur_goal and cur_phases:
        segments.append((cur_goal, cur_phases))
    return segments


def _sub_segment_by_attempt(segments: list[tuple[Any, list[str]]]) -> list[tuple[Any, list[str]]]:
    """
    Split goal segments into sub-segments where each represents one attempt at the goal.
    An attempt is a sequence of 'continue' phases followed by a terminal phase ('succeeded' or 'failed').

    This is necessary because the Markov model assumes 'succeeded' and 'failed' are absorbing states,
    but in practice users can retry the same goal multiple times in one trajectory, creating segments
    like [continue, continue, succeeded, continue, failed]. This function splits such segments at
    each terminal phase to get: [continue, continue, succeeded] and [continue, failed].

    Args:
        segments: List of (goal_id, [phases]) where phases may contain multiple attempts

    Returns:
        List of (goal_id, [phases]) where each sub-segment is one attempt (no 'succeeded' or 'failed'
        in non-final positions)
    """
    sub_segments = []
    for goal_id, phases in segments:
        if not phases:
            continue

        # Split at each succeeded/failed state to create sub-segments
        current_sub = []
        for phase in phases:
            current_sub.append(phase)
            # If we hit a terminal state, end this sub-segment
            if phase in ('succeeded', 'failed'):
                if current_sub:
                    sub_segments.append((goal_id, current_sub))
                current_sub = []

        # Add any remaining phases as a sub-segment (shouldn't normally happen,
        # but handle the case where a segment ends mid-attempt)
        if current_sub:
            sub_segments.append((goal_id, current_sub))

    return sub_segments


def run_em(
    trajectory_paths: list[Path],
    persona_ids: list[str],
    goal_ids: list[str],
    *,
    max_iters: int = 50,
    tol: float = 1e-5,
    estimate_confusion: bool = True,
    initial_error_rate: float = 0.05,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]] | None]:
    """
    Run EM to estimate nested transition matrices per (persona_id, goal_id) and optionally
    phase confusion matrix. persona_ids and goal_ids define the support (indices/ids used in data).

    Returns:
      nested: nested[persona_id][goal_id][phase_to] = dict of transition from each phase (row sums to 1)
      confusion: confusion[true_phase][obs_phase] or None if not estimated
    """
    pid_to_idx = {p: i for i, p in enumerate(persona_ids)}
    gid_to_idx = {g: i for i, g in enumerate(goal_ids)}

    # Load labels
    all_persona: list[int] = []
    all_goal_phase: list[list[tuple[Any, str]]] = []
    for path in trajectory_paths:
        pid, gp = _get_label_sequences(path)
        all_persona.append(pid_to_idx.get(pid, 0))
        all_goal_phase.append([(g, ph) for g, ph in gp])

    # Initialize: transition uniform, confusion with small error
    nested: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for pid in persona_ids:
        nested[pid] = {}
        for gid in goal_ids:
            nested[pid][gid] = {f: {t: 1.0 / len(PHASES) for t in PHASES} for f in PHASES}

    confusion: dict[str, dict[str, float]] = _default_confusion(initial_error_rate)
    prev_log_likelihood = -1e18

    logger.info("[EM] Starting: N=%d trajectories, %d personas, %d goals, max_iters=%d", len(trajectory_paths), len(persona_ids), len(goal_ids), max_iters)
    for iteration in range(max_iters):
        # E-step: expected transition counts from forward-backward per segment
        expected_counts: dict[tuple[str, str], dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
        confusion_counts: dict[tuple[str, str], float] = defaultdict(float)

        for traj_idx, (path, pid, gp) in enumerate(zip(trajectory_paths, all_persona, all_goal_phase)):
            persona_id = persona_ids[pid]
            segments = _segment_by_goal(gp)
            # Split segments into sub-segments (one attempt per sub-segment)
            sub_segments = _sub_segment_by_attempt(segments)
            for goal_id, phases in sub_segments:
                if not phases or goal_id not in goal_ids:
                    continue

                # For confusion matrix: use all segments including length-1
                for obs in phases:
                    # Use uniform prior for single observations (no transition info)
                    for true_s in PHASES:
                        confusion_counts[(true_s, obs)] += confusion.get(true_s, {}).get(obs, 1.0 / len(PHASES)) / len(PHASES)

                # For transition estimation: only use segments with length >= 2
                if len(phases) < 2:
                    continue

                # Build 3x3 transition: from "continue" use nested row (dict of float); succeeded/failed absorbing
                row_nested = nested[persona_id][goal_id]
                trans = {
                    "continue": dict(row_nested["continue"]),
                    "succeeded": {"succeeded": 1.0, "continue": 0.0, "failed": 0.0},
                    "failed": {"succeeded": 0.0, "continue": 0.0, "failed": 1.0},
                }
                gamma, xi = _forward_backward(phases, trans, confusion)
                key = (persona_id, goal_id)
                for t, xi_t in enumerate(xi):
                    for (s, s_next), v in xi_t.items():
                        expected_counts[key][(s, s_next)] += v
                # Don't double-count confusion matrix - already added above
                # for t, gam in enumerate(gamma):
                #     obs = phases[t]
                #     for true_s in PHASES:
                #         confusion_counts[(true_s, obs)] += gam[true_s]

        # M-step: normalize counts to get transition matrices and confusion
        for pid in persona_ids:
            for gid in goal_ids:
                key = (pid, gid)
                row_sums: dict[str, float] = defaultdict(float)
                for (f, t), c in expected_counts[key].items():
                    row_sums[f] += c
                for f in PHASES:
                    total = row_sums.get(f, 0.0)
                    if total > 0:
                        # Normalize using observed counts
                        for t in PHASES:
                            nested[pid][gid][f][t] = expected_counts[key].get((f, t), 0.0) / total
                    # If no data for this row, keep previous iteration's values (don't update)
                    # This prevents creating invalid all-zero transition matrices

        if estimate_confusion:
            row_sums_c: dict[str, float] = defaultdict(float)
            for (true_s, obs_s), c in confusion_counts.items():
                row_sums_c[true_s] += c
            for true_s in PHASES:
                total = row_sums_c.get(true_s, 1.0) or 1.0
                for obs_s in PHASES:
                    confusion[true_s][obs_s] = confusion_counts.get((true_s, obs_s), 0.0) / total

        # Log-likelihood (simplified: sum over trajectories of log P(obs))
        log_lik = 0.0
        for pid, gp in zip(all_persona, all_goal_phase):
            persona_id = persona_ids[pid]
            segments = _segment_by_goal(gp)
            # Use sub-segments for log-likelihood calculation (consistent with E-step)
            sub_segments = _sub_segment_by_attempt(segments)
            for goal_id, phases in sub_segments:
                if not phases or goal_id not in goal_ids:
                    continue
                trans = nested[persona_id][goal_id]
                gamma, _ = _forward_backward(phases, trans, confusion)
                for t, obs in enumerate(phases):
                    log_lik += sum(gamma[t][s] * (confusion.get(s, {}).get(obs, 1e-10)) for s in PHASES)
        if (iteration + 1) % 5 == 0 or iteration == 0:
            logger.info("[EM] iter %d/%d (log_lik=%.2f)", iteration + 1, max_iters, log_lik)
        if abs(log_lik - prev_log_likelihood) < tol:
            logger.info("[EM] Converged at iter %d", iteration + 1)
            break
        prev_log_likelihood = log_lik
    else:
        logger.info("[EM] Reached max_iters=%d without converging (log_lik still changing; try increasing --em-iters)", max_iters)

    # Convert nested to same shape as build_transition_matrix: nested[persona_id][goal_id] = {succeeded, continue, failed} as from "continue" only for compatibility
    result_nested: dict[str, dict[str, dict[str, float]]] = {}
    for pid in persona_ids:
        result_nested[pid] = {}
        for gid in goal_ids:
            result_nested[pid][gid] = {
                "succeeded": nested[pid][gid]["continue"].get("succeeded", 1.0 / 3),
                "continue": nested[pid][gid]["continue"].get("continue", 1.0 / 3),
                "failed": nested[pid][gid]["continue"].get("failed", 1.0 / 3),
            }
    return result_nested, confusion if estimate_confusion else None


def estimate_transitions_from_labels(
    trajectory_paths: list[Path],
    persona_ids: list[str],
    goal_ids: list[str],
    *,
    use_em: bool = True,
    em_iters: int = 50,
    initial_error_rate: float = 0.05,
) -> dict[str, Any]:
    """
    Estimate nested transition probabilities from labelled trajectories.
    If use_em=True, runs EM with optional confusion model; else uses raw counts (MLE).
    Returns dict compatible with theory (nested[persona_id][goal_id] = {succeeded, continue, failed}).
    """
    if use_em:
        nested, _ = run_em(
            trajectory_paths,
            persona_ids,
            goal_ids,
            max_iters=em_iters,
            estimate_confusion=True,
            initial_error_rate=initial_error_rate,
        )
        return {"nested": nested}
    # MLE from hard counts
    from expforge.estimator.counts import get_label_sequences_batch

    pid_to_idx = {p: i for i, p in enumerate(persona_ids)}
    gid_to_idx = {g: i for i, g in enumerate(goal_ids)}
    persona_indices, goal_phase = get_label_sequences_batch(
        trajectory_paths, pid_to_idx, gid_to_idx
    )
    goal_cluster_labels = [[g for g, _ in row] for row in goal_phase]
    goal_phase_with_ids = [
        [(goal_ids[g] if g >= 0 else '', ph) for g, ph in row] for row in goal_phase
    ]
    counts = compute_nested_transition_counts_from_labels(
        trajectory_paths,
        persona_indices,
        goal_phase_with_ids,
        goal_cluster_labels,
    )
    nested = {}
    for (pk, g), cdict in counts.items():
        pid = persona_ids[pk] if pk < len(persona_ids) else persona_ids[0]
        gid = goal_ids[g] if g < len(goal_ids) else goal_ids[0]
        if pid not in nested:
            nested[pid] = {}
        from_continue = sum(cdict.get(("continue", t), 0) for t in PHASES)
        nested[pid][gid] = {
            "succeeded": cdict.get(("continue", "succeeded"), 0) / (from_continue or 1),
            "continue": cdict.get(("continue", "continue"), 0) / (from_continue or 1),
            "failed": cdict.get(("continue", "failed"), 0) / (from_continue or 1),
        }
    return {"nested": nested}
