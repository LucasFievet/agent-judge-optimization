"""
Verifier EM: end-to-end check that the EM estimator recovers transition probabilities
from (optionally noisy) labelled trajectories.
"""

import logging
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from expforge.verifier.io import (
    load_experiment,
    load_existing_sample_paths,
    experiment_dir,
    DEFAULT_EXPERIMENTS_DIR,
)
from expforge.estimator.em import estimate_transitions_from_labels
from expforge.estimator.em import _segment_by_goal, _sub_segment_by_attempt  # for segment count diagnostic
from expforge.noise import add_label_noise_to_experiment


@dataclass
class EMVerificationResult:
    """Result of EM recovery verification."""

    experiment_id: str
    n_samples: int
    phase_error_rate: float
    persona_error_rate: float
    max_abs_diff: float
    mean_abs_diff: float
    tolerance: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


def _load_ground_truth_nested(transitions_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load nested transition matrix from transitions.yaml."""
    if not transitions_path.is_file():
        return {}
    with open(transitions_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("nested", {})


def _nested_max_and_mean_abs_diff(
    estimated: dict[str, dict[str, dict[str, float]]],
    ground_truth: dict[str, dict[str, dict[str, float]]],
) -> tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff) over (persona, goal, phase) entries."""
    diffs = []
    for pid, by_goal in ground_truth.items():
        for gid, probs in by_goal.items():
            est_row = (estimated.get(pid) or {}).get(gid) or {}
            for ph, true_val in probs.items():
                est_val = est_row.get(ph, 0.0)
                diffs.append(abs(float(est_val) - float(true_val)))
    if not diffs:
        return 0.0, 0.0
    return max(diffs), sum(diffs) / len(diffs)


def _nested_diff_details(
    estimated: dict[str, dict[str, dict[str, float]]],
    ground_truth: dict[str, dict[str, dict[str, float]]],
    top_k: int = 15,
) -> list[dict]:
    """
    Return list of entries (persona, goal, phase, P_true, P_est, diff) sorted by |diff| desc.
    Useful for understanding where the estimate disagrees most with ground truth.
    """
    entries = []
    for pid, by_goal in ground_truth.items():
        for gid, probs in by_goal.items():
            est_row = (estimated.get(pid) or {}).get(gid) or {}
            for ph, true_val in probs.items():
                est_val = est_row.get(ph, 0.0)
                diff = float(est_val) - float(true_val)
                entries.append({
                    "persona": pid,
                    "goal": gid,
                    "phase": ph,
                    "P_true": float(true_val),
                    "P_est": float(est_val),
                    "diff": diff,
                })
    entries.sort(key=lambda e: abs(e["diff"]), reverse=True)
    return entries[:top_k]


def _trajectory_length_distribution(
    trajectory_paths: list[Path],
) -> dict:
    """Return min, max, mean, and histogram buckets of trajectory length (number of steps)."""
    from expforge.trajectory.io import load_trajectory

    lengths = []
    for path in trajectory_paths:
        try:
            traj = load_trajectory(path)
            lengths.append(len(traj.steps))
        except Exception:
            continue
    if not lengths:
        return {"n": 0, "min": 0, "max": 0, "mean": 0.0, "histogram": []}
    lengths = sorted(lengths)
    n = len(lengths)
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 100)]
    hist = []
    for lo, hi in buckets:
        count = sum(1 for L in lengths if lo <= L <= hi)
        if count:
            hist.append({"range": f"{lo}-{hi}", "count": count})
    return {
        "n": n,
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / n, 2),
        "median": lengths[n // 2] if n else 0,
        "histogram": hist,
    }


def _outcome_counts(trajectory_paths: list[Path]) -> dict[str, int]:
    """Count how many trajectories ended in publish, subscribe, finished, abandoned."""
    from expforge.trajectory.io import load_trajectory

    counts: dict[str, int] = {}
    for path in trajectory_paths:
        try:
            traj = load_trajectory(path)
            o = traj.outcome or "unknown"
            counts[o] = counts.get(o, 0) + 1
        except Exception:
            pass
    return counts


def _transition_type_counts(
    trajectory_paths: list[Path],
    persona_ids: list[str],
    goal_ids: list[str],
) -> dict:
    """
    Count (from_phase, to_phase) within goal segments of length >= 2 (nested chain only).
    Also total_steps across all trajectories (for context: N trajs × mean length ≈ total steps).
    """
    from expforge.estimator.counts import _get_label_sequences
    from expforge.trajectory.io import load_trajectory

    pid_to_idx = {p: i for i, p in enumerate(persona_ids)}
    counts: dict[tuple[str, str], int] = {}
    per_goal: dict[str, dict[tuple[str, str], int]] = {gid: {} for gid in goal_ids}

    total_steps = 0
    for path in trajectory_paths:
        try:
            traj = load_trajectory(path)
            total_steps += len(traj.steps)
        except Exception:
            pass

    for path in trajectory_paths:
        pid, gp = _get_label_sequences(path)
        persona_id = persona_ids[pid_to_idx.get(pid, 0)] if pid in pid_to_idx else persona_ids[0]
        segments = _segment_by_goal(gp)
        # Apply sub-segmentation to split at terminal states (succeeded/failed)
        sub_segments = _sub_segment_by_attempt(segments)
        for goal_id, phases in sub_segments:
            if not phases or goal_id not in goal_ids or len(phases) < 2:
                continue
            for i in range(len(phases) - 1):
                key = (phases[i], phases[i + 1])
                counts[key] = counts.get(key, 0) + 1
                if goal_id in per_goal:
                    per_goal[goal_id][key] = per_goal[goal_id].get(key, 0) + 1

    return {
        "global": counts,
        "per_goal": per_goal,
        "total_steps": total_steps,
        "total_nested_transitions": sum(counts.values()),
    }


def _relabel_stats(trajectory_paths: list[Path]) -> dict:
    """
    From trajectories that have label_true (noisy run), count phase flips and persona flips.
    Returns n_trajectories_checked, n_phase_flips, n_phase_total, n_persona_flips.
    """
    from expforge.trajectory.io import load_trajectory

    n_phase_flips = 0
    n_phase_total = 0
    n_persona_flips = 0
    n_with_label_true = 0
    for path in trajectory_paths:
        try:
            traj = load_trajectory(path)
            if not traj.label or "label_true" not in traj.label:
                continue
            n_with_label_true += 1
            true_label = traj.label["label_true"]
            noisy_gp = traj.label.get("goal_phase", [])
            true_gp = true_label.get("goal_phase", [])
            if traj.label.get("persona_id") != true_label.get("persona_id"):
                n_persona_flips += 1
            for noisy_g, true_g in zip(noisy_gp, true_gp):
                n_phase_total += 1
                if noisy_g.get("phase") != true_g.get("phase"):
                    n_phase_flips += 1
        except Exception:
            continue
    return {
        "n_with_label_true": n_with_label_true,
        "n_phase_flips": n_phase_flips,
        "n_phase_total": n_phase_total,
        "n_persona_flips": n_persona_flips,
        "phase_flip_rate": round(n_phase_flips / n_phase_total, 4) if n_phase_total else 0.0,
    }


def _segment_counts_per_cell(
    trajectory_paths: list[Path],
    persona_ids: list[str],
    goal_ids: list[str],
) -> dict[tuple[str, str], tuple[int, int]]:
    """
    For each (persona_id, goal_id), return (n_segments, n_segments_with_transitions).
    Only segments of length >= 2 contribute to transition estimates; length-1 segments
    only inform the confusion matrix. So low n_segments_with_transitions explains large errors.
    """
    from expforge.estimator.counts import _get_label_sequences

    pid_to_idx = {p: i for i, p in enumerate(persona_ids)}
    counts: dict[tuple[str, str], list[int]] = {}  # (pid, gid) -> [lengths of segments]

    for path in trajectory_paths:
        pid, gp = _get_label_sequences(path)
        persona_id = persona_ids[pid_to_idx.get(pid, 0)] if pid in pid_to_idx else persona_ids[0]
        segments = _segment_by_goal(gp)
        # Apply sub-segmentation to split at terminal states
        sub_segments = _sub_segment_by_attempt(segments)
        for goal_id, phases in sub_segments:
            if not phases or goal_id not in goal_ids:
                continue
            key = (persona_id, goal_id)
            if key not in counts:
                counts[key] = []
            counts[key].append(len(phases))

    out = {}
    for key, lengths in counts.items():
        n_seg = len(lengths)
        n_with_trans = sum(1 for L in lengths if L >= 2)
        out[key] = (n_seg, n_with_trans)
    return out


def run_em_verification(
    experiment_id: str,
    *,
    base_dir: Path | str | None = None,
    n_samples: int = 500,
    phase_error_rate: float = 0.05,
    persona_error_rate: float = 0.02,
    seed: int = 42,
    tolerance: float = 0.08,
    use_em: bool = True,
    em_iters: int = 30,
    run_simulator_if_missing: bool = True,
    save_plot_path: Path | str | None = None,
) -> EMVerificationResult:
    """
    Run end-to-end EM verification: (optionally generate and) load trajectories with labels,
    optionally add noise, run EM to estimate nested transition matrix, compare to ground-truth.
    Pass if max_abs_diff <= tolerance.

    Sanity check: run with --phase-error 0 --persona-error 0 (and optionally --no-em) to verify
    recovery without label noise; large errors there would indicate a bug.
    """
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    exp_dir = experiment_dir(base_dir, experiment_id)
    transitions_path = exp_dir / "transitions.yaml"
    samples_dir = exp_dir / "samples"

    logger.info("[em verifier] Loading ground truth from %s", transitions_path)
    ground_truth = _load_ground_truth_nested(transitions_path)
    if not ground_truth:
        return EMVerificationResult(
            experiment_id=experiment_id,
            n_samples=0,
            phase_error_rate=phase_error_rate,
            persona_error_rate=persona_error_rate,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            tolerance=tolerance,
            passed=False,
            details={"error": "no transitions.yaml or empty nested"},
        )

    persona_ids = list(ground_truth.keys())
    goal_ids = list(next(iter(ground_truth.values()), {}).keys()) if ground_truth else []
    if not goal_ids and ground_truth:
        goal_ids = list(ground_truth.get(persona_ids[0], {}).keys())
    logger.info("[em verifier] %d personas, %d goals", len(persona_ids), len(goal_ids))

    sample_paths = load_existing_sample_paths(base_dir, experiment_id)
    logger.info("[em verifier] Found %d existing samples (need %d)", len(sample_paths), n_samples)
    if len(sample_paths) < n_samples and run_simulator_if_missing:
        logger.info("[em verifier] Running simulator to generate %d samples...", n_samples)
        from expforge.simulator.experiment_simulator import run_simulator
        run_simulator(
            experiment_id,
            n_samples,
            base_dir=base_dir,
            seed=seed,
            reuse_config=True,
            use_llm=False,
        )
        sample_paths = load_existing_sample_paths(base_dir, experiment_id)
        logger.info("[em verifier] Simulator done, %d samples", len(sample_paths))

    sample_paths = sample_paths[:n_samples]
    if len(sample_paths) < n_samples:
        return EMVerificationResult(
            experiment_id=experiment_id,
            n_samples=len(sample_paths),
            phase_error_rate=phase_error_rate,
            persona_error_rate=persona_error_rate,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            tolerance=tolerance,
            passed=False,
            details={"error": f"only {len(sample_paths)} samples, need {n_samples}"},
        )

    # Optionally add noise (writes to samples_noisy and we use those paths)
    if phase_error_rate > 0 or persona_error_rate > 0:
        logger.info("[em verifier] Adding label noise (phase=%.2f, persona=%.2f) to first %d samples...", phase_error_rate, persona_error_rate, n_samples)
        add_label_noise_to_experiment(
            exp_dir,
            persona_ids,
            phase_error_rate=phase_error_rate,
            persona_error_rate=persona_error_rate,
            seed=seed,
            in_place=False,
            max_samples=n_samples,
        )
        sample_paths = list((exp_dir / "samples_noisy").glob("sample_*.yaml"))
        sample_paths.sort(key=lambda p: int(re.search(r"\d+", p.stem).group(0)) if re.search(r"\d+", p.stem) else 0)
        sample_paths = sample_paths[:n_samples]
        logger.info("[em verifier] Using %d noisy samples", len(sample_paths))

    logger.info("[em verifier] Estimating transitions (%s, max_iters=%d)...", "EM" if use_em else "MLE", em_iters)

    # Debug logging: show raw transitions for persona_0/write_abstract (highest error cell)
    if not use_em and "persona_0" in persona_ids and "write_abstract" in goal_ids:
        logger.info("[em verifier] Debug: showing raw transitions for persona_0/write_abstract...")
        from expforge.estimator.counts import get_label_sequences_batch, compute_nested_transition_counts_from_labels
        pid_to_idx = {p: i for i, p in enumerate(persona_ids)}
        gid_to_idx = {g: i for i, g in enumerate(goal_ids)}
        persona_indices, goal_phase = get_label_sequences_batch(
            sample_paths, pid_to_idx, gid_to_idx
        )
        goal_cluster_labels = [[g for g, _ in row] for row in goal_phase]
        goal_phase_with_ids = [
            [(goal_ids[g] if g >= 0 else '', ph) for g, ph in row] for row in goal_phase
        ]
        # Use integer keys for MLE path (persona_idx, goal_idx)
        p0_idx = persona_ids.index("persona_0")
        wa_idx = goal_ids.index("write_abstract")
        debug_counts = compute_nested_transition_counts_from_labels(
            sample_paths,
            persona_indices,
            goal_phase_with_ids,
            goal_cluster_labels,
            debug_cell=(p0_idx, wa_idx),
        )
        logger.info("[em verifier] Debug counts returned: %s", {k: dict(v) for k, v in debug_counts.items() if k == (p0_idx, wa_idx)})

    result = estimate_transitions_from_labels(
        sample_paths,
        persona_ids,
        goal_ids,
        use_em=use_em,
        em_iters=em_iters,
        initial_error_rate=phase_error_rate,  # Use actual phase error rate for EM initialization
    )
    estimated = result.get("nested", {})
    logger.info("[em verifier] Comparing to ground truth...")
    max_diff, mean_diff = _nested_max_and_mean_abs_diff(estimated, ground_truth)
    diff_details = _nested_diff_details(estimated, ground_truth, top_k=15)
    segment_counts = _segment_counts_per_cell(sample_paths, persona_ids, goal_ids)
    traj_length_dist = _trajectory_length_distribution(sample_paths)
    transition_type_counts = _transition_type_counts(sample_paths, persona_ids, goal_ids)
    outcome_counts = _outcome_counts(sample_paths)
    relabel_stats = _relabel_stats(sample_paths)
    passed = max_diff <= tolerance

    if save_plot_path:
        try:
            from expforge.verifier.em_heatmaps import plot_em_heatmaps
            out_path = plot_em_heatmaps(
                ground_truth,
                estimated,
                Path(save_plot_path),
                persona_ids=persona_ids,
                goal_ids=goal_ids,
                transitions_path=transitions_path,
            )
            logger.info("[em verifier] Saved heatmaps to %s", out_path)
        except Exception as e:
            logger.warning("[em verifier] Could not save heatmaps: %s", e)

    return EMVerificationResult(
        experiment_id=experiment_id,
        n_samples=len(sample_paths),
        phase_error_rate=phase_error_rate,
        persona_error_rate=persona_error_rate,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        tolerance=tolerance,
        passed=passed,
        details={
            "estimated_keys": list(estimated.keys()),
            "ground_truth_keys": list(ground_truth.keys()),
            "diff_details": diff_details,
            "segment_counts": segment_counts,
            "traj_length_dist": traj_length_dist,
            "transition_type_counts": transition_type_counts,
            "outcome_counts": outcome_counts,
            "relabel_stats": relabel_stats,
        },
    )
