"""Score every sample in an experiment; infer clusters; compute metrics; write metrics.yaml."""

from pathlib import Path
from typing import Any

import yaml

from expforge.scoring.persona_clustering import assign_persona_hard, assign_persona_soft
from expforge.scoring.goal_scoring import score_goal_phases_for_trajectory
from expforge.scoring.goal_clustering import segment_trajectory, cluster_goal_segments
from expforge.scoring.transition_computation import (
    compute_nested_transition_counts,
    compute_top_level_transition_counts,
    normalize_transition_counts,
)
from expforge.scoring.metric_computation import (
    expected_metrics_from_transitions,
    metric_confidence_interval,
)


def _load_outcomes(samples_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (sample_path, outcome) for each sample YAML."""
    results = []
    for p in sorted(samples_dir.glob("sample_*.yaml")):
        with p.open() as f:
            data = yaml.safe_load(f)
        results.append((p, data.get("outcome", "unknown")))
    return results


def run_experiment_scoring(
    experiment_id: str,
    *,
    base_dir: Path | str | None = None,
    n_personas: int = 5,
    n_goal_clusters: int = 6,
    confidence: float = 0.95,
) -> Path:
    """
    Score all samples in experiment/{experiment_id}/samples/.
    Writes experiment/{experiment_id}/metrics.yaml.
    Use same base_dir as simulator so one experiment dir has persona, goals, transitions, samples, metrics.
    """
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    exp_dir = base_dir / "experiment" / experiment_id
    samples_dir = exp_dir / "samples"
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.yaml"

    sample_outcomes = _load_outcomes(samples_dir)
    if not sample_outcomes:
        metrics = {"experiment_id": experiment_id, "error": "no samples found"}
        with metrics_path.open("w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)
        return metrics_path

    paths = [p for p, _ in sample_outcomes]
    outcomes = [o for _, o in sample_outcomes]

    persona_assignments = [assign_persona_hard(p, n_personas=n_personas) for p in paths]
    phase_labels = [score_goal_phases_for_trajectory(p) for p in paths]
    all_segments = []
    segment_ranges = []
    for p in paths:
        segs = segment_trajectory(p)
        start = len(all_segments)
        all_segments.extend(segs)
        segment_ranges.append((start, len(all_segments)))
    global_clusters = cluster_goal_segments(all_segments, n_clusters=n_goal_clusters)
    goal_cluster_labels = []
    for idx, path in enumerate(paths):
        segs = segment_trajectory(path)
        phases = phase_labels[idx]
        s, e = segment_ranges[idx]
        traj_clusters = global_clusters[s:e]
        msg_clusters = []
        for i in range(len(phases)):
            for seg_idx, (start, end, _) in enumerate(segs):
                if start <= i <= end:
                    msg_clusters.append(traj_clusters[seg_idx] if seg_idx < len(traj_clusters) else 0)
                    break
            else:
                msg_clusters.append(0)
        goal_cluster_labels.append(msg_clusters)

    nested_counts = compute_nested_transition_counts(
        paths, persona_assignments, phase_labels, goal_cluster_labels
    )
    top_counts = compute_top_level_transition_counts(paths, outcomes)
    nested_probs = {
        k: normalize_transition_counts(v)
        for k, v in nested_counts.items()
    }

    metrics_by_name = {}
    for metric in ("publish", "subscribe", "finished", "abandoned"):
        point, lower, upper = metric_confidence_interval(outcomes, metric, confidence)
        metrics_by_name[metric] = {"point": point, "lower": lower, "upper": upper}

    persona_weights = [0.0] * n_personas
    for k in persona_assignments:
        if 0 <= k < n_personas:
            persona_weights[k] += 1
    n = len(persona_assignments)
    persona_weights = [w / n for w in persona_weights]

    expected = expected_metrics_from_transitions(
        nested_probs, persona_weights
    )

    metrics = {
        "experiment_id": experiment_id,
        "n_samples": len(paths),
        "n_personas": n_personas,
        "n_goal_clusters": n_goal_clusters,
        "persona_weights": persona_weights,
        "metrics": {
            m: {
                "observed": metrics_by_name.get(m, {}),
                "expected": expected.get(m),
            }
            for m in ["publish", "subscribe", "finished", "abandoned"]
        },
        "confidence": confidence,
    }
    with metrics_path.open("w") as f:
        yaml.safe_dump(metrics, f, default_flow_style=False, sort_keys=False)
    return metrics_path
