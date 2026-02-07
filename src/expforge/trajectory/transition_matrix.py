"""
Build transition matrix for an experiment from persona set + goal set.
Written to simulator/experiment/<experiment_id>/transitions.yaml (by default or when using build_and_save_transition_matrix).
"""

from pathlib import Path
from typing import Any

from expforge.persona import PersonaSet
from expforge.goal import GoalSet


# Same formula as TransitionSampler.sample_nested; wider range (0.25–1.0) so experiments can vary ~0–30%
# Coefficients: base + determined + tool_quality (tool quality has more weight so q1 vs q2 impact is visible)
P_FAILED = 0.3
P_CONTINUE_MAX = 0.12


def _nested_probs_for(persona_determined: float, tool_quality: float) -> dict[str, float]:
    """Compute nested outcome probabilities ensuring they sum to 1.0. p_success in [0.25, 1.0] for system variance."""
    p_success = 0.20 + 0.45 * persona_determined + 0.35 * tool_quality
    p_success = max(0.0, min(1.0, p_success))
    p_failed = P_FAILED
    p_continue_raw = 1.0 - p_success - p_failed
    p_continue = min(P_CONTINUE_MAX, max(0.0, p_continue_raw))

    # Adjust p_success if we capped p_continue
    if p_continue_raw > P_CONTINUE_MAX:
        p_success = 1.0 - p_failed - p_continue
    # Handle case where p_success + p_failed > 1.0 (p_continue_raw < 0)
    elif p_continue_raw < 0:
        # Normalize p_success and p_failed to sum to 1.0, keeping their relative proportions
        total = p_success + p_failed
        p_success = p_success / total
        p_failed = p_failed / total
        p_continue = 0.0

    return {"succeeded": round(p_success, 4), "failed": round(p_failed, 4), "continue": round(p_continue, 4)}


# Default top-level outcome weights: publish more likely than subscribe (users publish more often; subscribe once)
# Goals and other options use 1.0 if not listed.
DEFAULT_OUTCOME_WEIGHTS: dict[str, float] = {
    "publish": 2.0,
    "subscribe": 1.0,
    "finished": 2.0,
    "abandoned": 2.0,
}


def build_transition_matrix(
    persona_set: PersonaSet,
    goal_set: GoalSet,
    *,
    outcome_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Build the full transition matrix for the experiment from persona set and goal set.
    Returns a dict suitable for YAML: nested[persona_id][goal_id] -> {succeeded, continue, failed};
    top_level: from_start, from_goal_succeeded, from_goal_failed, from_goal_continue.

    outcome_weights: optional weights for top-level outcomes (publish, subscribe, finished, abandoned).
    Default gives publish weight 2 and subscribe 1 so P(ever publish) > P(ever subscribe) (more realistic).
    """
    outcome_weights = outcome_weights or DEFAULT_OUTCOME_WEIGHTS
    goal_ids = [g.id for g in goal_set.goals]
    n_goals = len(goal_ids)

    nested: dict[str, dict[str, dict[str, float]]] = {}
    for p in persona_set.personas:
        nested[p.id] = {}
        for g in goal_set.goals:
            tq = goal_set.tool_quality_for_goal(g.id)
            nested[p.id][g.id] = _nested_probs_for(p.determined, tq)

    from_start = {gid: round(1.0 / n_goals, 4) for gid in goal_ids} if goal_ids else {}
    next_succeeded = goal_ids + ["publish", "subscribe", "finished"]
    next_failed = goal_ids + ["abandoned"]
    next_continue = goal_ids + ["publish", "subscribe", "finished", "abandoned"]
    from_publish = goal_ids + ["subscribe", "finished", "abandoned"]
    from_subscribe = goal_ids + ["publish", "finished", "abandoned"]

    def _weighted_probs(options: list[str]) -> dict[str, float]:
        w_sum = sum(outcome_weights.get(s, 1.0) for s in options)
        return {s: round(outcome_weights.get(s, 1.0) / w_sum, 4) for s in options}

    top_level = {
        "from_start": from_start,
        "from_goal_succeeded": _weighted_probs(next_succeeded),
        "from_goal_failed": _weighted_probs(next_failed),
        "from_goal_continue": _weighted_probs(next_continue),
        "from_publish": _weighted_probs(from_publish),
        "from_subscribe": _weighted_probs(from_subscribe),
    }

    return {
        "experiment_id": persona_set.experiment_id,
        "nested": nested,
        "top_level": top_level,
    }


def write_transition_matrix(matrix: dict[str, Any], path: Path | str) -> None:
    """Write transition matrix to a YAML file."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(matrix, f, default_flow_style=False, sort_keys=False)


def build_and_save_transition_matrix(
    persona_set: PersonaSet,
    goal_set: GoalSet,
    experiment_id: str,
    *,
    base_dir: Path | str | None = None,
) -> Path:
    """
    Build transition matrix from persona set + goal set and save to
    base_dir/experiment/<experiment_id>/transitions.yaml.
    Returns the path to the written file.
    """
    from expforge.verifier.io import DEFAULT_EXPERIMENTS_DIR
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    out_dir = base_dir / "experiment" / experiment_id
    path = out_dir / "transitions.yaml"
    matrix = build_transition_matrix(persona_set, goal_set)
    matrix["experiment_id"] = experiment_id
    write_transition_matrix(matrix, path)
    return path
