"""
Build transition matrix for an experiment from persona set + goal set.
Written to simulator/experiment/<experiment_id>/transitions.yaml (by default or when using build_and_save_transition_matrix).
"""

from pathlib import Path
from typing import Any

from expforge.persona import PersonaSet
from expforge.goal import GoalSet


# Same formula as TransitionSampler.sample_nested (tuned for ~10-message trajectories)
P_FAILED = 0.3
P_CONTINUE_MAX = 0.12


def _nested_probs_for(persona_determined: float, tool_quality: float) -> dict[str, float]:
    p_success = 0.5 + 0.35 * persona_determined + 0.15 * tool_quality
    p_success = max(0.0, min(1.0, p_success))
    p_failed = P_FAILED
    p_continue_raw = 1.0 - p_success - p_failed
    p_continue = min(P_CONTINUE_MAX, max(0.0, p_continue_raw))
    if p_continue_raw > P_CONTINUE_MAX:
        p_success = 1.0 - p_failed - p_continue
    return {"succeeded": round(p_success, 4), "failed": round(p_failed, 4), "continue": round(p_continue, 4)}


def build_transition_matrix(persona_set: PersonaSet, goal_set: GoalSet) -> dict[str, Any]:
    """
    Build the full transition matrix for the experiment from persona set and goal set.
    Returns a dict suitable for YAML: nested[persona_id][goal_id] -> {succeeded, continue, failed};
    top_level: from_start, from_goal_succeeded, from_goal_failed, from_goal_continue.
    """
    goal_ids = [g.id for g in goal_set.goals]
    n_goals = len(goal_ids)

    nested: dict[str, dict[str, dict[str, float]]] = {}
    for p in persona_set.personas:
        nested[p.id] = {}
        for g in goal_set.goals:
            tq = goal_set.tool_quality_for_goal(g.id)
            nested[p.id][g.id] = _nested_probs_for(p.determined, tq)

    # Top-level: bias toward finished/abandoned (weight 2) vs others (1) for ~10-message trajectories
    from_start = {gid: round(1.0 / n_goals, 4) for gid in goal_ids} if goal_ids else {}
    next_succeeded = goal_ids + ["publish", "subscribe", "finished"]
    next_failed = goal_ids + ["abandoned"]
    next_continue = goal_ids + ["publish", "subscribe", "finished", "abandoned"]
    def _weighted_probs(options: list[str]) -> dict[str, float]:
        w_sum = sum(2.0 if s in ("finished", "abandoned") else 1.0 for s in options)
        return {s: round((2.0 if s in ("finished", "abandoned") else 1.0) / w_sum, 4) for s in options}
    top_level = {
        "from_start": from_start,
        "from_goal_succeeded": _weighted_probs(next_succeeded),
        "from_goal_failed": _weighted_probs(next_failed),
        "from_goal_continue": _weighted_probs(next_continue),
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
    simulator/experiment/<experiment_id>/transitions.yaml.
    base_dir defaults to the simulator package dir so the file lives with the experiment config.
    Returns the path to the written file.
    """
    if base_dir is None:
        # Default: simulator/experiment/<experiment_id>/transitions.yaml
        base_dir = Path(__file__).resolve().parent.parent / "simulator"
    base_dir = Path(base_dir)
    out_dir = base_dir / "experiment" / experiment_id
    path = out_dir / "transitions.yaml"
    matrix = build_transition_matrix(persona_set, goal_set)
    matrix["experiment_id"] = experiment_id
    write_transition_matrix(matrix, path)
    return path
