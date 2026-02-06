"""Load/save goal set (goals + tools) from/to YAML."""

from pathlib import Path

from expforge.goal.model import GoalSet, GoalSpec, Tool


def load_goal_set(path: Path | str) -> GoalSet:
    """Load a GoalSet from a YAML file."""
    import yaml

    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    experiment_id = data.get("experiment_id", path.stem)
    tools = [
        Tool(
            id=t["id"],
            quality=float(t["quality"]),
            name=t.get("name", ""),
            meta=t.get("meta", {}),
        )
        for t in data.get("tools", [])
    ]
    goals = [
        GoalSpec(
            id=g["id"],
            name=g["name"],
            tools=list(g.get("tools", [])),
            meta=g.get("meta", {}),
        )
        for g in data["goals"]
    ]
    return GoalSet(
        experiment_id=experiment_id,
        goals=goals,
        tools=tools,
        terminal_states=data.get("terminal_states", ["finished", "abandoned"]),
        outcome_states=data.get("outcome_states", ["publish", "subscribe"]),
        required_goals_for_finish=data.get("required_goals_for_finish", []),
    )


def save_goal_set(goal_set: GoalSet, path: Path | str) -> None:
    """Save a GoalSet to a YAML file."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "experiment_id": goal_set.experiment_id,
        "tools": [
            {
                "id": t.id,
                "quality": t.quality,
                **({"name": t.name} if t.name else {}),
                **({"meta": t.meta} if t.meta else {}),
            }
            for t in goal_set.tools
        ],
        "goals": [
            {
                "id": g.id,
                "name": g.name,
                "tools": g.tools,
                **({"meta": g.meta} if g.meta else {}),
            }
            for g in goal_set.goals
        ],
        "terminal_states": goal_set.terminal_states,
        "outcome_states": goal_set.outcome_states,
        **({"required_goals_for_finish": goal_set.required_goals_for_finish} if goal_set.required_goals_for_finish else {}),
    }
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
