"""Load/save trajectory from/to YAML."""

from pathlib import Path

from expforge.trajectory.steps import Trajectory, TrajectoryStep


def load_trajectory(path: Path | str) -> Trajectory:
    """Load a single trajectory from YAML."""
    import yaml

    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    steps = [
        TrajectoryStep(
            user_message=s["user_message"],
            tools_used=s.get("tools_used", []),
            agent_message=s.get("agent_message", s.get("assistant_message", "")),
            top_level_state=s["top_level_state"],
            nested_state=s.get("nested_state"),
            quality=float(s.get("quality", 0.5)),
            terminal=s.get("terminal", False),
        )
        for s in data["steps"]
    ]
    return Trajectory(
        trajectory_id=data["trajectory_id"],
        persona_id=data["persona_id"],
        sample_goal=data.get("sample_goal", ""),
        steps=steps,
        outcome=data.get("outcome"),
    )


def save_trajectory(trajectory: Trajectory, path: Path | str) -> None:
    """Save a trajectory to YAML."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(trajectory.to_dict(), f, default_flow_style=False, sort_keys=False)
