"""Score each message with g(trajectory, message) -> start | continue | success | fail."""

from pathlib import Path
from typing import Any

# Stub: g(trajectory, message) produces phase of current goal for that message


def score_goal_phase(trajectory: dict[str, Any], message_index: int) -> str:
    """
    Return phase label for the message: start, continue, success, or fail.

    Default: round-robin stub. Replace with LLM that takes (trajectory, message).
    """
    steps = trajectory.get("steps", [])
    if message_index < 0 or message_index >= len(steps):
        return "continue"
    # Stub: simple pattern
    phases = ["start", "continue", "continue", "success"]
    return phases[message_index % len(phases)]


def score_goal_phases_for_trajectory(trajectory_path: Path) -> list[str]:
    """Load trajectory and return phase label for each user message."""
    import yaml

    with trajectory_path.open() as f:
        data = yaml.safe_load(f)
    steps = data.get("steps", [])
    return [score_goal_phase(data, i) for i in range(len(steps))]
