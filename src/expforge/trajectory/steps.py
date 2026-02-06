"""Trajectory step and full trajectory data structures."""

from dataclasses import dataclass, field


@dataclass
class TrajectoryStep:
    """A single turn: user message, tools used, agent response (with actual artifacts), and source state for this turn."""

    user_message: str
    tools_used: list[str]
    agent_message: str
    # Source state (at the start of this turn); next state is the next step's top_level_state or outcome
    top_level_state: str
    nested_state: str | None
    quality: float
    terminal: bool = False


@dataclass
class Trajectory:
    """Full trajectory: list of steps and terminal outcome."""

    trajectory_id: str
    persona_id: str
    sample_goal: str = ""  # Overarching goal/plan for this sample (LLM-generated)
    steps: list[TrajectoryStep] = field(default_factory=list)
    outcome: str | None = None  # publish | subscribe | finished | abandoned

    def to_dict(self) -> dict:
        """Serialize for YAML storage. sample_goal first so it appears at the top of the file."""
        steps_data = [
            {
                "user_message": s.user_message,
                "tools_used": s.tools_used,
                "agent_message": s.agent_message,
                "top_level_state": s.top_level_state,
                "nested_state": s.nested_state,
                "quality": s.quality,
                "terminal": s.terminal,
            }
            for s in self.steps
        ]
        out = {
            "sample_goal": self.sample_goal or "",
            "trajectory_id": self.trajectory_id,
            "persona_id": self.persona_id,
            "outcome": self.outcome,
            "steps": steps_data,
        }
        return out
