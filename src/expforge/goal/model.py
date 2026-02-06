"""Data structures for goal and tool configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    """A tool used by goals. quality impacts all goals using this tool equally."""

    id: str
    quality: float
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalSpec:
    """Specification for a single goal (e.g. write abstract, create table)."""

    id: str
    name: str
    tools: list[str]  # tool ids from GoalSet.tools
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalSet:
    """Configuration for the set of goals, tools, and top-level transitions."""

    experiment_id: str
    goals: list[GoalSpec]
    tools: list[Tool] = field(default_factory=list)
    terminal_states: list[str] = field(default_factory=lambda: ["finished", "abandoned"])
    outcome_states: list[str] = field(default_factory=lambda: ["publish", "subscribe"])
    # If set, "finished" is only allowed once all these goals have succeeded at least once.
    required_goals_for_finish: list[str] = field(default_factory=list)

    def tool_by_id(self, tool_id: str) -> Tool | None:
        """Return the Tool with the given id, or None."""
        return next((t for t in self.tools if t.id == tool_id), None)

    def tool_quality_for_goal(self, goal_id: str) -> float:
        """Effective tool quality for a goal: mean quality of tools used by that goal."""
        goal = next((g for g in self.goals if g.id == goal_id), None)
        if not goal or not goal.tools:
            return 0.5
        qualities = []
        for tid in goal.tools:
            t = self.tool_by_id(tid)
            if t is not None:
                qualities.append(t.quality)
        return sum(qualities) / len(qualities) if qualities else 0.5
