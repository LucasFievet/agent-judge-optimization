"""Sample next state from nested and top-level chains."""

import random
from typing import Any

from expforge.persona import PersonaSpec
from expforge.goal import GoalSet


class TransitionSampler:
    """Draws next top-level and nested state given current state, persona, and goals."""

    def __init__(self, goal_set: GoalSet, seed: int | None = None) -> None:
        self.goal_set = goal_set
        if seed is not None:
            random.seed(seed)
        self._goal_ids = [g.id for g in goal_set.goals]

    def sample_nested(self, persona: PersonaSpec, goal_id: str) -> str:
        """Sample nested state (succeeded, continue, failed) for the goal.
        Probabilities tuned so trajectories average ~10 messages (low p_continue)."""
        goal = next((g for g in self.goal_set.goals if g.id == goal_id), None)
        if not goal:
            return "continue"
        tool_quality = self.goal_set.tool_quality_for_goal(goal_id)
        p_success = 0.5 + 0.35 * persona.determined + 0.15 * tool_quality
        p_success = max(0.0, min(1.0, p_success))
        p_failed = 0.3
        p_continue_raw = 1.0 - p_success - p_failed
        # Cap continue so we resolve (succeeded/failed) often → shorter trajectories
        p_continue = min(0.12, max(0.0, p_continue_raw))
        if p_continue_raw > 0.12:
            p_success = 1.0 - p_failed - p_continue
        r = random.random()
        if r < p_success:
            return "succeeded"
        if r < p_success + p_failed:
            return "failed"
        return "continue"

    def allowed_next_top_levels(self, current: str, nested_outcome: str | None) -> list[str]:
        """Return allowed next top-level states (for persona to choose from)."""
        if current == "start":
            return list(self._goal_ids) if self._goal_ids else ["finished"]
        if current in ("publish", "subscribe", "finished", "abandoned"):
            return [current]
        if nested_outcome == "succeeded":
            return self._goal_ids + ["publish", "subscribe", "finished"]
        if nested_outcome == "failed":
            return self._goal_ids + ["abandoned"]
        return self._goal_ids + ["publish", "subscribe", "finished", "abandoned"]

    def sample_top_level(
        self, current: str, nested_outcome: str | None, persona: Any
    ) -> str:
        """Sample next top-level state; bias toward finished/abandoned for ~10-message trajectories."""
        allowed = self.allowed_next_top_levels(current, nested_outcome)
        return self.sample_from_allowed(allowed)

    def sample_from_allowed(self, allowed: list[str]) -> str:
        """Sample from an allowed list (e.g. after filtering for required goals) with terminal bias."""
        if not allowed:
            return "finished"
        weights = []
        for a in allowed:
            if a == "finished" or a == "abandoned":
                weights.append(2.0)
            else:
                weights.append(1.0)
        total = sum(weights)
        r = random.random() * total
        for a, w in zip(allowed, weights):
            r -= w
            if r <= 0:
                return a
        return allowed[-1]
