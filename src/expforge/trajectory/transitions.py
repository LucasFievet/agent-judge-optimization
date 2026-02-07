"""Sample next state from nested and top-level chains."""

import logging
import random
from typing import Any

from expforge.persona import PersonaSpec
from expforge.goal import GoalSet

logger = logging.getLogger(__name__)


class TransitionSampler:
    """Draws next top-level and nested state given current state, persona, and goals."""

    def __init__(self, goal_set: GoalSet, seed: int | None = None, outcome_weights: dict[str, float] | None = None) -> None:
        self.goal_set = goal_set
        if seed is not None:
            random.seed(seed)
        self._goal_ids = [g.id for g in goal_set.goals]
        # Store outcome weights for sampling (default: publish=2, subscribe=1, terminals=2)
        from expforge.trajectory.transition_matrix import DEFAULT_OUTCOME_WEIGHTS
        self.outcome_weights = outcome_weights or DEFAULT_OUTCOME_WEIGHTS

    def sample_nested(self, persona: PersonaSpec, goal_id: str) -> str:
        """Sample nested state (succeeded, continue, failed) for the goal.
        p_success in [0.25, 1.0] so experiments can vary ~0–30% (low determined + weak tools → more abandon)."""
        goal = next((g for g in self.goal_set.goals if g.id == goal_id), None)
        if not goal:
            return "continue"
        tool_quality = self.goal_set.tool_quality_for_goal(goal_id)
        p_success = 0.25 + 0.50 * persona.determined + 0.25 * tool_quality
        p_success = max(0.0, min(1.0, p_success))
        p_failed = 0.3
        p_continue_raw = 1.0 - p_success - p_failed
        # Cap continue so we resolve (succeeded/failed) often → shorter trajectories
        p_continue = min(0.12, max(0.0, p_continue_raw))

        # Adjust p_success if we capped p_continue
        if p_continue_raw > 0.12:
            p_success = 1.0 - p_failed - p_continue
        # Handle case where p_success + p_failed > 1.0 (p_continue_raw < 0)
        elif p_continue_raw < 0:
            # Normalize p_success and p_failed to sum to 1.0, keeping their relative proportions
            total = p_success + p_failed
            p_success = p_success / total
            p_failed = p_failed / total
            p_continue = 0.0

        r = random.random()
        if r < p_success:
            return "succeeded"
        if r < p_success + p_failed:
            return "failed"
        return "continue"

    def allowed_next_top_levels(self, current: str, nested_outcome: str | None) -> list[str]:
        """Return allowed next top-level states (for persona to choose from).

        Per PRD: Terminal states are finished and abandoned only.
        Publish and subscribe are non-terminal and can transition to goals or terminal states.
        """
        if current == "start":
            return list(self._goal_ids) if self._goal_ids else ["finished"]
        # Only finished and abandoned are terminal/absorbing
        if current in ("finished", "abandoned"):
            return [current]
        # Publish and subscribe are non-terminal, can go to goals or terminal states
        if current == "publish":
            return self._goal_ids + ["subscribe", "finished", "abandoned"]
        if current == "subscribe":
            return self._goal_ids + ["publish", "finished", "abandoned"]
        # From a goal state:
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
        """Sample from an allowed list using outcome_weights for publish/subscribe/finished/abandoned."""
        if not allowed:
            return "finished"
        weights = []
        for a in allowed:
            # Use outcome_weights for special states, default 1.0 for goals
            weights.append(self.outcome_weights.get(a, 1.0))
        total = sum(weights)
        r = random.random() * total
        for a, w in zip(allowed, weights):
            r -= w
            if r <= 0:
                return a
        return allowed[-1]
