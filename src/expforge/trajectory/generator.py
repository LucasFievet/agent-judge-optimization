"""Generate a single trajectory from persona + goals."""

import random
import uuid
from typing import Callable

from expforge.persona import PersonaSpec
from expforge.goal import GoalSet
from expforge.trajectory.states import TrajectoryState
from expforge.trajectory.steps import TrajectoryStep, Trajectory
from expforge.trajectory.transitions import TransitionSampler

# Optional: (persona, top_level, nested_state) -> user_message
UserMessageFn = Callable[[PersonaSpec, str, str | None], str]
# Optional: (user_message, tools_used, nested_outcome, goal_name, goal_id) -> agent_message
AgentMessageFn = Callable[[str, list[str], str | None, str, str], str]
# Optional: () -> overarching goal string for this sample
SampleGoalFn = Callable[[GoalSet, PersonaSpec | None], str]
# Optional: (persona, goal_set, conversation, sample_goal, top_level, nested_state, nested_outcome, allowed_next) -> (user_message, next_top)
PersonaTurnFn = Callable[
    [
        PersonaSpec,
        GoalSet,
        list[tuple[str, str]],
        str,
        str,
        str | None,
        str | None,
        list[str],
    ],
    tuple[str, str],
]

DECIMALS = 2


def _round_quality(v: float) -> float:
    return round(v, DECIMALS)


def _fallback_user_message(top_level: str, nested: str | None) -> str:
    if top_level == "start":
        return "I need help with my document."
    if top_level in ("publish", "subscribe", "finished", "abandoned"):
        return "Thanks."
    if nested == "succeeded":
        return "That worked."
    if nested == "failed":
        return "That didn't work. What now?"
    return "Working on it."


def _fallback_agent_message(tools_used: list[str], nested_outcome: str | None) -> str:
    t = ", ".join(tools_used) if tools_used else "no tools"
    if nested_outcome == "succeeded":
        return f"I used {t} and it succeeded."
    if nested_outcome == "failed":
        return f"I tried {t} but it failed."
    return f"I used {t}; still in progress."


class TrajectoryGenerator:
    """Generates one trajectory: samples states (transition model), then fills user/agent messages (optional LLM)."""

    def __init__(
        self,
        goal_set: GoalSet,
        persona: PersonaSpec,
        *,
        max_steps: int = 50,
        seed: int | None = None,
        sample_goal_fn: SampleGoalFn | None = None,
        persona_turn_fn: PersonaTurnFn | None = None,
        user_message_fn: UserMessageFn | None = None,
        agent_message_fn: AgentMessageFn | None = None,
    ) -> None:
        self.goal_set = goal_set
        self.persona = persona
        self.max_steps = max_steps
        self.sampler = TransitionSampler(goal_set, seed=seed)
        self.sample_goal_fn = sample_goal_fn
        self.persona_turn_fn = persona_turn_fn
        self.user_message_fn = user_message_fn
        self.agent_message_fn = agent_message_fn
        self._goal_ids = [g.id for g in goal_set.goals]
        if seed is not None:
            random.seed(seed)

    def _tool_quality_for_step(self, top_level: str) -> float:
        """Fixed quality for this step: tool quality for the current goal, or 0.5 if not in a goal."""
        if top_level in self._goal_ids:
            return _round_quality(self.goal_set.tool_quality_for_goal(top_level))
        return 0.5

    def _tools_used_for_goal(self, goal_id: str) -> list[str]:
        """Tools used for this goal (from goal config)."""
        g = next((x for x in self.goal_set.goals if x.id == goal_id), None)
        return list(g.tools) if g else []

    def _goal_name(self, goal_id: str) -> str:
        g = next((x for x in self.goal_set.goals if x.id == goal_id), None)
        return g.name if g else goal_id

    def generate(self, trajectory_id: str | None = None) -> Trajectory:
        """Produce one trajectory: optional sample_goal, then steps with persona choosing message + next action (or sampler)."""
        trajectory_id = trajectory_id or str(uuid.uuid4())
        steps: list[TrajectoryStep] = []
        state = TrajectoryState(top_level="start", nested=None, quality=0.5)
        sample_goal = ""
        if self.sample_goal_fn:
            sample_goal = self.sample_goal_fn(self.goal_set, self.persona)
        conversation_so_far: list[tuple[str, str]] = []

        for _ in range(self.max_steps):
            nested_outcome = None
            if state.top_level not in ("start", "publish", "subscribe", "finished", "abandoned"):
                nested_outcome = self.sampler.sample_nested(self.persona, state.top_level)
                state.nested = nested_outcome

            allowed_next = self.sampler.allowed_next_top_levels(state.top_level, nested_outcome)
            # Require minimal completion (e.g. abstract + section + conclusion) before allowing "finished"
            required = getattr(self.goal_set, "required_goals_for_finish", None) or ()
            if required and "finished" in allowed_next:
                goals_succeeded = {
                    s.top_level_state for s in steps
                    if s.top_level_state in self._goal_ids and s.nested_state == "succeeded"
                }
                if state.top_level in self._goal_ids and nested_outcome == "succeeded":
                    goals_succeeded = goals_succeeded | {state.top_level}
                if not all(g in goals_succeeded for g in required):
                    allowed_next = [a for a in allowed_next if a != "finished"]
            if self.persona_turn_fn:
                user_message, next_top = self.persona_turn_fn(
                    self.persona,
                    self.goal_set,
                    conversation_so_far,
                    sample_goal,
                    state.top_level,
                    state.nested,
                    nested_outcome,
                    allowed_next,
                )
            else:
                next_top = self.sampler.sample_from_allowed(allowed_next)
                if self.user_message_fn:
                    user_message = self.user_message_fn(self.persona, state.top_level, state.nested)
                else:
                    user_message = _fallback_user_message(state.top_level, state.nested)

            quality = self._tool_quality_for_step(state.top_level)
            tools_used = self._tools_used_for_goal(state.top_level) if state.top_level in self._goal_ids else []

            if self.agent_message_fn:
                goal_name = self._goal_name(state.top_level) if state.top_level in self._goal_ids else ""
                goal_id = state.top_level if state.top_level in self._goal_ids else ""
                agent_message = self.agent_message_fn(
                    user_message, tools_used, nested_outcome, goal_name, goal_id
                )
            else:
                agent_message = _fallback_agent_message(tools_used, nested_outcome)

            step = TrajectoryStep(
                user_message=user_message,
                tools_used=tools_used,
                agent_message=agent_message,
                top_level_state=state.top_level,
                nested_state=state.nested,
                quality=quality,
                terminal=next_top in ("finished", "abandoned"),
            )
            steps.append(step)
            conversation_so_far.append((user_message, agent_message))
            state.top_level = next_top
            state.nested = None if next_top in ("start", "publish", "subscribe", "finished", "abandoned") else "continue"
            state.quality = quality

            if step.terminal:
                break

        outcome = state.top_level if steps and steps[-1].terminal else None
        return Trajectory(
            trajectory_id=trajectory_id,
            persona_id=self.persona.id,
            sample_goal=sample_goal,
            steps=steps,
            outcome=outcome,
        )


def generate_trajectory(
    goal_set: GoalSet,
    persona: PersonaSpec,
    *,
    max_steps: int = 50,
    seed: int | None = None,
    sample_goal_fn: SampleGoalFn | None = None,
    persona_turn_fn: PersonaTurnFn | None = None,
    user_message_fn: UserMessageFn | None = None,
    agent_message_fn: AgentMessageFn | None = None,
) -> Trajectory:
    """Convenience: generate one trajectory."""
    gen = TrajectoryGenerator(
        goal_set,
        persona,
        max_steps=max_steps,
        seed=seed,
        sample_goal_fn=sample_goal_fn,
        persona_turn_fn=persona_turn_fn,
        user_message_fn=user_message_fn,
        agent_message_fn=agent_message_fn,
    )
    return gen.generate()
