"""Run simulator for an experiment: (re)use config, draw persona, generate N samples."""

from pathlib import Path
import random

from expforge.persona import (
    PersonaSet,
    PersonaSpec,
    load_persona_set,
    save_persona_set,
    generate_persona_set,
)
from expforge.goal import GoalSet, load_goal_set, save_goal_set, generate_goal_set
from expforge.trajectory import TrajectoryGenerator, save_trajectory
from expforge.trajectory.transition_matrix import build_transition_matrix, write_transition_matrix
from expforge.simulator.persona_simulator import generate_sample_goal, generate_user_message_and_next_action
from expforge.simulator.agent_simulator import generate_agent_message
from expforge.verifier.io import experiment_dir, DEFAULT_EXPERIMENTS_DIR


def run_simulator(
    experiment_id: str,
    n_samples: int,
    *,
    base_dir: Path | str | None = None,
    seed: int | None = None,
    reuse_config: bool = True,
    use_llm: bool = True,
) -> tuple[PersonaSet, GoalSet, list[Path]]:
    """
    Run the simulator for experiment `experiment_id` with `n_samples` trajectories.

    All outputs live under simulator/experiment/<experiment_id>/:
    - persona.yaml, goals.yaml, transitions.yaml; samples/sample_1.yaml, ...
    Reuses persona/goals if present and reuse_config=True.

    If use_llm=False (fast mode), user and agent messages use fallbacks and the transition
    sampler chooses the next action; no LLM calls. Use for verification against theory.
    """
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    exp_dir = experiment_dir(base_dir, experiment_id)
    persona_path = exp_dir / "persona.yaml"
    goal_path = exp_dir / "goals.yaml"
    transitions_path = exp_dir / "transitions.yaml"
    samples_path = exp_dir / "samples"

    exp_dir.mkdir(parents=True, exist_ok=True)
    samples_path.mkdir(parents=True, exist_ok=True)

    if reuse_config and persona_path.exists():
        persona_set = load_persona_set(persona_path)
    else:
        persona_set = generate_persona_set(experiment_id, n_personas=5, seed=seed)
        save_persona_set(persona_set, persona_path)

    if reuse_config and goal_path.exists():
        goal_set = load_goal_set(goal_path)
    else:
        goal_set = generate_goal_set(experiment_id, seed=seed)
        save_goal_set(goal_set, goal_path)

    matrix = build_transition_matrix(persona_set, goal_set)
    matrix["experiment_id"] = experiment_id
    write_transition_matrix(matrix, transitions_path)

    if seed is not None:
        random.seed(seed)

    weights = persona_set.get_weights()
    saved_paths: list[Path] = []

    if use_llm:
        def sample_goal_fn(gs: GoalSet, p: PersonaSpec | None) -> str:
            return generate_sample_goal(gs, p)

        def persona_turn_fn(
            persona: PersonaSpec,
            gs: GoalSet,
            conversation: list[tuple[str, str]],
            sample_goal: str,
            top_level: str,
            nested_state: str | None,
            nested_outcome: str | None,
            allowed_next: list[str],
        ) -> tuple[str, str]:
            return generate_user_message_and_next_action(
                persona, gs, conversation, sample_goal,
                top_level, nested_state, nested_outcome, allowed_next,
            )

        def agent_message_fn(user_message: str, tools_used: list[str], nested_outcome: str | None, goal_name: str, goal_id: str = "") -> str:
            return generate_agent_message(user_message, tools_used, nested_outcome, goal_name, goal_id)
    else:
        sample_goal_fn = None
        persona_turn_fn = None
        agent_message_fn = None

    # In fast mode, disable required_goals_for_finish so dynamics match theory (theory assumes no filter)
    goal_set_for_gen = goal_set
    if not use_llm and getattr(goal_set, "required_goals_for_finish", None):
        from copy import copy
        goal_set_for_gen = copy(goal_set)
        goal_set_for_gen.required_goals_for_finish = []

    for i in range(n_samples):
        persona: PersonaSpec = random.choices(persona_set.personas, weights=weights)[0]
        gen = TrajectoryGenerator(
            goal_set_for_gen,
            persona,
            seed=None,  # Don't reseed - use global RNG state seeded above
            sample_goal_fn=sample_goal_fn,
            persona_turn_fn=persona_turn_fn,
            agent_message_fn=agent_message_fn,
        )
        traj = gen.generate(trajectory_id=f"sample_{i+1}")
        out_path = samples_path / f"sample_{i+1}.yaml"
        save_trajectory(traj, out_path)
        saved_paths.append(out_path)

    return persona_set, goal_set, saved_paths
