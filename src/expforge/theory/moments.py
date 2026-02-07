"""
Trajectory length moments: E[T], E[T^2] (doc/main.tex eq. fp-length-mean, phase-type).
"""

from typing import Any
import numpy as np

from expforge.theory.chain import build_chain_matrices, fundamental_matrix


def expected_trajectory_length(
    transition_matrix: dict[str, Any],
    persona_id: str,
    goal_ids: list[str],
) -> float:
    """E[T] = expected number of TOP-LEVEL state transitions (simulator steps).

    The raw calculation alpha' N 1 counts all transient state visits, but the simulator
    counts only top-level transitions. We need to adjust for the nested sub-states.

    In the model:
    - Start state: counts as 1 step
    - Each goal visit: enter continue → sample nested → outcome state = 1 step (not 2!)
    - Outcome states (publish/subscribe): count as 1 step each if visited

    Correction: Count only visits to "decision points":
    - Start (state 0): always visited once
    - Goal outcome states (succeeded/failed sub-states): each represents completing a goal
    - Outcome states (publish/subscribe): each visit is a step
    - The goal "continue" entry states are NOT separate steps

    So: E[steps] = E[visits to start] + E[visits to goal outcomes] + E[visits to outcome states]
    """
    from expforge.theory.chain import goal_nested_state_index, outcome_state_index

    Q, R, alpha = build_chain_matrices(transition_matrix, persona_id, goal_ids)
    N, _ = fundamental_matrix(Q, R, alpha)

    n_goals = len(goal_ids)
    n_transient = N.shape[0]

    # Count expected visits to each type of state
    expected_visits = alpha @ N  # Vector of expected visits to each transient state

    # Start state (index 0): always 1 visit
    steps_from_start = expected_visits[0]

    # Goal states: count visits to succeeded/failed outcome states only (not continue entry)
    # Each visit to succeeded or failed represents one completed goal attempt
    steps_from_goals = 0.0
    for i in range(n_goals):
        idx_s = goal_nested_state_index(i, "succeeded", n_goals)
        idx_f = goal_nested_state_index(i, "failed", n_goals)
        # Don't count idx_c (continue entry) as it's part of the same step as the outcome
        steps_from_goals += expected_visits[idx_s] + expected_visits[idx_f]

    # Outcome states (publish, subscribe): each visit is a step
    steps_from_outcomes = 0.0
    try:
        pub_idx = outcome_state_index("publish", n_goals)
        sub_idx = outcome_state_index("subscribe", n_goals)
        steps_from_outcomes += expected_visits[pub_idx] + expected_visits[sub_idx]
    except (ValueError, IndexError):
        pass  # Outcome states might not exist in all models

    total_steps = steps_from_start + steps_from_goals + steps_from_outcomes
    return float(total_steps)


def expected_trajectory_length_squared(
    transition_matrix: dict[str, Any],
    persona_id: str,
    goal_ids: list[str],
) -> float:
    """E[T^2] = 2 alpha' N^2 1 - E[T] (phase-type second moment)."""
    Q, R, alpha = build_chain_matrices(transition_matrix, persona_id, goal_ids)
    N, _ = fundamental_matrix(Q, R, alpha)
    n = N.shape[0]
    et = float(alpha @ N @ np.ones(n))
    et2 = 2.0 * float(alpha @ N @ N @ np.ones(n)) - et
    return et2
