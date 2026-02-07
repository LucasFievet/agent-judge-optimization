"""Unit tests for trajectory length calculation."""

from expforge.persona import PersonaSpec, PersonaSet
from expforge.goal import GoalSet, GoalSpec, Tool
from expforge.trajectory.transition_matrix import build_transition_matrix
from expforge.theory.moments import expected_trajectory_length


def test_mean_length_simple():
    """Test expected length on a simple 2-state chain."""
    # Setup: 1 goal with high success rate
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.8, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.8, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    matrix = build_transition_matrix(persona_set, goal_set)

    # Expected: start (1 step) + goal1 (1 step) + terminal (reached) = 2 steps
    # But theory might count differently due to nested states
    length = expected_trajectory_length(matrix, "p1", ["goal1"])

    print(f"Expected length: {length:.2f}")
    # This will show us what the theory actually computes
    # We expect: 1 (start) + at least 1 (goal) + absorption = at least 2
    assert length >= 2.0, f"Expected length >= 2, got {length}"
    assert length <= 10.0, f"Expected length <= 10, got {length}"  # Sanity check


def test_nested_state_counting():
    """Verify how nested states are counted in theory."""
    from expforge.theory.chain import build_chain_matrices, fundamental_matrix
    import numpy as np

    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    matrix = build_transition_matrix(persona_set, goal_set)
    Q, R, alpha = build_chain_matrices(matrix, "p1", ["goal1"])
    N, _ = fundamental_matrix(Q, R, alpha)

    print(f"\nTransient state count: {Q.shape[0]}")
    print(f"Alpha (initial dist): {alpha}")
    print(f"N diagonal (expected visits per state):\n{np.diag(N)}")
    print(f"Total expected visits: {alpha @ N @ np.ones(N.shape[0]):.2f}")

    # The issue: we have multiple transient states per conceptual "step"
    # Start state + goal's 3 sub-states (succeeded/failed/continue) + 2 outcome states
    # Expected: 1 + 3 + 2 = 6 transient states for 1 goal
    assert Q.shape[0] == 6, f"Expected 6 transient states, got {Q.shape[0]}"


if __name__ == "__main__":
    print("Test 1: Simple mean length")
    test_mean_length_simple()
    print("✓ PASSED\n")

    print("Test 2: Nested state counting")
    test_nested_state_counting()
    print("✓ PASSED")
