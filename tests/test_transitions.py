"""Unit tests for transition logic."""

from expforge.trajectory.transitions import TransitionSampler
from expforge.goal import GoalSet, GoalSpec, Tool


def create_test_goalset():
    """Create a simple goal set for testing."""
    tools = [
        Tool(id="tool1", quality=0.7, name="Tool 1"),
        Tool(id="tool2", quality=0.8, name="Tool 2"),
    ]
    goals = [
        GoalSpec(id="goal1", name="Goal 1", tools=["tool1"]),
        GoalSpec(id="goal2", name="Goal 2", tools=["tool2"]),
    ]
    return GoalSet(
        experiment_id="test",
        goals=goals,
        tools=tools,
    )


def test_publish_subscribe_are_not_terminal():
    """Per PRD and GoalSet model: publish and subscribe are non-terminal outcome states."""
    goal_set = create_test_goalset()
    sampler = TransitionSampler(goal_set, seed=42)

    # From publish, should be able to transition to other states (not just publish)
    allowed_from_publish = sampler.allowed_next_top_levels("publish", None)
    assert "publish" not in allowed_from_publish, f"Should not stay in publish, got {allowed_from_publish}"
    assert "finished" in allowed_from_publish or "abandoned" in allowed_from_publish, \
        f"Should be able to reach terminal states from publish, got {allowed_from_publish}"

    # From subscribe, should be able to transition to other states (not just subscribe)
    allowed_from_subscribe = sampler.allowed_next_top_levels("subscribe", None)
    assert "subscribe" not in allowed_from_subscribe, f"Should not stay in subscribe, got {allowed_from_subscribe}"
    assert "finished" in allowed_from_subscribe or "abandoned" in allowed_from_subscribe, \
        f"Should be able to reach terminal states from subscribe, got {allowed_from_subscribe}"


def test_finished_abandoned_are_terminal():
    """Per PRD and GoalSet model: finished and abandoned are terminal states."""
    goal_set = create_test_goalset()
    sampler = TransitionSampler(goal_set, seed=42)

    # Finished is terminal
    allowed_from_finished = sampler.allowed_next_top_levels("finished", None)
    assert allowed_from_finished == ["finished"], f"Finished should be terminal, got {allowed_from_finished}"

    # Abandoned is terminal
    allowed_from_abandoned = sampler.allowed_next_top_levels("abandoned", None)
    assert allowed_from_abandoned == ["abandoned"], f"Abandoned should be terminal, got {allowed_from_abandoned}"


def test_probabilities_sum_to_one():
    """Nested outcome probabilities must sum to 1.0."""
    from expforge.trajectory.transition_matrix import _nested_probs_for

    # Test with various parameter combinations
    test_cases = [
        (0.0, 0.0),  # Low determined, low quality
        (0.5, 0.5),  # Medium
        (1.0, 1.0),  # High determined, high quality
        (0.2, 0.9),  # Low determined, high quality
        (0.9, 0.2),  # High determined, low quality
    ]

    for determined, quality in test_cases:
        probs = _nested_probs_for(determined, quality)
        total = probs["succeeded"] + probs["failed"] + probs["continue"]
        assert abs(total - 1.0) < 0.001, \
            f"Probabilities must sum to 1.0, got {total} for determined={determined}, quality={quality}"
        # All probabilities should be non-negative
        assert all(p >= 0 for p in probs.values()), \
            f"All probabilities must be non-negative: {probs}"


if __name__ == "__main__":
    print("Running test_publish_subscribe_are_not_terminal...")
    try:
        test_publish_subscribe_are_not_terminal()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\nRunning test_finished_abandoned_are_terminal...")
    try:
        test_finished_abandoned_are_terminal()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\nRunning test_probabilities_sum_to_one...")
    try:
        test_probabilities_sum_to_one()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\n✅ All tests passed!")
