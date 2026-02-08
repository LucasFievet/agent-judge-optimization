"""Unit tests for theory package edge cases.

Tests special cases that check boundary behavior:
- Zero publish probability
- Zero subscribe probability
- Very high/low nested state probabilities
- Absorption and hitting probability edge cases
"""

import numpy as np
from expforge.persona import PersonaSpec, PersonaSet
from expforge.goal import GoalSet, GoalSpec, Tool
from expforge.trajectory.transition_matrix import build_transition_matrix
from expforge.theory.moments import expected_trajectory_length
from expforge.theory.absorption import absorption_probabilities, hitting_probabilities
from expforge.theory.correlation import correlation_publish_subscribe


def test_zero_publish_probability():
    """Test theory with zero publish probability (outcome_weights["publish"] = 0)."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    # Set publish weight to 0
    outcome_weights = {"publish": 0.0, "subscribe": 1.0, "finished": 2.0, "abandoned": 2.0}
    matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=outcome_weights)

    # Check that publish probability is 0 in the transition matrix
    from_succeeded = matrix["top_level"]["from_goal_succeeded"]
    assert from_succeeded.get("publish", 0.0) == 0.0, "Publish probability should be 0"

    # Check that probabilities sum to 1
    total = sum(from_succeeded.values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities should sum to 1, got {total}"

    # Theory should handle this without errors
    length = expected_trajectory_length(matrix, "p1", ["goal1"])
    assert length > 0, f"Expected positive length, got {length}"

    # Hitting probability for publish should be 0 or very small
    p_pub, p_sub = hitting_probabilities(matrix, "p1", ["goal1"])
    assert p_pub < 0.01, f"P(ever publish) should be ~0, got {p_pub}"
    assert p_sub > 0, f"P(ever subscribe) should be positive, got {p_sub}"

    print(f"✓ Zero publish probability: length={length:.2f}, p_pub={p_pub:.4f}, p_sub={p_sub:.4f}")


def test_zero_subscribe_probability():
    """Test theory with zero subscribe probability (outcome_weights["subscribe"] = 0)."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    # Set subscribe weight to 0
    outcome_weights = {"publish": 2.0, "subscribe": 0.0, "finished": 2.0, "abandoned": 2.0}
    matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=outcome_weights)

    # Check that subscribe probability is 0
    from_succeeded = matrix["top_level"]["from_goal_succeeded"]
    assert from_succeeded.get("subscribe", 0.0) == 0.0, "Subscribe probability should be 0"

    # Theory should handle this without errors
    length = expected_trajectory_length(matrix, "p1", ["goal1"])
    assert length > 0, f"Expected positive length, got {length}"

    # Hitting probability for subscribe should be 0 or very small
    p_pub, p_sub = hitting_probabilities(matrix, "p1", ["goal1"])
    assert p_pub > 0, f"P(ever publish) should be positive, got {p_pub}"
    assert p_sub < 0.01, f"P(ever subscribe) should be ~0, got {p_sub}"

    print(f"✓ Zero subscribe probability: length={length:.2f}, p_pub={p_pub:.4f}, p_sub={p_sub:.4f}")


def test_zero_both_publish_and_subscribe():
    """Test theory with both publish and subscribe probabilities set to 0."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    # Set both to 0
    outcome_weights = {"publish": 0.0, "subscribe": 0.0, "finished": 2.0, "abandoned": 2.0}
    matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=outcome_weights)

    # Theory should handle this without errors (trajectories go straight to terminal states)
    length = expected_trajectory_length(matrix, "p1", ["goal1"])
    assert length > 0, f"Expected positive length, got {length}"

    # Both hitting probabilities should be ~0
    p_pub, p_sub = hitting_probabilities(matrix, "p1", ["goal1"])
    assert p_pub < 0.01, f"P(ever publish) should be ~0, got {p_pub}"
    assert p_sub < 0.01, f"P(ever subscribe) should be ~0, got {p_sub}"

    # Correlation should be 0 or undefined (both are never visited)
    try:
        corr = correlation_publish_subscribe(
            matrix,
            persona_weights=[1.0],
            persona_ids=["p1"],
            goal_ids=["goal1"]
        )
        # If both are 0, correlation might be 0 or NaN
        assert np.isnan(corr) or abs(corr) < 0.01, f"Correlation should be 0 or NaN, got {corr}"
    except (ValueError, ZeroDivisionError):
        # Expected if variance is 0
        pass

    print(f"✓ Zero both probabilities: length={length:.2f}, p_pub={p_pub:.4f}, p_sub={p_sub:.4f}")


def test_very_high_publish_weight():
    """Test with very high publish weight (strongly favors publish over subscribe)."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    # Very high publish weight
    outcome_weights = {"publish": 100.0, "subscribe": 1.0, "finished": 2.0, "abandoned": 2.0}
    matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=outcome_weights)

    # Check that publish dominates
    from_succeeded = matrix["top_level"]["from_goal_succeeded"]
    p_publish = from_succeeded["publish"]
    p_subscribe = from_succeeded["subscribe"]
    assert p_publish > 0.9, f"Publish should dominate, got {p_publish}"
    assert p_subscribe < 0.05, f"Subscribe should be small, got {p_subscribe}"

    # Hitting probabilities should reflect this
    p_pub, p_sub = hitting_probabilities(matrix, "p1", ["goal1"])
    assert p_pub > p_sub, f"P(publish) should be > P(subscribe): {p_pub} vs {p_sub}"

    print(f"✓ Very high publish weight: p_matrix={p_publish:.4f}, p_hit={p_pub:.4f}")


def test_very_low_determined_persona():
    """Test with very low determined value (tests lower bound of success probability).

    Current formula: p_success = 0.20 + 0.45*determined + 0.35*quality
    With determined=0, quality=0: p_success_raw = 0.20, p_failed = 0.3
    p_continue_raw = 1.0 - 0.20 - 0.3 = 0.50, capped at P_CONTINUE_MAX = 0.40
    After capping: p_success = 1.0 - 0.3 - 0.4 = 0.30
    """
    personas = [PersonaSpec(
        id="p1", name="Low Determined", weight=1.0, technical=0.5,
        determined=0.0, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.0, name="Low Quality Tool")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    matrix = build_transition_matrix(persona_set, goal_set)

    # Check nested probabilities
    nested_probs = matrix["nested"]["p1"]["goal1"]
    p_success = nested_probs["succeeded"]
    p_failed = nested_probs["failed"]
    p_continue = nested_probs["continue"]

    # Expected: p_success = 0.30, p_failed = 0.3, p_continue = 0.40 (after capping)
    assert abs(p_success - 0.30) < 0.01, f"Expected p_success=0.30, got {p_success}"
    assert abs(p_failed - 0.30) < 0.01, f"Expected p_failed=0.30, got {p_failed}"
    assert abs(p_continue - 0.40) < 0.01, f"Expected p_continue=0.40, got {p_continue}"

    # Theory should handle this
    length = expected_trajectory_length(matrix, "p1", ["goal1"])
    assert length > 0, f"Expected positive length, got {length}"

    # Absorption probabilities should favor abandoned (due to failures)
    absorb = absorption_probabilities(matrix, "p1", ["goal1"])
    p_abandoned = absorb["abandoned"]
    assert p_abandoned > 0, f"P(abandoned) should be positive, got {p_abandoned}"

    print(f"✓ Very low determined: p_success={p_success:.4f}, p_abandoned={p_abandoned:.4f}, length={length:.2f}")


def test_very_high_determined_persona():
    """Test with very high determined value (tests upper bound of success probability).

    Formula: p_success = 0.25 + 0.50*determined + 0.25*quality
    With determined=1, quality=1: p_success_raw = 1.0, p_failed = 0.3
    This gives p_continue_raw = -0.3 (negative), triggering normalization:
    p_success = 1.0 / 1.3 = 0.7692, p_failed = 0.3 / 1.3 = 0.2308, p_continue = 0.0
    """
    personas = [PersonaSpec(
        id="p1", name="High Determined", weight=1.0, technical=0.5,
        determined=1.0, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=1.0, name="High Quality Tool")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    matrix = build_transition_matrix(persona_set, goal_set)

    # Check nested probabilities
    nested_probs = matrix["nested"]["p1"]["goal1"]
    p_success = nested_probs["succeeded"]
    p_failed = nested_probs["failed"]
    p_continue = nested_probs["continue"]

    # Expected: p_success = 0.7692, p_failed = 0.2308, p_continue = 0.0
    assert abs(p_success - 0.7692) < 0.01, f"Expected p_success=0.7692, got {p_success}"
    assert abs(p_failed - 0.2308) < 0.01, f"Expected p_failed=0.2308, got {p_failed}"
    assert p_continue == 0.0, f"Expected p_continue=0.0, got {p_continue}"

    # Theory should handle this
    length = expected_trajectory_length(matrix, "p1", ["goal1"])
    assert length > 0, f"Expected positive length, got {length}"

    # Absorption probabilities should favor finished (due to successes)
    absorb = absorption_probabilities(matrix, "p1", ["goal1"])
    p_finished = absorb["finished"]
    assert p_finished > 0.5, f"P(finished) should be > 0.5, got {p_finished}"

    print(f"✓ Very high determined: p_success={p_success:.4f}, p_finished={p_finished:.4f}, length={length:.2f}")


def test_single_goal_no_cycles():
    """Test with configuration that should minimize cycles (high terminal weights)."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.8, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.8, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    # High terminal weights, low publish/subscribe weights
    outcome_weights = {"publish": 0.1, "subscribe": 0.1, "finished": 10.0, "abandoned": 10.0}
    matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=outcome_weights)

    # Trajectory should be short (goes to terminal quickly)
    length = expected_trajectory_length(matrix, "p1", ["goal1"])
    assert length > 1.0, f"Expected length > 1, got {length}"
    assert length < 5.0, f"Expected short trajectory with high terminal weights, got {length}"

    # Hitting probabilities should be small
    p_pub, p_sub = hitting_probabilities(matrix, "p1", ["goal1"])
    assert p_pub < 0.5, f"P(publish) should be small, got {p_pub}"
    assert p_sub < 0.5, f"P(subscribe) should be small, got {p_sub}"

    print(f"✓ Minimal cycles: length={length:.2f}, p_pub={p_pub:.4f}, p_sub={p_sub:.4f}")


def test_absorption_probabilities_sum_to_one():
    """Test that absorption probabilities sum to 1 for various configurations."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    # Test with different outcome weights
    configs = [
        {"publish": 2.0, "subscribe": 1.0, "finished": 2.0, "abandoned": 2.0},
        {"publish": 0.0, "subscribe": 1.0, "finished": 2.0, "abandoned": 2.0},
        {"publish": 10.0, "subscribe": 10.0, "finished": 1.0, "abandoned": 1.0},
        {"publish": 1.0, "subscribe": 1.0, "finished": 1.0, "abandoned": 1.0},
    ]

    for i, weights in enumerate(configs):
        matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=weights)
        absorb = absorption_probabilities(matrix, "p1", ["goal1"])

        total = absorb["finished"] + absorb["abandoned"]
        assert abs(total - 1.0) < 1e-6, \
            f"Config {i}: Absorption probs should sum to 1, got {total} ({absorb})"

    print(f"✓ Absorption probabilities sum to 1 for all {len(configs)} configurations")


def test_hitting_probabilities_bounded():
    """Test that hitting probabilities are in [0, 1] for various configurations."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [GoalSpec(id="goal1", name="Goal 1", tools=["t1"])]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    configs = [
        {"publish": 2.0, "subscribe": 1.0, "finished": 2.0, "abandoned": 2.0},
        {"publish": 0.0, "subscribe": 1.0, "finished": 2.0, "abandoned": 2.0},
        {"publish": 10.0, "subscribe": 0.1, "finished": 1.0, "abandoned": 1.0},
    ]

    for i, weights in enumerate(configs):
        matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=weights)
        p_pub, p_sub = hitting_probabilities(matrix, "p1", ["goal1"])

        assert 0 <= p_pub <= 1.0, f"Config {i}: P(publish) should be in [0,1], got {p_pub}"
        assert 0 <= p_sub <= 1.0, f"Config {i}: P(subscribe) should be in [0,1], got {p_sub}"

    print(f"✓ Hitting probabilities bounded in [0, 1] for all {len(configs)} configurations")


def test_multiple_goals_edge_case():
    """Test with multiple goals and edge case weights."""
    personas = [PersonaSpec(
        id="p1", name="Persona 1", weight=1.0, technical=0.5,
        determined=0.5, swearing=0.0, baseline_sentiment=0.5
    )]
    tools = [Tool(id="t1", quality=0.5, name="Tool 1")]
    goals = [
        GoalSpec(id="goal1", name="Goal 1", tools=["t1"]),
        GoalSpec(id="goal2", name="Goal 2", tools=["t1"]),
        GoalSpec(id="goal3", name="Goal 3", tools=["t1"]),
    ]

    persona_set = PersonaSet(experiment_id="test", personas=personas)
    goal_set = GoalSet(experiment_id="test", goals=goals, tools=tools)

    # Zero publish
    outcome_weights = {"publish": 0.0, "subscribe": 1.0, "finished": 2.0, "abandoned": 2.0}
    matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=outcome_weights)

    # Theory should handle multiple goals
    length = expected_trajectory_length(matrix, "p1", ["goal1", "goal2", "goal3"])
    assert length > 0, f"Expected positive length with multiple goals, got {length}"

    p_pub, p_sub = hitting_probabilities(matrix, "p1", ["goal1", "goal2", "goal3"])
    assert 0 <= p_pub <= 1.0, f"P(publish) should be in [0,1], got {p_pub}"
    assert 0 <= p_sub <= 1.0, f"P(subscribe) should be in [0,1], got {p_sub}"

    print(f"✓ Multiple goals edge case: length={length:.2f}, p_pub={p_pub:.4f}, p_sub={p_sub:.4f}")


if __name__ == "__main__":
    print("Running theory edge case tests...")
    print("=" * 60)

    tests = [
        test_zero_publish_probability,
        test_zero_subscribe_probability,
        test_zero_both_publish_and_subscribe,
        test_very_high_publish_weight,
        test_very_low_determined_persona,
        test_very_high_determined_persona,
        test_single_goal_no_cycles,
        test_absorption_probabilities_sum_to_one,
        test_hitting_probabilities_bounded,
        test_multiple_goals_edge_case,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"✗ FAILED: {test_fn.__name__}")
            print(f"  {e}")
        except Exception as e:
            print(f"✗ ERROR: {test_fn.__name__}")
            print(f"  {type(e).__name__}: {e}")

    print("=" * 60)
    print("✅ All theory edge case tests completed")
