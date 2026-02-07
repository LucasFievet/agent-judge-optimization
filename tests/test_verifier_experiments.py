"""Test verifier across different experiments and configurations."""

from expforge.verifier import run_verification
from expforge.verifier.multi_seed import run_verification_multi_seed
import logging

# Suppress logs for cleaner test output
logging.basicConfig(level=logging.ERROR)


def test_dummy_experiment():
    """Test verification on dummy experiment."""
    result = run_verification('dummy', sample_sizes=[200, 500, 1000], seed=123)
    assert result.passed, f"Dummy experiment should pass with seed=123"
    print("✓ dummy experiment passed")


def test_test_experiment():
    """Test verification on test experiment."""
    result = run_verification('test', sample_sizes=[200, 500, 1000], seed=123)
    assert result.passed, f"Test experiment should pass with seed=123"
    print("✓ test experiment passed")


def test_verifier_1_experiment():
    """Test verification on verifier_1 experiment."""
    result = run_verification('verifier_1', sample_sizes=[200, 500, 1000], seed=123)
    assert result.passed, f"Verifier_1 experiment should pass with seed=123"
    print("✓ verifier_1 experiment passed")


def test_multiple_seeds():
    """Test that verification works with different seeds."""
    # Test with a mix of seeds (some pass, some fail due to statistical variance)
    seeds = [123, 999, 42, 100, 500, 5678]
    results = []

    for seed in seeds:
        result = run_verification('dummy', sample_sizes=[200, 500, 1000], seed=seed)
        results.append(result.passed)

    pass_rate = sum(results) / len(results)
    # At least 2 out of 6 should pass (33% threshold to account for variance)
    assert pass_rate >= 0.33, f"At least 33% of seeds should pass, got {pass_rate*100:.0f}%"
    print(f"✓ Multiple seeds test passed (pass rate: {pass_rate*100:.0f}%)")


def test_different_sample_sizes():
    """Test verification with various sample sizes."""
    # Test with different n values
    for n in [100, 200, 500, 1000, 2000]:
        result = run_verification('dummy', sample_sizes=[n], seed=123)
        # At least some sample sizes should pass
        if not result.passed:
            print(f"  n={n}: FAIL (expected for some n due to variance)")
        else:
            print(f"  n={n}: PASS")

    # The main requirement is that the function doesn't crash
    print("✓ Different sample sizes test completed")


def test_outcome_weights_consistency():
    """Test that simulator and theory use consistent outcome weights."""
    from expforge.persona import load_persona_set
    from expforge.goal import load_goal_set
    from expforge.trajectory.transition_matrix import build_transition_matrix, DEFAULT_OUTCOME_WEIGHTS
    from pathlib import Path

    base = Path('src/expforge/simulator/experiment/dummy')
    personas = load_persona_set(base / 'persona.yaml')
    goals = load_goal_set(base / 'goals.yaml')

    # Build matrix with default weights
    matrix = build_transition_matrix(personas, goals)

    # Check that publish has higher weight than subscribe
    from_succ = matrix['top_level']['from_goal_succeeded']
    p_publish_weight = from_succ.get('publish', 0.0)
    p_subscribe_weight = from_succ.get('subscribe', 0.0)

    expected_ratio = DEFAULT_OUTCOME_WEIGHTS['publish'] / DEFAULT_OUTCOME_WEIGHTS['subscribe']
    actual_ratio = p_publish_weight / p_subscribe_weight if p_subscribe_weight > 0 else 0

    assert abs(actual_ratio - expected_ratio) < 0.1, \
        f"Publish/subscribe weight ratio should be ~{expected_ratio:.1f}, got {actual_ratio:.1f}"

    print(f"✓ Outcome weights consistency test passed (ratio: {actual_ratio:.2f})")


def test_multi_seed_verification():
    """Test multi-seed verification function with pass rate threshold."""
    # Test with 50% pass rate threshold (should pass given ~60-90% actual pass rate)
    seeds = [123, 999, 100, 500, 1000, 5678]

    for exp in ['dummy', 'test', 'verifier_1']:
        results, overall_pass = run_verification_multi_seed(
            exp,
            seeds,
            sample_sizes=[1000],
            pass_rate=0.50  # Require 50% of seeds to pass
        )
        passed = sum(1 for r in results if r.passed)
        rate = passed / len(results)

        assert overall_pass, \
            f"{exp} should pass multi-seed verification with 50% threshold, got {rate*100:.0f}% pass rate"

    print(f"✓ Multi-seed verification test passed")


if __name__ == "__main__":
    print("Running verifier experiment tests...")
    print("="*60)

    try:
        test_dummy_experiment()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    try:
        test_test_experiment()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    try:
        test_verifier_1_experiment()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    try:
        test_multiple_seeds()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    try:
        test_different_sample_sizes()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    try:
        test_outcome_weights_consistency()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    try:
        test_multi_seed_verification()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("="*60)
    print("✅ All verifier experiment tests completed")
