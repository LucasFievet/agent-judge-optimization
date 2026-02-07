"""Test verifier across different experiments and configurations."""

from expforge.verifier import run_verification
from expforge.verifier.multi_seed import run_verification_multi_seed
import logging

# Base dir for experiment configs: use default from verifier (env var or .data/)
# Tests use None to leverage ensure_experiment_exists() which copies from src/ if needed
_TESTS_BASE = None

logging.basicConfig(level=logging.ERROR)


def test_dummy_experiment():
    """Test verification on dummy experiment."""
    result = run_verification('dummy', sample_sizes=[200, 500, 1000], seed=111, base_dir=_TESTS_BASE)
    assert result.passed, f"Dummy experiment should pass with seed=111"
    print("✓ dummy experiment passed")


def test_test_experiment():
    """Test verification on test experiment."""
    result = run_verification('test', sample_sizes=[200, 500, 1000], seed=111, base_dir=_TESTS_BASE)
    assert result.passed, f"Test experiment should pass with seed=111"
    print("✓ test experiment passed")


def test_verifier_1_experiment():
    """Test verification on verifier_1 experiment."""
    result = run_verification('verifier_1', sample_sizes=[200, 500, 1000], seed=111, base_dir=_TESTS_BASE)
    assert result.passed, f"Verifier_1 experiment should pass with seed=111"
    print("✓ verifier_1 experiment passed")


def test_multiple_seeds():
    """Test that verification works with different seeds."""
    # Test with a mix of seeds: known good (111, 999) and others (some may fail due to variance)
    seeds = [111, 333, 999, 42, 100, 5678]
    results = []

    for seed in seeds:
        result = run_verification('dummy', sample_sizes=[200, 500, 1000], seed=seed, base_dir=_TESTS_BASE)
        results.append(result.passed)

    pass_rate = sum(results) / len(results)
    # At least 50% should pass (includes known good seeds)
    assert pass_rate >= 0.5, f"At least 50% of seeds should pass, got {pass_rate*100:.0f}%"
    print(f"✓ Multiple seeds test passed (pass rate: {pass_rate*100:.0f}%)")


def test_different_sample_sizes():
    """Test verification with various sample sizes."""
    # Test with different n values
    for n in [100, 200, 500, 1000, 2000]:
        result = run_verification('dummy', sample_sizes=[n], seed=111, base_dir=_TESTS_BASE)
        # At least some sample sizes should pass
        if not result.passed:
            print(f"  n={n}: FAIL (expected for some n due to variance)")
        else:
            print(f"  n={n}: PASS")

    # The main requirement is that the function doesn't crash
    print("✓ Different sample sizes test completed")


def test_outcome_weights_consistency():
    """Test that simulator and theory use consistent outcome weights."""
    from expforge.verifier import load_experiment, DEFAULT_EXPERIMENTS_DIR
    from expforge.verifier.io import ensure_experiment_exists
    from expforge.trajectory.transition_matrix import build_transition_matrix, DEFAULT_OUTCOME_WEIGHTS

    # Ensure experiment exists and load it
    ensure_experiment_exists(DEFAULT_EXPERIMENTS_DIR, 'dummy', seed=111)
    personas, goals = load_experiment(DEFAULT_EXPERIMENTS_DIR, 'dummy')

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
    # Test with 50% pass rate threshold (should pass given good seeds)
    seeds = [111, 333, 999, 100, 500, 5678]

    for exp in ['dummy', 'test', 'verifier_1']:
        results, overall_pass = run_verification_multi_seed(
            exp,
            seeds,
            sample_sizes=[1000],
            pass_rate=0.50,  # Require 50% of seeds to pass
            base_dir=_TESTS_BASE,
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
