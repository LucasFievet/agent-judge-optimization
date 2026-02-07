"""
Run verification: compare theory to simulator (fast mode) at several sample sizes.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

from expforge.theory import TheoreticalValues
from expforge.simulator.experiment_simulator import run_simulator

from expforge.verifier.io import load_experiment, copy_experiment, experiment_dir, ensure_experiment_exists, DEFAULT_EXPERIMENTS_DIR
from expforge.verifier.empirical import empirical_from_trajectories, empirical_correlation
from expforge.verifier.checks import append_checks, z_for_confidence


@dataclass
class VerificationResult:
    """Result of theory vs simulator verification."""

    experiment_id: str
    sample_sizes: list[int]
    theory: TheoreticalValues
    empirical_by_n: dict[int, dict[str, float]] = field(default_factory=dict)
    correlation_by_n: dict[int, float] = field(default_factory=dict)
    checks: dict[str, list[bool]] = field(default_factory=dict)
    confidence: float = 0.95
    passed: bool = False


def run_verification(
    experiment_id: str,
    sample_sizes: list[int] | None = None,
    *,
    base_dir: Path | str | None = None,
    seed: int = 42,
    confidence: float = 0.95,
    max_samples: int | None = None,
) -> VerificationResult:
    """
    Load experiment config, compute theory, run simulator (fast mode).
    If max_samples is set: run once with max_samples, then verify at each n by subsampling.

    Reproducibility: the same seed yields the same result. The seed is used to bootstrap
    missing config (persona/goals via effective_seed(seed, experiment_id)), and to seed
    the simulator RNG for trajectory sampling (and subsampling when max_samples is set).
    """
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    sample_sizes = sample_sizes or [200, 500, 1000]

    ensure_experiment_exists(base_dir, experiment_id, seed=seed)
    persona_set, goal_set = load_experiment(base_dir, experiment_id)
    theory = TheoreticalValues.compute(persona_set, goal_set)

    var_length = max(0.0, theory.expected_length_sq - theory.expected_length ** 2)
    z = z_for_confidence(confidence)

    empirical_by_n: dict[int, dict[str, float]] = {}
    correlation_by_n: dict[int, float] = {}
    checks: dict[str, list[bool]] = {
        "mean_length": [],
        "p_finished": [],
        "p_abandoned": [],
        "p_publish": [],
        "p_subscribe": [],
        "correlation": [],
    }

    if max_samples is not None:
        M = max(max_samples, max(sample_sizes))
        _, _, all_paths = run_simulator(
            experiment_id,
            M,
            base_dir=base_dir,
            seed=seed,
            reuse_config=True,
            use_llm=False,
        )
        rng = random.Random(seed)
        for n in sample_sizes:
            subset = rng.sample(all_paths, n) if n <= len(all_paths) else all_paths
            emp = empirical_from_trajectories(subset)
            empirical_by_n[n] = emp
            correlation_by_n[n] = empirical_correlation(subset)
            append_checks(checks, theory, emp, correlation_by_n[n], n, var_length, z)
    else:
        for n in sample_sizes:
            _, _, paths = run_simulator(
                experiment_id, n,
                base_dir=base_dir, seed=seed, reuse_config=True, use_llm=False,
            )
            emp = empirical_from_trajectories(paths)
            empirical_by_n[n] = emp
            rho = empirical_correlation(paths)
            correlation_by_n[n] = rho
            append_checks(checks, theory, emp, rho, n, var_length, z)

    all_passed = all(all(v) for v in checks.values())
    return VerificationResult(
        experiment_id=experiment_id,
        sample_sizes=sample_sizes,
        theory=theory,
        empirical_by_n=empirical_by_n,
        correlation_by_n=correlation_by_n,
        checks=checks,
        confidence=confidence,
        passed=all_passed,
    )


def run_n_verifications(
    n_experiments: int = 1,
    *,
    source_experiment: str = "dummy",
    base_dir: Path | str | None = None,
    sample_sizes: list[int] | None = None,
    seed: int = 42,
    confidence: float = 0.95,
) -> list[VerificationResult]:
    """Create verifier_1..verifier_N, run simulator once per experiment, verify by subsampling."""
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    sample_sizes = sample_sizes or [200, 500, 1000]
    max_samples = max(sample_sizes)
    results = []
    for i in range(1, n_experiments + 1):
        exp_id = f"verifier_{i}"
        copy_experiment(source_experiment, exp_id, base_dir)
        res = run_verification(
            exp_id,
            sample_sizes=sample_sizes,
            base_dir=base_dir,
            seed=seed + i,
            confidence=confidence,
            max_samples=max_samples,
        )
        results.append(res)
    return results
