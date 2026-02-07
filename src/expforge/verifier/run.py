"""
Run verification: compare theory to simulator (fast mode) at several sample sizes.
"""

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from expforge.theory import TheoreticalValues
from expforge.simulator.experiment_simulator import run_simulator

from expforge.verifier.io import (
    load_experiment,
    copy_experiment,
    experiment_dir,
    ensure_experiment_exists,
    load_existing_sample_paths,
    delete_sample_files,
    DEFAULT_EXPERIMENTS_DIR,
)
from expforge.verifier.empirical import (
    empirical_from_trajectories,
    empirical_correlation,
    batch_empirical_stats,
)
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


@dataclass
class BatchReportData:
    """Theory plus per-batch statistics for distribution plots."""

    experiment_id: str
    theory: TheoreticalValues
    batch_means_length: list[float]
    batch_p_finished: list[float]
    batch_p_abandoned: list[float]
    batch_p_publish: list[float]
    batch_p_subscribe: list[float]
    batch_correlations: list[float]
    batch_size: int
    total_samples: int


def _load_batch_cache(cache_path: Path, total_samples: int, batch_size: int) -> BatchReportData | None:
    """Load BatchReportData from JSON if file exists and matches total_samples/batch_size."""
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if raw.get("total_samples") != total_samples or raw.get("batch_size") != batch_size:
        return None
    theory_dict = raw.get("theory") or {}
    theory = SimpleNamespace(
        expected_length=theory_dict.get("expected_length", 0.0),
        expected_length_sq=theory_dict.get("expected_length_sq", 0.0),
        p_finished=theory_dict.get("p_finished", 0.0),
        p_abandoned=theory_dict.get("p_abandoned", 0.0),
        p_publish=theory_dict.get("p_publish", 0.0),
        p_subscribe=theory_dict.get("p_subscribe", 0.0),
        correlation_publish_subscribe=theory_dict.get("correlation_publish_subscribe", 0.0),
    )
    return BatchReportData(
        experiment_id=raw.get("experiment_id", ""),
        theory=theory,
        batch_means_length=raw.get("batch_means_length", []),
        batch_p_finished=raw.get("batch_p_finished", []),
        batch_p_abandoned=raw.get("batch_p_abandoned", []),
        batch_p_publish=raw.get("batch_p_publish", []),
        batch_p_subscribe=raw.get("batch_p_subscribe", []),
        batch_correlations=raw.get("batch_correlations", []),
        batch_size=batch_size,
        total_samples=total_samples,
    )


def _save_batch_cache(cache_path: Path, data: BatchReportData) -> None:
    """Save BatchReportData to JSON (theory as dict with plot-needed fields only)."""
    t = data.theory
    payload = {
        "experiment_id": data.experiment_id,
        "total_samples": data.total_samples,
        "batch_size": data.batch_size,
        "theory": {
            "expected_length": t.expected_length,
            "expected_length_sq": t.expected_length_sq,
            "p_finished": t.p_finished,
            "p_abandoned": t.p_abandoned,
            "p_publish": t.p_publish,
            "p_subscribe": t.p_subscribe,
            "correlation_publish_subscribe": t.correlation_publish_subscribe,
        },
        "batch_means_length": data.batch_means_length,
        "batch_p_finished": data.batch_p_finished,
        "batch_p_abandoned": data.batch_p_abandoned,
        "batch_p_publish": data.batch_p_publish,
        "batch_p_subscribe": data.batch_p_subscribe,
        "batch_correlations": data.batch_correlations,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0)


def run_verification_batch_data(
    experiment_id: str,
    *,
    total_samples: int = 10_000,
    batch_size: int = 100,
    base_dir: Path | str | None = None,
    seed: int = 42,
    override: bool = False,
    cache_path: Path | str | None = None,
) -> BatchReportData:
    """
    Get per-batch statistics for distribution plots. By default reuses existing
    sample_*.yaml if at least total_samples are present; with override=True
    deletes samples and runs the simulator afresh. If cache_path is set and
    cache is valid (same total_samples, batch_size), load from cache for fast re-render.
    """
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    cache_file = Path(cache_path) if cache_path else None

    if override and cache_file and cache_file.is_file():
        cache_file.unlink()

    if cache_file and not override:
        t0 = time.perf_counter()
        cached = _load_batch_cache(cache_file, total_samples, batch_size)
        if cached is not None:
            elapsed = time.perf_counter() - t0
            print(f"[verifier] Loaded batch data from cache in {elapsed:.2f}s")
            return cached

    t0 = time.perf_counter()
    ensure_experiment_exists(base_dir, experiment_id, seed=seed)
    persona_set, goal_set = load_experiment(base_dir, experiment_id)
    theory = TheoreticalValues.compute(persona_set, goal_set)
    t1 = time.perf_counter()
    print(f"[verifier] Load config + theory in {t1 - t0:.2f}s")

    if override:
        delete_sample_files(base_dir, experiment_id)
        all_paths = None
    else:
        t_load = time.perf_counter()
        existing = load_existing_sample_paths(base_dir, experiment_id)
        print(f"[verifier] Loaded {len(existing)} existing paths in {time.perf_counter() - t_load:.2f}s")
        all_paths = existing[:total_samples] if len(existing) >= total_samples else None

    if all_paths is None:
        t_sim = time.perf_counter()
        _, _, all_paths = run_simulator(
            experiment_id,
            total_samples,
            base_dir=base_dir,
            seed=seed,
            reuse_config=True,
            use_llm=False,
        )
        print(f"[verifier] Simulator run ({total_samples} samples) in {time.perf_counter() - t_sim:.2f}s")
    else:
        all_paths = list(all_paths)

    n_use = (len(all_paths) // batch_size) * batch_size
    paths = all_paths[:n_use] if n_use < len(all_paths) else all_paths
    num_batches = len(paths) // batch_size
    t_batch = time.perf_counter()
    stats = batch_empirical_stats(paths, batch_size)
    print(f"[verifier] Batch stats ({num_batches} batches of {batch_size}) in {time.perf_counter() - t_batch:.2f}s")

    data = BatchReportData(
        experiment_id=experiment_id,
        theory=theory,
        batch_means_length=stats["batch_means_length"],
        batch_p_finished=stats["batch_p_finished"],
        batch_p_abandoned=stats["batch_p_abandoned"],
        batch_p_publish=stats["batch_p_publish"],
        batch_p_subscribe=stats["batch_p_subscribe"],
        batch_correlations=stats["batch_correlations"],
        batch_size=batch_size,
        total_samples=len(paths),
    )
    if cache_file:
        _save_batch_cache(cache_file, data)
        print(f"[verifier] Saved batch data to cache")
    return data
