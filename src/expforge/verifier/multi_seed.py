"""
Multi-seed verification: run verification over several seeds and require a pass rate.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from expforge.verifier.run import run_verification, VerificationResult

if TYPE_CHECKING:
    pass


def run_verification_multi_seed(
    experiment_id: str,
    seeds: list[int],
    *,
    base_dir: Path | str | None = None,
    sample_sizes: list[int] | None = None,
    max_samples: int | None = None,
    confidence: float = 0.95,
    pass_rate: float = 0.95,
) -> tuple[list[VerificationResult], bool]:
    """
    Run verification for each seed. Return (list of VerificationResult, overall_pass).
    overall_pass is True iff at least pass_rate fraction of seeds passed (e.g. 0.95 => 95%).
    """
    base_dir = Path(base_dir or Path(__file__).resolve().parent.parent / "simulator")
    sample_sizes = sample_sizes or [200, 500, 1000]
    if max_samples is None:
        max_samples = max(sample_sizes)

    results = []
    for seed in seeds:
        res = run_verification(
            experiment_id,
            sample_sizes=sample_sizes,
            base_dir=base_dir,
            seed=seed,
            confidence=confidence,
            max_samples=max_samples,
        )
        results.append(res)

    passed = sum(1 for r in results if r.passed)
    overall_pass = (passed / len(results)) >= pass_rate if results else False
    return results, overall_pass
