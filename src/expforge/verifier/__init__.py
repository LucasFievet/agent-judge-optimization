"""
Cross-check theory vs simulator for the finite-persona nested Markov model.

Runs the simulator in fast mode (no LLM) at various sample sizes, computes
theoretical expectations, and checks whether empirical means fall within
expected confidence intervals.
"""

from expforge.verifier.run import (
    run_verification,
    run_n_verifications,
    VerificationResult,
)
from expforge.verifier.io import copy_experiment, load_experiment, experiment_dir
from expforge.verifier.multi_seed import run_verification_multi_seed
from expforge.verifier.report import (
    summary_table,
    pass_rate_table,
    write_summary_table,
    figures,
)

__all__ = [
    "run_verification",
    "run_n_verifications",
    "VerificationResult",
    "copy_experiment",
    "load_experiment",
    "experiment_dir",
    "run_verification_multi_seed",
    "summary_table",
    "pass_rate_table",
    "write_summary_table",
    "figures",
]
