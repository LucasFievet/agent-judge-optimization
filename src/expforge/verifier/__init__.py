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
    run_verification_batch_data,
    BatchReportData,
)
from expforge.verifier.io import copy_experiment, load_experiment, experiment_dir, DEFAULT_EXPERIMENTS_DIR
from expforge.verifier.multi_seed import run_verification_multi_seed
from expforge.verifier.report import (
    summary_table,
    pass_rate_table,
    write_summary_table,
    figures,
    figures_batch_distributions,
)

__all__ = [
    "run_verification",
    "run_n_verifications",
    "VerificationResult",
    "run_verification_batch_data",
    "BatchReportData",
    "copy_experiment",
    "load_experiment",
    "experiment_dir",
    "DEFAULT_EXPERIMENTS_DIR",
    "run_verification_multi_seed",
    "summary_table",
    "pass_rate_table",
    "write_summary_table",
    "figures",
    "figures_batch_distributions",
]
