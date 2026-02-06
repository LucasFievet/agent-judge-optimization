"""Scoring: infer persona/goal clusters from trajectories and compute metrics."""

from expforge.scoring.experiment_scoring import run_experiment_scoring
from expforge.scoring.experiment_compare import run_experiment_compare

__all__ = [
    "run_experiment_scoring",
    "run_experiment_compare",
]
