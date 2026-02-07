"""
Theoretical quantities for the discrete (finite-persona) nested Markov model.

Implements closed-form formulas from doc/main.tex §Method (finite personas).
"""

from expforge.theory.chain import (
    ABSORBING_ORDER,
    build_chain_matrices,
    fundamental_matrix,
    goal_nested_state_index,
)
from expforge.theory.absorption import (
    absorption_probabilities,
    hitting_probabilities,
)
from expforge.theory.moments import (
    expected_trajectory_length,
    expected_trajectory_length_squared,
)
from expforge.theory.correlation import (
    correlation_publish_subscribe,
    hitting_both_probability,
)
from expforge.theory.sample_size import (
    prob_subscribe_rate_larger,
    sample_size_subscribe,
    sample_size_publish_proxy,
)
from expforge.theory.values import TheoreticalValues

__all__ = [
    "ABSORBING_ORDER",
    "build_chain_matrices",
    "fundamental_matrix",
    "goal_nested_state_index",
    "absorption_probabilities",
    "hitting_probabilities",
    "expected_trajectory_length",
    "expected_trajectory_length_squared",
    "correlation_publish_subscribe",
    "hitting_both_probability",
    "prob_subscribe_rate_larger",
    "sample_size_subscribe",
    "sample_size_publish_proxy",
    "TheoreticalValues",
]
