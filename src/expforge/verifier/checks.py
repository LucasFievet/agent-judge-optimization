"""CI checks: compare empirical to theory and record pass/fail."""

import logging
import numpy as np
from scipy import stats

from expforge.theory import TheoreticalValues

logger = logging.getLogger(__name__)


def append_checks(
    checks: dict[str, list[bool]],
    theory: TheoreticalValues,
    emp: dict[str, float],
    rho_emp: float,
    n: int,
    var_length: float,
    z: float,
    correlation_tolerance: float = 0.25,
) -> None:
    """Append one set of checks for sample size n."""
    logger.debug(f"Checking n={n}")

    se_length = np.sqrt(var_length / n)
    ci_length_lo = theory.expected_length - z * se_length
    ci_length_hi = theory.expected_length + z * se_length
    passed_length = ci_length_lo <= emp["mean_length"] <= ci_length_hi
    checks["mean_length"].append(passed_length)
    if not passed_length:
        logger.info(f"mean_length check failed at n={n}: empirical={emp['mean_length']:.3f}, theory={theory.expected_length:.3f}, CI=[{ci_length_lo:.3f}, {ci_length_hi:.3f}]")

    for key, theory_val in [
        ("p_finished", theory.p_finished),
        ("p_abandoned", theory.p_abandoned),
        ("p_publish", theory.p_publish),
        ("p_subscribe", theory.p_subscribe),
    ]:
        p = max(1e-6, min(1 - 1e-6, theory_val))
        se_p = np.sqrt(p * (1 - p) / n)
        ci_lo = theory_val - z * se_p
        ci_hi = theory_val + z * se_p
        passed = ci_lo <= emp[key] <= ci_hi
        checks[key].append(passed)
        if not passed:
            logger.info(f"{key} check failed at n={n}: empirical={emp[key]:.3f}, theory={theory_val:.3f}, CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

    passed_corr = abs(rho_emp - theory.correlation_publish_subscribe) <= correlation_tolerance
    checks["correlation"].append(passed_corr)
    if not passed_corr:
        logger.info(f"correlation check failed at n={n}: empirical={rho_emp:.3f}, theory={theory.correlation_publish_subscribe:.3f}, tolerance={correlation_tolerance}")


def z_for_confidence(confidence: float) -> float:
    """Half-width z for symmetric CI."""
    return stats.norm.ppf(0.5 + confidence / 2)
