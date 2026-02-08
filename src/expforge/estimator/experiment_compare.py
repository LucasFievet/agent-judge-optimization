"""Compare two experiments and output confidence interval on one being better than the other."""

from pathlib import Path
from typing import Any

import yaml

from expforge.verifier.io import DEFAULT_EXPERIMENTS_DIR


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load metrics.yaml."""
    with metrics_path.open() as f:
        return yaml.safe_load(f)


def run_experiment_compare(
    experiment_id_a: str,
    experiment_id_b: str,
    metric: str = "subscribe",
    *,
    base_dir: Path | str | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Compare experiments A and B on the given metric (e.g. subscribe).
    Returns dict with point difference (B - A), confidence interval for the difference,
    and P(metric_B > metric_A) or similar.
    """
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    path_a = base_dir / "experiment" / experiment_id_a / "metrics.yaml"
    path_b = base_dir / "experiment" / experiment_id_b / "metrics.yaml"

    if not path_a.exists() or not path_b.exists():
        return {
            "experiment_a": experiment_id_a,
            "experiment_b": experiment_id_b,
            "metric": metric,
            "error": "missing metrics.yaml for one or both experiments",
        }

    m_a = load_metrics(path_a)
    m_b = load_metrics(path_b)

    obs_a = m_a.get("metrics", {}).get(metric, {}).get("observed", {})
    obs_b = m_b.get("metrics", {}).get(metric, {}).get("observed", {})

    point_a = obs_a.get("point", 0.0)
    point_b = obs_b.get("point", 0.0)
    diff = point_b - point_a
    lower_a, upper_a = obs_a.get("lower", 0), obs_a.get("upper", 0)
    lower_b, upper_b = obs_b.get("lower", 0), obs_b.get("upper", 0)
    # Approximate CI for difference (independent)
    import math
    se_a = (upper_a - lower_a) / (2 * 1.96) if (upper_a - lower_a) else 0
    se_b = (upper_b - lower_b) / (2 * 1.96) if (upper_b - lower_b) else 0
    se_diff = math.sqrt(se_a**2 + se_b**2)
    z = 1.96 if confidence >= 0.95 else 1.645
    diff_lower = diff - z * se_diff
    diff_upper = diff + z * se_diff

    return {
        "experiment_a": experiment_id_a,
        "experiment_b": experiment_id_b,
        "metric": metric,
        "point_a": point_a,
        "point_b": point_b,
        "difference_b_minus_a": diff,
        "confidence_interval_difference": [diff_lower, diff_upper],
        "confidence": confidence,
        "conclusion": "B better than A" if diff_lower > 0 else ("A better than B" if diff_upper < 0 else "inconclusive"),
    }
