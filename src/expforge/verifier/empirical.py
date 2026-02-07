"""Compute empirical statistics from trajectory files."""

from pathlib import Path

from expforge.trajectory import load_trajectory


def empirical_from_trajectories(trajectory_paths: list[Path]) -> dict[str, float]:
    """Compute empirical means from saved trajectories.

    Per PRD: Only finished and abandoned are terminal states (outcomes).
    p_publish and p_subscribe are "hitting probabilities" - probability of ever visiting those states.
    """
    lengths = []
    lengths_sq = []
    n_finished = 0
    n_abandoned = 0
    n_publish = 0
    n_subscribe = 0
    for path in trajectory_paths:
        traj = load_trajectory(path)
        n_steps = len(traj.steps)
        lengths.append(n_steps)
        lengths_sq.append(n_steps * n_steps)
        outcome = traj.outcome or ""
        if outcome == "finished":
            n_finished += 1
        elif outcome == "abandoned":
            n_abandoned += 1
        # For publish/subscribe, check if we ever visited that state
        ever_pub = any(s.top_level_state == "publish" for s in traj.steps)
        ever_sub = any(s.top_level_state == "subscribe" for s in traj.steps)
        if ever_pub:
            n_publish += 1
        if ever_sub:
            n_subscribe += 1
    n = len(trajectory_paths) or 1
    return {
        "mean_length": sum(lengths) / n,
        "mean_length_sq": sum(lengths_sq) / n,
        "p_finished": n_finished / n,
        "p_abandoned": n_abandoned / n,
        "p_publish": n_publish / n,
        "p_subscribe": n_subscribe / n,
        "n": n,
    }


def empirical_correlation(trajectory_paths: list[Path]) -> float:
    """Sample correlation between ever-publish and ever-subscribe."""
    y_pub = []
    y_sub = []
    for path in trajectory_paths:
        traj = load_trajectory(path)
        y_pub.append(1.0 if any(s.top_level_state == "publish" for s in traj.steps) else 0.0)
        y_sub.append(1.0 if any(s.top_level_state == "subscribe" for s in traj.steps) else 0.0)
    n = len(y_pub)
    if n < 2:
        return 0.0
    m_pub = sum(y_pub) / n
    m_sub = sum(y_sub) / n
    var_pub = sum((y - m_pub) ** 2 for y in y_pub) / (n - 1) or 1e-12
    var_sub = sum((y - m_sub) ** 2 for y in y_sub) / (n - 1) or 1e-12
    cov = sum((y_pub[i] - m_pub) * (y_sub[i] - m_sub) for i in range(n)) / (n - 1)
    if var_pub * var_sub <= 0:
        return 0.0
    return float(max(-1.0, min(1.0, cov / (var_pub * var_sub) ** 0.5)))
