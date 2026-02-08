"""
Sample size for subscribe-rate comparison; publish as proxy (doc eq. fp-N, fp-moments).
P(q2 beats q1) for subscription rate (doc eq. fp-beats).
"""

from typing import Any
import numpy as np

from expforge.persona import PersonaSet
from expforge.goal import GoalSet
from expforge.trajectory.transition_matrix import _nested_probs_for
from expforge.theory.absorption import hitting_probabilities


def prob_subscribe_rate_larger(
    transition_matrix: dict[str, Any],
    persona_weights: list[float],
    persona_ids: list[str],
    goal_ids: list[str],
    q1: float,
    q2: float,
    goal_set: GoalSet,
    persona_set: PersonaSet,
) -> float:
    """P(p_sub(u; q2) > p_sub(u; q1)) for random user (doc eq. fp-beats)."""
    top_level = transition_matrix["top_level"]
    p_sub_q1 = []
    p_sub_q2 = []
    for p in persona_set.personas:
        nested_q1 = {g.id: _nested_probs_for(p.determined, q1) for g in goal_set.goals}
        nested_q2 = {g.id: _nested_probs_for(p.determined, q2) for g in goal_set.goals}
        mat_q1 = {"nested": {p.id: nested_q1}, "top_level": top_level}
        mat_q2 = {"nested": {p.id: nested_q2}, "top_level": top_level}
        _, h1 = hitting_probabilities(mat_q1, p.id, goal_ids)
        _, h2 = hitting_probabilities(mat_q2, p.id, goal_ids)
        p_sub_q1.append(h1)
        p_sub_q2.append(h2)
    p_sub_q1 = np.array(p_sub_q1)
    p_sub_q2 = np.array(p_sub_q2)
    p_weights = np.array(persona_weights)
    return float(np.sum(p_weights[p_sub_q2 > p_sub_q1]))


def sample_size_subscribe(
    transition_matrix: dict[str, Any],
    persona_weights: list[float],
    persona_ids: list[str],
    goal_ids: list[str],
    alpha: float = 0.05,
    power: float = 0.8,
    delta: float | None = None,
    q1: float = 0.4,
    q2: float = 0.6,
    goal_set: GoalSet | None = None,
    persona_set: PersonaSet | None = None,
    *,
    return_continuous: bool = False,
) -> float:
    """Required N per system to detect subscribe-rate difference (doc eq. fp-N).

    Outcome: P(ever subscribe) — user subscribes at most once. When return_continuous
    is True, returns the raw N (before ceiling) for smooth plots.
    """
    from scipy import stats

    if goal_set is None or persona_set is None:
        return np.nan
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    mus, sigs = [], []
    for q in (q1, q2):
        p_subs = []
        for p in persona_set.personas:
            nested_q = {g.id: _nested_probs_for(p.determined, q) for g in goal_set.goals}
            mat = {"nested": {p.id: nested_q}, "top_level": transition_matrix["top_level"]}
            _, h_sub = hitting_probabilities(mat, p.id, goal_ids)
            p_subs.append(h_sub)
        p_subs = np.array(p_subs)
        p_weights = np.array(persona_weights)
        mu = float(p_weights @ p_subs)
        sig2 = float(p_weights @ (p_subs - mu) ** 2)
        mus.append(mu)
        sigs.append(sig2)
    delta_val = delta if delta is not None else (mus[1] - mus[0])
    if delta_val <= 0:
        return np.inf
    N = (z_alpha + z_beta) ** 2 * (sigs[0] + sigs[1]) / (delta_val ** 2)
    if return_continuous:
        return float(N)
    return float(np.ceil(N))


def sample_size_publish_proxy(
    transition_matrix: dict[str, Any],
    persona_weights: list[float],
    persona_ids: list[str],
    goal_ids: list[str],
    alpha: float = 0.05,
    power: float = 0.8,
    q1: float = 0.4,
    q2: float = 0.6,
    goal_set: GoalSet | None = None,
    persona_set: Any = None,
    *,
    return_continuous: bool = False,
) -> tuple[float, float]:
    """N_pub for publish-as-proxy and ratio N_sub/N_pub. Returns (N_pub, ratio).

    Proxy outcome: P(ever publish). We care about publish actions *before* subscribe
    (user subscribes at most once); ever-publish is a tractable stand-in. When
    return_continuous is True, N_pub and ratio use raw N (no ceiling) for smooth plots.
    """
    from scipy import stats

    if goal_set is None or persona_set is None:
        return np.nan, np.nan
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    top = transition_matrix["top_level"]
    results = []
    for q in (q1, q2):
        h_pubs = []
        for p in persona_set.personas:
            nested_q = {g.id: _nested_probs_for(p.determined, q) for g in goal_set.goals}
            mat = {"nested": {p.id: nested_q}, "top_level": top}
            h_pub, _ = hitting_probabilities(mat, p.id, goal_ids)
            h_pubs.append(h_pub)
        h_pubs = np.array(h_pubs)
        p_weights = np.array(persona_weights)
        mu = float(p_weights @ h_pubs)
        sig2 = float(p_weights @ (h_pubs - mu) ** 2)
        results.append((mu, sig2))
    mu1, s1 = results[0]
    mu2, s2 = results[1]
    delta_pub = mu2 - mu1
    if delta_pub <= 0:
        return np.inf, np.nan
    N_pub = (z_alpha + z_beta) ** 2 * (s1 + s2) / (delta_pub ** 2)
    N_sub = sample_size_subscribe(
        transition_matrix, persona_weights, persona_ids, goal_ids,
        alpha=alpha, power=power, delta=None, q1=q1, q2=q2, goal_set=goal_set, persona_set=persona_set,
        return_continuous=return_continuous,
    )
    ratio = N_sub / N_pub if N_pub > 0 else np.nan
    if return_continuous:
        return float(N_pub), float(ratio)
    return float(np.ceil(N_pub)), float(ratio)
