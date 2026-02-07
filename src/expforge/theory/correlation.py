"""
Correlation between Y_pub (ever publish) and Y_sub (ever subscribe) (doc eq. fp-corr).
Exact P(both hit) via augmented chain when both can be visited; otherwise proxy.
"""

from typing import Any
import numpy as np

from expforge.theory.chain import ABSORBING_ORDER, build_chain_matrices, fundamental_matrix
from expforge.theory.absorption import hitting_probabilities


def hitting_both_probability(
    transition_matrix: dict[str, Any],
    persona_id: str,
    goal_ids: list[str],
) -> float:
    """
    P(ever hit publish AND ever hit subscribe) for one persona.

    Since publish and subscribe are now transient states (not absorbing), we can visit both.
    Method: Create augmented chain that tracks whether we've visited each state.
    State space: (base_state, visited_pub, visited_sub) where visited_* ∈ {0,1}

    We compute the probability of reaching a state where both flags are 1.
    """
    from expforge.theory.chain import outcome_state_index

    Q, R, alpha = build_chain_matrices(transition_matrix, persona_id, goal_ids)
    n_transient = Q.shape[0]
    n_absorbing = R.shape[1]
    n_goals = len(goal_ids)

    try:
        pub_idx = outcome_state_index("publish", n_goals)
        sub_idx = outcome_state_index("subscribe", n_goals)
    except (ValueError, IndexError):
        # If outcome states don't exist in model, can't visit both
        return 0.0

    # Augmented state: (base_state, visited_pub, visited_sub)
    # Encoding: base * 4 + visited_pub * 2 + visited_sub
    n_aug = n_transient * 4

    Q_aug = np.zeros((n_aug, n_aug))
    R_aug = np.zeros((n_aug, n_absorbing))

    # Build augmented transition matrix
    for s in range(n_transient):
        for visited_pub in [0, 1]:
            for visited_sub in [0, 1]:
                aug_s = s * 4 + visited_pub * 2 + visited_sub

                # Transitions to other transient states
                for s_next in range(n_transient):
                    if Q[s, s_next] > 0:
                        # Update flags if we visit publish/subscribe
                        new_pub = visited_pub or (s_next == pub_idx)
                        new_sub = visited_sub or (s_next == sub_idx)
                        aug_s_next = s_next * 4 + new_pub * 2 + new_sub
                        Q_aug[aug_s, aug_s_next] = Q[s, s_next]

                # Transitions to absorbing states
                for abs_idx in range(n_absorbing):
                    if R[s, abs_idx] > 0:
                        R_aug[aug_s, abs_idx] = R[s, abs_idx]

    # Initial distribution: start in base initial state with flags (0,0)
    alpha_aug = np.zeros(n_aug)
    for s in range(n_transient):
        if alpha[s] > 0:
            aug_s = s * 4  # visited_pub=0, visited_sub=0
            alpha_aug[aug_s] = alpha[s]

    # Compute fundamental matrix for augmented chain
    N_aug = np.linalg.inv(np.eye(n_aug) - Q_aug)

    # Probability of "both" = sum over all states with both flags set
    # At absorption, sum over (s, 1, 1) states weighted by expected visits
    prob_both = 0.0
    for s in range(n_transient):
        aug_s_both = s * 4 + 1 * 2 + 1  # visited_pub=1, visited_sub=1
        # Expected visits to this augmented state
        expected_visits = alpha_aug @ N_aug[:, aug_s_both]
        # Probability of absorbing from this state
        prob_absorb_from_s = R[s, :].sum()
        prob_both += expected_visits * prob_absorb_from_s

    return float(np.clip(prob_both, 0.0, 1.0))


def correlation_publish_subscribe(
    transition_matrix: dict[str, Any],
    persona_weights: list[float],
    persona_ids: list[str],
    goal_ids: list[str],
    use_exact_both: bool = True,
) -> float:
    """
    Correlation rho between Y_pub and Y_sub over the population (doc eq. fp-corr).
    If use_exact_both: use hitting_both_probability per persona (exact); else use min(h_pub, h_sub) proxy.
    """
    h_pub = np.array([
        hitting_probabilities(transition_matrix, pid, goal_ids)[0]
        for pid in persona_ids
    ])
    h_sub = np.array([
        hitting_probabilities(transition_matrix, pid, goal_ids)[1]
        for pid in persona_ids
    ])
    p = np.array(persona_weights)
    P_pub = float(p @ h_pub)
    P_sub = float(p @ h_sub)

    if use_exact_both:
        h_both = np.array([
            hitting_both_probability(transition_matrix, pid, goal_ids)
            for pid in persona_ids
        ])
        P_both = float(p @ h_both)
    else:
        h_both = np.minimum(h_pub, h_sub)
        P_both = float(p @ h_both)

    var_pub = P_pub * (1 - P_pub)
    var_sub = P_sub * (1 - P_sub)
    if var_pub <= 0 or var_sub <= 0:
        return 0.0
    cov = P_both - P_pub * P_sub
    rho = cov / np.sqrt(var_pub * var_sub)
    return float(np.clip(rho, -1.0, 1.0))
