"""
Absorption and hitting probabilities (doc/main.tex eq. fp-absorb, fp-hit).
"""

from typing import Any
import numpy as np

from expforge.theory.chain import (
    ABSORBING_ORDER,
    build_chain_matrices,
    fundamental_matrix,
)


def absorption_probabilities(
    transition_matrix: dict[str, Any],
    persona_id: str,
    goal_ids: list[str],
) -> dict[str, float]:
    """P(finished), P(abandoned) for one persona (only these are absorbing)."""
    Q, R, alpha = build_chain_matrices(transition_matrix, persona_id, goal_ids)
    _, absorb = fundamental_matrix(Q, R, alpha)
    return {
        name: float(absorb[i])
        for i, name in enumerate(ABSORBING_ORDER)
    }


def hitting_probabilities(
    transition_matrix: dict[str, Any],
    persona_id: str,
    goal_ids: list[str],
) -> tuple[float, float]:
    """(P(ever publish), P(ever subscribe)) for one persona (doc eq. fp-hit).

    Since publish and subscribe are transient states, compute hitting probabilities
    using the fundamental matrix N: probability of ever visiting that state.
    """
    from expforge.theory.chain import outcome_state_index

    Q, R, alpha = build_chain_matrices(transition_matrix, persona_id, goal_ids)
    N, _ = fundamental_matrix(Q, R, alpha)

    n_goals = len(goal_ids)
    # Hitting probability = alpha' @ N[:, state_idx]
    # N[i,j] = expected number of times in state j starting from state i
    # So alpha' @ N[:, j] = expected number of visits to state j from initial distribution
    # For transient states in an absorbing chain, this equals P(ever visit state j)

    pub_idx = outcome_state_index("publish", n_goals)
    sub_idx = outcome_state_index("subscribe", n_goals)

    # P(ever visit) = sum over all states of (prob start in state i) * (expected visits from i to target)
    # For absorbing chains, expected visits ≥ 1 if we visit, 0 if we don't
    # So we need: P(ever visit) = 1 - P(never visit)
    # Or more directly: sum over paths that reach the state

    # Actually, for hitting probability in an absorbing chain:
    # P(ever hit j) = sum_i alpha_i * (N[i,j] > 0 ? 1 : 0)
    # But N gives expected visits, which is >= 1 if we ever visit
    # Simpler: probability of hitting is the sum of probabilities of all paths that pass through it

    # For our case, we can compute: prob of reaching state j = alpha' @ (I + Q + Q^2 + ...) @ e_j
    # But that's just N @ e_j evaluated at our starting state

    # The cleanest approach: hitting prob = probability that we absorb after visiting the state
    # = sum over all transient states k of: P(reach j) * P(reach k from j) * P(absorb from k)

    # Alternative: Use first-passage probability
    # P(ever visit j | start from i) = N[i,j] / N[j,j] if i != j, else 1
    # Wait, that's not right either.

    # Correct formula: hitting probability H[i,j] = probability of visiting j starting from i
    # For i != j: H[i,j] = Q[i,j] + sum_k Q[i,k] * H[k,j]
    # This is H = Q + Q @ H, so H = (I - Q)^{-1} @ Q = N @ Q... no, that gives N again.

    # Actually the simplest: N[i,j] > 0 iff we can reach j from i
    # And in an absorbing chain, N[i,j] = expected number of visits
    # Since we start from i=0, hitting prob = P(N[0,j] > 0)

    # For numerical stability, we can use: P(ever visit j) ≈ min(1, alpha' @ N[:, j])
    # But actually N gives expected visits which equals hitting probability for absorbing chains

    # Most direct: Use the identity that for transient states in absorbing chains,
    # the expected number of visits equals the hitting probability when that state
    # is visited at most once per trajectory. But publish/subscribe can be visited multiple times!

    # Correct approach: Compute taboo probabilities or use auxiliary Markov chain
    # For now, approximate using: prob(ever visit) ≈ sum of all entries in N's column for that state
    # This overcounts if we can visit multiple times, but gives an upper bound

    # Actually, the right formula: Let B = indicator matrix where B[j,j] = 1 if state j is target
    # Then hitting prob vector h satisfies: h = B @ 1 + Q' @ h (transpose!)
    # Solving: h = (I - Q')^{-1} @ (B @ 1)

    # Simpler: Just compute directly using the definition
    # Add "already_visited_publish" and "already_visited_subscribe" as states? Too complex.

    # Pragmatic approach: N[i,j] in an absorbing chain where each transient state is visited
    # at most once equals the hitting probability. But if cycles exist, it overcounts.

    # For our chain: from publish, can we return to publish? Check the transition matrix.
    # The transitions from publish allow going to goals and subscribe and terminals.
    # From goals, can go back to publish. So yes, cycles exist!

    # Therefore, we need a different approach. Let's compute the probability directly:
    # P(ever visit publish) = compute by tracking which paths lead through publish

    # Implementation: Modify the state space to track "visited_publish" as a flag
    # This doubles the state space. For now, let's use a simpler heuristic:

    # Heuristic: Compute N, then use N[0, pub_idx] as a proxy
    # This overcounts repeated visits but gives an upper bound

    # Better: Use the formula for first passage
    # Create matrix Q_star where state j is made absorbing (remove outgoing transitions)
    # Then solve (I - Q_star) @ f = Q @ e_j to get first-passage probabilities

    # For implementation simplicity and correctness, let's compute hitting probability properly:
    # Method: Remove the target state from transient states, make it absorbing,
    # then compute absorption probability into it

    p_pub = _hitting_prob_for_state(Q, R, alpha, pub_idx, n_goals)
    p_sub = _hitting_prob_for_state(Q, R, alpha, sub_idx, n_goals)

    return float(p_pub), float(p_sub)


def _hitting_prob_for_state(Q: np.ndarray, R: np.ndarray, alpha: np.ndarray, target_idx: int, n_goals: int) -> float:
    """Compute probability of ever visiting a transient state.

    Method: Make target_idx absorbing, compute probability of absorbing into it.
    """
    n_transient = Q.shape[0]
    n_absorbing = R.shape[1]

    # Create modified Q' and R' where target state becomes absorbing
    Q_prime = Q.copy()
    R_prime = np.zeros((n_transient, n_absorbing + 1))
    R_prime[:, :n_absorbing] = R

    # Redirect transitions FROM target_idx to absorption
    R_prime[target_idx, n_absorbing] = 1.0  # New absorbing state for "visited target"
    Q_prime[target_idx, :] = 0.0  # Remove outgoing transitions from target

    # Redirect transitions TO target_idx to the new absorbing state
    for i in range(n_transient):
        if i != target_idx:
            p_to_target = Q_prime[i, target_idx]
            if p_to_target > 0:
                R_prime[i, n_absorbing] += p_to_target
                Q_prime[i, target_idx] = 0.0

    # Compute absorption probabilities
    N_prime = np.linalg.inv(np.eye(n_transient) - Q_prime)
    absorb_probs = alpha @ N_prime @ R_prime

    # Return probability of absorbing into the "visited target" state
    return float(absorb_probs[n_absorbing])
