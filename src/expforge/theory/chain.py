"""
Build transient/absorbing chain matrices (Q, R, alpha) for one persona from transition matrix.

Updated model:
- Transient states: start, goal×{succeeded,failed,continue}, publish, subscribe
- Absorbing states: finished, abandoned
"""

from typing import Any
import numpy as np

# Only finished and abandoned are absorbing (terminal) states
ABSORBING_ORDER = ("finished", "abandoned")
# Publish and subscribe are transient "outcome" states
OUTCOME_STATES = ("publish", "subscribe")


def goal_nested_state_index(goal_idx: int, nested: str, n_goals: int) -> int:
    """Transient index for (goal_idx, nested). nested in ('succeeded', 'failed', 'continue')."""
    n = {"succeeded": 0, "failed": 1, "continue": 2}[nested]
    return 1 + goal_idx * 3 + n


def outcome_state_index(outcome: str, n_goals: int) -> int:
    """Transient index for outcome states (publish, subscribe)."""
    # Outcome states come after: start (0) + goals (3*n_goals)
    base = 1 + 3 * n_goals
    if outcome == "publish":
        return base
    elif outcome == "subscribe":
        return base + 1
    else:
        raise ValueError(f"Unknown outcome state: {outcome}")


def build_chain_matrices(
    transition_matrix: dict[str, Any],
    persona_id: str,
    goal_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Q (transient x transient), R (transient x absorbing), and alpha (initial dist)
    for one persona from a transition_matrix dict (nested[persona_id][goal_id], top_level).

    Transient states: start (0), goal×{succeeded,failed,continue} (1..3n), publish (3n+1), subscribe (3n+2)
    Absorbing states: finished (0), abandoned (1)
    """
    nested = transition_matrix.get("nested", {}).get(persona_id, {})
    top = transition_matrix.get("top_level", {})
    from_start = top.get("from_start", {})
    from_succ = top.get("from_goal_succeeded", {})
    from_fail = top.get("from_goal_failed", {})
    from_cont = top.get("from_goal_continue", {})
    from_pub = top.get("from_publish", {})
    from_sub = top.get("from_subscribe", {})

    n_goals = len(goal_ids)
    # Transient: start + 3*goals + publish + subscribe
    n_transient = 1 + 3 * n_goals + 2
    # Absorbing: finished, abandoned only
    n_absorbing = 2

    Q = np.zeros((n_transient, n_transient))
    R = np.zeros((n_transient, n_absorbing))

    # From start
    for j, gid in enumerate(goal_ids):
        Q[0, goal_nested_state_index(j, "continue", n_goals)] = from_start.get(gid, 0.0)

    # Helper to add transition probabilities
    def add_transitions(from_idx: int, transition_probs: dict[str, float]) -> None:
        for next_id, p in transition_probs.items():
            if next_id in goal_ids:
                j = goal_ids.index(next_id)
                Q[from_idx, goal_nested_state_index(j, "continue", n_goals)] += p
            elif next_id in OUTCOME_STATES:
                Q[from_idx, outcome_state_index(next_id, n_goals)] += p
            elif next_id in ABSORBING_ORDER:
                R[from_idx, ABSORBING_ORDER.index(next_id)] += p

    # From goal states
    # Model: When entering a goal, we immediately sample nested outcome, then transition to next top-level.
    # We use a 2-step process: entry -> outcome_state -> next_top
    # idx_c is the entry state, idx_s/idx_f are outcome states
    # We reuse idx_c as both entry AND as the "continue outcome" state (this is a bit hacky but works)

    for i, gid in enumerate(goal_ids):
        idx_s = goal_nested_state_index(i, "succeeded", n_goals)
        idx_f = goal_nested_state_index(i, "failed", n_goals)
        idx_c = goal_nested_state_index(i, "continue", n_goals)

        nested_probs = nested.get(gid, {})
        p_succ = nested_probs.get("succeeded", 0.0)
        p_fail = nested_probs.get("failed", 0.0)
        p_cont = nested_probs.get("continue", 0.0)

        # From goal entry (idx_c), sample nested outcome to transition to outcome states
        Q[idx_c, idx_s] = p_succ
        Q[idx_c, idx_f] = p_fail
        # For "continue" outcome, we could create a separate state, but instead we'll
        # transition directly to next top-level states from idx_c

        # From each outcome state, transition to next top-level state
        add_transitions(idx_s, from_succ)
        add_transitions(idx_f, from_fail)
        # For continue outcome, add transitions directly from idx_c (weighted by p_cont)
        # Actually, we need to handle this carefully. The issue is idx_c serves dual purpose.

        # Clean approach: Don't use idx_c for "continue" outcome. Instead, from idx_c,
        # transition directly with combined probability: p(nested) * p(next | nested)
        # This collapses the 2-step into 1-step for "continue" outcome only.

        for next_id, p_next in from_cont.items():
            # Probability of: sample "continue", then go to next_id
            combined_p = p_cont * p_next
            if next_id in goal_ids:
                j = goal_ids.index(next_id)
                Q[idx_c, goal_nested_state_index(j, "continue", n_goals)] += combined_p
            elif next_id in OUTCOME_STATES:
                Q[idx_c, outcome_state_index(next_id, n_goals)] += combined_p
            elif next_id in ABSORBING_ORDER:
                R[idx_c, ABSORBING_ORDER.index(next_id)] += combined_p

    # From outcome states (publish, subscribe)
    add_transitions(outcome_state_index("publish", n_goals), from_pub)
    add_transitions(outcome_state_index("subscribe", n_goals), from_sub)

    # Normalize rows
    row_sums = Q.sum(axis=1) + R.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    Q = Q / row_sums[:, np.newaxis]
    R = R / row_sums[:, np.newaxis]

    alpha = np.zeros(n_transient)
    alpha[0] = 1.0

    return Q, R, alpha


def fundamental_matrix(Q: np.ndarray, R: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N = (I - Q)^{-1}, absorption probs = alpha' N R."""
    n = Q.shape[0]
    N = np.linalg.inv(np.eye(n) - Q)
    absorb = alpha @ N @ R
    return N, absorb
