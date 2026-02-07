"""
Aggregate theoretical values (finite personas): single entry point.
"""

from dataclasses import dataclass

from expforge.persona import PersonaSet
from expforge.goal import GoalSet
from expforge.trajectory.transition_matrix import build_transition_matrix

from expforge.theory.absorption import absorption_probabilities, hitting_probabilities
from expforge.theory.moments import expected_trajectory_length, expected_trajectory_length_squared
from expforge.theory.correlation import correlation_publish_subscribe
from expforge.theory.sample_size import (
    prob_subscribe_rate_larger,
    sample_size_subscribe,
    sample_size_publish_proxy,
)


@dataclass
class TheoreticalValues:
    """Container for all theoretical quantities (finite personas)."""

    expected_length: float
    expected_length_sq: float
    p_finished: float
    p_abandoned: float
    p_publish: float
    p_subscribe: float
    correlation_publish_subscribe: float
    prob_subscribe_q2_beats_q1: float | None
    N_subscribe: float | None
    N_publish_proxy: float | None
    N_sub_over_N_pub: float | None

    @classmethod
    def compute(
        cls,
        persona_set: PersonaSet,
        goal_set: GoalSet,
        q1: float = 0.4,
        q2: float = 0.6,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> "TheoreticalValues":
        """Compute all theoretical values from persona set and goal set."""
        matrix = build_transition_matrix(persona_set, goal_set)
        goal_ids = [g.id for g in goal_set.goals]
        weights = persona_set.get_weights()
        persona_ids = [p.id for p in persona_set.personas]

        elens = [expected_trajectory_length(matrix, pid, goal_ids) for pid in persona_ids]
        elen = sum(w * e for w, e in zip(weights, elens))
        elen2_list = [expected_trajectory_length_squared(matrix, pid, goal_ids) for pid in persona_ids]
        elen2 = sum(w * e for w, e in zip(weights, elen2_list))

        p_fin = sum(
            w * absorption_probabilities(matrix, pid, goal_ids)["finished"]
            for w, pid in zip(weights, persona_ids)
        )
        p_aban = sum(
            w * absorption_probabilities(matrix, pid, goal_ids)["abandoned"]
            for w, pid in zip(weights, persona_ids)
        )
        p_pub = sum(
            w * hitting_probabilities(matrix, pid, goal_ids)[0]
            for w, pid in zip(weights, persona_ids)
        )
        p_sub = sum(
            w * hitting_probabilities(matrix, pid, goal_ids)[1]
            for w, pid in zip(weights, persona_ids)
        )

        rho = correlation_publish_subscribe(matrix, weights, persona_ids, goal_ids)

        try:
            prob_beats = prob_subscribe_rate_larger(
                matrix, weights, persona_ids, goal_ids, q1, q2, goal_set, persona_set
            )
        except Exception:
            prob_beats = None
        try:
            N_sub = sample_size_subscribe(
                matrix, weights, persona_ids, goal_ids,
                alpha=alpha, power=power, q1=q1, q2=q2, goal_set=goal_set, persona_set=persona_set,
            )
        except Exception:
            N_sub = None
        try:
            N_pub, ratio = sample_size_publish_proxy(
                matrix, weights, persona_ids, goal_ids,
                alpha=alpha, power=power, q1=q1, q2=q2, goal_set=goal_set, persona_set=persona_set,
            )
        except Exception:
            N_pub = None
            ratio = None

        return cls(
            expected_length=float(elen),
            expected_length_sq=float(elen2),
            p_finished=float(p_fin),
            p_abandoned=float(p_aban),
            p_publish=float(p_pub),
            p_subscribe=float(p_sub),
            correlation_publish_subscribe=rho,
            prob_subscribe_q2_beats_q1=prob_beats,
            N_subscribe=N_sub,
            N_publish_proxy=N_pub,
            N_sub_over_N_pub=ratio,
        )
