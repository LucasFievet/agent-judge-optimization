"""
Theory-based figures: sample size vs publish-heaviness.

Target: subscribe (once). Proxy: publish; we care about publish actions *before*
subscribe; the plot uses P(ever publish) as a tractable stand-in.
"""

import logging
from pathlib import Path

import numpy as np

from expforge.persona import PersonaSet
from expforge.goal import GoalSet
from expforge.trajectory.transition_matrix import build_transition_matrix, DEFAULT_OUTCOME_WEIGHTS
from expforge.theory.sample_size import sample_size_subscribe, sample_size_publish_proxy

logger = logging.getLogger(__name__)


def figure_sample_size_vs_publish_heaviness(
    persona_set: PersonaSet,
    goal_set: GoalSet,
    out_dir: Path,
    *,
    q1: float = 0.4,
    q2: float = 0.6,
    power: float = 0.8,
    alpha: float = 0.05,
    weight_ratios: list[float] | None = None,
    dpi: int = 150,
) -> Path:
    """Plot minimal N for subscribe vs publish (proxy) and ratio. Uses continuous N so curves can cross.

    Target outcome: subscribe (once). Proxy: ever publish (stand-in for publish-before-subscribe).
    Left axis: required N per system (smooth curves; round up in practice). When orange is below
    blue, the publish proxy needs fewer samples. Right axis: N_sub/N_pub; when above 1, publish
    is the more efficient proxy. Gray line at ratio = 1 is the break-even.
    """
    import matplotlib.pyplot as plt

    # Wider range so ratio has room to vary (e.g. 0.25 = subscribe-heavy, 5 = publish-heavy)
    if weight_ratios is None:
        weight_ratios = np.linspace(0.25, 5.0, 40).tolist()

    goal_ids = [g.id for g in goal_set.goals]
    weights = np.array(persona_set.get_weights())
    persona_ids = [p.id for p in persona_set.personas]

    x_ratio = []
    n_sub_list = []
    n_pub_list = []
    ratio_list = []

    # Use continuous N (no ceiling) so blue and orange can cross and ratio crosses 1 smoothly
    for w_ratio in weight_ratios:
        ow = {**DEFAULT_OUTCOME_WEIGHTS, "publish": float(w_ratio), "subscribe": 1.0}
        matrix = build_transition_matrix(persona_set, goal_set, outcome_weights=ow)

        try:
            n_sub = sample_size_subscribe(
                matrix,
                weights.tolist(),
                persona_ids,
                goal_ids,
                alpha=alpha,
                power=power,
                q1=q1,
                q2=q2,
                goal_set=goal_set,
                persona_set=persona_set,
                return_continuous=True,
            )
            n_pub, n_sub_over_n_pub = sample_size_publish_proxy(
                matrix,
                weights.tolist(),
                persona_ids,
                goal_ids,
                alpha=alpha,
                power=power,
                q1=q1,
                q2=q2,
                goal_set=goal_set,
                persona_set=persona_set,
                return_continuous=True,
            )
        except Exception as e:
            logger.warning("sample_size failed at w_ratio=%.2f: %s", w_ratio, e)
            n_sub = np.nan
            n_pub = np.nan
            n_sub_over_n_pub = np.nan

        x_ratio.append(w_ratio)
        n_sub_list.append(n_sub)
        n_pub_list.append(n_pub)
        ratio_list.append(n_sub_over_n_pub)

    x_ratio = np.array(x_ratio)
    n_sub_list = np.array(n_sub_list, dtype=float)
    n_pub_list = np.array(n_pub_list, dtype=float)
    ratio_list = np.array(ratio_list, dtype=float)
    valid = np.isfinite(n_sub_list) & np.isfinite(n_pub_list) & (n_sub_list > 0) & (n_pub_list > 0)
    valid_ratio = valid & np.isfinite(ratio_list) & (ratio_list > 0)

    # Left axis (N): scale so N lines use ~60–80% of vertical space, with padding
    n_max = float(np.max(np.r_[n_sub_list[valid], n_pub_list[valid]]))
    n_min = float(np.min(np.r_[n_sub_list[valid], n_pub_list[valid]]))
    n_span = max(n_max - n_min, 1.0)
    n_lo = max(0, n_min - 0.15 * n_span)
    n_hi = n_max + 0.25 * n_span
    # Round to nice tick-friendly bounds
    n_hi = max(n_hi, 10)  # avoid tiny scale when N is single-digit
    n_lo = 0

    # Right axis (ratio): center around 1.0 so the ratio line has visible slope
    r_min = float(np.min(ratio_list[valid_ratio]))
    r_max = float(np.max(ratio_list[valid_ratio]))
    r_span = max(r_max - r_min, 0.05)
    r_margin = max(0.12, r_span * 0.6)
    r_lo = min(1.0 - r_margin, r_min - 0.05)
    r_hi = max(1.0 + r_margin, r_max + 0.05)
    r_lo = max(0.85, r_lo)  # keep above 0 for readability

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x_ratio[valid], n_sub_list[valid], color="C0", label="Target: subscribe (once)", linewidth=2)
    ax1.plot(x_ratio[valid], n_pub_list[valid], color="C1", label="Proxy: ever publish", linewidth=2)
    # x-axis = system design (transition weights), not sample ratio
    ax1.set_xlabel(
        "Publish / subscribe outcome weight ratio (system design: how often chain goes to publish vs subscribe)"
    )
    ax1.set_ylabel("Required $N$ per system (80% power; round up in practice)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_ylim(n_lo, n_hi)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x_ratio[valid_ratio],
        ratio_list[valid_ratio],
        color="C2",
        linestyle="--",
        linewidth=2,
        label=r"$N_{\mathrm{sub}}/N_{\mathrm{pub}}$",
    )
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.8, linewidth=1.5, label="Ratio = 1 (break-even)")
    # Right y-axis = sample-size ratio (outcome of the model), not the design weight
    ax2.set_ylabel(
        r"Sample-size ratio $N_{\mathrm{sub}}/N_{\mathrm{pub}}$ (above 1 $\Rightarrow$ publish proxy needs fewer samples)",
        color="C2",
    )
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim(r_lo, r_hi)
    ax2.legend(loc="upper right")

    ax1.set_title(
        f"Target: subscribe (once). Proxy: ever publish. Required $N$ to detect $q_2 > q_1$ "
        f"($q_1={q1}$, $q_2={q2}$, power={power:.0%}). Orange below blue $\\Rightarrow$ proxy needs fewer samples."
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "sample_size_vs_publish_heaviness.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p
