"""
Verifier confidence: two experiments with different tool quality (q1, q2).
Runs batch data for both, plots distributions + theory, confidence vs N, and P(sub/pub) vs q.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from expforge.persona import PersonaSet
from expforge.goal import GoalSet, save_goal_set
from expforge.theory import TheoreticalValues
from expforge.theory.absorption import hitting_probabilities
from expforge.trajectory.transition_matrix import build_transition_matrix, _nested_probs_for
from expforge.verifier.io import (
    load_experiment,
    copy_experiment,
    experiment_dir,
    ensure_experiment_exists,
    DEFAULT_EXPERIMENTS_DIR,
)
from expforge.verifier.run import run_verification_batch_data, BatchReportData

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBatchData:
    """Batch data and theory for two tool qualities q1 and q2."""

    experiment_id: str
    q1: float
    q2: float
    batch_data_q1: BatchReportData
    batch_data_q2: BatchReportData


def _moments_for_q(
    persona_set: PersonaSet,
    goal_set: GoalSet,
    q: float,
) -> tuple[float, float, float, float, float, float]:
    """(mu_sub, sigma_sub_sq, mu_pub, sigma_pub_sq, E[p_sub(1-p_sub)], E[p_pub(1-p_pub)]) for one tool quality q.

    The variance of the sample mean from N trajectories is (Var_u[p] + E[p(1-p)])/N (doc: between-persona
    variance plus within-persona Bernoulli variance). So we return both for correct power/SE.
    """
    import numpy as np

    goal_set_q = goal_set.with_uniform_tool_quality(q)
    matrix = build_transition_matrix(persona_set, goal_set_q)
    goal_ids = [g.id for g in goal_set.goals]
    weights = np.array(persona_set.get_weights())
    top = matrix["top_level"]

    p_subs = []
    p_pubs = []
    for p in persona_set.personas:
        nested_q = {g.id: _nested_probs_for(p.determined, q) for g in goal_set.goals}
        mat = {"nested": {p.id: nested_q}, "top_level": top}
        h_pub, h_sub = hitting_probabilities(mat, p.id, goal_ids)
        p_pubs.append(h_pub)
        p_subs.append(h_sub)
    p_subs = np.array(p_subs)
    p_pubs = np.array(p_pubs)

    mu_sub = float(weights @ p_subs)
    mu_pub = float(weights @ p_pubs)
    sig2_sub = float(weights @ (p_subs - mu_sub) ** 2)
    sig2_pub = float(weights @ (p_pubs - mu_pub) ** 2)
    # Within-persona Bernoulli variance E[p(1-p)] (needed for variance of sample mean)
    ep_sub = float(weights @ (p_subs * (1 - p_subs)))
    ep_pub = float(weights @ (p_pubs * (1 - p_pubs)))
    return mu_sub, sig2_sub, mu_pub, sig2_pub, ep_sub, ep_pub


def run_confidence_batch_data(
    experiment_id: str,
    q1: float = 0.4,
    q2: float = 0.6,
    *,
    total_samples: int = 10_000,
    batch_size: int = 100,
    base_dir: Path | str | None = None,
    seed: int = 42,
    override: bool = False,
    cache_path_q1: Path | str | None = None,
    cache_path_q2: Path | str | None = None,
) -> ConfidenceBatchData:
    """
    Create two experiment copies with tool quality q1 and q2, run batch data for each.
    Experiment ids will be <experiment_id>_conf_q1 and <experiment_id>_conf_q2.
    """
    base_dir = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    ensure_experiment_exists(base_dir, experiment_id, seed=seed)
    persona_set, goal_set = load_experiment(base_dir, experiment_id)

    exp_q1 = f"{experiment_id}_conf_q1"
    exp_q2 = f"{experiment_id}_conf_q2"
    copy_experiment(experiment_id, exp_q1, base_dir)
    copy_experiment(experiment_id, exp_q2, base_dir)

    goal_set_q1 = goal_set.with_uniform_tool_quality(q1)
    goal_set_q2 = goal_set.with_uniform_tool_quality(q2)
    save_goal_set(goal_set_q1, experiment_dir(base_dir, exp_q1) / "goals.yaml")
    save_goal_set(goal_set_q2, experiment_dir(base_dir, exp_q2) / "goals.yaml")

    out_dir = base_dir / "experiment" / experiment_id / "confidence_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache1 = Path(cache_path_q1) if cache_path_q1 else out_dir / "batch_data_q1.json"
    cache2 = Path(cache_path_q2) if cache_path_q2 else out_dir / "batch_data_q2.json"

    batch_data_q1 = run_verification_batch_data(
        exp_q1,
        total_samples=total_samples,
        batch_size=batch_size,
        base_dir=base_dir,
        seed=seed,
        override=override,
        cache_path=cache1,
    )
    batch_data_q2 = run_verification_batch_data(
        exp_q2,
        total_samples=total_samples,
        batch_size=batch_size,
        base_dir=base_dir,
        seed=seed,
        override=override,
        cache_path=cache2,
    )

    return ConfidenceBatchData(
        experiment_id=experiment_id,
        q1=q1,
        q2=q2,
        batch_data_q1=batch_data_q1,
        batch_data_q2=batch_data_q2,
    )


def figures_confidence(
    data: ConfidenceBatchData,
    out_dir: Path | str,
    *,
    base_dir: Path | str | None = None,
    dpi: int = 150,
) -> list[Path]:
    """
    Generate confidence report figures:
    1. Batch distributions (same 2x3 as report) with both q1 and q2 empirical + theory in different colors.
    2. Confidence that q2 > q1 vs sample size N (subscribe and publish proxy).
    3. Theoretical P(subscribe) and P(publish) vs tool quality q in [0, 1].
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        import matplotlib
        import numpy as np
        from scipy import stats as scipy_stats

        matplotlib.use("Agg")
    except ImportError:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Resolve base_dir for loading source experiment (persona + goal structure)
    if base_dir is not None:
        base_dir = Path(base_dir)
    else:
        # Default: .../experiment/<id>/confidence_report -> project base contains "experiment"
        if "experiment" in out_dir.parts:
            idx = list(out_dir.parts).index("experiment")
            base_dir = Path(*out_dir.parts[:idx])
        else:
            base_dir = out_dir.parent
    persona_set, goal_set = load_experiment(base_dir, data.experiment_id)

    # --- 1. Batch distributions: both q1 and q2, both theory ---
    paths.extend(
        _fig_batch_distributions(data, out_dir, dpi=dpi)
    )

    # --- 2. Confidence (power) that we conclude q2 > q1 vs N ---
    max_samples = max(data.batch_data_q1.total_samples, data.batch_data_q2.total_samples)
    logger.info("[confidence] Computing confidence-vs-N and theory-vs-q (q1=%.2f, q2=%.2f, N up to %d)", data.q1, data.q2, max_samples)
    p = _fig_confidence_vs_n(
        persona_set, goal_set, data.q1, data.q2, out_dir,
        max_n=max_samples,
        dpi=dpi,
    )
    if p:
        paths.append(p)

    # --- 3. P(subscribe) and P(publish) vs q in [0, 1] ---
    p = _fig_theory_vs_q(persona_set, goal_set, out_dir, dpi=dpi)
    if p:
        paths.append(p)

    return paths


def _fig_batch_distributions(
    data: ConfidenceBatchData,
    out_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    """Same 2x3 panels as report: two histograms (q1, q2) and two theory lines per panel."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import numpy as np
    from scipy import stats as scipy_stats

    d1 = data.batch_data_q1
    d2 = data.batch_data_q2
    t1 = d1.theory
    t2 = d2.theory
    b = d1.batch_size

    def std_length(t):
        var = max(0.0, getattr(t, "expected_length_sq", 0) - (t.expected_length ** 2))
        return math.sqrt(var / b) if b else 0.0

    def std_prop(p):
        return math.sqrt(p * (1 - p) / b) if b else 0.0

    panels = [
        ("batch_means_length", t1.expected_length, std_length(t1), t2.expected_length, std_length(t2), r"$\mathbb{E}[L]$", True),
        ("batch_p_finished", t1.p_finished, std_prop(t1.p_finished), t2.p_finished, std_prop(t2.p_finished), r"$P(\text{finished})$", True),
        ("batch_p_abandoned", t1.p_abandoned, std_prop(t1.p_abandoned), t2.p_abandoned, std_prop(t2.p_abandoned), r"$P(\text{abandoned})$", True),
        ("batch_p_publish", t1.p_publish, std_prop(t1.p_publish), t2.p_publish, std_prop(t2.p_publish), r"$P(\text{publish})$", True),
        ("batch_p_subscribe", t1.p_subscribe, std_prop(t1.p_subscribe), t2.p_subscribe, std_prop(t2.p_subscribe), r"$P(\text{subscribe})$", True),
        ("batch_correlations", t1.correlation_publish_subscribe, None, t2.correlation_publish_subscribe, None, r"$\rho(\text{pub},\text{sub})$", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes_flat = axes.flatten()

    for idx, (key, mu1, std1, mu2, std2, ylabel, overlay_normal) in enumerate(panels):
        ax = axes_flat[idx]
        v1 = getattr(d1, key)
        v2 = getattr(d2, key)
        if not v1 and not v2:
            ax.set_title(ylabel)
            continue
        bins = min(25, max(10, (len(v1 or []) + len(v2 or [])) // 6))
        if v1:
            ax.hist(v1, bins=bins, density=True, alpha=0.5, color="C0", edgecolor="black", linewidth=0.3, label=f"q1={data.q1}")
        if v2:
            ax.hist(v2, bins=bins, density=True, alpha=0.5, color="C1", edgecolor="black", linewidth=0.3, label=f"q2={data.q2}")
        ax.axvline(mu1, color="C0", linestyle="--", linewidth=2)
        ax.axvline(mu2, color="C1", linestyle="--", linewidth=2)
        if overlay_normal and std1 is not None and std1 > 0:
            xmin, xmax = ax.get_xlim()
            xx = np.linspace(xmin, xmax, 200)
            ax.plot(xx, scipy_stats.norm.pdf(xx, mu1, std1), color="C0", linestyle="-", linewidth=1.5)
        if overlay_normal and std2 is not None and std2 > 0:
            xmin, xmax = ax.get_xlim()
            xx = np.linspace(xmin, xmax, 200)
            ax.plot(xx, scipy_stats.norm.pdf(xx, mu2, std2), color="C1", linestyle="-", linewidth=1.5)
        ax.set_ylabel("Density")
        ax.set_xlabel(ylabel)

    fig.suptitle(
        f"Batch statistics: q1={data.q1}, q2={data.q2} (batch size={b}, {d1.total_samples + d2.total_samples} total samples)",
        fontsize=11,
        y=0.98,
    )
    legend_handles = [
        mpatches.Patch(facecolor="C0", alpha=0.5, edgecolor="black", label=f"Empirical q1"),
        mpatches.Patch(facecolor="C1", alpha=0.5, edgecolor="black", label=f"Empirical q2"),
        mlines.Line2D([], [], color="C0", linestyle="--", linewidth=2, label="Theory q1"),
        mlines.Line2D([], [], color="C1", linestyle="--", linewidth=2, label="Theory q2"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = out_dir / "batch_distributions.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [p]


def _fig_confidence_vs_n(
    persona_set: PersonaSet,
    goal_set: GoalSet,
    q1: float,
    q2: float,
    out_dir: Path,
    *,
    max_n: int = 5000,
    alpha: float = 0.05,
    dpi: int = 150,
) -> Path | None:
    """Plot probability (power) that we conclude q2 > q1 as function of sample size N per system.

    Variance of sample mean p̂ from N trajectories: Var(p̂) = (Var_u[p] + E[p(1-p)])/N (between-persona
    + within-persona Bernoulli). Using only Var_u[p] gives SE too small and power ≈ 1 for small N.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    out1 = _moments_for_q(persona_set, goal_set, q1)
    out2 = _moments_for_q(persona_set, goal_set, q2)
    mu_sub_1, sig2_sub_1, mu_pub_1, sig2_pub_1, ep_sub_1, ep_pub_1 = out1
    mu_sub_2, sig2_sub_2, mu_pub_2, sig2_pub_2, ep_sub_2, ep_pub_2 = out2

    delta_sub = mu_sub_2 - mu_sub_1
    delta_pub = mu_pub_2 - mu_pub_1
    z_alpha = stats.norm.ppf(1 - alpha)  # one-sided

    # Total variance of (p̂2 - p̂1): (v1 + v2)/N with v_q = Var_u[p] + E[p(1-p)]
    v_sub_1 = sig2_sub_1 + ep_sub_1
    v_sub_2 = sig2_sub_2 + ep_sub_2
    v_pub_1 = sig2_pub_1 + ep_pub_1
    v_pub_2 = sig2_pub_2 + ep_pub_2

    logger.info(
        "[confidence_vs_N] q1=%.2f q2=%.2f | subscribe: mu1=%.4f mu2=%.4f delta=%.4f | "
        "sig2_1=%.6f sig2_2=%.6f E[p(1-p)]_1=%.4f E[p(1-p)]_2=%.4f | v1+v2=%.4f",
        q1, q2, mu_sub_1, mu_sub_2, delta_sub,
        sig2_sub_1, sig2_sub_2, ep_sub_1, ep_sub_2, v_sub_1 + v_sub_2,
    )
    logger.info(
        "[confidence_vs_N] publish: mu1=%.4f mu2=%.4f delta=%.4f | "
        "sig2_1=%.6f sig2_2=%.6f E[p(1-p)]_1=%.4f E[p(1-p)]_2=%.4f | v1+v2=%.4f",
        mu_pub_1, mu_pub_2, delta_pub,
        sig2_pub_1, sig2_pub_2, ep_pub_1, ep_pub_2, v_pub_1 + v_pub_2,
    )

    step = max(50, max_n // 200)  # ~200 points or step 50, whichever is coarser
    N_grid = np.arange(50, max_n + 1, step)
    if N_grid.size == 0 or N_grid[-1] < max_n:
        N_grid = np.append(N_grid, max_n)
    power_sub = []
    power_pub = []
    for N in N_grid:
        if delta_sub > 0 and (v_sub_1 + v_sub_2) > 0:
            # power = P(reject H0 | H1) = Phi( delta*sqrt(N)/sqrt(v1+v2) - z_alpha )
            p_sub = stats.norm.cdf(delta_sub * np.sqrt(N) / np.sqrt(v_sub_1 + v_sub_2) - z_alpha)
        else:
            p_sub = 0.0
        power_sub.append(p_sub)
        if delta_pub > 0 and (v_pub_1 + v_pub_2) > 0:
            p_pub = stats.norm.cdf(delta_pub * np.sqrt(N) / np.sqrt(v_pub_1 + v_pub_2) - z_alpha)
        else:
            p_pub = 0.0
        power_pub.append(p_pub)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_grid, power_sub, color="C0", label="Subscribe rate", linewidth=2)
    ax.plot(N_grid, power_pub, color="C1", label="Publish rate (proxy)", linewidth=2)
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.8, label="80% power")
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.8, label="95% power")
    ax.set_xlabel("Sample size per system ($N$)")
    ax.set_ylabel("Confidence (power) that we conclude $q_2 > q_1$")
    ax.set_title(f"One-sided test power ($q_1={q1}$, $q_2={q2}$, $\\alpha={alpha}$)")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "confidence_vs_sample_size.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def _fig_theory_vs_q(
    persona_set: PersonaSet,
    goal_set: GoalSet,
    out_dir: Path,
    *,
    n_q: int = 101,
    dpi: int = 150,
) -> Path | None:
    """Plot theoretical P(subscribe) and P(publish) as function of tool quality q in [0, 1]."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Log sensitivity: nested p_success depends on 0.25 + 0.5*determined + 0.25*q, so q contributes at most 0.25
    for q_log in (0.0, 0.5, 1.0):
        p_success_vals = [
            _nested_probs_for(p.determined, q_log)["succeeded"]
            for p in persona_set.personas
        ]
        logger.info(
            "[theory_vs_q] q=%.1f nested p_success per persona: %s (range %.3f–%.3f)",
            q_log, [round(x, 3) for x in p_success_vals], min(p_success_vals), max(p_success_vals),
        )

    q_grid = np.linspace(0.0, 1.0, n_q)
    p_sub = []
    p_pub = []
    for q in q_grid:
        gs_q = goal_set.with_uniform_tool_quality(float(q))
        theory = TheoreticalValues.compute(persona_set, gs_q)
        p_sub.append(theory.p_subscribe)
        p_pub.append(theory.p_publish)

    p_sub_arr = np.array(p_sub)
    p_pub_arr = np.array(p_pub)
    logger.info(
        "[theory_vs_q] P(subscribe): min=%.4f max=%.4f range=%.4f at q in [0,1]",
        float(p_sub_arr.min()), float(p_sub_arr.max()), float(p_sub_arr.max() - p_sub_arr.min()),
    )
    logger.info(
        "[theory_vs_q] P(publish):   min=%.4f max=%.4f range=%.4f at q in [0,1]",
        float(p_pub_arr.min()), float(p_pub_arr.max()), float(p_pub_arr.max() - p_pub_arr.min()),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q_grid, p_sub, color="C0", label=r"$P(\text{subscribe})$", linewidth=2)
    ax.plot(q_grid, p_pub, color="C1", label=r"$P(\text{publish})$", linewidth=2)
    ax.set_xlabel("Tool quality $q$")
    ax.set_ylabel("Probability")
    ax.set_title("Theoretical outcome probabilities vs tool quality")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "theory_vs_quality.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p
