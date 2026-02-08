"""
Heatmaps for EM verifier: full transition matrix (top-level + nested) aggregated over persona,
with same state labels on x and y axes.
"""

import yaml
from pathlib import Path

import numpy as np


PHASES = ("succeeded", "continue", "failed")
TERMINALS = ("finished", "abandoned")
TOP_NON_TERMINALS = ("publish", "subscribe")


def _state_list(goal_ids: list[str]) -> list[str]:
    """Full state space: start, then for each goal (continue, succeeded, failed), then publish, subscribe, finished, abandoned."""
    states = ["start"]
    for gid in goal_ids:
        states.append(f"{gid}_continue")
        states.append(f"{gid}_succeeded")
        states.append(f"{gid}_failed")
    states.extend(["publish", "subscribe", "finished", "abandoned"])
    return states


def _state_labels_hierarchical(states: list[str], goal_ids: list[str]) -> list[str]:
    """Goal name (bold) only on first row of each goal block; then just phase (continue, succeeded, failed)."""
    labels = []
    for s in states:
        if s == "start":
            labels.append("start")
        elif s in ("publish", "subscribe", "finished", "abandoned"):
            labels.append(s)
        else:
            for gid in goal_ids:
                if s.startswith(gid + "_") and s[len(gid) + 1 :] in ("continue", "succeeded", "failed"):
                    ph = s[len(gid) + 1 :]
                    if ph == "continue":
                        goal_display = gid.replace("_", " ")
                        labels.append(r"$\mathbf{" + goal_display.replace(" ", r"\ ") + r"}$" + "\n" + ph)
                    else:
                        labels.append(ph)  # succeeded or failed, no repeated goal name
                    break
            else:
                labels.append(s.replace("_", " "))
    return labels


def _build_full_matrix(
    nested: dict[str, dict[str, dict[str, float]]],
    top_level: dict[str, dict[str, float]],
    goal_ids: list[str],
    persona_ids: list[str],
) -> np.ndarray:
    """
    Build full transition matrix P(s'|s) for one persona (nested is per-persona) or already aggregated nested.
    nested[goal_id] = {succeeded, continue, failed}; top_level = from_start, from_goal_succeeded, etc.
    """
    states = _state_list(goal_ids)
    n = len(states)
    stoi = {s: i for i, s in enumerate(states)}
    P = np.zeros((n, n))

    from_start = top_level.get("from_start", {})
    from_succ = top_level.get("from_goal_succeeded", {})
    from_fail = top_level.get("from_goal_failed", {})
    from_pub = top_level.get("from_publish", {})
    from_sub = top_level.get("from_subscribe", {})

    # start -> goal_continue
    for gid in goal_ids:
        idx = stoi.get(f"{gid}_continue")
        if idx is not None:
            P[stoi["start"], idx] = float(from_start.get(gid, 0.0))

    for gid in goal_ids:
        probs = nested.get(gid, {})
        p_succ = float(probs.get("succeeded", 0.0))
        p_cont = float(probs.get("continue", 0.0))
        p_fail = float(probs.get("failed", 0.0))
        i_cont = stoi[f"{gid}_continue"]
        i_succ = stoi[f"{gid}_succeeded"]
        i_fail = stoi[f"{gid}_failed"]
        P[i_cont, i_succ] = p_succ
        P[i_cont, i_cont] = p_cont
        P[i_cont, i_fail] = p_fail

        # goal_succeeded -> next (goals as _continue, or publish, subscribe, finished)
        for to_gid in goal_ids:
            j = stoi.get(f"{to_gid}_continue")
            if j is not None:
                P[i_succ, j] = float(from_succ.get(to_gid, 0.0))
        P[i_succ, stoi["publish"]] = float(from_succ.get("publish", 0.0))
        P[i_succ, stoi["subscribe"]] = float(from_succ.get("subscribe", 0.0))
        P[i_succ, stoi["finished"]] = float(from_succ.get("finished", 0.0))

        # goal_failed -> next (goals or abandoned)
        for to_gid in goal_ids:
            j = stoi.get(f"{to_gid}_continue")
            if j is not None:
                P[i_fail, j] = float(from_fail.get(to_gid, 0.0))
        P[i_fail, stoi["abandoned"]] = float(from_fail.get("abandoned", 0.0))
        # goal_continue: only nested outcomes (succeeded, continue, failed) — no from_goal_continue
    # publish -> from_publish
    for to_gid in goal_ids:
        j = stoi.get(f"{to_gid}_continue")
        if j is not None:
            P[stoi["publish"], j] = float(from_pub.get(to_gid, 0.0))
    P[stoi["publish"], stoi["subscribe"]] = float(from_pub.get("subscribe", 0.0))
    P[stoi["publish"], stoi["finished"]] = float(from_pub.get("finished", 0.0))
    P[stoi["publish"], stoi["abandoned"]] = float(from_pub.get("abandoned", 0.0))

    # subscribe -> from_subscribe
    for to_gid in goal_ids:
        j = stoi.get(f"{to_gid}_continue")
        if j is not None:
            P[stoi["subscribe"], j] = float(from_sub.get(to_gid, 0.0))
    P[stoi["subscribe"], stoi["publish"]] = float(from_sub.get("publish", 0.0))
    P[stoi["subscribe"], stoi["finished"]] = float(from_sub.get("finished", 0.0))
    P[stoi["subscribe"], stoi["abandoned"]] = float(from_sub.get("abandoned", 0.0))

    # absorbing
    P[stoi["finished"], stoi["finished"]] = 1.0
    P[stoi["abandoned"], stoi["abandoned"]] = 1.0

    return P


def _aggregate_nested_over_persona(
    nested: dict[str, dict[str, dict[str, float]]],
    persona_ids: list[str],
    goal_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Average nested[persona][goal] over personas -> nested_avg[goal] = {succeeded, continue, failed}."""
    out: dict[str, dict[str, float]] = {gid: {"succeeded": 0.0, "continue": 0.0, "failed": 0.0} for gid in goal_ids}
    n_p = len(persona_ids) or 1
    for pid in persona_ids:
        for gid in goal_ids:
            probs = (nested.get(pid) or {}).get(gid) or {}
            for ph in ("succeeded", "continue", "failed"):
                out[gid][ph] += float(probs.get(ph, 0.0)) / n_p
    return out


def _load_transitions(transitions_path: Path) -> tuple[dict, dict]:
    """Load nested and top_level from transitions.yaml."""
    if not transitions_path.is_file():
        return {}, {}
    with open(transitions_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("nested", {}), data.get("top_level", {})


def plot_em_heatmaps(
    ground_truth: dict[str, dict[str, dict[str, float]]],
    estimated: dict[str, dict[str, dict[str, float]]],
    output_path: Path | str,
    *,
    persona_ids: list[str] | None = None,
    goal_ids: list[str] | None = None,
    transitions_path: Path | str | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """
    Plot two heatmaps side by side with same state labels on x and y:

    1. Full transition matrix (top-level + nested) aggregated over persona → estimated P(s'|s).
    2. Difference P_true − P_est (same layout).

    State space: start, then for each goal (goal_continue, goal_succeeded, goal_failed), then publish, subscribe, finished, abandoned.

    If transitions_path is provided, top_level is loaded and the full combined matrix is built and averaged over personas. If not, falls back to nested-only (persona × goal × phase) heatmaps.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    persona_ids = persona_ids or list(ground_truth.keys())
    goal_ids = goal_ids or list(next(iter(ground_truth.values()), {}).keys())
    if not goal_ids and persona_ids:
        goal_ids = list(ground_truth.get(persona_ids[0], {}).keys())

    nested_true_avg = _aggregate_nested_over_persona(ground_truth, persona_ids, goal_ids)
    nested_est_avg = _aggregate_nested_over_persona(estimated, persona_ids, goal_ids)

    use_full_matrix = False
    if transitions_path:
        _, top_level = _load_transitions(Path(transitions_path))
        if top_level:
            P_true = _build_full_matrix(nested_true_avg, top_level, goal_ids, [])
            P_est = _build_full_matrix(nested_est_avg, top_level, goal_ids, [])
            states = _state_list(goal_ids)
            state_labels = _state_labels_hierarchical(states, goal_ids)
            n = len(states)
            P_diff = P_true - P_est
            use_full_matrix = True

    if use_full_matrix:
        # A4 portrait: two heatmaps stacked vertically, filling the page
        a4_portrait = (8.27, 11.69)  # inches
        if figsize is None:
            figsize = a4_portrait
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.patch.set_facecolor("white")

        im1 = ax1.imshow(P_true, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax1.set_xticks(np.arange(n))
        ax1.set_xticklabels(state_labels, fontsize=9, rotation=45, ha="right")
        ax1.set_yticks(np.arange(n))
        ax1.set_yticklabels(state_labels, fontsize=9)
        ax1.set_ylabel("current state", fontsize=11)
        ax1.set_title("True $P(s'\\mid s)$, aggregated over persona", fontsize=12)
        plt.colorbar(im1, ax=ax1, shrink=0.6, label="probability")

        v_abs = max(abs(float(np.nanmin(P_diff))), abs(float(np.nanmax(P_diff))), 0.01)
        im2 = ax2.imshow(P_diff, aspect="auto", cmap="RdBu_r", vmin=-v_abs, vmax=v_abs)
        ax2.set_xticks(np.arange(n))
        ax2.set_xticklabels(state_labels, fontsize=9, rotation=45, ha="right")
        ax2.set_yticks(np.arange(n))
        ax2.set_yticklabels(state_labels, fontsize=9)
        ax2.set_xlabel("next state", fontsize=11)
        ax2.set_ylabel("current state", fontsize=11)
        ax2.set_title(r"$P_{\mathrm{true}} - P_{\mathrm{est}}$", fontsize=12)
        plt.colorbar(im2, ax=ax2, shrink=0.6, label="difference")

        for ax in (ax1, ax2):
            ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        return output_path

    # Fallback: original nested-only (persona × goal) × phase
    M_true = np.array([
        [float((ground_truth.get(pid) or {}).get(gid, {}).get(ph, 0.0)) for ph in ("succeeded", "continue", "failed")]
        for pid in persona_ids for gid in goal_ids
    ])
    M_est = np.array([
        [float((estimated.get(pid) or {}).get(gid, {}).get(ph, 0.0)) for ph in ("succeeded", "continue", "failed")]
        for pid in persona_ids for gid in goal_ids
    ])
    M_diff = M_true - M_est
    short_rows = []
    for pid in persona_ids:
        short_p = pid.replace("persona_", "p") if pid.startswith("persona_") else pid[:6]
        for gid in goal_ids:
            short_rows.append(f"{short_p}  {gid.replace('_', ' ')}")
    n_rows = len(short_rows)
    col_labels_display = ["succeeded", "continue", "failed"]
    if figsize is None:
        figsize = (10, max(8, n_rows * 0.35))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.patch.set_facecolor("white")
    im1 = ax1.imshow(M_est, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(col_labels_display, fontsize=12)
    ax1.set_yticks(np.arange(n_rows))
    ax1.set_yticklabels(short_rows, fontsize=10)
    ax1.set_ylabel("persona · goal", fontsize=12)
    ax1.set_title("Estimated P(phase)", fontsize=13)
    plt.colorbar(im1, ax=ax1, shrink=0.7, label="probability")
    v_abs = max(abs(float(np.nanmin(M_diff))), abs(float(np.nanmax(M_diff))), 0.01)
    im2 = ax2.imshow(M_diff, aspect="auto", cmap="RdBu_r", vmin=-v_abs, vmax=v_abs)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(col_labels_display, fontsize=12)
    ax2.set_yticks(np.arange(n_rows))
    ax2.set_yticklabels(short_rows, fontsize=10)
    ax2.set_title("P_true − P_est", fontsize=13)
    plt.colorbar(im2, ax=ax2, shrink=0.7, label="difference")
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(4) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path
