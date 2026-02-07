"""
Publication-ready tables and figures from verification results.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expforge.verifier.run import VerificationResult, BatchReportData

_METRIC_LABELS = {
    "mean_length": r"$\mathbb{E}[L]$",
    "p_finished": r"$P(\text{finished})$",
    "p_abandoned": r"$P(\text{abandoned})$",
    "p_publish": r"$P(\text{publish})$",
    "p_subscribe": r"$P(\text{subscribe})$",
    "correlation": r"$\rho(\text{pub},\text{sub})$",
}


def summary_table(
    result: "VerificationResult",
    *,
    fmt: str = "markdown",
) -> str:
    """
    One table: rows = (metric, theory, n_1 emp, pass_1, n_2 emp, pass_2, ...).
    fmt: "markdown" | "latex" | "plain"
    """
    theory = result.theory
    theory_vals = {
        "mean_length": theory.expected_length,
        "p_finished": theory.p_finished,
        "p_abandoned": theory.p_abandoned,
        "p_publish": theory.p_publish,
        "p_subscribe": theory.p_subscribe,
        "correlation": theory.correlation_publish_subscribe,
    }
    rows = []
    for metric in ["mean_length", "p_finished", "p_abandoned", "p_publish", "p_subscribe", "correlation"]:
        row = [metric, theory_vals[metric]]
        for n in result.sample_sizes:
            emp = result.empirical_by_n.get(n, {})
            if metric == "correlation":
                val = result.correlation_by_n.get(n, 0.0)
            else:
                val = emp.get(metric, 0.0)
            passed = _check_passed(result, metric, n)
            row.extend([val, "✓" if passed else "✗"])
        rows.append(row)

    if fmt == "plain":
        return _plain_table(rows, result.sample_sizes)
    if fmt == "latex":
        return _latex_table(rows, result.sample_sizes)
    return _markdown_table(rows, result.sample_sizes)


def _check_passed(result: "VerificationResult", metric: str, n: int) -> bool:
    """Whether the check for this metric at this n passed."""
    idx = result.sample_sizes.index(n) if n in result.sample_sizes else -1
    if idx < 0:
        return False
    return bool(result.checks.get(metric, [False] * (idx + 1))[idx])


def _plain_table(rows: list, sample_sizes: list[int]) -> str:
    lines = []
    header = ["metric", "theory"] + [f"n={n}" for n in sample_sizes for _ in ("emp", "pass")]
    lines.append("\t".join(header))
    for row in rows:
        lines.append("\t".join(str(x) for x in row))
    return "\n".join(lines)


def _markdown_table(rows: list, sample_sizes: list[int]) -> str:
    cols = ["Metric", "Theory"] + [f"n={n} (emp / pass)" for n in sample_sizes]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for row in rows:
        metric = row[0]
        label = _METRIC_LABELS.get(metric, metric)
        cells = [label, f"{row[1]:.4f}"]
        for i in range(0, len(row) - 2, 2):
            cells.append(f"{row[2 + i]:.4f} / {row[3 + i]}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _latex_table(rows: list, sample_sizes: list[int]) -> str:
    lines = ["\\begin{tabular}{l r" + " r r" * len(sample_sizes) + "}", "\\toprule"]
    lines.append(
        "Metric & Theory & "
        + " & ".join("\\multicolumn{2}{c}{$n=" + str(n) + "$}" for n in sample_sizes)
        + " \\\\"
    )
    lines.append(" & & " + " & ".join("emp & pass" for _ in sample_sizes) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        label = _METRIC_LABELS.get(row[0], row[0])
        cells = [label, f"{row[1]:.4f}"]
        for i in range(0, len(row) - 2, 2):
            cells.append(f"{row[2 + i]:.4f} & {row[3 + i]}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def pass_rate_table(
    results_by_n: dict[int, list[bool]],
    *,
    fmt: str = "markdown",
) -> str:
    """
    For multi-seed: rows = (metric, pass_rate at n_1, n_2, ...).
    results_by_n: map n -> list of pass booleans per seed (or per check).
    """
    if not results_by_n:
        return ""
    ns = sorted(results_by_n.keys())
    total = len(next(iter(results_by_n.values())))
    rows = []
    for n in ns:
        passed = sum(results_by_n[n])
        rate = passed / total if total else 0.0
        rows.append((n, rate, passed, total))
    if fmt == "latex":
        return (
            "\\begin{tabular}{rrrr}\n"
            "\\toprule\n$n$ & Pass rate & Passed & Total \\\\\n\\midrule\n"
            + "\n".join(f"{n} & {r:.2%} & {p} & {t} \\\\" for n, r, p, t in rows)
            + "\n\\bottomrule\n\\end{tabular}"
        )
    if fmt == "markdown":
        return (
            "| $n$ | Pass rate | Passed | Total |\n|-----|----------|--------|-------|\n"
            + "\n".join(f"| {n} | {r:.2%} | {p} | {t} |" for n, r, p, t in rows)
        )
    return "\n".join(f"n={n}\t{r:.2%}\t{p}/{t}" for n, r, p, t in rows)


def write_summary_table(
    result: "VerificationResult",
    out_path: Path | str,
    *,
    fmt: str = "markdown",
) -> None:
    """Write summary table to file. fmt: markdown | latex."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(summary_table(result, fmt=fmt), encoding="utf-8")


def figures(
    result: "VerificationResult",
    out_dir: Path | str,
    *,
    dpi: int = 150,
) -> list[Path]:
    """
    No-op: vs-n plots were removed. Use figures_batch_distributions for batch distribution plots.
    Returns empty list for backward compatibility.
    """
    return []


def figures_batch_distributions(
    batch_data: "BatchReportData",
    out_dir: Path | str,
    *,
    dpi: int = 150,
) -> list[Path]:
    """
    Plot distributions of per-batch statistics (from one large run) with
    theoretical sampling distributions overlaid. One 2×3 figure: mean length,
    four proportions, correlation. One legend at top. Saves batch_distributions.pdf.
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
    theory = batch_data.theory
    b = batch_data.batch_size
    var_length = max(0.0, theory.expected_length_sq - theory.expected_length ** 2)
    std_length = math.sqrt(var_length / b) if b else 0.0

    panels = [
        ("batch_means_length", theory.expected_length, std_length, r"$\mathbb{E}[L]$ (batch mean)", True),
        ("batch_p_finished", theory.p_finished, math.sqrt(theory.p_finished * (1 - theory.p_finished) / b) if b else 0, r"$P(\text{finished})$", True),
        ("batch_p_abandoned", theory.p_abandoned, math.sqrt(theory.p_abandoned * (1 - theory.p_abandoned) / b) if b else 0, r"$P(\text{abandoned})$", True),
        ("batch_p_publish", theory.p_publish, math.sqrt(theory.p_publish * (1 - theory.p_publish) / b) if b else 0, r"$P(\text{publish})$", True),
        ("batch_p_subscribe", theory.p_subscribe, math.sqrt(theory.p_subscribe * (1 - theory.p_subscribe) / b) if b else 0, r"$P(\text{subscribe})$", True),
        ("batch_correlations", theory.correlation_publish_subscribe, None, r"$\rho(\text{pub},\text{sub})$", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes_flat = axes.flatten()

    for idx, (key, mu, std, ylabel, overlay_normal) in enumerate(panels):
        ax = axes_flat[idx]
        values = getattr(batch_data, key)
        if not values:
            ax.set_title(ylabel)
            continue
        ax.hist(values, bins=min(25, max(10, len(values) // 4)), density=True, alpha=0.7, color="C1", edgecolor="black", linewidth=0.3)
        ax.axvline(mu, color="C0", linestyle="--", linewidth=2)
        if overlay_normal and std is not None and std > 0:
            xmin, xmax = ax.get_xlim()
            xx = np.linspace(xmin, xmax, 200)
            ax.plot(xx, scipy_stats.norm.pdf(xx, mu, std), color="C0", linestyle="-", linewidth=1.5)
        ax.set_ylabel("Density")
        ax.set_xlabel(ylabel)

    # Title on top, legend below title, then plots; compact top margin
    fig.suptitle(f"Batch statistics (batch size={b}, {batch_data.total_samples} samples)", fontsize=11, y=0.98)
    legend_handles = [
        mpatches.Patch(facecolor="C1", alpha=0.7, edgecolor="black", label="Empirical"),
        mlines.Line2D([], [], color="C0", linestyle="--", linewidth=2, label="Theoretical mean"),
        mlines.Line2D([], [], color="C0", linestyle="-", linewidth=1.5, label="Normal approx. (mean & variance)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = out_dir / "batch_distributions.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [p]
