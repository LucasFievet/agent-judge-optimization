"""
Publication-ready tables and figures from verification results.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expforge.verifier.run import VerificationResult

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
    Save publication figures: trajectory length, proportions, correlation vs n.
    Returns list of saved paths. Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    ns = result.sample_sizes
    theory = result.theory

    # 1) Mean trajectory length vs n
    emp_lengths = [result.empirical_by_n[n]["mean_length"] for n in ns]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axhline(theory.expected_length, color="C0", linestyle="--", label="Theory")
    ax.plot(ns, emp_lengths, "o-", color="C1", label="Empirical")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(r"$\mathbb{E}[L]$")
    ax.legend()
    ax.set_title("Mean trajectory length")
    fig.tight_layout()
    p = out_dir / "mean_length_vs_n.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 2) Proportions (p_finished, p_abandoned, p_publish, p_subscribe) vs n
    metrics = ["p_finished", "p_abandoned", "p_publish", "p_subscribe"]
    theory_vals = [
        theory.p_finished,
        theory.p_abandoned,
        theory.p_publish,
        theory.p_subscribe,
    ]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m, tv in zip(metrics, theory_vals):
        ax.axhline(tv, linestyle="--", alpha=0.7)
    emp_vals = {m: [result.empirical_by_n[n][m] for n in ns] for m in metrics}
    for i, m in enumerate(metrics):
        ax.plot(ns, emp_vals[m], "o-", label=_METRIC_LABELS.get(m, m))
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.set_title("Outcome and hitting probabilities")
    fig.tight_layout()
    p = out_dir / "proportions_vs_n.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 3) Correlation vs n
    rho_emp = [result.correlation_by_n.get(n, 0.0) for n in ns]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axhline(theory.correlation_publish_subscribe, color="C0", linestyle="--", label="Theory")
    ax.plot(ns, rho_emp, "o-", color="C1", label="Empirical")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(r"$\rho(\text{publish}, \text{subscribe})$")
    ax.legend()
    ax.set_title("Correlation")
    fig.tight_layout()
    p = out_dir / "correlation_vs_n.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved
