"""CLI for expforge simulator and scoring."""

import random
from datetime import datetime
from pathlib import Path

import typer

from expforge.simulator import run_simulator
from expforge.scoring import run_experiment_scoring, run_experiment_compare
from expforge.verifier import (
    run_verification,
    run_n_verifications,
    VerificationResult,
    run_verification_batch_data,
    DEFAULT_EXPERIMENTS_DIR,
    figures_batch_distributions,
)

app = typer.Typer(help="Experiment Forge: simulator and scoring for nested Markov experiments.")

# Metric labels for verification output
_VERIFY_METRIC_LABELS = {
    "mean_length": "E[T]",
    "p_finished": "P(finished)",
    "p_abandoned": "P(abandoned)",
    "p_publish": "P(publish)",
    "p_subscribe": "P(subscribe)",
    "correlation": "ρ(pub,sub)",
}


def _echo_verification_details(r: VerificationResult) -> None:
    """Print theory vs empirical and list failed checks for a VerificationResult."""
    t = r.theory
    typer.echo("  Theory: E[T]=%.4f  P(fin)=%.4f  P(aban)=%.4f  P(pub)=%.4f  P(sub)=%.4f  ρ=%.4f" % (
        t.expected_length, t.p_finished, t.p_abandoned, t.p_publish, t.p_subscribe, t.correlation_publish_subscribe,
    ))
    for n in r.sample_sizes:
        emp = r.empirical_by_n.get(n, {})
        rho = r.correlation_by_n.get(n, 0.0)
        typer.echo("  N=%d: mean_len=%.4f  p_fin=%.4f  p_aban=%.4f  p_pub=%.4f  p_sub=%.4f  ρ=%.4f" % (
            n,
            emp.get("mean_length", 0),
            emp.get("p_finished", 0),
            emp.get("p_abandoned", 0),
            emp.get("p_publish", 0),
            emp.get("p_subscribe", 0),
            rho,
        ))
    _theory_val = {
        "mean_length": lambda: t.expected_length,
        "p_finished": lambda: t.p_finished,
        "p_abandoned": lambda: t.p_abandoned,
        "p_publish": lambda: t.p_publish,
        "p_subscribe": lambda: t.p_subscribe,
        "correlation": lambda: t.correlation_publish_subscribe,
    }
    typer.echo("  Failed checks:")
    for metric, outcomes in r.checks.items():
        label = _VERIFY_METRIC_LABELS.get(metric, metric)
        get_theory = _theory_val.get(metric, lambda: 0.0)
        for i, ok in enumerate(outcomes):
            if i >= len(r.sample_sizes):
                break
            if not ok:
                n = r.sample_sizes[i]
                if metric == "correlation":
                    val = r.correlation_by_n.get(n, 0.0)
                else:
                    val = r.empirical_by_n.get(n, {}).get(metric, 0.0)
                typer.echo("    %s @ n=%d: FAIL (emp=%.4f, theory=%.4f)" % (label, n, val, get_theory()))

simulator_app = typer.Typer(help="Run simulator for an experiment.")
app.add_typer(simulator_app, name="simulator")

scoring_app = typer.Typer(help="Score or compare experiments.")
app.add_typer(scoring_app, name="scoring")

verifier_app = typer.Typer(
    help="Theory vs simulator: run (single experiment), batch (N experiments), report (tables/figures).",
)
app.add_typer(verifier_app, name="verifier")

vertex_app = typer.Typer(help="Vertex AI: config, resources, and custom training jobs.")
app.add_typer(vertex_app, name="vertex")


def _run_batch_verification(
    n: int,
    source: str,
    sample_sizes: list[int],
    base_dir: Path,
    seed: int,
    confidence: float,
) -> None:
    """Shared logic: run N verifications and print pass/fail summary."""
    results = run_n_verifications(
        n_experiments=n,
        source_experiment=source,
        base_dir=base_dir,
        sample_sizes=sample_sizes,
        seed=seed,
        confidence=confidence,
    )
    for r in results:
        typer.echo(f"{r.experiment_id}: {'PASS' if r.passed else 'FAIL'}")
        if not r.passed:
            _echo_verification_details(r)
            typer.echo("  (One or more metrics fell outside the 95% CI at some n; can be sampling variance.)")
    passed = sum(1 for r in results if r.passed)
    typer.echo(f"Passed {passed}/{n}")


@app.command("verify")
def verify_cmd(
    n: int = typer.Option(1, "--n", "-n", help="Number of verification experiments (verifier_1, verifier_2, ...)"),
    source: str = typer.Option("dummy", "--experiment", "-e", help="Source experiment to copy (persona, goals)"),
    sample_sizes: str = typer.Option(
        "200,500,1000",
        "--sample-sizes",
        help="Comma-separated sample sizes; one run per experiment with max(sizes) samples, then subsample",
    ),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", path_type=Path),
    seed: int = typer.Option(42, "--seed", "-s", help="Base random seed (seed+i per verifier_i)"),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence level for intervals"),
) -> None:
    """Alias for 'verifier batch': run N verification experiments (verifier_1 .. verifier_N), print pass X/N."""
    base = base_dir or DEFAULT_EXPERIMENTS_DIR
    sizes = [int(x.strip()) for x in sample_sizes.split(",") if x.strip()] or [200, 500, 1000]
    _run_batch_verification(n, source, sizes, base, seed, confidence)


@simulator_app.command("run")
def run_simulator_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment id (e.g. abc or experiment_abc)"),
    sample: int = typer.Option(10, "--sample", "-n", help="Number of trajectories to generate"),
    base_dir: Path = typer.Option(
        None,
        "--base-dir",
        "-d",
        path_type=Path,
        help="Base directory for experiment outputs (default: <project_root>/.data or EXPFORGE_EXPERIMENTS_DIR)",
    ),
    seed: int = typer.Option(None, "--seed", "-s", help="Random seed"),
    no_reuse_config: bool = typer.Option(False, "--no-reuse-config", help="Regenerate persona and goal config"),
    no_llm: bool = typer.Option(False, "--no-llm", "--fast", help="Fast mode: no LLM for user/agent messages (for verification)"),
) -> None:
    """
    Run simulator: (re)use config, draw random persona per sample, generate trajectories.
    Writes to base_dir/experiment/<id>/: persona.yaml, goals.yaml, transitions.yaml, samples/.
    """
    base = base_dir or DEFAULT_EXPERIMENTS_DIR
    persona_set, goal_set, paths = run_simulator(
        experiment_id,
        sample,
        base_dir=base,
        seed=seed,
        reuse_config=not no_reuse_config,
        use_llm=not no_llm,
    )
    exp_dir = base / "experiment" / experiment_id
    typer.echo(f"Experiment dir: {exp_dir}")
    typer.echo(f"  persona.yaml, goals.yaml, transitions.yaml, samples/")
    typer.echo(f"Generated {len(paths)} samples for {experiment_id}")
    typer.echo(f"Personas: {len(persona_set.personas)}, Goals: {len(goal_set.goals)}")


@verifier_app.command("run")
def verifier_run_cmd(
    experiment_id: str | None = typer.Argument(
        None,
        help="Experiment id (default: <timestamp>_verifier, e.g. 20250207_143022_verifier)",
    ),
    sample_sizes: str = typer.Option(
        "200,500,1000",
        "--sample-sizes",
        help="Comma-separated sample sizes (e.g. 200,500,1000)",
    ),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", path_type=Path),
    seed: int | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Seed for reproducibility; omit for a random seed (re-run with this seed to reproduce)",
    ),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence level for intervals"),
) -> None:
    """Run theory vs simulator verification (fast mode, no LLM). By default creates a new timestamped experiment with a random seed; use --seed to re-run a specific case."""
    base = base_dir or DEFAULT_EXPERIMENTS_DIR
    if experiment_id is None:
        experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_verifier"
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        typer.echo(f"Experiment: {experiment_id}  Seed: {seed} (use --seed {seed} to re-run this case)")
    sizes = [int(x.strip()) for x in sample_sizes.split(",") if x.strip()]
    result = run_verification(
        experiment_id,
        sample_sizes=sizes,
        base_dir=base,
        seed=seed,
        confidence=confidence,
    )
    typer.echo(f"Theory (expected):")
    typer.echo(f"  E[T] = {result.theory.expected_length:.4f}, E[T²] = {result.theory.expected_length_sq:.4f}")
    typer.echo(f"  P(finished) = {result.theory.p_finished:.4f}, P(abandoned) = {result.theory.p_abandoned:.4f}")
    typer.echo(f"  P(publish) = {result.theory.p_publish:.4f}, P(subscribe) = {result.theory.p_subscribe:.4f}")
    typer.echo(f"  ρ(pub, sub) = {result.theory.correlation_publish_subscribe:.4f}")
    for n in result.sample_sizes:
        emp = result.empirical_by_n.get(n, {})
        typer.echo(f"  N={n}: mean_len={emp.get('mean_length', 0):.4f}, p_fin={emp.get('p_finished', 0):.4f}, ρ={result.correlation_by_n.get(n, 0):.4f}")
    typer.echo(f"Checks (pass): {result.checks}")
    typer.echo(f"Overall: {'PASS' if result.passed else 'FAIL'}")
    if not result.passed:
        _echo_verification_details(result)
        typer.echo("  (One or more metrics fell outside the 95% CI at some n; can be sampling variance.)")


@verifier_app.command("report")
def verifier_report_cmd(
    experiment_id: str = typer.Argument(
        ...,
        help="Experiment id; uses existing samples if enough exist, else runs simulator",
    ),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        "-o",
        path_type=Path,
        help="Directory for figures (default: base_dir/experiment/<id>/verification_report)",
    ),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", path_type=Path),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed (used when running simulator)"),
    total_samples: int = typer.Option(10_000, "--total-samples", help="Total samples (reuse up to this many from disk, or generate this many)"),
    batch_size: int = typer.Option(100, "--batch-size", help="Batch size for distribution histograms"),
    override: bool = typer.Option(False, "--override", help="Delete existing samples and re-run simulator before plotting"),
) -> None:
    """Regenerate batch distribution plot from existing data by default; use --override to re-run simulator."""
    import time
    base = base_dir or DEFAULT_EXPERIMENTS_DIR
    out = out_dir or base / "experiment" / experiment_id / "verification_report"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "batch_data.json"
    if override:
        typer.echo("Override: deleting existing samples and re-running simulator...")
    t0 = time.perf_counter()
    batch_data = run_verification_batch_data(
        experiment_id,
        total_samples=total_samples,
        batch_size=batch_size,
        base_dir=base,
        seed=seed,
        override=override,
        cache_path=cache_path,
    )
    typer.echo(f"[verifier] Batch data ready in {time.perf_counter() - t0:.2f}s total")
    t1 = time.perf_counter()
    batch_figs = figures_batch_distributions(batch_data, out, dpi=150)
    typer.echo(f"[verifier] Rendered figure in {time.perf_counter() - t1:.2f}s")
    for p in batch_figs:
        typer.echo(f"Figure: {p}")


@verifier_app.command("batch")
def verifier_batch_cmd(
    n: int = typer.Option(1, "--n", "-n", help="Number of verification experiments (verifier_1, verifier_2, ...)"),
    source: str = typer.Option("dummy", "--experiment", "-e", help="Source experiment to copy (persona, goals)"),
    sample_sizes: str = typer.Option(
        "200,500,1000",
        "--sample-sizes",
        help="Comma-separated sample sizes; one run per experiment with max(sizes) samples, then subsample",
    ),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", path_type=Path),
    seed: int = typer.Option(42, "--seed", "-s", help="Base random seed (seed+i per verifier_i)"),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence level for intervals"),
) -> None:
    """Run N verification experiments (verifier_1 .. verifier_N), print pass/fail per experiment and Passed X/N."""
    base = base_dir or DEFAULT_EXPERIMENTS_DIR
    sizes = [int(x.strip()) for x in sample_sizes.split(",") if x.strip()] or [200, 500, 1000]
    _run_batch_verification(n, source, sizes, base, seed, confidence)


@scoring_app.command("experiment")
def score_experiment_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment id to score"),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", path_type=Path),
    n_personas: int = typer.Option(5, "--personas", help="Number of persona clusters"),
    n_goals: int = typer.Option(6, "--goals", help="Number of goal clusters"),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence level for intervals"),
) -> None:
    """Score every sample; write experiment/{id}/metrics.yaml (same base_dir as simulator)."""
    base = base_dir or DEFAULT_EXPERIMENTS_DIR
    path = run_experiment_scoring(
        experiment_id,
        base_dir=base,
        n_personas=n_personas,
        n_goal_clusters=n_goals,
        confidence=confidence,
    )
    typer.echo(f"Metrics written to {path}")


@scoring_app.command("compare")
def compare_experiments_cmd(
    experiment_a: str = typer.Argument(..., help="First experiment id"),
    experiment_b: str = typer.Argument(..., help="Second experiment id"),
    metric: str = typer.Option("subscribe", "--metric", "-m", help="Metric to compare"),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", path_type=Path),
    confidence: float = typer.Option(0.95, "--confidence"),
) -> None:
    """Compare two experiments; output confidence interval on B - A and conclusion."""
    base = base_dir or DEFAULT_EXPERIMENTS_DIR
    result = run_experiment_compare(
        experiment_a,
        experiment_b,
        metric=metric,
        base_dir=base,
        confidence=confidence,
    )
    if "error" in result:
        typer.echo(result["error"], err=True)
        raise typer.Exit(1)
    typer.echo(f"Metric: {result['metric']}")
    typer.echo(f"  {experiment_a}: {result['point_a']:.4f}")
    typer.echo(f"  {experiment_b}: {result['point_b']:.4f}")
    typer.echo(f"  Difference (B - A): {result['difference_b_minus_a']:.4f}")
    typer.echo(f"  CI: {result['confidence_interval_difference']}")
    typer.echo(f"  Conclusion: {result['conclusion']}")


# ---- Vertex AI commands ----

@vertex_app.command("config")
def vertex_config_cmd(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    project_id: str | None = typer.Option(None, "--project-id", help="GCP project ID"),
    location: str | None = typer.Option(None, "--location", help="GCP region (e.g. us-central1)"),
    bucket_name: str | None = typer.Option(None, "--bucket-name", help="GCS bucket name"),
    experiment_name: str | None = typer.Option(None, "--experiment-name", help="Vertex AI experiment name"),
    tensorboard_name: str | None = typer.Option(None, "--tensorboard-name", help="TensorBoard resource name"),
    machine_type: str | None = typer.Option(None, "--machine-type", help="Default machine type for training jobs"),
) -> None:
    """View or update Vertex AI configuration (stored in config.json)."""
    from expforge.config import get_config_path, load_config, save_config, ExpforgeConfig
    path = get_config_path()
    if show:
        try:
            config = load_config()
            typer.echo(__import__("json").dumps(config.to_dict(), indent=2))
        except FileNotFoundError:
            typer.echo(f"No config at {path}. Create config.json from config.json.example or set options.", err=True)
            raise typer.Exit(1)
        return
    updates = {k: v for k, v in [
        ("project_id", project_id),
        ("location", location),
        ("bucket_name", bucket_name),
        ("experiment_name", experiment_name),
        ("tensorboard_name", tensorboard_name),
        ("machine_type", machine_type),
    ] if v is not None}
    if not updates:
        typer.echo("Use --show to view config, or pass --project-id, --location, etc. to update.")
        return
    try:
        config = load_config()
        d = config.to_dict()
    except FileNotFoundError:
        d = {
            "project_id": "your-project-id",
            "location": "us-central1",
            "bucket_name": "your-bucket",
            "experiment_name": "expforge-experiments",
            "tensorboard_name": "expforge-tensorboard",
        }
    for k, v in updates.items():
        d[k] = v
    config = ExpforgeConfig.from_dict(d)
    saved = save_config(config)
    typer.echo(f"Saved configuration to {saved}")


@vertex_app.command("check-resources")
def vertex_check_resources_cmd(
    fix: bool = typer.Option(False, "--fix", "-f", help="Create missing bucket and TensorBoard"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed errors"),
) -> None:
    """Check Vertex AI resources (credentials, bucket, experiment, TensorBoard)."""
    from expforge.vertex.manager import VertexManager
    manager = VertexManager()
    result = manager.check()
    typer.echo(result["output"])
    if fix and not result["results"]["all_passed"]:
        typer.echo("")
        create_result = manager.create()
        typer.echo(create_result["output"])
    raise typer.Exit(0 if result["results"]["all_passed"] else 1)


@vertex_app.command("train-job")
def vertex_train_job_cmd(
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(0.001, "--learning-rate", help="Learning rate"),
    sync: bool = typer.Option(False, "--sync", help="Wait for job completion"),
) -> None:
    """Submit a Custom Training Job to Vertex AI (placeholder training script)."""
    from expforge.training.custom_job import CustomTrainingJobManager
    manager = CustomTrainingJobManager()
    manager.create_and_submit_job(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sync=sync,
    )
    typer.echo("Job submitted. Use Vertex AI console or logs to monitor.")


@app.command("check-account")
def check_account_cmd() -> None:
    """Show which Google account is authenticated (gcloud)."""
    import subprocess
    try:
        r = subprocess.run(
            ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            typer.echo(f"Active account: {r.stdout.strip()}")
        else:
            typer.echo("No active gcloud account. Run: gcloud auth login")
    except FileNotFoundError:
        typer.echo("gcloud not found. Install Google Cloud SDK.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
