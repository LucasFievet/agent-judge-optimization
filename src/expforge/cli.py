"""CLI for expforge simulator and scoring."""

from pathlib import Path

import typer

from expforge.simulator import run_simulator
from expforge.scoring import run_experiment_scoring, run_experiment_compare

app = typer.Typer(help="Experiment Forge: simulator and scoring for nested Markov experiments.")

simulator_app = typer.Typer(help="Run simulator for an experiment.")
app.add_typer(simulator_app, name="simulator")

scoring_app = typer.Typer(help="Score or compare experiments.")
app.add_typer(scoring_app, name="scoring")


@simulator_app.command("run")
def run_simulator_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment id (e.g. abc or experiment_abc)"),
    sample: int = typer.Option(10, "--sample", "-n", help="Number of trajectories to generate"),
    base_dir: Path = typer.Option(
        None,
        "--base-dir",
        "-d",
        path_type=Path,
        help="Base directory (default: simulator package data dir)",
    ),
    seed: int = typer.Option(None, "--seed", "-s", help="Random seed"),
    no_reuse_config: bool = typer.Option(False, "--no-reuse-config", help="Regenerate persona and goal config"),
) -> None:
    """
    Run simulator: (re)use config, draw random persona per sample, generate trajectories.
    Writes to simulator/experiment/<id>/: persona.yaml, goals.yaml, transitions.yaml, samples/.
    """
    base = base_dir or Path(__file__).resolve().parent / "simulator"
    persona_set, goal_set, paths = run_simulator(
        experiment_id,
        sample,
        base_dir=base,
        seed=seed,
        reuse_config=not no_reuse_config,
    )
    exp_dir = base / "experiment" / experiment_id
    typer.echo(f"Experiment dir: {exp_dir}")
    typer.echo(f"  persona.yaml, goals.yaml, transitions.yaml, samples/")
    typer.echo(f"Generated {len(paths)} samples for {experiment_id}")
    typer.echo(f"Personas: {len(persona_set.personas)}, Goals: {len(goal_set.goals)}")


@scoring_app.command("experiment")
def score_experiment_cmd(
    experiment_id: str = typer.Argument(..., help="Experiment id to score"),
    base_dir: Path = typer.Option(None, "--base-dir", "-d", path_type=Path),
    n_personas: int = typer.Option(5, "--personas", help="Number of persona clusters"),
    n_goals: int = typer.Option(6, "--goals", help="Number of goal clusters"),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence level for intervals"),
) -> None:
    """Score every sample; write experiment/{id}/metrics.yaml (same base_dir as simulator)."""
    base = base_dir or Path(__file__).resolve().parent / "simulator"
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
    base = base_dir or Path(__file__).resolve().parent / "simulator"
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
