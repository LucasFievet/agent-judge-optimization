"""Command-line interface for Experiment Forge."""

from pathlib import Path
from typing import List, Optional

import typer

from expforge.synthetic.font_generator import build_font_manifest
from expforge.training.sklearn_classifier import train_logistic_regression
from expforge.visualization.overview import plot_sample_grid
from expforge.datasets import bootstrap_mnist

app = typer.Typer(help="Experiment Forge: manage synthetic data, training, and visualization")


@app.command()
def bootstrap_mnist(
    output: Path = typer.Option(Path("data/real/mnist"), help="Where to store the MNIST subset."),
    limit: int = typer.Option(2000, help="Number of MNIST samples to download for lightweight testing."),
    test_split: float = typer.Option(0.2, help="Fraction of records to reserve for testing."),
) -> None:
    """Download a compact MNIST subset and save it using the shared dataset format."""

    manifest = bootstrap_mnist(Path(output), limit=limit, test_split=test_split)
    typer.echo(f"Saved MNIST subset to {output} with {len(manifest)} rows.")


@app.command()
def generate_font_digits(
    count: int = typer.Option(1000, help="How many synthetic images to generate."),
    output: Path = typer.Option(Path("data/synthetic/fonts"), help="Directory to store images and labels."),
    fonts: Optional[List[Path]] = typer.Option(
        None,
        help="Optional list of font files (.ttf/.otf). Defaults to a curated set of system fonts.",
    ),
    seed: int = typer.Option(13, help="Seed for reproducibility."),
) -> None:
    """Generate a synthetic digit dataset by rendering digits with system fonts."""

    manifest = build_font_manifest(count=count, output_dir=output, fonts=fonts, seed=seed)
    typer.echo(f"Generated {len(manifest)} synthetic samples at {output}.")


@app.command()
def train_sklearn(
    train: List[Path] = typer.Option(..., help="One or more dataset directories to use for training."),
    model_out: Path = typer.Option(Path("runs/model.joblib"), help="Where to save the trained model."),
    test_ratio: float = typer.Option(0.2, help="Test split ratio applied after combining datasets."),
    max_iter: int = typer.Option(200, help="Max iterations for the solver."),
    c_value: float = typer.Option(1.0, help="Inverse of regularization strength (C parameter)."),
) -> None:
    """Train a simple scikit-learn logistic regression model using dataset manifests."""

    report = train_logistic_regression(
        dataset_dirs=train,
        model_out=model_out,
        test_ratio=test_ratio,
        max_iter=max_iter,
        c_value=c_value,
    )
    typer.echo("Training complete:")
    for key, value in report.items():
        typer.echo(f"  {key}: {value}")


@app.command()
def visualize_samples(
    dataset: Path = typer.Option(..., help="Dataset directory that contains labels.csv and images/"),
    output: Path = typer.Option(Path("runs/samples.png"), help="Where to save the grid visualization."),
    per_class: int = typer.Option(8, help="How many samples per class to show."),
) -> None:
    """Visualize a grid of digit samples from any compatible dataset."""

    plot_sample_grid(dataset, output=output, per_class=per_class)
    typer.echo(f"Saved sample grid to {output}.")


if __name__ == "__main__":
    app()
