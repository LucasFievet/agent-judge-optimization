"""Command-line interface for Experiment Forge."""

import contextlib
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Suppress "An error occurred" messages from Google Cloud libraries BEFORE any imports
# These are printed when importlib.metadata fails (Python 3.9 compatibility issue)
# We need to suppress output before importing modules that import Google Cloud libraries
# Use a context manager approach to keep devnull open for the lifetime of the process
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_devnull = open(os.devnull, "w")
sys.stdout = _devnull
sys.stderr = _devnull

try:
    import typer
    
    # Suppress annoying warnings from Google Cloud libraries
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
    warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
    warnings.filterwarnings("ignore", message=".*Do not pass an `input_shape`.*")
    
    # Import modules that may trigger Google Cloud imports
    from expforge.synthetic.triple_mnist_generator import generate_triple_mnist
    from expforge.training.triple_mnist_model import create_triple_mnist_model
    from expforge.training.vertex_training import train_model_local, prepare_data_for_training
    from expforge.training.vertex_integration import train_with_vertex_experiments
    from expforge.training.custom_job import CustomTrainingJobManager
    from expforge.visualization.overview import plot_sample_grid
    from expforge.datasets import bootstrap_mnist as bootstrap_mnist_dataset
    from expforge.config import load_vertex_config, save_vertex_config, VertexAIConfig
finally:
    # Restore stdout/stderr after imports complete
    # Keep devnull open - don't close it, as Google Cloud libraries may try to log later
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    # Note: We intentionally don't close _devnull here to avoid "I/O operation on closed file" errors
    # The file will be closed when the process exits

app = typer.Typer(help="Experiment Forge: Triple MNIST training with Vertex AI")


@app.command()
def bootstrap_mnist(
    output: Path = typer.Option(Path("data/real/mnist"), help="Where to store the MNIST subset."),
    limit: int = typer.Option(2000, help="Number of MNIST samples to download for lightweight testing."),
    test_split: float = typer.Option(0.2, help="Fraction of records to reserve for testing."),
) -> None:
    """Download a compact MNIST subset and save it using the shared dataset format."""

    manifest = bootstrap_mnist_dataset(output, limit=limit, test_split=test_split)
    typer.echo(f"Saved MNIST subset to {output} with {len(manifest)} rows.")


@app.command()
def generate_triple_mnist_data(
    mnist_dir: Path = typer.Option(Path("data/real/mnist"), help="Directory containing base MNIST dataset."),
    count: int = typer.Option(5000, help="How many triple MNIST samples to generate."),
    output: Path = typer.Option(Path("data/synthetic/triple_mnist"), help="Directory to store images and labels."),
    seed: int = typer.Option(42, help="Seed for reproducibility."),
    split: str = typer.Option("train", help="Dataset split name (train/test)."),
) -> None:
    """Generate triple MNIST dataset by concatenating 3 MNIST images horizontally."""

    manifest = generate_triple_mnist(
        mnist_dir=mnist_dir,
        count=count,
        output_dir=output,
        seed=seed,
        split=split,
    )
    typer.echo(f"Generated {len(manifest)} triple MNIST samples at {output}.")


@app.command()
def train_triple_mnist(
    train: Path = typer.Option(..., help="Directory containing training data."),
    val: Optional[Path] = typer.Option(None, help="Optional directory containing validation data."),
    test: Optional[Path] = typer.Option(None, help="Optional directory containing test data."),
    model_out: Path = typer.Option(Path("runs/triple_mnist_model"), help="Where to save the trained model."),
    epochs: int = typer.Option(10, help="Number of training epochs."),
    batch_size: int = typer.Option(32, help="Batch size for training."),
    tensorboard_log_dir: Optional[Path] = typer.Option(None, help="Directory for TensorBoard logs (local only)."),
    use_vertex: bool = typer.Option(False, help="Use Vertex AI Experiments and Model Registry."),
    upload_model: bool = typer.Option(True, help="Upload model to Vertex AI Model Registry."),
    resume_from: Optional[Path] = typer.Option(None, help="Path to a saved model checkpoint to resume training from."),
    initial_epoch: Optional[int] = typer.Option(None, help="Starting epoch number (auto-detected from checkpoint if not specified)."),
) -> None:
    """Train the triple MNIST model (locally or with Vertex AI)."""

    config = load_vertex_config()

    if use_vertex:
        model, history, run, uploaded_model, test_metrics = train_with_vertex_experiments(
            train_dir=train,
            val_dir=val,
            test_dir=test,
            model_out=model_out,
            epochs=epochs,
            batch_size=batch_size,
            config=config,
            upload_model=upload_model,
            resume_from=resume_from,
            initial_epoch=initial_epoch,
        )
        
        typer.echo("Training complete with Vertex AI integration!")
        typer.echo(f"Model saved to: {model_out}")
        if run:
            # Get run name - try multiple attributes
            run_name = None
            # Try stored run_name first (we store it during creation)
            if hasattr(run, '_run_name') and run._run_name:
                run_name = run._run_name
            elif hasattr(run, 'name') and run.name:
                # Extract name from resource name (format: projects/.../contexts/RUN_NAME)
                if '/' in run.name:
                    run_name = run.name.split('/')[-1]
                else:
                    run_name = run.name
            elif hasattr(run, 'resource_id') and run.resource_id:
                run_name = run.resource_id
            
            typer.echo(f"Experiment run: {run_name or 'N/A'}")
            if run_name:
                # Construct console URL - need to URL encode the run name
                import urllib.parse
                encoded_run_name = urllib.parse.quote(run_name, safe='')
                typer.echo(f"  View in console: https://console.cloud.google.com/vertex-ai/experiments/locations/{config.location}/experiments/{config.experiment_name}/runs/{encoded_run_name}?project={config.project_id}")
        if uploaded_model:
            typer.echo(f"Model uploaded to Vertex AI: {uploaded_model.resource_name}")
    else:
        model, history, test_metrics = train_model_local(
            train_dir=train,
            val_dir=val,
            test_dir=test,
            model_out=model_out,
            epochs=epochs,
            batch_size=batch_size,
            tensorboard_log_dir=tensorboard_log_dir,
            resume_from=resume_from,
            initial_epoch=initial_epoch,
        )
        
        typer.echo("Training complete!")
        typer.echo(f"Model saved to: {model_out}")
        if test_metrics:
            typer.echo(f"Test accuracy: {test_metrics['test_accuracy']:.4f}, Test loss: {test_metrics['test_loss']:.4f}")
    
    if history:
        final_acc = history.history.get('accuracy', [0])[-1]
        typer.echo(f"Final training accuracy: {final_acc:.4f}")
        if val:
            final_val_acc = history.history.get('val_accuracy', [0])[-1]
            typer.echo(f"Final validation accuracy: {final_val_acc:.4f}")
        if test_metrics:
            typer.echo(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")


@app.command()
def visualize_samples(
    dataset: Path = typer.Option(..., help="Dataset directory that contains labels.csv and images/"),
    output: Path = typer.Option(Path("runs/samples.png"), help="Where to save the grid visualization."),
    per_class: int = typer.Option(4, help="How many samples per class to show."),
) -> None:
    """Visualize a grid of triple MNIST samples (labels 0-27)."""

    plot_sample_grid(dataset, output=output, per_class=per_class)
    typer.echo(f"Saved sample grid to {output}.")


@app.command()
def config_vertex(
    project_id: Optional[str] = typer.Option(None, help="GCP project ID."),
    location: Optional[str] = typer.Option(None, help="GCP region."),
    bucket_name: Optional[str] = typer.Option(None, help="GCS bucket name."),
    experiment_name: Optional[str] = typer.Option(None, help="Vertex AI Experiment name."),
    tensorboard_name: Optional[str] = typer.Option(None, help="TensorBoard resource name."),
    machine_type: Optional[str] = typer.Option(None, help="Default machine type for training jobs."),
    show: bool = typer.Option(False, help="Show current configuration."),
) -> None:
    """Configure Vertex AI settings."""
    
    config = load_vertex_config()
    
    if show:
        typer.echo("Current Vertex AI configuration:")
        typer.echo(f"  Project ID: {config.project_id}")
        typer.echo(f"  Location: {config.location}")
        typer.echo(f"  Bucket: {config.bucket_name}")
        typer.echo(f"  Experiment: {config.experiment_name}")
        typer.echo(f"  TensorBoard: {config.tensorboard_name}")
        typer.echo(f"  Machine Type: {config.machine_type}")
        return
    
    # Update config with provided values
    if project_id:
        config.project_id = project_id
    if location:
        config.location = location
    if bucket_name:
        config.bucket_name = bucket_name
    if experiment_name:
        config.experiment_name = experiment_name
    if tensorboard_name:
        config.tensorboard_name = tensorboard_name
    if machine_type:
        config.machine_type = machine_type
    
    save_vertex_config(config)
    from expforge.config import get_config_path
    typer.echo(f"Configuration saved to {get_config_path()}")


@app.command()
def check_vertex_resources(
    fix: bool = typer.Option(False, help="Automatically create missing resources (bucket, TensorBoard)."),
    verbose: bool = typer.Option(False, help="Show detailed test results and error traces."),
) -> None:
    """Check and optionally fix Vertex AI resources (credentials, bucket, TensorBoard, experiments)."""
    from expforge.config import check_vertex_resources as check_resources, check_current_account
    
    typer.echo("Checking Vertex AI Resources...")
    typer.echo()
    
    # Check current account first
    account_info = check_current_account()
    if account_info["account"]:
        typer.echo(f"Using account: {account_info['account']}")
        typer.echo()
    
    check_result = check_resources(fix=fix, verbose=verbose)
    config = check_result["config"]
    
    typer.echo(f"Project ID: {config.project_id}")
    typer.echo(f"Location: {config.location}")
    typer.echo()
    
    # Print all messages
    for message in check_result["messages"]:
        typer.echo(message)
    
    # If credentials are expired, provide clear next steps
    if "credentials_expired" in check_result["results"]["errors"] or "credentials_error" in check_result["results"]["errors"]:
        typer.echo()
        typer.echo("🔧 Debugging Credential Issues:")
        typer.echo()
        typer.echo("1. Check which account is authenticated:")
        typer.echo("   expforge check-account")
        typer.echo()
        typer.echo("2. Verify your gcloud account:")
        typer.echo("   gcloud auth list")
        typer.echo()
        typer.echo("3. If needed, switch to the correct account:")
        typer.echo("   gcloud config set account YOUR_EMAIL@gmail.com")
        typer.echo()
        typer.echo("4. Re-authenticate application-default credentials:")
        typer.echo("   gcloud auth application-default login")
        typer.echo()
        typer.echo("5. Verify the credentials work:")
        typer.echo("   gcloud auth application-default print-access-token")
        typer.echo()
        typer.echo("6. Re-run the check:")
        typer.echo("   expforge check-vertex-resources")


@app.command()
def check_account() -> None:
    """Check which Google account is currently authenticated."""
    from expforge.config import check_current_account, load_vertex_config
    
    account_info = check_current_account()
    config = load_vertex_config()
    
    typer.echo("Current Google Account Status:")
    typer.echo()
    
    if account_info["account"]:
        typer.echo(f"  Account: {account_info['account']}")
    else:
        typer.echo("  Account: Unable to determine")
    
    if account_info["project"]:
        typer.echo(f"  Active Project: {account_info['project']}")
    else:
        typer.echo("  Active Project: Not set")
    
    typer.echo(f"  Config Project: {config.project_id}")
    typer.echo()
    
    if account_info["project"] and account_info["project"] != config.project_id:
        typer.echo("⚠ Warning: Active project doesn't match config project!")
        typer.echo()
    
    typer.echo("To switch accounts:")
    typer.echo("1. List available accounts:")
    typer.echo("   gcloud auth list")
    typer.echo()
    typer.echo("2. Set the account you want to use:")
    typer.echo("   gcloud config set account YOUR_EMAIL@example.com")
    typer.echo()
    typer.echo("3. Re-authenticate application-default credentials:")
    typer.echo("   gcloud auth application-default login")
    typer.echo()
    typer.echo("4. Verify the account:")
    typer.echo("   expforge check-account")


@app.command()
def train_vertex_job(
    train_data_gcs: str = typer.Option(..., help="GCS path to training data."),
    val_data_gcs: Optional[str] = typer.Option(None, help="GCS path to validation data."),
    test_data_gcs: Optional[str] = typer.Option(None, help="GCS path to test data."),
    model_output_gcs: Optional[str] = typer.Option(None, help="GCS path for model output."),
    epochs: int = typer.Option(10, help="Number of training epochs."),
    batch_size: int = typer.Option(32, help="Batch size for training."),
    learning_rate: float = typer.Option(0.001, help="Learning rate."),
    machine_type: Optional[str] = typer.Option(None, help="Machine type (overrides config)."),
    accelerator_type: Optional[str] = typer.Option(None, help="Accelerator type (e.g., NVIDIA_TESLA_K80)."),
    accelerator_count: int = typer.Option(0, help="Number of accelerators."),
    sync: bool = typer.Option(False, help="Wait for job completion."),
    local: bool = typer.Option(False, "--local", help="Run locally with Docker for debugging (uses gcloud ai custom-jobs local-run)."),
) -> None:
    """Submit a Custom Training Job to Vertex AI, or run locally for debugging."""
    
    config = load_vertex_config()
    manager = CustomTrainingJobManager(config=config)
    
    if local:
        # Run locally with Docker
        typer.echo("🐳 Running training job locally with Docker...")
        typer.echo("   This uses gcloud ai custom-jobs local-run for debugging.")
        typer.echo("   The same code and container logic will run in Vertex AI.\n")
        
        manager.run_local(
            train_data_gcs=train_data_gcs,
            val_data_gcs=val_data_gcs,
            test_data_gcs=test_data_gcs,
            model_output_gcs=model_output_gcs,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    else:
        # Submit to Vertex AI
        job = manager.create_and_submit_job(
            train_data_gcs=train_data_gcs,
            val_data_gcs=val_data_gcs,
            test_data_gcs=test_data_gcs,
            model_output_gcs=model_output_gcs,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            sync=sync,
        )
        
        # Get job info - lightweight approach
        display_name = getattr(job, '_display_name', None) or (job.display_name if hasattr(job, 'display_name') else "Custom Training Job")
        
        typer.echo(f"Custom Training Job submitted: {display_name}")
        
        if sync:
            # When sync=True, resource is fully available
            job_id = job.resource_name.split("/")[-1]
            typer.echo(f"Job ID: {job_id}")
            typer.echo(f"\nTo view logs, run:")
            typer.echo(f"  expforge view-job-logs --job-id {job_id}")
            typer.echo("Job completed.")
        else:
            # When sync=False, try to get job_id but don't fail if not available
            try:
                job_id = job.resource_name.split("/")[-1]
                typer.echo(f"Job ID: {job_id}")
                typer.echo(f"\nTo view logs, run:")
                typer.echo(f"  expforge view-job-logs --job-id {job_id}")
            except (RuntimeError, AttributeError):
                # Resource not immediately available - that's okay for async
                pass
            typer.echo("Job is running asynchronously. Check Vertex AI Console for status.")


@app.command()
def view_job_logs(
    job_id: str = typer.Option(..., help="Custom Job ID (from job resource name)"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output (like tail -f)"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of recent log lines to show"),
) -> None:
    """View logs for a Vertex AI Custom Training Job."""
    import subprocess
    
    config = load_vertex_config()
    
    # Extract job ID if full resource name is provided
    if "/" in job_id:
        job_id = job_id.split("/")[-1]
    
    # Build gcloud logging command
    # Use a simple format that works with gcloud logging
    filter_expr = (
        f'resource.type="ml_job" '
        f'resource.labels.job_id="{job_id}"'
    )
    
    # Use default format which shows all log fields
    # gcloud will automatically display textPayload or jsonPayload appropriately
    cmd = [
        "gcloud", "logging", "read",
        filter_expr,
        f"--project={config.project_id}",
        f"--limit={lines}",
    ]
    
    if follow:
        cmd.append("--follow")
    else:
        # For non-following, show newest first
        cmd.append("--order=desc")
    
    typer.echo(f"Fetching logs for job: {job_id}")
    typer.echo(f"Project: {config.project_id}")
    typer.echo("")
    
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        typer.echo("Error: gcloud CLI not found. Please install it or view logs in the console:")
        typer.echo(f"https://console.cloud.google.com/logs/viewer?project={config.project_id}&resource=ml_job%2Fjob_id%2F{job_id}")
    except Exception as e:
        typer.echo(f"Error viewing logs: {e}")
        typer.echo(f"View logs in console: https://console.cloud.google.com/logs/viewer?project={config.project_id}&resource=ml_job%2Fjob_id%2F{job_id}")


if __name__ == "__main__":
    app()
