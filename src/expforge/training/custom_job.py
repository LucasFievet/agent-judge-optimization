"""Custom Training Job for Vertex AI using manual packaging with packageUris and pythonModule."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from google.cloud import aiplatform
from google.cloud import storage

from expforge.config import ExpforgeConfig
from expforge.vertex.context import get_config
from expforge.vertex.experiment import get_or_create_experiment
from expforge.vertex.tensorboard import get_or_create_tensorboard
from expforge.vertex.run import create_run


class CustomTrainingJobManager:
    """Manager for Custom Training Jobs using manual packaging with packageUris and pythonModule."""

    def __init__(self, config: Optional[ExpforgeConfig] = None):
        """
        Initialize Custom Training Job manager.
        
        Args:
            config: Vertex AI configuration (defaults to loaded config)
        """
        self.config = config

    def _build_and_upload_package(self, project_root: Path) -> str:
        """
        Build source distribution and upload to GCS.
        
        Args:
            project_root: Path to project root directory
            
        Returns:
            GCS URI of uploaded package
        """
        # Build source distribution
        print("Building source distribution...", flush=True)
        setup_py = project_root / "setup.py"
        if not setup_py.exists():
            raise FileNotFoundError(
                f"setup.py not found at {setup_py}. "
                "Please ensure setup.py exists in project root."
            )
        
        # Run setup.py sdist
        result = subprocess.run(
            ["python3", "setup.py", "sdist", "--formats=gztar"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        
        # Find the created distribution file
        dist_dir = project_root / "dist"
        dist_files = list(dist_dir.glob("expforge-*.tar.gz"))
        if not dist_files:
            raise FileNotFoundError(
                f"No source distribution found in {dist_dir}. "
                "Build may have failed."
            )
        
        # Get the most recent distribution file
        dist_file = max(dist_files, key=lambda p: p.stat().st_mtime)
        print(f"✓ Built source distribution: {dist_file.name}", flush=True)
        
        # Upload to GCS
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        gcs_path = f"packages/expforge-{timestamp}.tar.gz"
        gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"
        
        print(f"Uploading to {gcs_uri}...", flush=True)
        client = storage.Client(project=self.config.project_id)
        bucket = client.bucket(self.config.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(dist_file))
        
        print(f"✓ Uploaded package to {gcs_uri}", flush=True)
        return gcs_uri

    def create_and_submit_job(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        resume_from: Optional[str] = None,
        sync: bool = False,
    ):
        """
        Create and submit a Custom Training Job using manual packaging.
        
        Builds a source distribution from setup.py, uploads it to GCS, and uses
        CustomPythonPackageTrainingJob with packageUris and pythonModule to ensure
        the full expforge package structure is available.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            resume_from: Optional checkpoint name to resume from (None = start fresh, "latest" = use latest)
            sync: Whether to wait for job completion
        
        Returns:
            CustomPythonPackageTrainingJob object
        """
        # Find project root
        def find_project_root(start_path: Path) -> Path:
            """Find project root by looking for common marker files."""
            current = start_path.resolve()
            markers = ["pyproject.toml", "README.md", "setup.py"]
            
            while current != current.parent:
                for marker in markers:
                    if (current / marker).exists():
                        return current
                current = current.parent
            
            # Fallback: assume we're in src/expforge/training, go up 4 levels
            return start_path.resolve().parent.parent.parent.parent
        
        project_root = find_project_root(Path(__file__))
        
        # Build script arguments - same as train.py main()
        script_args = [
            f"--epochs={epochs}",
            f"--batch-size={batch_size}",
            f"--learning-rate={learning_rate}",
        ]
        
        if resume_from == "latest":
            script_args.append("--resume")
        elif resume_from:
            script_args.append(f"--resume-from={resume_from}")
        
        display_name = f"{self.config.experiment_name}-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create experiment run BEFORE launching job (uses local credentials with permissions)
        # This avoids authentication scope issues inside the container
        print("Creating experiment run before job launch...", flush=True)
        experiment, _ = get_or_create_experiment(self.config, create=True)
        tensorboard, _ = get_or_create_tensorboard(self.config, create=True)
        
        run = create_run(
            experiment=experiment,
            tensorboard=tensorboard,
            metadata={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
        )

        # Log run name
        print(f"✓ Created experiment run: {run.name}", flush=True)
        
        # Build source distribution and upload to GCS
        # This properly packages the entire expforge package structure
        package_uri = self._build_and_upload_package(project_root)
        
        # Use CustomPythonPackageTrainingJob which is designed for packageUris + pythonModule
        # This ensures the full expforge package is available and imports work correctly
        job = aiplatform.CustomPythonPackageTrainingJob(
            display_name=display_name,
            python_package_gcs_uri=package_uri,
            python_module_name="expforge.training.train",
            container_uri=self.config.serving_container_image_uri,
            project=self.config.project_id,
            location=self.config.location,
            staging_bucket=self.config.bucket_name,
        )
        
        # Build environment variables as a dict (format expected by run() method)
        env_vars = {
            "GOOGLE_CLOUD_PROJECT": self.config.project_id,
            "GOOGLE_CLOUD_REGION": self.config.location,
            "EXPFORGE_BUCKET_NAME": self.config.bucket_name,
            "EXPFORGE_EXPERIMENT_NAME": self.config.experiment_name,
            "EXPFORGE_TENSORBOARD_NAME": self.config.tensorboard_name,
            "EXPFORGE_RUN_NAME": run.name,
        }
        
        # Add service account if configured
        if self.config.service_account:
            env_vars["EXPFORGE_SERVICE_ACCOUNT"] = self.config.service_account
        
        # Build run arguments directly from config
        run_kwargs = {
            "args": script_args,
            "replica_count": 1,
            "machine_type": self.config.machine_type,
            "sync": sync,
            "environment_variables": env_vars,
        }
        
        # Add accelerator if configured
        if self.config.accelerator_type and self.config.accelerator_count > 0:
            run_kwargs["accelerator_type"] = self.config.accelerator_type
            run_kwargs["accelerator_count"] = self.config.accelerator_count
        
        result = job.run(**run_kwargs)
        
        # If sync=True, run() returns a Model (or None if no model was created)
        # If sync=False, run() returns None but the job is submitted
        return result if sync else job

    def _extract_job_id(self, resource_name: str) -> str:
        """Extract job ID from full resource name."""
        return resource_name.split("/")[-1] if "/" in resource_name else resource_name

    def get_latest_job_id(self) -> Optional[str]:
        """Get the ID of the most recent custom training job."""
        try:
            # Use gcloud to list jobs and get the latest one
            cmd = [
                "gcloud", "ai", "custom-jobs", "list",
                f"--project={self.config.project_id}",
                f"--region={self.config.location}",
                f"--filter=displayName:{self.config.experiment_name}-training-*",
                "--sort-by=~createTime",
                "--limit=1",
                "--format=value(name)",
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return self._extract_job_id(result.stdout.strip())
            
            return None
        except Exception as e:
            print(f"Error finding latest job: {e}", flush=True)
            return None

    def _get_console_url(self, job_id: str) -> str:
        """Get console URL for viewing logs."""
        return f"https://console.cloud.google.com/logs/viewer?project={self.config.project_id}&resource=ml_job%2Fjob_id%2F{job_id}"

    def view_logs(self, job_id: Optional[str] = None, follow: bool = False, lines: int = 100):
        """View logs for a custom training job."""
        if job_id is None:
            job_id = self.get_latest_job_id()
            if job_id is None:
                print("No jobs found.", flush=True)
                return
        
        # Extract job ID if full resource name is provided
        job_id = self._extract_job_id(job_id)
        
        # Build gcloud logging command - use value format for compact one-line output
        filter_expr = (
            f'resource.type="ml_job" '
            f'resource.labels.job_id="{job_id}"'
        )
        
        cmd = [
            "gcloud", "logging", "read",
            filter_expr,
            f"--project={self.config.project_id}",
            f"--limit={lines}",
            "--format=table(timestamp, severity, jsonPayload.message)",  # Compact one-line format!
        ]
        
        if follow:
            cmd.append("--follow")
        else:
            cmd.append("--order=desc")
        
        try:
            if follow:
                # For follow mode, stream output directly
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                except KeyboardInterrupt:
                    process.terminate()
                    print("\nStopped following logs.", flush=True)
            else:
                # For non-follow mode, just run and print output
                result = subprocess.run(cmd, check=False)
                if result.returncode != 0:
                    print(f"\nView logs in console: {self._get_console_url(job_id)}", flush=True)
        except FileNotFoundError:
            print("Error: gcloud CLI not found. Please install it or view logs in the console:", flush=True)
            print(self._get_console_url(job_id), flush=True)
        except Exception as e:
            print(f"Error viewing logs: {e}", flush=True)
            print(f"View logs in console: {self._get_console_url(job_id)}", flush=True)


def main():
    """Main entry point for custom_job module."""
    parser = argparse.ArgumentParser(description="Submit Custom Training Job to Vertex AI")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint name")
    parser.add_argument("--sync", action="store_true", help="Wait for job completion")
    parser.add_argument("--logs", action="store_true", help="View logs of the latest job")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output (use with --logs)")
    parser.add_argument("--lines", "-n", type=int, default=100, help="Number of recent log lines to show (use with --logs)")
    
    args = parser.parse_args()
    
    config = get_config()
    manager = CustomTrainingJobManager(config=config)
    
    # Handle --logs flag
    if args.logs:
        manager.view_logs(follow=args.follow, lines=args.lines)
        return
    
    # Determine resume_from
    resume_from = "latest" if args.resume else args.resume_from
    
    print("=" * 80, flush=True)
    print("Submitting Custom Training Job to Vertex AI", flush=True)
    print(f"Project: {config.project_id}, Location: {config.location}", flush=True)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}", flush=True)
    if resume_from:
        print(f"Resume from: {resume_from}", flush=True)
    print("=" * 80, flush=True)
    
    # Submit job
    job = manager.create_and_submit_job(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=resume_from,
        sync=args.sync,
    )
    
    # Get job info
    display_name = getattr(job, '_display_name', None) or (job.display_name if hasattr(job, 'display_name') else "Custom Training Job")
    print(f"\nCustom Training Job submitted: {display_name}", flush=True)
    
    try:
        job_id = job.resource_name.split("/")[-1]
        print(f"Job ID: {job_id}", flush=True)
        print(f"\nTo view logs, run:", flush=True)
        print(f"  python -m expforge.training.custom_job --logs", flush=True)
    except (RuntimeError, AttributeError):
        pass
    
    if args.sync:
        print("Job completed.", flush=True)
    else:
        print("Job is running asynchronously. Check Vertex AI Console for status.", flush=True)


if __name__ == "__main__":
    main()
