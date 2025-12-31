"""Simplified Custom Training Job for Vertex AI using aiplatform.CustomTrainingJob."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from google.cloud import aiplatform

from expforge.config import ExpforgeConfig, load_config


class CustomTrainingJobManager:
    """Simplified manager for Custom Training Jobs using aiplatform.CustomTrainingJob."""

    def __init__(self, config: Optional[ExpforgeConfig] = None):
        """
        Initialize Custom Training Job manager.
        
        Args:
            config: Vertex AI configuration (defaults to loaded config)
        """
        self.config = config or load_config()
        staging_bucket = f"gs://{self.config.bucket_name}"
        aiplatform.init(
            project=self.config.project_id,
            location=self.config.location,
            staging_bucket=staging_bucket,
        )

    def create_and_submit_job(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        resume_from: Optional[str] = None,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: int = 0,
        sync: bool = False,
    ):
        """
        Create and submit a Custom Training Job using aiplatform.CustomTrainingJob.
        
        This automatically packages the training script and all dependencies,
        uploads to GCS, and runs it in Vertex AI.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            resume_from: Optional checkpoint name to resume from (None = start fresh, "latest" = use latest)
            machine_type: Machine type (defaults to config)
            accelerator_type: Accelerator type (defaults to config)
            accelerator_count: Number of accelerators (defaults to config)
            sync: Whether to wait for job completion
        
        Returns:
            CustomTrainingJob object
        """
        # Find project root to locate training script
        def find_project_root(start_path: Path) -> Path:
            """Find project root by looking for common marker files."""
            current = start_path.resolve()
            markers = ["pyproject.toml", "README.md", "requirements.txt"]
            
            while current != current.parent:
                for marker in markers:
                    if (current / marker).exists():
                        return current
                current = current.parent
            
            # Fallback: assume we're in src/expforge/training, go up 4 levels
            return start_path.resolve().parent.parent.parent.parent
        
        project_root = find_project_root(Path(__file__))
        
        # Path to the training script - CustomTrainingJob will package everything automatically
        script_path = project_root / "src" / "expforge" / "training" / "train.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")
        
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
        
        # Requirements - CustomTrainingJob will install these
        requirements = [
            "tensorflow>=2.11.0",
            "numpy>=1.23.0",
            "pandas>=2.0.0",
            "pillow>=9.0.0",
            "google-cloud-aiplatform>=1.38.0",
            "google-cloud-storage>=2.10.0",
        ]
        
        # Create CustomTrainingJob - it handles packaging automatically!
        display_name = f"triple-mnist-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job = aiplatform.CustomTrainingJob(
            display_name=display_name,
            script_path=str(script_path),
            container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
            requirements=requirements,
        )
        
        # Build machine spec
        machine_spec = {
            "machine_type": machine_type or self.config.machine_type,
        }
        
        # Determine accelerator settings
        final_accelerator_type = accelerator_type or self.config.accelerator_type
        final_accelerator_count = accelerator_count if accelerator_count is not None else self.config.accelerator_count
        
        # Only add accelerator if we have both a valid type and count > 0
        if final_accelerator_type and final_accelerator_count > 0:
            machine_spec["accelerator_type"] = final_accelerator_type
            machine_spec["accelerator_count"] = final_accelerator_count
        
        # Run the job
        # CustomTrainingJob.run() handles everything: packaging, uploading, and running
        # Only pass accelerator parameters if we have valid values
        run_kwargs = {
            "args": script_args,
            "replica_count": 1,
            "machine_type": machine_spec["machine_type"],
            "sync": sync,
        }
        
        # Only add accelerator parameters if we have valid values
        if "accelerator_type" in machine_spec:
            run_kwargs["accelerator_type"] = machine_spec["accelerator_type"]
            run_kwargs["accelerator_count"] = machine_spec["accelerator_count"]
        
        result = job.run(**run_kwargs)
        
        # If sync=True, run() returns a Model (or None if no model was created)
        # If sync=False, run() returns None but the job is submitted
        return result if sync else job

    def get_latest_job_id(self) -> Optional[str]:
        """Get the ID of the most recent custom training job."""
        try:
            # Use gcloud to list jobs and get the latest one
            cmd = [
                "gcloud", "ai", "custom-jobs", "list",
                f"--project={self.config.project_id}",
                f"--region={self.config.location}",
                "--filter=displayName:triple-mnist-training-*",
                "--sort-by=~createTime",
                "--limit=1",
                "--format=value(name)",
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                # Extract job ID from full resource name
                resource_name = result.stdout.strip()
                if "/" in resource_name:
                    return resource_name.split("/")[-1]
                return resource_name
            
            return None
        except Exception as e:
            print(f"Error finding latest job: {e}", flush=True)
            return None

    def view_logs(self, job_id: Optional[str] = None, follow: bool = False, lines: int = 100):
        """View logs for a custom training job."""
        if job_id is None:
            job_id = self.get_latest_job_id()
            if job_id is None:
                print("No jobs found.", flush=True)
                return
        
        # Extract job ID if full resource name is provided
        if "/" in job_id:
            job_id = job_id.split("/")[-1]
        
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
                    print(f"\nView logs in console: https://console.cloud.google.com/logs/viewer?project={self.config.project_id}&resource=ml_job%2Fjob_id%2F{job_id}", flush=True)
        except FileNotFoundError:
            print("Error: gcloud CLI not found. Please install it or view logs in the console:", flush=True)
            print(f"https://console.cloud.google.com/logs/viewer?project={self.config.project_id}&resource=ml_job%2Fjob_id%2F{job_id}", flush=True)
        except Exception as e:
            print(f"Error viewing logs: {e}", flush=True)
            print(f"View logs in console: https://console.cloud.google.com/logs/viewer?project={self.config.project_id}&resource=ml_job%2Fjob_id%2F{job_id}", flush=True)


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
    
    config = load_config()
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
