"""Fully implemented Custom Training Job for Vertex AI."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress "An error occurred" messages from Google Cloud libraries during import
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Import Google Cloud modules with stderr suppressed
with suppress_stderr():
    from google.cloud import aiplatform, storage
    from google.cloud.aiplatform import CustomJob

from expforge.config import VertexAIConfig, load_vertex_config


class CustomTrainingJobManager:
    """Manages Custom Training Jobs on Vertex AI."""

    def __init__(self, config: Optional[VertexAIConfig] = None):
        """
        Initialize Custom Training Job manager.
        
        Args:
            config: Vertex AI configuration (defaults to loaded config)
        """
        self.config = config or load_vertex_config()
        aiplatform.init(project=self.config.project_id, location=self.config.location)

    def package_training_code(
        self,
        source_dir: Path,
        output_dir: Path,
        package_name: str = "triple_mnist_training",
    ) -> Path:
        """
        Package training code into a directory structure for Vertex AI.
        
        Args:
            source_dir: Directory containing training code
            output_dir: Directory to create package in
            package_name: Name of the package
        
        Returns:
            Path to the packaged code
        """
        package_dir = output_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy necessary files
        # Use relative paths from project root
        files_to_copy = [
            ("src/expforge/training/triple_mnist_model.py", "training/triple_mnist_model.py"),
            ("src/expforge/datasets.py", "datasets.py"),
            ("src/expforge/config.py", "expforge/config.py"),
            ("vertex_config.json", "vertex_config.json"),
        ]
        
        for src_path_str, dst_path_str in files_to_copy:
            src = source_dir / src_path_str
            if not src.exists():
                raise FileNotFoundError(
                    f"Required file not found: {src}\n"
                    f"  Source dir: {source_dir}\n"
                    f"  Expected path: {src_path_str}"
                )
            
            # Use explicit destination path
            dst = package_dir / dst_path_str
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"✓ Copied {src.name} -> {dst_path_str}")
        
        # Create __init__.py for expforge package
        (package_dir / "expforge" / "__init__.py").touch()
        
        # Copy training script from source
        training_script_source = Path(__file__).parent / "vertex_train_script.py"
        training_script = package_dir / "train.py"
        if training_script_source.exists():
            shutil.copy2(training_script_source, training_script)
        else:
            raise FileNotFoundError(f"Training script not found: {training_script_source}")
        
        # Create __init__.py files to make directories proper Python packages
        # This ensures imports work correctly
        (package_dir / "__init__.py").touch()
        
        # Ensure training directory exists before creating __init__.py
        training_dir = package_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        (training_dir / "__init__.py").touch()
        
        # Ensure expforge directory exists (created above, but ensure __init__.py exists)
        expforge_dir = package_dir / "expforge"
        expforge_dir.mkdir(parents=True, exist_ok=True)
        (expforge_dir / "__init__.py").touch()
        
        # Verify structure and that all required files exist
        print(f"Package structure created:")
        print(f"  {package_dir}/")
        for item in package_dir.iterdir():
            if item.is_dir():
                print(f"    {item.name}/")
                for subitem in item.iterdir():
                    print(f"      {subitem.name}")
            else:
                print(f"    {item.name}")
        
        # Verify critical files exist
        required_files = [
            package_dir / "train.py",
            package_dir / "training" / "triple_mnist_model.py",
            package_dir / "datasets.py",
            package_dir / "expforge" / "config.py",
            package_dir / "vertex_config.json",
        ]
        for req_file in required_files:
            if not req_file.exists():
                raise FileNotFoundError(f"Required packaged file missing: {req_file}")
            print(f"✓ Verified {req_file.name} exists")
        
        # Create requirements.txt
        requirements = package_dir / "requirements.txt"
        requirements.write_text(
            "tensorflow>=2.11.0\n"
            "numpy>=1.23.0\n"
            "pandas>=2.0.0\n"
            "pillow>=9.0.0\n"
            "google-cloud-aiplatform>=1.38.0\n"
            "google-cloud-storage>=2.10.0\n"
        )
        
        return package_dir

    def upload_package_to_gcs(self, package_dir: Path, gcs_path: str) -> str:
        """
        Upload packaged training code to GCS as a tarball.
        
        Args:
            package_dir: Local directory containing packaged code
            gcs_path: GCS path (gs://bucket/path) - will append .tar.gz
        
        Returns:
            GCS path to the uploaded tarball
        """
        client = storage.Client(project=self.config.project_id)
        bucket_name = gcs_path.split("/")[2]
        prefix = "/".join(gcs_path.split("/")[3:])
        
        # Create tarball
        # Ensure the root directory in tarball is package_dir.name
        tarball_path = package_dir.parent / f"{package_dir.name}.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            # Add all files and directories, preserving structure with package_dir.name as root
            for item in package_dir.rglob("*"):
                if item.is_file():
                    # Get path relative to package_dir, then prepend package_dir.name
                    relative_path = item.relative_to(package_dir)
                    arcname = f"{package_dir.name}/{relative_path}"
                    tar.add(item, arcname=arcname)
        
        # Upload tarball to GCS
        bucket = client.bucket(bucket_name)
        blob_name = f"{prefix}.tar.gz"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(tarball_path))
        
        gcs_tarball_path = f"{gcs_path}.tar.gz"
        print(f"✓ Uploaded package tarball to {gcs_tarball_path}")
        return gcs_tarball_path

    def create_and_submit_job(
        self,
        train_data_gcs: str,
        val_data_gcs: Optional[str] = None,
        test_data_gcs: Optional[str] = None,
        model_output_gcs: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        machine_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: int = 0,
        tensorboard_resource_name: Optional[str] = None,
        sync: bool = False,
    ) -> CustomJob:
        """
        Create and submit a Custom Training Job.
        
        Args:
            train_data_gcs: GCS path to training data
            val_data_gcs: Optional GCS path to validation data
            test_data_gcs: Optional GCS path to test data
            model_output_gcs: GCS path for model output (defaults to bucket/models/)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            machine_type: Machine type (defaults to config)
            accelerator_type: Accelerator type (defaults to config)
            accelerator_count: Number of accelerators (defaults to config)
            tensorboard_resource_name: TensorBoard resource name
            sync: Whether to wait for job completion
        
        Returns:
            CustomJob object
        """
        # Find project root by looking for a marker file (pyproject.toml, setup.py, or README.md)
        # This is more robust than counting parent directories
        def find_project_root(start_path: Path) -> Path:
            """Find project root by looking for common marker files."""
            current = start_path.resolve()
            markers = ["pyproject.toml", "setup.py", "README.md", "requirements.txt"]
            
            while current != current.parent:  # Stop at filesystem root
                for marker in markers:
                    if (current / marker).exists():
                        return current
                current = current.parent
            
            # Fallback: assume we're in src/expforge/training, go up 4 levels
            return start_path.resolve().parent.parent.parent.parent
        
        # Package training code
        project_root = find_project_root(Path(__file__))
        package_dir = self.package_training_code(
            source_dir=project_root,
            output_dir=Path("/tmp") / "vertex_packages",
            package_name="triple_mnist_training",
        )
        
        # Upload package to GCS
        package_gcs_base = f"gs://{self.config.bucket_name}/training_packages/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        package_gcs = self.upload_package_to_gcs(package_dir, package_gcs_base)  # Returns path with .tar.gz
        
        # Prepare model output path
        if model_output_gcs is None:
            model_output_gcs = f"gs://{self.config.bucket_name}/models/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Build command arguments
        args = [
            f"--train-data={train_data_gcs}",
            f"--model-output={model_output_gcs}",
            f"--epochs={epochs}",
            f"--batch-size={batch_size}",
            f"--learning-rate={learning_rate}",
        ]
        
        if val_data_gcs:
            args.append(f"--val-data={val_data_gcs}")
        
        if test_data_gcs:
            args.append(f"--test-data={test_data_gcs}")
        
        # Build machine spec
        machine_spec = {
            "machine_type": machine_type or self.config.machine_type,
        }
        
        # Add accelerator if specified
        if (accelerator_type or self.config.accelerator_type) and (accelerator_count or self.config.accelerator_count):
            machine_spec["accelerator_type"] = accelerator_type or self.config.accelerator_type
            machine_spec["accelerator_count"] = accelerator_count or self.config.accelerator_count
        
        # Build worker pool specs using container_spec with pre-built TensorFlow container
        # Per Vertex AI docs: https://docs.cloud.google.com/vertex-ai/docs/training/pre-built-containers
        # For TensorFlow 2.11 CPU, use: us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest
        # The package is uploaded as a tarball, so we download, extract, and run it
        
        # Build command to download package from GCS and run training
        # package_gcs already includes .tar.gz extension
        package_name = "triple_mnist_training"  # This is the directory name inside the tarball
        # Create a single-line bash command
        # Important: We need to be in the package directory and ensure Python can find the modules
        bash_script = (
            f"gsutil cp {package_gcs} /tmp/package.tar.gz && "
            f"mkdir -p /tmp/package && "
            f"cd /tmp/package && "
            f"tar -xzf /tmp/package.tar.gz && "
            f"cd {package_name} && "
            f"echo 'Current directory:' && pwd && "
            f"echo 'Directory contents:' && ls -la && "
            f"echo 'Training directory:' && ls -la training/ 2>&1 || echo 'Training dir not found' && "
            f"export PYTHONPATH=/tmp/package/{package_name}:$PYTHONPATH && "
            f"[ -f requirements.txt ] && pip install -r requirements.txt || true && "
            f"python train.py {' '.join(args)}"
        )
        
        container_spec = {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest",
            "command": ["bash", "-c"],
            "args": [bash_script],
        }
        
        worker_pool_specs = [{
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": container_spec,
        }]
        
        # Create job using the new worker_pool_specs API
        display_name = f"triple-mnist-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        job = CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
            base_output_dir=model_output_gcs,
            staging_bucket=f"gs://{self.config.bucket_name}",
        )
        
        # Submit job
        # When sync=False, use submit() which creates the resource immediately
        # When sync=True, use run() which waits for completion
        if sync:
            job.run(sync=True)
        else:
            # submit() creates the resource and returns immediately
            job.submit()
        
        # Store display_name on job for easy access
        job._display_name = display_name
        
        return job
    
    def run_local(
        self,
        train_data_gcs: str,
        val_data_gcs: Optional[str] = None,
        test_data_gcs: Optional[str] = None,
        model_output_gcs: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> None:
        """
        Run training job locally using gcloud ai custom-jobs local-run.
        
        This builds a Docker image and runs it locally, mimicking how Vertex AI
        will run it in the cloud. Useful for debugging before submitting to Vertex AI.
        
        Args:
            train_data_gcs: GCS path to training data
            val_data_gcs: Optional GCS path to validation data
            test_data_gcs: Optional GCS path to test data
            model_output_gcs: GCS path for model output (defaults to bucket/models/)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Find project root
        def find_project_root(start_path: Path) -> Path:
            """Find project root by looking for common marker files."""
            current = start_path.resolve()
            markers = ["pyproject.toml", "setup.py", "README.md", "requirements.txt"]
            
            while current != current.parent:
                for marker in markers:
                    if (current / marker).exists():
                        return current
                current = current.parent
            
            return start_path.resolve().parent.parent.parent.parent
        
        # Package training code (same as create_and_submit_job)
        project_root = find_project_root(Path(__file__))
        package_dir = self.package_training_code(
            source_dir=project_root,
            output_dir=Path("/tmp") / "vertex_packages",
            package_name="triple_mnist_training",
        )
        
        # Prepare model output path
        if model_output_gcs is None:
            model_output_gcs = f"gs://{self.config.bucket_name}/models/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Build script arguments (same format as create_and_submit_job)
        script_args = [
            f"--train-data={train_data_gcs}",
            f"--model-output={model_output_gcs}",
            f"--epochs={epochs}",
            f"--batch-size={batch_size}",
            f"--learning-rate={learning_rate}",
        ]
        
        if val_data_gcs:
            script_args.append(f"--val-data={val_data_gcs}")
        
        if test_data_gcs:
            script_args.append(f"--test-data={test_data_gcs}")
        
        # Build gcloud command
        executor_image = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest"
        local_image_name = "triple-mnist-training-local"
        
        # Build command: gcloud flags come first, then -- separator, then script arguments
        # The -- separator tells gcloud to pass everything after it directly to the script
        cmd = [
            "gcloud", "ai", "custom-jobs", "local-run",
            f"--executor-image-uri={executor_image}",
            f"--local-package-path={package_dir}",
            f"--script=train.py",
            f"--output-image-uri={local_image_name}",
        ]
        
        # Add script arguments after the separator
        # Note: -- is the standard separator for passing arguments to scripts in gcloud commands
        if script_args:
            cmd.append("--")
            cmd.extend(script_args)
        
        print(f"Running: {' '.join(cmd)}\n")
        print("Note: Platform mismatch warnings (amd64 vs arm64) are expected on Mac M1/M2 and harmless.\n")
        print("Note: Training output will appear below. This may take a moment to start...\n")
        
        # Run the command directly without capturing output
        # gcloud ai custom-jobs local-run handles Docker output itself
        # By not capturing stdout/stderr, we let gcloud display output directly to the terminal
        # This ensures we see all Docker container output in real-time
        try:
            import warnings
            
            # Suppress Python 3.13 subprocess warnings (they're from gcloud, not our code)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='subprocess')
                
                # Run directly without capturing pipes - let output go to terminal
                # This is the most reliable way to see Docker container output
                return_code = subprocess.call(cmd)
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Local run failed with exit code {e.returncode}.\n"
                f"Make sure Docker is running and gcloud CLI is installed."
            ) from e
        except FileNotFoundError:
            raise RuntimeError(
                "gcloud CLI not found. Please install Google Cloud SDK:\n"
                "  https://cloud.google.com/sdk/docs/install"
            )
