"""Training script for Vertex AI Custom Training Job.

This script is packaged and uploaded to GCS for execution in Vertex AI Custom Training Jobs.
It can also be run locally for testing.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Force unbuffered output for real-time logging in containers
# This ensures training progress is visible immediately
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import tensorflow as tf
from google.cloud import storage

# Add package to path for imports
# The script is in triple_mnist_training/train.py
# We need to ensure the current directory (triple_mnist_training) is in the path
# so Python can find the 'training' subdirectory
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

# Debug: Print current directory and verify structure
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {package_dir}")
print(f"Contents of package_dir: {list(package_dir.iterdir())}")
if (package_dir / "training").exists():
    print(f"Training directory exists: {list((package_dir / 'training').iterdir())}")
else:
    print("ERROR: Training directory does not exist!")

# Now we can import from the training subdirectory
from training.triple_mnist_model import create_triple_mnist_model
from datasets import load_images, load_manifest

# Import config module - it's in the package
from expforge.config import VertexAIConfig


def _download_from_gcs_python_client(gcs_path: str, local_path: Path):
    """
    Fallback: Download data from GCS using Python client (slower than gsutil).
    Only used if gsutil is not available.
    """
    print(f"Using Python client to download from {gcs_path}...", flush=True)
    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    prefix = "/".join(gcs_path.split("/")[3:])
    
    bucket = client.bucket(bucket_name)
    print(f"Listing blobs with prefix: {prefix}", flush=True)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    if not blobs:
        raise ValueError(f"No files found at {gcs_path}")
    
    print(f"Found {len(blobs)} files to download", flush=True)
    local_path.mkdir(parents=True, exist_ok=True)
    
    for i, blob in enumerate(blobs, 1):
        # Preserve directory structure
        relative_path = blob.name[len(prefix):].lstrip("/")
        if not relative_path:
            continue
        
        local_file = local_path / relative_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        if i % 10 == 0 or i == len(blobs):
            print(f"Downloading file {i}/{len(blobs)}: {relative_path}", flush=True)
        blob.download_to_filename(str(local_file))
    
    print(f"✓ Downloaded {len(blobs)} files to {local_path}", flush=True)


def download_from_gcs(gcs_path: str, local_path: Path):
    """
    Download data from GCS to local path using gsutil with parallel processing.
    
    This uses gsutil -m (parallel processing) which is the recommended approach
    for Vertex AI custom jobs. It's much faster than sequential downloads.
    """
    if not gcs_path.startswith("gs://"):
        print(f"Using local path: {gcs_path}", flush=True)
        return  # Assume local path
    
    print(f"Downloading data from {gcs_path} to {local_path}...", flush=True)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Use gsutil -m for parallel downloads (recommended for Vertex AI)
    # -m: parallel processing (multiple threads)
    # -r: recursive
    # -n: no-clobber (skip existing files)
    # The trailing slash ensures we copy the directory contents, not the directory itself
    gcs_path_with_slash = gcs_path if gcs_path.endswith("/") else f"{gcs_path}/"
    
    try:
        result = subprocess.run(
            ["gsutil", "-m", "rsync", "-r", gcs_path_with_slash, str(local_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        print(f"✓ Downloaded data to {local_path}", flush=True)
        if result.stdout:
            print(result.stdout, flush=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(
            f"Failed to download data from {gcs_path} to {local_path}: {error_msg}\n"
            f"Make sure gsutil is available and you have access to the GCS bucket."
        ) from e
    except FileNotFoundError:
        # Fallback to Python client if gsutil is not available
        print("Warning: gsutil not found, falling back to Python client (slower)", flush=True)
        _download_from_gcs_python_client(gcs_path, local_path)


def upload_to_gcs(local_path: Path, gcs_path: str):
    """Upload model to GCS."""
    if not gcs_path.startswith("gs://"):
        return
    
    client = storage.Client()
    bucket_name = gcs_path.split("/")[2]
    prefix = "/".join(gcs_path.split("/")[3:])
    
    bucket = client.bucket(bucket_name)
    
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            blob_name = f"{prefix}/{file_path.relative_to(local_path)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))


def main():
    parser = argparse.ArgumentParser(description="Train triple MNIST model on Vertex AI")
    parser.add_argument("--train-data", type=str, required=True, help="GCS path to training data")
    parser.add_argument("--val-data", type=str, default=None, help="GCS path to validation data")
    parser.add_argument("--test-data", type=str, default=None, help="GCS path to test data")
    parser.add_argument("--model-output", type=str, required=True, help="GCS path for model output")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    # Load config from package directory
    # Config file is in the same directory as this script
    config_path = package_dir / "vertex_config.json"
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults", flush=True)
        config = VertexAIConfig()
    else:
        config = VertexAIConfig.load(config_path)
    
    print("=" * 80, flush=True)
    print("Starting training script", flush=True)
    print(f"Arguments: {args}", flush=True)
    print("=" * 80, flush=True)
    
    # Download training data
    print("\n[1/4] Downloading training data...", flush=True)
    train_local = Path("/tmp/train_data")
    download_from_gcs(args.train_data, train_local)
    
    val_local = None
    if args.val_data:
        print("\n[2/4] Downloading validation data...", flush=True)
        val_local = Path("/tmp/val_data")
        download_from_gcs(args.val_data, val_local)
    
    test_local = None
    if args.test_data:
        print("\n[3/4] Downloading test data...", flush=True)
        test_local = Path("/tmp/test_data")
        download_from_gcs(args.test_data, test_local)
    
    # Load data
    print("\n[4/4] Loading data into memory...", flush=True)
    print("Loading training manifest...", flush=True)
    train_manifest = load_manifest(train_local)
    print(f"Training samples: {len(train_manifest)}", flush=True)
    print("Loading training images...", flush=True)
    train_images = load_images(train_manifest, train_local)
    train_labels = train_manifest["label"].astype(int).to_numpy()
    print(f"Training images shape: {train_images.shape}", flush=True)
    
    train_images = train_images.astype(np.float32) / 255.0
    train_images = np.expand_dims(train_images, axis=-1)
    
    val_images = None
    val_labels = None
    if val_local:
        val_manifest = load_manifest(val_local)
        val_images = load_images(val_manifest, val_local)
        val_labels = val_manifest["label"].astype(int).to_numpy()
        val_images = val_images.astype(np.float32) / 255.0
        val_images = np.expand_dims(val_images, axis=-1)
    
    test_images = None
    test_labels = None
    if test_local:
        test_manifest = load_manifest(test_local)
        test_images = load_images(test_manifest, test_local)
        test_labels = test_manifest["label"].astype(int).to_numpy()
        test_images = test_images.astype(np.float32) / 255.0
        test_images = np.expand_dims(test_images, axis=-1)
    
    # Create model
    print("\n" + "=" * 80, flush=True)
    print("Creating model...", flush=True)
    model = create_triple_mnist_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Model created with {model.count_params():,} parameters", flush=True)
    
    # Setup model output path - use same approach as vertex_integration.py
    model_out = Path("/tmp/model/model.keras")
    # Ensure the path has a .keras extension for Keras format
    if model_out.suffix not in ['.keras', '.h5']:
        model_out = model_out.with_suffix('.keras')
    model_out.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup callbacks
    callbacks = []
    
    # Setup TensorBoard callback - write logs to GCS path
    # The TensorBoard resource should already exist and be linked to this GCS path
    from datetime import datetime
    run_name = f"custom-job-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"gs://{config.bucket_name}/tensorboard/{config.experiment_name}/{run_name}"
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        )
    )
    print(f"TensorBoard logs will be written to: {log_dir}", flush=True)
    
    # Model checkpoint - use same path as final model save (like vertex_integration.py)
    if val_images is not None:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_out),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
            )
        )
    
    # Train
    validation_data = (val_images, val_labels) if val_images is not None else None
    
    print("\n" + "=" * 80, flush=True)
    print(f"Starting training for {args.epochs} epochs...", flush=True)
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    history = model.fit(
        train_images,
        train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,  # Show progress bar for each epoch
    )
    
    print("\n" + "=" * 80, flush=True)
    print("Training completed!", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    # Evaluate on test set if available
    if test_images is not None:
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
    
    # Save final model - use same approach as vertex_integration.py
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))
    
    # Upload to GCS (upload the directory containing the model file)
    upload_to_gcs(model_out.parent, args.model_output)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

