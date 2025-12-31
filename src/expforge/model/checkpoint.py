"""Checkpoint management utilities for GCS-only storage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

from expforge.config import load_config


def _gsutil_args(*args: str) -> list[str]:
    """Build gsutil command with macOS multiprocessing warning suppression."""
    return [
        "gsutil",
        "-o", "GSUtil:parallel_process_count=1",  # Suppress macOS multiprocessing warnings
        *args
    ]


def get_checkpoint_dir_gcs() -> str:
    """Get GCS checkpoint directory."""
    config = load_config()
    return f"gs://{config.bucket_name}/checkpoints"


def list_checkpoints() -> list[dict]:
    """List all checkpoints in GCS."""
    gcs_path = get_checkpoint_dir_gcs()
    
    try:
        # Use gsutil to list checkpoints
        result = subprocess.run(
            _gsutil_args("ls", gcs_path),
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            return []
        
        checkpoints = []
        for line in result.stdout.strip().split("\n"):
            if not line or not line.startswith("gs://"):
                continue
            
            checkpoint_name = line.rstrip("/").split("/")[-1]
            if checkpoint_name:
                # Try to get metadata
                metadata = get_checkpoint_metadata(checkpoint_name)
                
                checkpoints.append({
                    "name": checkpoint_name,
                    "gcs_path": line,
                    "epoch": metadata.get("epoch") if metadata else None,
                    "created": metadata.get("created") if metadata else None,
                })
        
        return sorted(checkpoints, key=lambda x: x.get("created") or "", reverse=True)
    except Exception as e:
        print(f"Error listing checkpoints: {e}", flush=True)
        return []


def get_checkpoint_metadata(checkpoint_name: str) -> Optional[dict]:
    """Load metadata for a checkpoint from GCS."""
    gcs_metadata_path = f"{get_checkpoint_dir_gcs()}/{checkpoint_name}/metadata.json"
    
    try:
        result = subprocess.run(
            _gsutil_args("cat", gcs_metadata_path),
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception:
        return None


def get_latest_checkpoint_name() -> Optional[str]:
    """Get the name of the latest checkpoint from GCS."""
    # First try to read the "latest" marker
    latest_path = f"{get_checkpoint_dir_gcs()}/latest"
    try:
        result = subprocess.run(
            _gsutil_args("cat", latest_path),
            capture_output=True,
            text=True,
            check=True,
        )
        checkpoint_name = result.stdout.strip()
        if checkpoint_name:
            return checkpoint_name
    except Exception:
        pass
    
    # Fall back to most recently created checkpoint
    checkpoints = list_checkpoints()
    if not checkpoints:
        return None
    
    # Return the most recently created checkpoint
    return checkpoints[0]["name"]


def save_checkpoint(
    model: tf.keras.Model,
    epoch: int,
    checkpoint_name: Optional[str] = None,
) -> str:
    """Save a checkpoint to GCS."""
    if checkpoint_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        checkpoint_name = f"checkpoint-{timestamp}"
    
    # Save model to temp location
    base_dir = Path("/tmp/triple-mnist/checkpoints")
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / checkpoint_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = temp_dir / "model.keras"
    model.save(str(checkpoint_path))
    
    # Upload model and metadata to GCS
    gcs_checkpoint_dir = f"{get_checkpoint_dir_gcs()}/{checkpoint_name}"
    
    # Upload model
    subprocess.run(
        _gsutil_args("cp", str(checkpoint_path), f"{gcs_checkpoint_dir}/model.keras"),
        check=True,
    )
    
    # Upload metadata
    metadata = {
        "epoch": epoch,
        "created": datetime.now().isoformat(),
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f, indent=2)
        temp_metadata = f.name
    
    subprocess.run(
        _gsutil_args("cp", temp_metadata, f"{gcs_checkpoint_dir}/metadata.json"),
        check=True,
    )
    Path(temp_metadata).unlink()
    shutil.rmtree(temp_dir)
    
    # Update latest checkpoint marker
    latest_path = f"{get_checkpoint_dir_gcs()}/latest"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(checkpoint_name)
        temp_latest = f.name
    
    subprocess.run(
        _gsutil_args("cp", temp_latest, latest_path),
        check=True,
    )
    Path(temp_latest).unlink()
    
    return checkpoint_name


def load_checkpoint(checkpoint_name: str) -> Tuple[tf.keras.Model, int]:
    """Load a checkpoint from GCS."""
    # Download from GCS to temp location
    base_dir = Path("/tmp/triple-mnist/checkpoints")
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / checkpoint_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    gcs_checkpoint_dir = f"{get_checkpoint_dir_gcs()}/{checkpoint_name}"
    
    # Download model
    subprocess.run(
        _gsutil_args("-m", "rsync", "-r", f"{gcs_checkpoint_dir}/", str(temp_dir)),
        check=True,
    )
    
    checkpoint_path = temp_dir / "model.keras"
    if not checkpoint_path.exists():
        # Try .h5
        checkpoint_path = temp_dir / "model.h5"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model file not found in checkpoint {checkpoint_name}")
    
    model = tf.keras.models.load_model(str(checkpoint_path), safe_mode=False)
    
    # Load metadata
    metadata = get_checkpoint_metadata(checkpoint_name)
    epoch = metadata.get("epoch", 0) if metadata else 0
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return model, epoch


def main():
    """CLI entry point for checkpoint management."""
    parser = argparse.ArgumentParser(description="Manage model checkpoints in GCS")
    parser.add_argument("--list", action="store_true", help="List all available checkpoints")
    args = parser.parse_args()
    
    if args.list:
        print("\n=== GCS Checkpoints ===", flush=True)
        checkpoints = list_checkpoints()
        if checkpoints:
            for cp in checkpoints:
                epoch_info = f"epoch {cp['epoch']}" if cp['epoch'] is not None else "unknown epoch"
                print(f"  {cp['name']} ({epoch_info})", flush=True)
        else:
            print("  No checkpoints found in GCS", flush=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

