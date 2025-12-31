"""GCS utilities for data upload and download."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _gsutil_args(*args: str) -> list[str]:
    """Build gsutil command with macOS multiprocessing warning suppression."""
    return [
        "gsutil",
        "-o", "GSUtil:parallel_process_count=1",  # Suppress macOS multiprocessing warnings
        *args
    ]


def download_from_gcs(gcs_path: str, local_path: Path) -> Path:
    """Download data from GCS to local path using gsutil."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Expected GCS path (gs://...), got: {gcs_path}")
    
    print(f"Downloading data from {gcs_path} to {local_path}...", flush=True)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Use gsutil -m for parallel downloads
    gcs_path_with_slash = gcs_path if gcs_path.endswith("/") else f"{gcs_path}/"
    
    subprocess.run(
        _gsutil_args("-m", "rsync", "-r", gcs_path_with_slash, str(local_path)),
        check=True,
        timeout=3600,
    )
    print(f"✓ Downloaded data to {local_path}", flush=True)
    
    return local_path


def upload_to_gcs(local_path: Path, gcs_path: str) -> None:
    """Upload directory contents to GCS using gsutil."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Expected GCS path (gs://...), got: {gcs_path}")
    
    gcs_path_with_slash = gcs_path if gcs_path.endswith("/") else f"{gcs_path}/"
    
    subprocess.run(
        _gsutil_args("-m", "rsync", "-r", str(local_path), gcs_path_with_slash),
        check=True,
    )
    print(f"✓ Uploaded to {gcs_path}", flush=True)

