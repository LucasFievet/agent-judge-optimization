"""GCS bucket management for Vertex AI."""

from __future__ import annotations

from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound, Forbidden

from expforge.config import ExpforgeConfig


def get_or_create_bucket(config: ExpforgeConfig, create: bool = False) -> tuple[Optional[object], bool]:
    """
    Get or create GCS bucket.
    
    Args:
        config: Configuration
        create: If True, create bucket if it doesn't exist
    
    Returns:
        Tuple of (bucket object or None, was_created: bool)
    """
    client = storage.Client(project=config.project_id)
    bucket = client.bucket(config.bucket_name)
    
    try:
        bucket.reload()
        return bucket, False
    except NotFound:
        if create:
            bucket.create(location=config.location)
            return bucket, True
        return None, False
    except Forbidden:
        raise


def check_bucket_access(config: ExpforgeConfig) -> tuple[bool, Optional[str]]:
    """
    Check if bucket is accessible.
    
    Args:
        config: Configuration
    
    Returns:
        Tuple of (is_accessible: bool, error_message: Optional[str])
    """
    bucket, _ = get_or_create_bucket(config, create=False)
    if bucket:
        return True, None
    return False, f"Bucket '{config.bucket_name}' not accessible"

