"""Centralized Vertex AI context management with automatic initialization."""

from __future__ import annotations

from typing import Optional

from google.cloud import aiplatform

from expforge.config import ExpforgeConfig, load_config as _load_config


# Module-level singleton state
_config: Optional[ExpforgeConfig] = None
_initialized: bool = False


def get_config() -> ExpforgeConfig:
    """
    Get Vertex AI configuration, loading and initializing if needed.
    
    This function ensures that:
    1. Config is loaded only once (lazy loading)
    2. aiplatform is initialized automatically when config is first loaded
    
    Returns:
        ExpforgeConfig: The configuration object
    """
    global _config, _initialized
    
    if _config is None:
        _config = _load_config()
        _ensure_initialized()
    
    return _config


def _ensure_initialized() -> None:
    """Initialize aiplatform if not already done."""
    global _config, _initialized
    
    if not _initialized:
        if _config is None:
            _config = _load_config()
        
        aiplatform.init(
            project=_config.project_id,
            location=_config.location,
            staging_bucket=_config.bucket_name,
        )
        _initialized = True

