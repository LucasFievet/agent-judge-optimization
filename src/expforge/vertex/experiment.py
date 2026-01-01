"""Vertex AI Experiment management."""

from __future__ import annotations

from typing import Optional, Union

from google.cloud.aiplatform import Experiment

from expforge.config import ExpforgeConfig


def get_or_create_experiment(config: ExpforgeConfig, create: bool = False) -> tuple[Optional[Union[Experiment, str]], bool]:
    """
    Get or create Vertex AI Experiment.
    
    Args:
        config: Configuration
        create: If True, create experiment if it doesn't exist
    
    Returns:
        Tuple of (experiment object/name or None, was_created: bool)
    """
    if create:
        try:
            experiment = Experiment.create(experiment_name=config.experiment_name)
            return experiment, True
        except Exception:
            try:
                experiment = Experiment(experiment_name=config.experiment_name)
                return experiment, False
            except Exception:
                return config.experiment_name, False
    
    return config.experiment_name, False


def check_experiment_access(config: ExpforgeConfig) -> tuple[bool, Optional[str]]:
    """
    Check if experiment is accessible.
    
    Args:
        config: Configuration
    
    Returns:
        Tuple of (is_accessible: bool, error_message: Optional[str])
    """
    try:
        _ = Experiment(experiment_name=config.experiment_name)
        return True, None
    except Exception:
        return False, f"Experiment '{config.experiment_name}' not accessible"

