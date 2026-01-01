"""Vertex AI Experiment management."""

from __future__ import annotations

from typing import Optional, Union

from google.cloud import aiplatform
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
    aiplatform.init(project=config.project_id, location=config.location)
    
    if create:
        # Try to create - will raise if already exists
        try:
            experiment = Experiment.create(experiment_name=config.experiment_name)
            return experiment, True
        except Exception:
            # Already exists, try to get it
            try:
                experiment = Experiment(experiment_name=config.experiment_name)
                return experiment, False
            except Exception:
                # If we can't get it either, return name as fallback
                return config.experiment_name, False
    
    # Return experiment name as string (assumes it exists)
    # ExperimentRun.create() accepts either Experiment object or name string
    return config.experiment_name, False


def check_experiment_access(config: ExpforgeConfig) -> tuple[bool, Optional[str]]:
    """
    Check if experiment is accessible.
    
    Args:
        config: Configuration
    
    Returns:
        Tuple of (is_accessible: bool, error_message: Optional[str])
    """
    aiplatform.init(project=config.project_id, location=config.location)
    try:
        # Try to create an Experiment object from name - will fail if doesn't exist
        _ = Experiment(experiment_name=config.experiment_name)
        return True, None
    except Exception:
        return False, f"Experiment '{config.experiment_name}' not accessible"

