"""Vertex AI Experiment management."""

from __future__ import annotations

from typing import Optional

from google.cloud import aiplatform
from google.cloud.aiplatform import Experiment

from expforge.config import ExpforgeConfig


def get_or_create_experiment(config: ExpforgeConfig, create: bool = False) -> tuple[Optional[object], bool]:
    """
    Get or create Vertex AI Experiment.
    
    Args:
        config: Configuration
        create: If True, create experiment if it doesn't exist
    
    Returns:
        Tuple of (experiment object or None, was_created: bool)
    """
    aiplatform.init(project=config.project_id, location=config.location)
    exp_list = Experiment.list(filter=f'display_name="{config.experiment_name}"')
    
    if exp_list:
        return exp_list[0], False
    elif create:
        experiment = Experiment.create(experiment_name=config.experiment_name)
        return experiment, True
    return None, False


def check_experiment_access(config: ExpforgeConfig) -> tuple[bool, Optional[str]]:
    """
    Check if experiment is accessible.
    
    Args:
        config: Configuration
    
    Returns:
        Tuple of (is_accessible: bool, error_message: Optional[str])
    """
    experiment, _ = get_or_create_experiment(config, create=False)
    if experiment:
        return True, None
    return False, f"Experiment '{config.experiment_name}' not accessible"

