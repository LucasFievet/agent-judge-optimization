"""Vertex AI Experiment Run management."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Union

from google.cloud.aiplatform import ExperimentRun


def create_run(
    experiment: Union[str, Any],
    tensorboard=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentRun:
    """
    Create a new experiment run.
    
    Args:
        experiment: Vertex AI Experiment object or experiment name (str)
        tensorboard: Optional TensorBoard object
        metadata: Optional metadata to log as parameters
    
    Returns:
        ExperimentRun object
    """
    run_name = f"triple-mnist-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    run = ExperimentRun.create(
        run_name=run_name,
        experiment=experiment,
        tensorboard=tensorboard,
    )
    
    if metadata:
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
        if filtered_metadata:
            run.log_params(filtered_metadata)
    
    return run
