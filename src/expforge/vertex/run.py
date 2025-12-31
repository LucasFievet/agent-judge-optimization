"""Vertex AI Experiment Run management."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from google.cloud.aiplatform import ExperimentRun


def create_run(
    experiment,
    run_name: Optional[str] = None,
    tensorboard=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentRun:
    """
    Create a new experiment run.
    
    Args:
        experiment: Vertex AI Experiment object
        run_name: Name for the run (auto-generated if None)
        tensorboard: Optional TensorBoard object
        metadata: Optional metadata to log as parameters
    
    Returns:
        ExperimentRun object
    """
    if run_name is None:
        run_name = f"triple-mnist-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    run = ExperimentRun.create(
        run_name=run_name,
        experiment=experiment,
        tensorboard=tensorboard if tensorboard else None,
    )
    run._run_name = run_name
    
    if metadata:
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
        if filtered_metadata:
            run.log_params(filtered_metadata)
    
    return run


def end_run(run: ExperimentRun):
    """
    End an experiment run.
    
    Args:
        run: ExperimentRun object
    """
    try:
        from google.cloud.aiplatform import gapic
        run.update_state(state=gapic.Execution.State.COMPLETE)
        run.end_run()
    except Exception:
        run.end_run()

