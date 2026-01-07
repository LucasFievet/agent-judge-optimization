"""Metrics logging for Vertex AI Experiment Runs."""

from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf
from google.cloud.aiplatform import ExperimentRun


def log_metrics(run: ExperimentRun, metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log metrics to an experiment run.
    
    Args:
        run: ExperimentRun object
        metrics: Dictionary of metric names to values
        step: Optional step number for time series metrics
    """
    filtered_metrics = {k: float(v) for k, v in metrics.items()}
    
    if step is not None:
        try:
            run.log_time_series_metrics(filtered_metrics, step=step)
        except RuntimeError:
            pass  # TensorBoard may not be linked
    
    run.log_metrics(filtered_metrics)


def create_metrics_callback(run: ExperimentRun) -> tf.keras.callbacks.Callback:
    """
    Create Keras callback for logging metrics to Vertex AI.
    
    Args:
        run: ExperimentRun object
    
    Returns:
        Keras callback for logging metrics
    """
    class VertexAIMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, run: ExperimentRun):
            self.run = run
        
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                metrics = {}
                for k, v in logs.items():
                    try:
                        metrics[k] = float(v)
                    except (ValueError, TypeError):
                        continue
                
                if metrics:
                    log_metrics(self.run, metrics, step=epoch)
    
    return VertexAIMetricsCallback(run)

