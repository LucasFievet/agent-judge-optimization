"""Vertex AI Custom Job training entrypoint.

This script runs inside Vertex AI Custom Jobs. It demonstrates the Vertex
contract: load config, create an experiment run, optionally use GCS data,
and save checkpoints. Replace the placeholder training with your own
model and data loading.
"""

import argparse
import os
from pathlib import Path

import tensorflow as tf

from expforge.config import ExpforgeConfig
from expforge.vertex.context import get_config
from expforge.vertex.tensorboard import get_or_create_tensorboard
from expforge.vertex.run import create_run
from expforge.vertex.metrics import create_metrics_callback, log_metrics
from expforge.model.checkpoint import save_checkpoint
from google.cloud.aiplatform import ExperimentRun


def _placeholder_model():
    """Minimal Keras model for placeholder training (no dataset)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(1),
    ])
    return model


def train(
    epochs: int = 1,
    config: ExpforgeConfig | None = None,
) -> tuple[tf.keras.Model, ExperimentRun]:
    """
    Placeholder training with Vertex AI integration.

    Creates a run, trains a minimal model for one step, saves a checkpoint,
    and ends the run. Replace with your own data and model.
    """
    if config is None:
        config = get_config()

    experiment = config.experiment_name
    tensorboard, _ = get_or_create_tensorboard(config, create=True)

    existing_run_name = os.environ.get("EXPFORGE_RUN_NAME")
    if existing_run_name:
        run = ExperimentRun.get(run_name=existing_run_name, experiment=experiment)
    else:
        run = create_run(
            experiment=experiment,
            tensorboard=tensorboard,
            metadata={"epochs": epochs},
        )

    model = _placeholder_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Placeholder step (no real data)
    dummy_x = tf.zeros((2, 4))
    dummy_y = tf.zeros((2, 1))
    model.fit(dummy_x, dummy_y, epochs=epochs, verbose=1)

    log_metrics(run, {"loss": 0.0, "mae": 0.0}, step=0)
    save_checkpoint(model=model, epoch=0)
    run.end_run()

    return model, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI Custom Job training entrypoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (ignored in placeholder)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (ignored in placeholder)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint")
    args = parser.parse_args()

    config = get_config()
    print("Vertex AI training (placeholder)", flush=True)
    print(f"Project: {config.project_id}, Location: {config.location}", flush=True)
    train(epochs=args.epochs, config=config)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
