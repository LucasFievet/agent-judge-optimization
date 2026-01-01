"""Unified training script for Vertex AI Custom Jobs.

This script is designed to run in Vertex AI Custom Jobs.
It downloads data from GCS, trains the model, and uploads results back to GCS.
Vertex AI integration (Experiments, TensorBoard, Model Registry) is always enabled.
"""

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Suppress FutureWarning about np.object from keras/tf2onnx
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.object.*")

import tensorflow as tf

# Normal imports - work when packaged by CustomTrainingJob
from expforge.data.gcs_utils import download_from_gcs
from expforge.training.triple_mnist_model import create_triple_mnist_model
from expforge.training.data_utils import prepare_data_for_training
from expforge.model.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    get_latest_checkpoint_name,
)
from expforge.config import load_config, ExpforgeConfig
from expforge.vertex.experiment import get_or_create_experiment
from expforge.vertex.tensorboard import get_or_create_tensorboard
from expforge.vertex.run import create_run, end_run
from expforge.vertex.metrics import create_metrics_callback, log_metrics


def train(
    data_root: Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    resume_from: Optional[str] = None,
    config: Optional[ExpforgeConfig] = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Optional[Any], Optional[Dict[str, float]]]:
    """
    Train the triple MNIST model with Vertex AI integration.
    
    Args:
        data_root: Root directory containing train/, val/, and test/ subdirectories
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        resume_from: Optional checkpoint name to resume from (None = start fresh, "latest" = use latest)
        config: Vertex AI configuration (defaults to loaded config)
    
    Returns:
        Tuple of (model, history, experiment_run, test_metrics)
    """
    # Setup Vertex AI resources
    if config is None:
        config = load_config()
    
    # Initialize Vertex AI
    from google.cloud import aiplatform
    aiplatform.init(project=config.project_id, location=config.location)
    
    # Get or create experiment and tensorboard
    experiment, _ = get_or_create_experiment(config, create=True)
    tensorboard, _ = get_or_create_tensorboard(config, create=True)
    
    # Link tensorboard to experiment
    if experiment and tensorboard:
        try:
            experiment.assign_backing_tensorboard(tensorboard)
        except Exception:
            pass  # Already linked or not needed
    
    # Determine checkpoint to resume from
    if resume_from == "latest":
        checkpoint_name = get_latest_checkpoint_name()
        if not checkpoint_name:
            print("No checkpoint found, starting from scratch", flush=True)
            checkpoint_name = None
        else:
            print(f"Resuming from latest checkpoint: {checkpoint_name}", flush=True)
    elif resume_from:
        checkpoint_name = resume_from
        print(f"Resuming from checkpoint: {checkpoint_name}", flush=True)
    else:
        checkpoint_name = None
    
    initial_epoch = 0
    
    # Load model from checkpoint or create new
    if checkpoint_name:
        model, initial_epoch = load_checkpoint(checkpoint_name)
        initial_epoch += 1  # Start from next epoch
        print(f"Loaded checkpoint from epoch {initial_epoch - 1}, resuming from epoch {initial_epoch}", flush=True)
    else:
        model = create_triple_mnist_model()
        print("Starting training from scratch", flush=True)
    
    # Start experiment run
    metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if checkpoint_name:
        metadata.update({"resume_from": checkpoint_name, "initial_epoch": initial_epoch})
    
    run = create_run(
        experiment=experiment,
        tensorboard=tensorboard,
        metadata=metadata,
    )
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_training(data_root)
    
    # Validate training data
    if X_train is None or y_train is None:
        raise ValueError(f"Training data not found or empty in {data_root / 'train'}")
    
    print(f"Training samples: {len(y_train)}", flush=True)
    print(f"Training images shape: {X_train.shape}", flush=True)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Setup callbacks
    callbacks = []
    
    # TensorBoard callback - extract run name from run object
    run_name = run.name.split('/')[-1] if '/' in run.name else run.name
    tensorboard_log_dir = f"{config.tensorboard_log_dir}/{config.experiment_name}/{run_name}"
    
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        )
    )
    
    # Vertex AI metrics callback
    callbacks.append(create_metrics_callback(run))
    
    # Train
    validation_data = (X_val, y_val) if X_val is not None else None
    
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Calculate final epoch
    num_epochs_trained = len(history.history.get('loss', []))
    last_epoch = initial_epoch + num_epochs_trained - 1
    
    # Evaluate on test set
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
        log_metrics(run, test_metrics, step=initial_epoch + num_epochs_trained)
    
    # Save final checkpoint
    final_checkpoint_name = save_checkpoint(
        model=model,
        epoch=last_epoch,
    )
    print(f"✓ Saved final checkpoint: {final_checkpoint_name} (epoch {last_epoch})", flush=True)
    
    # End Vertex AI run
    end_run(run)
    
    return model, history, run, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train triple MNIST model with Vertex AI")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint name")
    args = parser.parse_args()
    
    # Determine resume_from
    resume_from = "latest" if args.resume else args.resume_from
    
    # Load config
    config = load_config()
    
    print("=" * 80, flush=True)
    print("Starting training script with Vertex AI", flush=True)
    print(f"Project: {config.project_id}, Location: {config.location}", flush=True)
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.learning_rate}", flush=True)
    print("=" * 80, flush=True)
    
    # Download all data from GCS (contains train/, val/, test/ subdirectories)
    data_root_gcs = f"gs://{config.bucket_name}/{config.data_root_gcs}"
    data_root = Path("/tmp/triple-mnist/data")
    
    print("\nDownloading data from GCS...", flush=True)
    download_from_gcs(data_root_gcs, data_root)
    
    # Call unified train function
    model, history, run, test_metrics = train(
        data_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=resume_from,
        config=config,
    )
    
    print("Training complete!", flush=True)


if __name__ == "__main__":
    main()
