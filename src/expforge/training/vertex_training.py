"""Vertex AI training job for triple MNIST model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from expforge.datasets import load_images, load_manifest
from expforge.training.triple_mnist_model import create_triple_mnist_model


def _get_metadata_path(model_path: Path) -> Path:
    """Get path to metadata file for a model."""
    return model_path.with_suffix('.json')


def _load_epoch_from_metadata(checkpoint_path: Path) -> Optional[int]:
    """Load the last epoch number from metadata file."""
    metadata_path = _get_metadata_path(checkpoint_path)
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('last_epoch')
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_epoch_metadata(model_path: Path, epoch: int):
    """Save epoch number to metadata file."""
    metadata_path = _get_metadata_path(model_path)
    metadata = {'last_epoch': epoch}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def prepare_data_for_training(
    train_dir: Path,
    val_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
):
    """
    Load and prepare triple MNIST data for training.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        Missing splits will be None.
    """
    def load_split(directory: Path):
        if not directory.exists():
            return None, None
        manifest = load_manifest(directory)
        images = load_images(manifest, directory)
        labels = manifest["label"].astype(int).to_numpy()
        
        # Normalize images to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Add channel dimension: (N, 28, 84) -> (N, 28, 84, 1)
        images = np.expand_dims(images, axis=-1)
        
        return images, labels
    
    X_train, y_train = load_split(train_dir)
    X_val, y_val = load_split(val_dir) if val_dir else (None, None)
    X_test, y_test = load_split(test_dir) if test_dir else (None, None)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model_local(
    train_dir: Path,
    val_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    model_out: Path = Path("runs/triple_mnist_model"),
    epochs: int = 10,
    batch_size: int = 32,
    tensorboard_log_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    initial_epoch: Optional[int] = None,
):
    """
    Train the triple MNIST model locally (for testing).
    
    Args:
        train_dir: Directory containing training data
        val_dir: Optional directory containing validation data
        test_dir: Optional directory containing test data
        model_out: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        tensorboard_log_dir: Directory for TensorBoard logs
        resume_from: Optional path to a saved model to resume training from
        initial_epoch: Starting epoch number (used when resuming)
    
    Returns:
        Tuple of (model, history, test_metrics)
    """
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_training(train_dir, val_dir, test_dir)
    
    # Load model from checkpoint or create new one
    if resume_from and resume_from.exists():
        # Ensure the path has the correct extension
        checkpoint_path = resume_from
        if checkpoint_path.suffix not in ['.keras', '.h5']:
            # Try both extensions
            keras_path = checkpoint_path.with_suffix('.keras')
            h5_path = checkpoint_path.with_suffix('.h5')
            if keras_path.exists():
                checkpoint_path = keras_path
            elif h5_path.exists():
                checkpoint_path = h5_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {resume_from} (tried .keras and .h5)")
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # Use safe_mode=False to allow Lambda layer deserialization (we trust our own models)
        model = tf.keras.models.load_model(str(checkpoint_path), safe_mode=False)
        
        # Auto-detect epoch from metadata if initial_epoch not explicitly set
        if initial_epoch is None:
            detected_epoch = _load_epoch_from_metadata(checkpoint_path)
            if detected_epoch is not None:
                initial_epoch = detected_epoch + 1  # Start from next epoch
                print(f"Auto-detected last epoch: {detected_epoch}, resuming from epoch {initial_epoch}")
            else:
                initial_epoch = 0
                print("No epoch metadata found, starting from epoch 0")
        else:
            print(f"Resuming training from epoch {initial_epoch} (manually specified)")
    else:
        if resume_from:
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        # Create new model
        model = create_triple_mnist_model()
        if initial_epoch is None:
            initial_epoch = 0
        print(f"Starting training from scratch (epoch {initial_epoch})")
    
    # Ensure the path has a .keras extension for Keras format
    if model_out.suffix not in ['.keras', '.h5']:
        model_out = model_out.with_suffix('.keras')
    
    # Setup callbacks
    callbacks = []
    
    if tensorboard_log_dir:
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_log_dir),
                histogram_freq=1,
            )
        )
    
    if val_dir:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_out),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
            )
        )
    
    # Train model
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
    
    # Evaluate on test set if available
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
        print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    
    # Save final model
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))
    
    # Save epoch metadata (last epoch completed = epochs - 1, but we track the actual last epoch)
    last_epoch = initial_epoch + len(history.history.get('loss', [])) - 1
    _save_epoch_metadata(model_out, last_epoch)
    
    return model, history, test_metrics

