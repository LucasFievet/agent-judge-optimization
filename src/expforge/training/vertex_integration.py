"""Comprehensive Vertex AI integration for triple MNIST training."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf

# Suppress "An error occurred" messages from Google Cloud libraries during import
# These are printed to stderr when importlib.metadata fails (Python 3.9 compatibility issue)
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Import Google Cloud modules with stderr suppressed
with suppress_stderr():
    from google.cloud import aiplatform
    from google.cloud.aiplatform import (
        CustomJob,
        Model,
        Tensorboard,
    )
    # Note: Experiment is available directly from aiplatform, not from a separate module
    try:
        from google.cloud.aiplatform import Experiment, ExperimentRun
        EXPERIMENTS_AVAILABLE = True
    except ImportError:
        EXPERIMENTS_AVAILABLE = False
        Experiment = None
        ExperimentRun = None

from expforge.config import VertexAIConfig, load_vertex_config
from expforge.training.custom_job import CustomTrainingJobManager
from expforge.training.vertex_training import _load_epoch_from_metadata, _save_epoch_metadata


class VertexAITrainingManager:
    """Manages Vertex AI resources for training triple MNIST models."""

    def __init__(
        self,
        config: Optional[VertexAIConfig] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        bucket_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize Vertex AI training manager.
        
        Args:
            config: Vertex AI configuration (defaults to loaded config)
            project_id: GCP project ID (overrides config)
            location: GCP region (overrides config)
            bucket_name: GCS bucket name (overrides config)
            experiment_name: Name for Vertex AI Experiment (overrides config)
        """
        self.config = config or load_vertex_config()
        self.project_id = project_id or self.config.project_id
        self.location = location or self.config.location
        self.bucket_name = bucket_name or self.config.bucket_name
        self.experiment_name = experiment_name or self.config.experiment_name
        
        # Initialize Vertex AI with the actual values (not the parameters which might be None)
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Get or create experiment
        self.experiment = self._get_or_create_experiment()
        
        # Link TensorBoard to experiment if available
        try:
            tensorboard = self.get_tensorboard_resource()
            if tensorboard and self.experiment:
                self.experiment.assign_backing_tensorboard(tensorboard)
        except Exception as e:
            # If TensorBoard linking fails, we'll handle it when setting up the callback
            pass
        
        self.current_run = None

    def _get_or_create_experiment(self):
        """
        Get or create a Vertex AI Experiment.
        
        Raises:
            RuntimeError: If experiments are not available or creation fails
        """
        if not EXPERIMENTS_AVAILABLE:
            raise RuntimeError(
                "Vertex AI Experiments API is not available. "
                "Please ensure you have the latest google-cloud-aiplatform package installed: "
                "pip install --upgrade google-cloud-aiplatform"
            )
        
        try:
            # Try to get existing experiment
            exp_list = Experiment.list(filter=f'display_name="{self.experiment_name}"')
            if exp_list:
                return exp_list[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to list Vertex AI experiments: {e}\n"
                f"Please check your Vertex AI configuration and permissions."
            ) from e
        
        # Create new experiment
        try:
            return Experiment.create(experiment_name=self.experiment_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Vertex AI experiment '{self.experiment_name}': {e}\n"
                f"Please check your Vertex AI configuration and permissions."
            ) from e

    def start_run(
        self,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a new experiment run.
        
        Args:
            run_name: Name for this run (defaults to timestamp)
            description: Description of the run
            metadata: Additional metadata to log
        
        Returns:
            ExperimentRun object
        
        Raises:
            RuntimeError: If experiment run cannot be started
        """
        if not self.experiment:
            raise RuntimeError(
                "No Vertex AI experiment available. "
                "Cannot start experiment run without a valid experiment."
            )
        
        if run_name is None:
            run_name = f"triple-mnist-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            # Get TensorBoard resource to link with the run
            tensorboard = self.get_tensorboard_resource()
            
            # Create experiment run using ExperimentRun.create()
            # Link TensorBoard to the run if available
            self.current_run = ExperimentRun.create(
                run_name=run_name,
                experiment=self.experiment,
                tensorboard=tensorboard if tensorboard else None,
            )
            
            # Store run_name for later reference (run_name is the display name)
            # The run object doesn't expose display_name directly, so we store it
            if not hasattr(self.current_run, '_run_name'):
                self.current_run._run_name = run_name
            
            if description:
                self.current_run.log_params({"description": description})
            
            if metadata:
                # Filter out None values - log_params only accepts float, int, or str
                filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                if filtered_metadata:
                    self.current_run.log_params(filtered_metadata)
        except Exception as e:
            raise RuntimeError(
                f"Failed to start Vertex AI experiment run '{run_name}': {e}\n"
                f"Please check your Vertex AI configuration and permissions."
            ) from e
        
        return self.current_run

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to the current experiment run.
        
        Note: The step parameter is kept for API compatibility but is not used.
        For time series metrics with steps, use log_time_series_metrics() instead.
        """
        if self.current_run is None:
            return  # Silently skip if no run
        
        try:
            # Filter out non-numeric values and ensure all values are floats
            filtered_metrics = {}
            for k, v in metrics.items():
                try:
                    filtered_metrics[k] = float(v)
                except (ValueError, TypeError):
                    continue  # Skip non-numeric metrics
            
            if filtered_metrics:
                # log_metrics() doesn't accept step argument, only log_time_series_metrics() does
                self.current_run.log_metrics(filtered_metrics)
        except Exception as e:
            # Log error for debugging but don't fail
            print(f"Warning: Failed to log metrics to Vertex AI: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters to the current experiment run."""
        if self.current_run is None:
            return  # Silently skip if no run
        
        try:
            self.current_run.log_params(params)
        except Exception:
            pass  # Silently fail if logging doesn't work

    def log_model(self, model_path: Path, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Upload model to Vertex AI Model Registry.
        
        Following Vertex AI best practices:
        1. Convert model to SavedModel format (required by Vertex AI)
        2. Upload SavedModel artifacts to GCS bucket
        3. Register model in Vertex AI Model Registry using GCS path
        
        Args:
            model_path: Local path to saved model (file or directory)
            model_name: Display name for the model
            metadata: Additional metadata to include in model description
        
        Raises:
            FileNotFoundError: If model_path doesn't exist
            Exception: If conversion, upload, or registration fails
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create timestamped GCS path for model artifacts
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        gcs_model_path = f"gs://{self.bucket_name}/models/{model_name}/{timestamp}"
        
        # Initialize GCS client
        from google.cloud import storage
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.bucket_name)
        
        # Vertex AI Model Registry requires SavedModel format (directory with saved_model.pb)
        # Convert .keras/.h5 files to SavedModel format if needed
        if model_path.is_file() and model_path.suffix in ['.keras', '.h5']:
            # Load the model and convert to SavedModel format
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_savedmodel_path = Path(temp_dir) / "saved_model"
                print(f"Converting {model_path.name} to SavedModel format...")
                
                try:
                    # Load model (use safe_mode=False for Lambda layers)
                    model = tf.keras.models.load_model(str(model_path), safe_mode=False)
                    
                    # Save in SavedModel format using Keras export API
                    # This works with Keras 3/TensorFlow 2.20+ and avoids Python 3.13 compatibility issues
                    # with tf.saved_model.save()
                    # Loaded models should already be built, but we verify
                    if not model.built:
                        # If model isn't built, we need to build it first
                        # This shouldn't happen with saved models, but handle it gracefully
                        print("Warning: Model not built, attempting to build...")
                        # Try to infer input shape from model
                        if hasattr(model, 'input_shape') and model.input_shape:
                            dummy_input = tf.zeros((1,) + tuple(model.input_shape[1:]))
                            _ = model(dummy_input)
                        else:
                            raise ValueError("Cannot build model: input shape unknown")
                    
                    # Export to SavedModel format
                    model.export(str(temp_savedmodel_path))
                    
                    # Verify SavedModel was created correctly
                    saved_model_pb = temp_savedmodel_path / "saved_model.pb"
                    if not saved_model_pb.exists():
                        raise ValueError(f"SavedModel conversion failed: saved_model.pb not found at {temp_savedmodel_path}")
                    
                    # Upload SavedModel directory to GCS
                    uploaded_files = []
                    for file_path in temp_savedmodel_path.rglob("*"):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(temp_savedmodel_path)
                            blob_name = f"models/{model_name}/{timestamp}/{relative_path}"
                            blob = bucket.blob(blob_name)
                            blob.upload_from_filename(str(file_path))
                            uploaded_files.append(blob_name)
                    
                    print(f"✓ Uploaded {len(uploaded_files)} SavedModel files to GCS: {gcs_model_path}")
                except Exception as e:
                    raise Exception(f"Failed to convert and upload model: {e}") from e
                    
        elif model_path.is_dir():
            # Directory is already in SavedModel format
            # Verify it contains saved_model.pb
            saved_model_pb = model_path / "saved_model.pb"
            if not saved_model_pb.exists():
                # Check if it's a nested saved_model directory
                nested_saved_model = model_path / "saved_model" / "saved_model.pb"
                if nested_saved_model.exists():
                    model_path = model_path / "saved_model"
                else:
                    raise ValueError(f"Directory {model_path} does not contain saved_model.pb. Vertex AI requires SavedModel format.")
            
            # Upload SavedModel directory to GCS
            uploaded_files = []
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_path)
                    blob_name = f"models/{model_name}/{timestamp}/{relative_path}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))
                    uploaded_files.append(blob_name)
            print(f"✓ Uploaded {len(uploaded_files)} SavedModel files to GCS: {gcs_model_path}")
        else:
            raise ValueError(f"Unsupported model format: {model_path}. Vertex AI requires SavedModel format (directory with saved_model.pb).")
        
        # Ensure Vertex AI is initialized before registering model
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Build model description with metadata
        description_parts = [f"Triple MNIST model"]
        if metadata:
            metadata_str = ", ".join([f"{k}={v}" for k, v in metadata.items()])
            description_parts.append(f"Metadata: {metadata_str}")
        description = " - ".join(description_parts)
        
        # Register model in Vertex AI Model Registry
        # This is the recommended approach per Vertex AI best practices
        print(f"Registering model in Vertex AI Model Registry...")
        try:
            vertex_model = Model.upload(
            display_name=model_name,
                artifact_uri=gcs_model_path,  # GCS path to SavedModel directory
                serving_container_image_uri=self.config.serving_container_image_uri,
                description=description,
            )
            print(f"✓ Model registered successfully: {vertex_model.resource_name}")
        except Exception as e:
            raise Exception(f"Failed to register model in Vertex AI Model Registry: {e}") from e
        
        # Store model reference in run metadata for lineage tracking
        # Note: log_model() expects a tf.Module/sklearn/xgboost model object, not a Vertex AI Model object
        # So we store the model reference in run params instead
        if self.current_run:
            try:
                self.current_run.log_params({
                    "model_resource_name": vertex_model.resource_name,
                    "model_gcs_path": gcs_model_path,
                })
            except Exception:
                # Not critical - model is still registered
                pass
        
        return vertex_model

    def get_tensorboard_resource(self) -> Tensorboard:
        """
        Get or create a TensorBoard resource.
        
        Raises:
            Exception: If TensorBoard resource cannot be retrieved or created
        """
        tensorboard_name = self.config.tensorboard_name
        try:
            tensorboards = Tensorboard.list(filter=f'display_name="{tensorboard_name}"')
            if tensorboards:
                return tensorboards[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to list TensorBoard resources: {e}\n"
                f"Please check your Vertex AI configuration and permissions."
            ) from e
        
        # Create new TensorBoard
        try:
                    return Tensorboard.create(display_name=tensorboard_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create TensorBoard resource '{tensorboard_name}': {e}\n"
                f"Please check your Vertex AI configuration and permissions."
            ) from e

    def end_run(self):
        """
        End the current experiment run and update its state to COMPLETE.
        
        This ensures the run status is properly synced in Vertex AI dashboard.
        """
        if self.current_run:
            try:
                # Update run state to COMPLETE before ending
                # This ensures the status is properly synced
                from google.cloud.aiplatform import gapic
                self.current_run.update_state(state=gapic.Execution.State.COMPLETE)
                self.current_run.end_run()
                print("Experiment run completed and synced to Vertex AI")
            except Exception as e:
                # If update_state fails, still try to end the run
                try:
                    self.current_run.end_run()
                    print(f"Experiment run ended (state update failed: {e})")
                except Exception as e2:
                    print(f"Warning: Failed to end experiment run: {e2}")
            finally:
                self.current_run = None


def train_with_vertex_experiments(
    train_dir: Path,
    val_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    model_out: Path = Path("runs/triple_mnist_model"),
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    config: Optional[VertexAIConfig] = None,
    run_name: Optional[str] = None,
    upload_model: bool = True,
    resume_from: Optional[Path] = None,
    initial_epoch: Optional[int] = None,
    resume_from_model_registry: Optional[str] = None,
) -> tuple:
    """
    Train model with full Vertex AI Experiments integration.
    
    Args:
        train_dir: Directory containing training data
        val_dir: Optional directory containing validation data
        test_dir: Optional directory containing test data
        model_out: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        config: Vertex AI configuration (defaults to loaded config)
        run_name: Name for experiment run
        upload_model: Whether to upload model to Vertex AI Model Registry
        resume_from: Optional path to a saved model to resume training from (local file)
        initial_epoch: Starting epoch number (used when resuming)
        resume_from_model_registry: Optional model resource name or ID from Vertex AI Model Registry to resume from
    
    Returns:
        Tuple of (model, history, experiment_run, test_metrics)
    """
    from expforge.training.vertex_training import prepare_data_for_training
    from expforge.training.triple_mnist_model import create_triple_mnist_model
    
    # Initialize Vertex AI manager
    manager = VertexAITrainingManager(config=config)
    
    # Start experiment run
    # Build metadata dict, only including non-None values
    metadata = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if resume_from:
        metadata["resume_from"] = str(resume_from)
    if initial_epoch is not None:
        metadata["initial_epoch"] = initial_epoch
    
    run = manager.start_run(
        run_name=run_name,
        description=f"Triple MNIST training - {epochs} epochs, batch_size={batch_size}",
        metadata=metadata,
    )
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_training(train_dir, val_dir, test_dir)
    
    # Load model from checkpoint or create new one
    if resume_from:
        # Resolve path to absolute for better error messages
        resume_from = Path(resume_from).resolve()
        
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
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # Use safe_mode=False to allow Lambda layer deserialization (we trust our own models)
        model = tf.keras.models.load_model(str(checkpoint_path), safe_mode=False)
        
        # Auto-detect epoch from metadata if initial_epoch not explicitly set
        if initial_epoch is None:
            detected_epoch = _load_epoch_from_metadata(checkpoint_path)
            if detected_epoch is not None:
                initial_epoch = detected_epoch + 1  # Start from next epoch
                print(f"✓ Auto-detected last epoch: {detected_epoch}, resuming from epoch {initial_epoch}")
            else:
                initial_epoch = 0
                print("⚠ No epoch metadata found, starting from epoch 0")
        else:
            print(f"Resuming training from epoch {initial_epoch} (manually specified)")
    else:
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
    
    # TensorBoard callback - simplified approach per Vertex AI docs
    # https://docs.cloud.google.com/vertex-ai/docs/experiments/configure-training-script
    # Write logs to GCS - can be any location accessible to Google Cloud
    if run and hasattr(run, '_run_name') and run._run_name:
        run_display_name = run._run_name
    elif run and hasattr(run, 'name') and run.name:
        run_display_name = run.name.split('/')[-1] if '/' in run.name else str(run.name)
    else:
        run_display_name = run_name if run_name else f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Write TensorBoard logs to GCS
    # Per docs: "set the log_dir variable to any location which can connect to Google Cloud"
    log_dir = f"gs://{manager.bucket_name}/tensorboard/{manager.experiment_name}/{run_display_name}"
    
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        )
    )
    print(f"TensorBoard logs will be written to: {log_dir}")
    
    # Model checkpoint
    if val_dir:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_out),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
            )
        )
    
    # Log metrics to Vertex AI Experiments
    # We need both log_metrics() for Experiments UI and log_time_series_metrics() for TensorBoard
    class VertexAIMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, manager: VertexAITrainingManager):
            self.manager = manager
        
        def on_epoch_end(self, epoch, logs=None):
            if logs and manager.current_run:
                try:
                    # Filter and convert metrics to floats
                    metrics = {}
                    for k, v in logs.items():
                        try:
                            metrics[k] = float(v)
                        except (ValueError, TypeError):
                            continue
                    
                    if metrics:
                        # Log time series metrics for TensorBoard visualization
                        # This is required for TensorBoard to show scalar data
                        try:
                            manager.current_run.log_time_series_metrics(metrics, step=epoch)
                        except RuntimeError as e:
                            # If TensorBoard isn't linked, this will fail
                            print(f"Warning: Could not log time series metrics (TensorBoard may not be linked): {e}")
                        
                        # Log regular metrics for Experiments UI
                        # Note: log_metrics() doesn't accept step argument, only log_time_series_metrics() does
                        manager.current_run.log_metrics(metrics)
                except Exception as e:
                    print(f"Warning: Failed to log metrics to Vertex AI: {e}")
    
    callbacks.append(VertexAIMetricsCallback(manager))
    
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
        
        # Log test metrics to Vertex AI Experiments
        if manager.current_run:
            try:
                # Use final epoch as step for test metrics
                final_step = initial_epoch + len(history.history.get('loss', []))
                # Log time series metrics for TensorBoard
                try:
                    manager.current_run.log_time_series_metrics(test_metrics, step=final_step)
                except RuntimeError:
                    pass  # TensorBoard may not be linked
                # Log regular metrics for Experiments UI
                # Note: log_metrics() doesn't accept step argument, only log_time_series_metrics() does
                manager.current_run.log_metrics(test_metrics)
            except Exception as e:
                print(f"Warning: Failed to log test metrics: {e}")
        
        print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    
    # Save final model
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))
    
    # Save epoch metadata
    last_epoch = initial_epoch + len(history.history.get('loss', [])) - 1
    _save_epoch_metadata(model_out, last_epoch)
    
    # Upload to Vertex AI Model Registry if requested
    uploaded_model = None
    if upload_model:
        metadata = {
            "final_train_accuracy": float(history.history.get('accuracy', [0])[-1]),
            "epochs": epochs,
            "batch_size": batch_size,
        }
        if val_dir:
            metadata["final_val_accuracy"] = float(history.history.get('val_accuracy', [0])[-1])
        if test_metrics:
            metadata.update(test_metrics)
        
        uploaded_model = manager.log_model(
            model_path=model_out,
            model_name="triple-mnist-model",
            metadata=metadata,
        )
    
    # End run
    manager.end_run()
    
    return model, history, run, uploaded_model, test_metrics
