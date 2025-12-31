"""Deploy checkpoints to Vertex AI Model Registry."""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import tensorflow as tf

from expforge.config import load_config
from expforge.model.checkpoint import (
    get_checkpoint_metadata,
    get_latest_checkpoint_name,
    load_checkpoint,
)


def deploy_checkpoint(
    checkpoint_name: Optional[str] = None,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[Any]:
    """Deploy a checkpoint to Vertex AI Model Registry.
    
    Args:
        checkpoint_name: Name of checkpoint to deploy. Defaults to "latest".
        model_name: Display name for the model (default: auto-generated).
        description: Description for the model.
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import Model
        from google.cloud import storage
    except ImportError:
        print("Error: google-cloud-aiplatform not installed", flush=True)
        return None
    
    config = load_config()
    
    # Initialize Vertex AI
    aiplatform.init(project=config.project_id, location=config.location)
    
    # Resolve checkpoint name (default to latest)
    if checkpoint_name is None or checkpoint_name == "latest":
        resolved_name = get_latest_checkpoint_name()
        if resolved_name is None:
            print("Error: No checkpoints found in GCS", flush=True)
            return None
        checkpoint_name = resolved_name
        print(f"Using latest checkpoint: {checkpoint_name}", flush=True)
    
    # Load checkpoint metadata
    metadata = get_checkpoint_metadata(checkpoint_name)
    if not metadata:
        print(f"Warning: No metadata found for checkpoint {checkpoint_name}", flush=True)
    
    # Load checkpoint from GCS
    print(f"Loading checkpoint {checkpoint_name}...", flush=True)
    model, epoch = load_checkpoint(checkpoint_name)
    
    # Build model if needed
    if not model.built:
        if hasattr(model, 'input_shape') and model.input_shape:
            dummy_input = tf.zeros((1,) + tuple(model.input_shape[1:]))
            _ = model(dummy_input)
        else:
            raise ValueError("Cannot build model: input shape unknown")
    
    # Convert to SavedModel format
    print("Converting to SavedModel format...", flush=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        savedmodel_path = Path(temp_dir) / "saved_model"
        model.export(str(savedmodel_path))
        
        # Upload to GCS
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        if model_name is None:
            model_name = f"triple-mnist-model-{checkpoint_name}"
        
        gcs_model_path = f"gs://{config.bucket_name}/models/{model_name}/{timestamp}"
        
        print(f"Uploading SavedModel to {gcs_model_path}...", flush=True)
        client = storage.Client(project=config.project_id)
        bucket = client.bucket(config.bucket_name)
        
        for file_path in savedmodel_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(savedmodel_path)
                blob_name = f"models/{model_name}/{timestamp}/{relative_path}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
        
        # Register in Vertex AI
        print("Registering model in Vertex AI...", flush=True)
        
        if description is None:
            desc_parts = [f"Checkpoint: {checkpoint_name}"]
            if metadata and metadata.get("epoch") is not None:
                desc_parts.append(f"Epoch: {metadata['epoch']}")
            description = " - ".join(desc_parts)
        
        vertex_model = Model.upload(
            display_name=model_name,
            artifact_uri=gcs_model_path,
            serving_container_image_uri=config.serving_container_image_uri,
            description=description,
        )
        
        print(f"✓ Model deployed: {vertex_model.resource_name}", flush=True)
        print(f"  GCS path: {gcs_model_path}", flush=True)
        
        return vertex_model


def main():
    """CLI entry point for deployment."""
    parser = argparse.ArgumentParser(description="Deploy checkpoint to Vertex AI Model Registry")
    parser.add_argument(
        "checkpoint_name",
        nargs="?",
        default="latest",
        help="Name of checkpoint to deploy (default: latest)",
    )
    parser.add_argument("--model-name", type=str, help="Display name for the model (default: auto-generated)")
    parser.add_argument("--description", type=str, help="Description for the model")
    args = parser.parse_args()
    
    deploy_checkpoint(
        checkpoint_name=args.checkpoint_name,
        model_name=args.model_name,
        description=args.description,
    )


if __name__ == "__main__":
    main()

