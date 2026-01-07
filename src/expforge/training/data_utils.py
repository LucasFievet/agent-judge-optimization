"""Data preparation utilities for training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from expforge.data.io import load_images, load_manifest


def prepare_data_for_training(
    data_root: Path,
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray],  # X_train, y_train
    Optional[np.ndarray], Optional[np.ndarray],  # X_val, y_val
    Optional[np.ndarray], Optional[np.ndarray],  # X_test, y_test
]:
    """
    Load and prepare triple MNIST data for training.
    
    Looks for train, val, and test subdirectories under data_root.
    
    Args:
        data_root: Root directory containing train/, val/, and test/ subdirectories
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        Missing splits will be None.
    """
    def load_split(directory: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
    
    X_train, y_train = load_split(data_root / "train")
    X_val, y_val = load_split(data_root / "val")
    X_test, y_test = load_split(data_root / "test")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
