"""Generate synthetic triple MNIST dataset by concatenating 3 MNIST images horizontally."""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from expforge.datasets import DatasetRecord, ensure_dataset_dirs, load_images, load_manifest, save_manifest


def generate_triple_mnist(
    mnist_dir: Path,
    count: int,
    output_dir: Path,
    seed: Optional[int] = None,
    split: str = "train",
) -> List[DatasetRecord]:
    """
    Generate triple MNIST samples by concatenating 3 random MNIST images horizontally.
    
    The label is the sum of the three digits (0-27).
    
    Args:
        mnist_dir: Directory containing the base MNIST dataset
        count: Number of triple MNIST samples to generate
        output_dir: Directory to save the generated samples
        seed: Random seed for reproducibility
        split: Dataset split name (train/test)
    
    Returns:
        List of DatasetRecord objects
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # Load base MNIST dataset
    manifest = load_manifest(mnist_dir)
    images = load_images(manifest, mnist_dir)
    labels = manifest["label"].astype(int).to_numpy()
    
    # Filter by split if needed
    if "split" in manifest.columns:
        split_mask = manifest["split"] == split
        images = images[split_mask]
        labels = labels[split_mask]
    
    images_dir = ensure_dataset_dirs(output_dir)
    records: List[DatasetRecord] = []
    
    for _ in range(count):
        # Randomly select 3 indices
        indices = rng.choices(range(len(images)), k=3)
        
        # Get the 3 images and their labels
        img1 = images[indices[0]]
        img2 = images[indices[1]]
        img3 = images[indices[2]]
        
        digit1 = labels[indices[0]]
        digit2 = labels[indices[1]]
        digit3 = labels[indices[2]]
        
        # Concatenate horizontally: 28x28 + 28x28 + 28x28 = 84x28
        triple_image = np.concatenate([img1, img2, img3], axis=1)
        
        # Calculate label (sum of digits, 0-27)
        label = int(digit1 + digit2 + digit3)
        
        # Save image
        file_name = f"{uuid.uuid4().hex}.png"
        img_pil = Image.fromarray(triple_image.astype(np.uint8), mode="L")
        rel_path = Path("images") / file_name
        img_pil.save(images_dir / file_name)
        
        records.append(
            DatasetRecord(
                path=str(rel_path),
                label=str(label),
                split=split,
            )
        )
    
    return save_manifest(records, output_dir)

