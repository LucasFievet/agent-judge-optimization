"""Generate synthetic triple MNIST dataset by concatenating 3 MNIST images."""

from __future__ import annotations

import argparse
import random
import uuid
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from expforge.data.io import DatasetRecord, ensure_dataset_dirs, load_images, load_manifest, save_manifest
from expforge.data.gcs_utils import download_from_gcs, upload_to_gcs
from expforge.data.util import get_data_root
from expforge.config import load_config


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
        split: Dataset split name (train/test/val)
    
    Returns:
        List of DatasetRecord objects
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # Load base MNIST dataset
    manifest = load_manifest(mnist_dir)
    images = load_images(manifest, mnist_dir)
    labels = manifest["label"].astype(int).to_numpy()
    
    # Filter by split if needed (MNIST only has train/test, so use train for val)
    if "split" in manifest.columns:
        mnist_split = split if split in ["train", "test"] else "train"
        split_mask = manifest["split"] == mnist_split
        images = images[split_mask]
        labels = labels[split_mask]
    
    if len(images) == 0:
        raise ValueError(f"No images found in MNIST dataset")
    
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


def visualize_triple_mnist(dataset_dir: Path, output: Path, per_class: int = 4) -> None:
    """Plot a grid of triple MNIST samples (labels 0-27)."""
    manifest = load_manifest(dataset_dir)
    images_dir = dataset_dir / "images"

    # Convert labels to numeric for consistent comparison
    manifest["label"] = pd.to_numeric(manifest["label"], errors="coerce")

    # For triple MNIST, labels range from 0 to 27
    num_classes = 28
    fig, axes = plt.subplots(nrows=num_classes, ncols=per_class, figsize=(per_class * 2, num_classes * 0.8))
    
    for label in range(num_classes):
        # Filter by numeric label value
        label_rows = manifest[manifest["label"] == label].head(per_class)
        
        for idx, (_, row) in enumerate(label_rows.iterrows()):
            # Extract filename from path (handles both "images/filename.png" and "filename.png")
            img_filename = Path(row["path"]).name
            img_path = images_dir / img_filename
            
            if not img_path.exists():
                # Try without images/ prefix if it's already in the path
                img_path = dataset_dir / row["path"]
            
            if img_path.exists():
                img = plt.imread(img_path)
                ax = axes[label, idx]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"Sum={label}", fontsize=8)
                ax.axis("off")
            else:
                # If image not found, show empty subplot
                axes[label, idx].axis("off")
        
        # Hide remaining empty subplots for this label
        for idx in range(len(label_rows), per_class):
            axes[label, idx].axis("off")

    fig.suptitle("Triple MNIST samples by sum (0-27)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Triple MNIST dataset utilities")
    parser.add_argument("--generate", action="store_true", help="Generate triple MNIST samples")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of triple MNIST samples")
    parser.add_argument("--gcs-upload", action="store_true", help="Upload triple_mnist dataset to GCS")
    parser.add_argument("--gcs-download", type=str, default=None, metavar="GCS_PATH", help="Download dataset from GCS (gs://bucket/path)")
    parser.add_argument("--count", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split (default: train)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization (default: .data/triple_mnist/{split}/overview.png)")
    args = parser.parse_args()
    
    data_root = get_data_root()
    mnist_dir = data_root / "mnist"
    triple_mnist_base = data_root / "triple_mnist"
    
    # Handle GCS download
    if args.gcs_download:
        split_dir = triple_mnist_base / args.split
        print(f"Downloading triple MNIST {args.split} from {args.gcs_download} to {split_dir}...", flush=True)
        download_from_gcs(args.gcs_download, split_dir)
        print(f"✓ Downloaded to {split_dir}", flush=True)
        return
    
    # Handle GCS upload
    if args.gcs_upload:
        if not triple_mnist_base.exists():
            print(f"Error: Triple MNIST dataset not found at {triple_mnist_base}. Generate it first.", flush=True)
            return
        
        config = load_config()
        gcs_path = f"gs://{config.bucket_name}/{config.data_path}"
        print(f"Uploading triple MNIST dataset from {triple_mnist_base} to {gcs_path}...", flush=True)
        upload_to_gcs(triple_mnist_base, gcs_path)
        return
    
    # Handle generation
    if args.generate:
        if not (mnist_dir / "labels.csv").exists():
            print(f"Error: MNIST dataset not found at {mnist_dir}. Run 'python -m expforge.data.mnist --load' first.", flush=True)
            return
        
        split_dir = triple_mnist_base / args.split
        print(f"Generating {args.count} triple MNIST samples for {args.split} split to {split_dir}...", flush=True)
        generate_triple_mnist(mnist_dir, args.count, split_dir, seed=args.seed, split=args.split)
        print(f"✓ Generated {args.count} samples to {split_dir}", flush=True)
    
    # Handle visualization
    if args.visualize:
        split_dir = triple_mnist_base / args.split
        if not (split_dir / "labels.csv").exists():
            print(f"Error: Triple MNIST {args.split} dataset not found at {split_dir}. Run with --generate first.", flush=True)
            return
        
        output_path = Path(args.output) if args.output else split_dir / "overview.png"
        print(f"Generating visualization to {output_path}...", flush=True)
        visualize_triple_mnist(split_dir, output_path)
        print(f"✓ Visualization saved to {output_path}", flush=True)
    
    if not args.generate and not args.visualize and not args.gcs_upload and not args.gcs_download:
        parser.print_help()


if __name__ == "__main__":
    main()

