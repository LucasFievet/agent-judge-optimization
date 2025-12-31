"""MNIST dataset loading and visualization."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from expforge.data.io import DatasetRecord, ensure_dataset_dirs, load_manifest, save_manifest
from expforge.data.util import get_data_root


def bootstrap_mnist(output_dir: Path, limit: int = 2000, test_split: float = 0.2) -> List[DatasetRecord]:
    """Download a compact MNIST subset and store it in the unified dataset format."""

    images_dir = ensure_dataset_dirs(output_dir)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data[:limit]
    y = mnist.target[:limit]

    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )

    records: List[DatasetRecord] = []
    for split_name, data, labels in [("train", train_data, train_labels), ("test", test_data, test_labels)]:
        for row, label in zip(data, labels):
            uid = uuid.uuid4().hex
            img = Image.fromarray(row.reshape(28, 28).astype(np.uint8), mode="L")
            rel_path = Path("images") / f"{uid}.png"
            img.save(images_dir / rel_path.name)
            records.append(DatasetRecord(path=str(rel_path), label=str(label), split=split_name))

    return save_manifest(records, output_dir)


def visualize_mnist(dataset_dir: Path, output: Path, per_class: int = 4) -> None:
    """Plot a grid of MNIST samples (digits 0-9)."""
    manifest = load_manifest(dataset_dir)
    images_dir = dataset_dir / "images"

    # Convert labels to numeric for consistent comparison
    manifest["label"] = pd.to_numeric(manifest["label"], errors="coerce")

    # For MNIST, labels range from 0 to 9
    num_classes = 10
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
                ax.set_title(f"Digit={label}", fontsize=8)
                ax.axis("off")
            else:
                # If image not found, show empty subplot
                axes[label, idx].axis("off")
        
        # Hide remaining empty subplots for this label
        for idx in range(len(label_rows), per_class):
            axes[label, idx].axis("off")

    fig.suptitle("MNIST samples by digit (0-9)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MNIST dataset utilities")
    parser.add_argument("--load", action="store_true", help="Download and load MNIST dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of MNIST samples")
    parser.add_argument("--limit", type=int, default=2000, help="Number of samples to download (default: 2000)")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization (default: .data/mnist/overview.png)")
    args = parser.parse_args()
    
    data_root = get_data_root()
    mnist_dir = data_root / "mnist"
    
    if args.load:
        print(f"Loading MNIST dataset to {mnist_dir}...", flush=True)
        bootstrap_mnist(mnist_dir, limit=args.limit)
        print(f"✓ MNIST dataset loaded to {mnist_dir}", flush=True)
    
    if args.visualize:
        if not (mnist_dir / "labels.csv").exists():
            print(f"Error: MNIST dataset not found at {mnist_dir}. Run with --load first.", flush=True)
            return
        
        output_path = Path(args.output) if args.output else mnist_dir / "overview.png"
        print(f"Generating visualization to {output_path}...", flush=True)
        visualize_mnist(mnist_dir, output_path)
        print(f"✓ Visualization saved to {output_path}", flush=True)
    
    if not args.load and not args.visualize:
        parser.print_help()


if __name__ == "__main__":
    main()

