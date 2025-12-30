"""Visualization helpers for datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from expforge.datasets import load_manifest


def plot_sample_grid(dataset_dir: Path, output: Path, per_class: int = 4) -> None:
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
