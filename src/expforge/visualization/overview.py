"""Visualization helpers for datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expforge.datasets import load_manifest


COLORS = ["#0b84a5", "#f6c85f", "#6f4e7c", "#9dd866", "#ca472f", "#ffa056", "#8dddd0", "#828282", "#b0cbe2", "#b3b3b3"]


def plot_sample_grid(dataset_dir: Path, output: Path, per_class: int = 8) -> None:
    manifest = load_manifest(dataset_dir)
    images_dir = dataset_dir / "images"

    fig, axes = plt.subplots(nrows=10, ncols=per_class, figsize=(per_class * 1.2, 12))
    for digit in range(10):
        digit_rows = manifest[manifest["label"] == str(digit)].head(per_class)
        for idx, (_, row) in enumerate(digit_rows.iterrows()):
            img = plt.imread(images_dir / row["path"])
            ax = axes[digit, idx]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        for idx in range(len(digit_rows), per_class):
            axes[digit, idx].axis("off")

    fig.suptitle("Sample grid per class", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close(fig)


def plot_label_distribution(manifest: pd.DataFrame) -> None:
    counts = manifest["label"].value_counts().sort_index()
    counts.plot(kind="bar", color=COLORS[: len(counts)], title="Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
