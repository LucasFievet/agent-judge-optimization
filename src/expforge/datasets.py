"""Dataset utilities for Experiment Forge."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image


@dataclass
class DatasetRecord:
    """Single row in the dataset manifest."""

    path: str
    label: str
    split: str


def ensure_dataset_dirs(root: Path) -> Path:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def save_manifest(records: Iterable[DatasetRecord], root: Path) -> List[DatasetRecord]:
    df = pd.DataFrame([r.__dict__ for r in records])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "labels.csv"
    df.to_csv(manifest_path, index=False)
    return list(records)


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


def load_manifest(root: Path) -> pd.DataFrame:
    manifest_path = root / "labels.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No labels.csv found in {root}")
    return pd.read_csv(manifest_path)


def load_images(manifest: pd.DataFrame, root: Path) -> np.ndarray:
    images_dir = root / "images"
    imgs = []
    for rel_path in manifest["path"]:
        # Extract just the filename from the path (which may include "images/" prefix)
        img_path = images_dir / Path(rel_path).name
        img = Image.open(img_path).convert("L")
        imgs.append(np.array(img))
    return np.stack(imgs, axis=0)
