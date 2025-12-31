"""Dataset I/O utilities for Experiment Forge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
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

