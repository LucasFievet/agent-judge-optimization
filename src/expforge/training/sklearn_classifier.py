"""Baseline training routine using scikit-learn."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from expforge.datasets import load_images, load_manifest


def _stack_datasets(dataset_dirs: Iterable[Path]):
    images = []
    labels = []

    for directory in dataset_dirs:
        manifest = load_manifest(directory)
        images.append(load_images(manifest, directory))
        labels.append(manifest["label"].astype(str).to_numpy())

    X = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def train_logistic_regression(
    dataset_dirs: List[Path],
    model_out: Path,
    test_ratio: float = 0.2,
    max_iter: int = 200,
    c_value: float = 1.0,
) -> Dict[str, float]:
    X, y = _stack_datasets(dataset_dirs)
    X = X.reshape(len(X), -1) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1, solver="lbfgs", multi_class="multinomial", C=c_value)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)

    return {
        "accuracy": float(accuracy),
        "model_path": str(model_out),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
