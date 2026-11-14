"""Metrics, the persistence baseline and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
)

from src.config import CLASS_NAMES

LABELS = list(range(len(CLASS_NAMES)))


def persistence_predict(X: pd.DataFrame) -> np.ndarray:
    """Baseline: drought class next week = drought class this week."""
    return X["score"].round().clip(0, 5).astype(int).to_numpy()


def compute_metrics(y_true, y_pred) -> dict:
    """Macro F1 treats every drought class equally, which matters because most
    weeks have no drought. MAE measures how many classes off a prediction is."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABELS, display_labels=CLASS_NAMES,
        normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
