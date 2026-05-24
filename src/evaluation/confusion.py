from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.class_mapping import ONE_TRACK_CLASS_ID
from src.utils.io import ensure_dir


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], path: str | Path, title: str, normalize: bool = False) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(8, 7))
    display = matrix.astype(float)
    if normalize:
        display = display / np.clip(display.sum(axis=1, keepdims=True), 1, None)
    image = ax.imshow(display, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = display[row, col]
            text = f"{value:.2f}" if normalize else str(int(value))
            ax.text(col, row, text, ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def save_confusion_outputs(matrix: np.ndarray, class_names: Sequence[str], output_dir: str | Path) -> None:
    root = ensure_dir(output_dir)
    plot_confusion_matrix(matrix, class_names, root / "confusion_matrix_counts.png", "Confusion Matrix", normalize=False)
    plot_confusion_matrix(matrix, class_names, root / "confusion_matrix_normalized.png", "Normalized Confusion Matrix", normalize=True)
    partly_ids = [index for index, name in enumerate(class_names) if "Partly" in name or name == "4 Fully"]
    if partly_ids:
        submatrix = matrix[np.ix_(partly_ids, partly_ids)]
        subnames = [class_names[index] for index in partly_ids]
        plot_confusion_matrix(submatrix, subnames, root / "partly_classes_confusion_matrix.png", "Partly/Fully Confusion", normalize=False)
    if ONE_TRACK_CLASS_ID < len(class_names):
        one_track_matrix = np.zeros_like(matrix)
        one_track_matrix[ONE_TRACK_CLASS_ID, :] = matrix[ONE_TRACK_CLASS_ID, :]
        one_track_matrix[:, ONE_TRACK_CLASS_ID] += matrix[:, ONE_TRACK_CLASS_ID]
        plot_confusion_matrix(one_track_matrix, class_names, root / "one_track_error_matrix.png", "One Track Error Matrix", normalize=False)
