from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, write_dataframe


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    if len(confidences) == 0:
        return 0.0, 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        gap = abs(bin_accuracy - bin_confidence)
        ece += gap * mask.mean()
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier_score_multiclass(probabilities: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    if len(labels) == 0:
        return 0.0
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def calibration_summary(probabilities: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[str, float]:
    confidences = probabilities.max(axis=1) if len(probabilities) else np.array([])
    predictions = probabilities.argmax(axis=1) if len(probabilities) else np.array([])
    correct = predictions == labels if len(labels) else np.array([])
    ece, mce = expected_calibration_error(confidences, correct)
    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier_score_multiclass(probabilities, labels, num_classes),
        "nll": negative_log_likelihood(probabilities, labels),
    }


def save_calibration_outputs(probabilities: np.ndarray, labels: np.ndarray, output_dir: str | Path, num_classes: int) -> Dict[str, float]:
    root = ensure_dir(output_dir)
    summary = calibration_summary(probabilities, labels, num_classes)
    write_dataframe(root / "ece_results.csv", pd.DataFrame([{"ece": summary["ece"], "mce": summary["mce"]}]))
    write_dataframe(root / "brier_score.csv", pd.DataFrame([{"brier_score": summary["brier_score"]}]))
    write_dataframe(root / "nll_results.csv", pd.DataFrame([{"nll": summary["nll"]}]))
    plot_reliability_diagram(probabilities, labels, root / "reliability_diagram.png")
    plot_reliability_diagram(probabilities, labels, root / "reliability_diagram_by_class.png", by_class=True, num_classes=num_classes)
    plot_scalar_by_epoch(root / "ece_by_epoch.png", "ECE by Epoch", "ECE")
    plot_scalar_by_epoch(root / "brier_score_by_epoch.png", "Brier Score by Epoch", "Brier Score")
    plot_reliability_diagram(probabilities, labels, root / "calibration_by_class.png", by_class=True, num_classes=num_classes)
    return summary


def plot_reliability_diagram(probabilities: np.ndarray, labels: np.ndarray, path: str | Path, by_class: bool = False, num_classes: int | None = None) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(labels) == 0:
        ax.text(0.5, 0.5, "No predictions available", ha="center", va="center")
    elif by_class and num_classes is not None:
        for class_id in range(num_classes):
            mask = labels == class_id
            if not np.any(mask):
                continue
            _plot_bin_curve(ax, probabilities[mask].max(axis=1), (probabilities[mask].argmax(axis=1) == labels[mask]).astype(float), label=f"class {class_id}")
        ax.legend(fontsize=7)
    else:
        _plot_bin_curve(ax, probabilities.max(axis=1), (probabilities.argmax(axis=1) == labels).astype(float), label="all")
        ax.legend()
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def _plot_bin_curve(ax, confidences: np.ndarray, correct: np.ndarray, label: str) -> None:
    bins = np.linspace(0.0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        accuracies.append(float(correct[mask].mean()) if np.any(mask) else np.nan)
    ax.plot(centers, accuracies, marker="o", label=label)


def plot_scalar_by_epoch(path: str | Path, title: str, ylabel: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Available when epoch calibration history is produced", ha="center", va="center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
