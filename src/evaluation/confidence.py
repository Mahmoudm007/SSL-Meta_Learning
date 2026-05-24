from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir, write_dataframe


def save_confidence_outputs(predictions: pd.DataFrame, output_dir: str | Path, class_names: Sequence[str]) -> None:
    root = ensure_dir(output_dir)
    write_dataframe(root / "confidence_predictions.csv", predictions)
    _hist(predictions, root / "confidence_histogram_all.png", "Confidence Histogram", "confidence")
    _hist(predictions, root / "confidence_margin_histogram.png", "Top-1/Top-2 Confidence Margin", "top1_top2_margin")
    _hist(predictions, root / "top1_top2_margin_plot.png", "Top-1/Top-2 Margin", "top1_top2_margin")
    _hist(predictions, root / "entropy_by_class.png", "Entropy Distribution", "entropy")
    _correct_wrong_hist(predictions, root / "correct_vs_wrong_confidence.png")
    _correct_wrong_hist(predictions, root / "correct_vs_wrong_confidence_histogram.png")
    _box_by_class(predictions, root / "confidence_by_class.png")
    _box_by_class(predictions, root / "confidence_by_class_boxplot.png")
    _confidence_vs_accuracy(predictions, root / "confidence_vs_accuracy_curve.png")


def _hist(predictions: pd.DataFrame, path: Path, title: str, column: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if column in predictions and len(predictions):
        ax.hist(predictions[column].astype(float), bins=25, color="#4169e1", alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No prediction data", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _correct_wrong_hist(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(predictions):
        correct = predictions[predictions["correct"].astype(bool)]["confidence"].astype(float)
        wrong = predictions[~predictions["correct"].astype(bool)]["confidence"].astype(float)
        ax.hist(correct, bins=20, alpha=0.7, label="correct")
        ax.hist(wrong, bins=20, alpha=0.7, label="wrong")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No prediction data", ha="center", va="center")
    ax.set_title("Correct vs Wrong Confidence")
    ax.set_xlabel("confidence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _box_by_class(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(predictions):
        predictions.boxplot(column="confidence", by="true_label_name", rot=30, ax=ax)
        fig.suptitle("")
    else:
        ax.text(0.5, 0.5, "No prediction data", ha="center", va="center")
    ax.set_title("Confidence by True Class")
    ax.set_ylabel("confidence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _confidence_vs_accuracy(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(predictions):
        bins = pd.cut(predictions["confidence"].astype(float), bins=10, include_lowest=True)
        grouped = predictions.assign(bin=bins).groupby("bin", observed=False)["correct"].mean()
        ax.plot(range(len(grouped)), grouped.values, marker="o")
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels([str(interval) for interval in grouped.index], rotation=45, ha="right", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No prediction data", ha="center", va="center")
    ax.set_title("Confidence vs Accuracy")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
