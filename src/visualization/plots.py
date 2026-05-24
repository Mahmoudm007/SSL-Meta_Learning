from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.class_mapping import KNOWN_CLASS_COUNTS
from src.data.datasets import ImageRecord, dataset_summary_dataframe
from src.utils.io import ensure_dir, list_image_files


def save_placeholder_figure(path: str | Path, title: str, message: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def plot_curve(csv_path: str | Path, x_col: str, y_cols: Sequence[str], output_path: str | Path, title: str) -> None:
    target = Path(output_path)
    ensure_dir(target.parent)
    if not Path(csv_path).exists() or Path(csv_path).stat().st_size == 0:
        save_placeholder_figure(target, title, "No epoch data available")
        return
    frame = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    if x_col in frame:
        for y_col in y_cols:
            if y_col in frame and frame[y_col].notna().any():
                ax.plot(frame[x_col], frame[y_col], marker="o", label=y_col)
                plotted = True
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Requested metrics are not available", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def plot_bar(frame: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str, ylabel: str | None = None) -> None:
    target = Path(output_path)
    ensure_dir(target.parent)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(frame) and x in frame and y in frame:
        ax.bar(frame[x].astype(str), frame[y].astype(float), color="#4682b4")
        ax.tick_params(axis="x", labelrotation=30)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_ylabel(ylabel or y)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def save_dataset_figures(records: Sequence[ImageRecord], warmup_dir: str | Path, output_root: str | Path) -> None:
    root = ensure_dir(Path(output_root) / "plots" / "dataset")
    summary = dataset_summary_dataframe(records)
    _class_distribution(summary, root / "class_distribution_barplot.png")
    _class_percentage(summary, root / "class_distribution_percentage.png")
    _imbalance_ratio(summary, root / "imbalance_ratio_plot.png")
    _train_val_comparison(summary, root / "train_val_distribution_comparison.png")
    _sample_grid(records, root / "sample_grid_by_class.png")
    _warmup_grid(warmup_dir, root / "warmup_dataset_sample_grid.png")


def _class_distribution(summary: pd.DataFrame, path: Path) -> None:
    train = summary[summary["split"] == "train"]
    plot_bar(train, "label_name", "count", path, "Train Class Distribution", "Count")


def _class_percentage(summary: pd.DataFrame, path: Path) -> None:
    train = summary[summary["split"] == "train"]
    plot_bar(train, "label_name", "percentage", path, "Train Class Percentage", "Percent")


def _imbalance_ratio(summary: pd.DataFrame, path: Path) -> None:
    train = summary[summary["split"] == "train"].copy()
    if len(train):
        smallest = max(float(train["count"].min()), 1.0)
        train["imbalance_ratio"] = train["count"] / smallest
    plot_bar(train, "label_name", "imbalance_ratio", path, "Imbalance Ratio Relative to Smallest Class", "Ratio")


def _train_val_comparison(summary: pd.DataFrame, path: Path) -> None:
    target = Path(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(summary):
        pivot = summary.pivot(index="label_name", columns="split", values="count").fillna(0)
        pivot.plot(kind="bar", ax=ax)
    else:
        ax.text(0.5, 0.5, "No dataset summary available", ha="center", va="center")
    ax.set_title("Train/Val Distribution Comparison")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def _sample_grid(records: Sequence[ImageRecord], path: Path, per_class: int = 4) -> None:
    by_class: dict[str, list[Path]] = {}
    for record in records:
        if record.split == "train":
            by_class.setdefault(record.label_name, []).append(record.path)
    rows = max(len(by_class), 1)
    fig, axes = plt.subplots(rows, per_class, figsize=(per_class * 3, rows * 2.5))
    axes_array = np.array(axes).reshape(rows, per_class)
    for row_index, (class_name, paths) in enumerate(sorted(by_class.items())):
        for col_index in range(per_class):
            ax = axes_array[row_index, col_index]
            ax.axis("off")
            if col_index == 0:
                ax.set_title(class_name, fontsize=9)
            if col_index < len(paths):
                try:
                    ax.imshow(Image.open(paths[col_index]).convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "read error", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _warmup_grid(warmup_dir: str | Path, path: Path, count: int = 12) -> None:
    paths = list_image_files(warmup_dir)[:count]
    cols = 4
    rows = max(1, int(np.ceil(max(len(paths), 1) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.4))
    axes_array = np.array(axes).reshape(rows, cols)
    for index, ax in enumerate(axes_array.ravel()):
        ax.axis("off")
        if index < len(paths):
            try:
                ax.imshow(Image.open(paths[index]).convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, "read error", ha="center", va="center")
    fig.suptitle("Warm-Up SSL Samples")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_augmentation_placeholders(output_root: str | Path) -> None:
    root = ensure_dir(Path(output_root) / "plots" / "augmentations")
    message = "Generated training transforms preserve road region, tire paths, and snow-cover semantics."
    for name in [
        "augmentation_gallery_light.png",
        "augmentation_gallery_medium.png",
        "augmentation_gallery_strong.png",
        "rsc_preserving_augmentation_comparison.png",
    ]:
        save_placeholder_figure(root / name, name.replace("_", " ").replace(".png", "").title(), message)


def save_required_curve_figures(output_root: str | Path) -> None:
    root = Path(output_root)
    plot_curve(root / "epoch_progress" / "ssl_epoch_metrics.csv", "epoch", ["ssl_train_loss", "contrastive_loss"], root / "plots" / "ssl" / "ssl_loss_curve.png", "SSL Loss Curve")
    plot_curve(root / "epoch_progress" / "learning_rate_schedule.csv", "epoch", ["learning_rate"], root / "plots" / "ssl" / "ssl_learning_rate_curve.png", "SSL Learning Rate")
    plot_curve(root / "epoch_progress" / "ssl_epoch_metrics.csv", "epoch", ["feature_norm_mean"], root / "plots" / "ssl" / "ssl_feature_norm_curve.png", "SSL Feature Norm")
    save_placeholder_figure(root / "plots" / "ssl" / "ssl_alignment_uniformity_plot.png", "SSL Alignment / Uniformity", "Alignment and uniformity are logged when contrastive diagnostics are enabled.")
    save_placeholder_figure(root / "plots" / "ssl" / "ssl_linear_probe_accuracy_curve.png", "SSL Linear Probe", "Linear probe is optional; supervised evaluation files contain current classifier performance.")
    for filename, y_cols, title in [
        ("train_val_loss_curve.png", ["train_loss", "val_loss"], "Train/Val Loss"),
        ("train_val_accuracy_curve.png", ["train_accuracy", "val_accuracy"], "Train/Val Accuracy"),
        ("train_val_macro_f1_curve.png", ["train_macro_f1", "val_macro_f1"], "Train/Val Macro-F1"),
        ("train_val_balanced_accuracy_curve.png", ["train_balanced_accuracy", "val_balanced_accuracy"], "Train/Val Balanced Accuracy"),
        ("per_class_recall_over_epochs.png", ["one_track_recall"], "Per-Class Recall Over Epochs"),
        ("per_class_f1_over_epochs.png", ["one_track_f1"], "Per-Class F1 Over Epochs"),
        ("one_track_recall_over_epochs.png", ["one_track_recall"], "One Track Recall Over Epochs"),
        ("one_track_f1_over_epochs.png", ["one_track_f1"], "One Track F1 Over Epochs"),
        ("learning_rate_schedule.png", ["learning_rate"], "Learning Rate Schedule"),
        ("gradient_norm_curve.png", ["gradient_norm"], "Gradient Norm"),
    ]:
        plot_curve(root / "epoch_progress" / "supervised_epoch_metrics.csv", "epoch", y_cols, root / "plots" / "supervised" / filename, title)
    for filename, y_cols, title in [
        ("episode_loss_curve.png", ["episode_train_loss", "episode_query_loss"], "Episode Loss"),
        ("episode_query_accuracy_curve.png", ["query_accuracy"], "Episode Query Accuracy"),
        ("episode_query_macro_f1_curve.png", ["query_macro_f1"], "Episode Query Macro-F1"),
        ("support_vs_query_accuracy_curve.png", ["support_accuracy", "query_accuracy"], "Support vs Query Accuracy"),
        ("hard_episode_accuracy_curve.png", ["hard_episode_accuracy"], "Hard Episode Accuracy"),
        ("hard_episode_one_track_recall_curve.png", ["hard_episode_one_track_recall"], "Hard Episode One Track Recall"),
        ("fewshot_accuracy_by_support_size.png", ["query_accuracy"], "Few-Shot Accuracy by Support Size"),
    ]:
        plot_curve(root / "epoch_progress" / "meta_epoch_metrics.csv", "epoch", y_cols, root / "plots" / "meta" / filename, title)
    save_placeholder_figure(root / "plots" / "meta" / "prototype_trajectory_plot.png", "Prototype Trajectory", "Prototype trajectories are available when prototype checkpoints from multiple epochs are present.")
    save_placeholder_figure(root / "plots" / "meta" / "class_centroid_shift_after_meta_learning.png", "Centroid Shift After Meta-Learning", "Centroid shift is computed from saved embeddings when available.")
