from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

EXPERIMENT_KEYS = [
    "SSL_Prototypical",
    "SSL_Hard_Prototypical",
    "SSL_ClassBalanced_FineTune",
    "SSL_MAML_ANIL",
    "SSL_MetricLearning",
    "SSL_Hybrid_FineTune_Episodic",
    "SSL_Simulated_FutureClass",
]

OUTPUT_SUBDIRS = [
    "checkpoints",
    "logs",
    "metrics",
    "plots",
    "plots/dataset",
    "plots/augmentations",
    "plots/ssl",
    "plots/supervised",
    "plots/meta",
    "plots/errors",
    "plots/model_comparison",
    "confusion_matrices",
    "calibration",
    "confidence",
    "embeddings",
    "gradcam",
    "gradcam/all_validation",
    "gradcam/all_query",
    "gradcam/correct",
    "gradcam/incorrect",
    "gradcam/by_true_class",
    "gradcam/by_predicted_class",
    "gradcam/one_track_false_negatives",
    "gradcam/blackice",
    "predictions",
    "configs",
    "epoch_progress",
    "ablations",
    "reports",
]

REQUIRED_LOG_FILES = [
    "train.log",
    "ssl.log",
    "supervised.log",
    "meta.log",
    "evaluation.log",
    "errors.log",
]


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def create_experiment_tree(root: str | Path) -> Path:
    root_path = ensure_dir(root)
    for subdir in OUTPUT_SUBDIRS:
        ensure_dir(root_path / subdir)
    for log_name in REQUIRED_LOG_FILES:
        log_path = root_path / "logs" / log_name
        if not log_path.exists():
            log_path.write_text("", encoding="utf-8")
    return root_path


def list_image_files(root: str | Path, recursive: bool = True) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    iterator = root_path.rglob("*") if recursive else root_path.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_yaml(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def write_dataframe(path: str | Path, dataframe: pd.DataFrame) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    dataframe.to_csv(target, index=False)


def append_dataframe(path: str | Path, row: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    frame = pd.DataFrame([row])
    frame.to_csv(target, mode="a", header=not target.exists(), index=False)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "not_available"


def save_run_config(output_root: str | Path, args_dict: Dict[str, Any], config: Dict[str, Any], class_mapping: Dict[str, Any]) -> None:
    root = Path(output_root)
    write_json(root / "configs" / "args.json", args_dict)
    write_yaml(root / "configs" / "config.yaml", config)
    write_json(root / "configs" / "class_mapping.json", class_mapping)
    write_text(root / "configs" / "git_commit.txt", get_git_commit() + "\n")


def initialize_epoch_csvs(output_root: str | Path) -> None:
    root = Path(output_root) / "epoch_progress"
    ensure_dir(root)
    csv_headers = {
        "ssl_epoch_metrics.csv": [
            "epoch",
            "ssl_train_loss",
            "ssl_val_loss",
            "contrastive_loss",
            "reconstruction_loss",
            "feature_norm_mean",
            "feature_norm_std",
            "embedding_alignment",
            "embedding_uniformity",
            "learning_rate",
            "epoch_time_seconds",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
        ],
        "supervised_epoch_metrics.csv": [
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "train_balanced_accuracy",
            "val_balanced_accuracy",
            "train_macro_f1",
            "val_macro_f1",
            "train_weighted_f1",
            "val_weighted_f1",
            "one_track_precision",
            "one_track_recall",
            "one_track_f1",
            "one_track_false_negative_rate",
            "top2_accuracy",
            "ece",
            "brier_score",
            "nll",
            "learning_rate",
            "gradient_norm",
            "epoch_time_seconds",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
        ],
        "meta_epoch_metrics.csv": [
            "epoch",
            "episode_train_loss",
            "episode_query_loss",
            "support_accuracy",
            "query_accuracy",
            "query_balanced_accuracy",
            "query_macro_f1",
            "query_weighted_f1",
            "one_track_query_recall",
            "one_track_query_f1",
            "hard_episode_accuracy",
            "hard_episode_loss",
            "hard_episode_one_track_recall",
            "prototype_mean_distance",
            "prototype_min_distance",
            "prototype_max_distance",
            "learning_rate",
            "epoch_time_seconds",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
        ],
        "maml_anil_epoch_metrics.csv": [
            "epoch",
            "outer_loss",
            "mean_inner_loss_step_1",
            "mean_inner_loss_step_2",
            "mean_inner_loss_step_3",
            "mean_inner_loss_step_4",
            "mean_inner_loss_step_5",
            "pre_adaptation_query_accuracy",
            "post_adaptation_query_accuracy",
            "adaptation_gain",
            "pre_adaptation_macro_f1",
            "post_adaptation_macro_f1",
            "one_track_pre_adaptation_recall",
            "one_track_post_adaptation_recall",
            "learning_rate_inner",
            "learning_rate_outer",
            "epoch_time_seconds",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
        ],
        "learning_rate_schedule.csv": ["epoch", "phase", "learning_rate"],
        "gpu_memory_log.csv": ["epoch", "phase", "allocated_mb", "reserved_mb"],
        "epoch_time_log.csv": ["epoch", "phase", "epoch_time_seconds"],
    }
    for filename, headers in csv_headers.items():
        path = root / filename
        if not path.exists():
            pd.DataFrame(columns=headers).to_csv(path, index=False)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def path_as_posix(path: str | Path) -> str:
    return Path(path).as_posix()
