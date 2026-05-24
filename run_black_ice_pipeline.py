from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.augmentations.rsc_augmentations import get_eval_transform, get_train_transform
from src.data.blackice_dataset import blackice_records, validate_blackice_dir
from src.data.class_mapping import default_class_mapping
from src.data.datasets import RSCImageDataset, scan_rsc_dataset, save_dataset_summaries, split_records
from src.evaluation.evaluate import evaluate_and_save_all
from src.evaluation.metrics import classification_metrics
from src.experiments import _fewshot_prototype_eval
from src.meta.hard_episodes import train_hard_prototypical_network
from src.meta.maml_anil import train_maml_anil
from src.meta.metric_learning import train_metric_learning
from src.meta.prototypical import install_prototype_classifier, train_prototypical_network
from src.models.backbones import build_model
from src.ssl.ssl_pretrain import train_ssl_encoder
from src.supervised.finetune import train_supervised_classifier
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.config import namespace_to_dict
from src.utils.io import create_experiment_tree, ensure_dir, initialize_epoch_csvs, save_run_config, write_dataframe
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import set_seed
from src.visualization.plots import save_augmentation_placeholders, save_dataset_figures, save_placeholder_figure


SUPPORTED_BLACKICE_EXPERIMENTS = [
    "SSL_Prototypical",
    "SSL_Hard_Prototypical",
    "SSL_ClassBalanced_FineTune",
    "SSL_MAML_ANIL",
    "SSL_MetricLearning",
    "SSL_Hybrid_FineTune_Episodic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future black-ice few-shot adaptation pipeline")
    parser.add_argument("--blackice_mode", choices=["adapt_existing", "train_from_start"], default="adapt_existing")
    parser.add_argument("--experiment", choices=SUPPORTED_BLACKICE_EXPERIMENTS, default="SSL_Prototypical")
    parser.add_argument("--models", nargs="+", choices=["convnext", "dino"], default=None)
    parser.add_argument("--model", choices=["convnext", "dino"], default=None)
    parser.add_argument("--blackice_dir", default="Black-ice")
    parser.add_argument("--blackice_shots", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for adapt_existing mode")
    parser.add_argument("--warmup_dir", default="Warm-up Dataset")
    parser.add_argument("--data_dir", default="Dataset_classes/1 Defined")
    parser.add_argument("--output_dir", default="Output")
    parser.add_argument("--support_per_class", type=int, default=60)
    parser.add_argument("--query_per_class", type=int, default=60)
    parser.add_argument("--epochs_ssl", type=int, default=1)
    parser.add_argument("--epochs_finetune", type=int, default=1)
    parser.add_argument("--epochs_meta", type=int, default=1)
    parser.add_argument("--episodes_per_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--outer_lr", type=float, default=1e-4)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_ssl_if_checkpoint_exists", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--convnext_name", default="convnext_base_in22k", choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_base_in22k", "convnext_large"])
    parser.add_argument("--dino_name", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--loss", default="weighted_ce", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--sampler", default="balanced", choices=["standard", "balanced"])
    parser.add_argument("--meta_algorithm", default="anil", choices=["anil", "maml"])
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--adapt_scope", default="head", choices=["head", "last_block", "full"])
    parser.add_argument("--metric_loss", default="supcon", choices=["supcon", "triplet"])
    parser.add_argument("--hard_negative_mining", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--prototype_distance", default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--augmentation_strength", default="medium", choices=["light", "medium", "strong"])
    parser.add_argument("--ssl_temperature", type=float, default=0.2)
    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--hard_episode_probability", type=float, default=0.5)
    parser.add_argument("--generate_gradcam", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--max_gradcam_images", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_warmup_samples", type=int, default=0)
    parser.add_argument("--train_eval_batches", type=int, default=None)
    return parser.parse_args()


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return args.models
    if args.model:
        return [args.model]
    return ["convnext", "dino"]


def resolve_device(requested: str, logger: logging.Logger) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_blackice_dir(args.blackice_dir)
    ensure_dir(Path(args.output_dir) / "BlackIce")
    logger = setup_logger("blackice_rsc", Path(args.output_dir) / "BlackIce" / "logs", "train.log")
    device = resolve_device(args.device, logger)
    base_records = scan_rsc_dataset(args.data_dir)
    black_records = blackice_records(args.blackice_dir, split="blackice")
    for model_key in resolve_models(args):
        output_root = Path(args.output_dir) / "BlackIce" / args.experiment / model_key
        create_experiment_tree(output_root)
        initialize_epoch_csvs(output_root)
        class_mapping6 = default_class_mapping(include_black_ice=True)
        save_run_config(output_root, namespace_to_dict(args), {"experiment": args.experiment, "model": model_key, "blackice_mode": args.blackice_mode}, class_mapping6.to_json_dict())
        save_dataset_summaries(list(base_records) + black_records, output_root)
        save_dataset_figures(base_records, args.warmup_dir, output_root)
        save_augmentation_placeholders(output_root)
        model = build_model(model_key, num_classes=5, convnext_name=args.convnext_name, dino_name=args.dino_name, logger=logger).to(device)
        if args.blackice_mode == "adapt_existing":
            checkpoint = args.checkpoint or _default_checkpoint(args.output_dir, args.experiment, model_key)
            if not checkpoint.exists():
                raise FileNotFoundError(f"No checkpoint found for adapt_existing mode: {checkpoint}")
            model.replace_classifier(6)
            load_checkpoint(checkpoint, model, map_location=device)
            logger.info("Loaded existing checkpoint for black-ice adaptation: %s", checkpoint)
        else:
            _train_selected_five_class_pipeline(model, base_records, args, output_root, device, logger)
            model.replace_classifier(6)
        train_dataset, val_dataset = _blackice_datasets(base_records, black_records, args)
        fewshot_eval = _run_blackice_fewshot(model, train_dataset, val_dataset, class_mapping6, output_root, args, device)
        install_prototype_classifier(model, train_dataset, class_mapping6, output_root, args, device)
        evaluate_and_save_all(model, val_dataset, class_mapping6, output_root, args, device, logger, args.experiment, model_key, eval_result=fewshot_eval, include_black_ice=True)
        _save_blackice_specific_outputs(fewshot_eval, class_mapping6, output_root)
        save_checkpoint(Path(output_root) / "checkpoints" / "blackice_adapted_final.pt", model, phase="blackice_adaptation")


def _default_checkpoint(output_dir: str, experiment: str, model_key: str) -> Path:
    candidate = Path(output_dir) / experiment / model_key / "checkpoints" / "best_model.pt"
    if candidate.exists():
        return candidate
    return Path(output_dir) / experiment / model_key / "checkpoints" / "final_model.pt"


def _train_selected_five_class_pipeline(model, base_records, args, output_root: Path, device: torch.device, logger: logging.Logger) -> None:
    train_records = split_records(base_records, "train")[: args.max_train_samples or None]
    val_records = split_records(base_records, "val")[: args.max_val_samples or None]
    class_mapping5 = default_class_mapping()
    train_dataset = RSCImageDataset(train_records, transform=get_train_transform(args.image_size, args.augmentation_strength))
    train_eval_dataset = RSCImageDataset(train_records, transform=get_eval_transform(args.image_size))
    val_dataset = RSCImageDataset(val_records, transform=get_eval_transform(args.image_size))
    train_ssl_encoder(model, args.warmup_dir, output_root, args, device, logger)
    if args.experiment == "SSL_Prototypical":
        train_prototypical_network(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger)
        install_prototype_classifier(model, train_eval_dataset, class_mapping5, output_root, args, device)
    elif args.experiment == "SSL_Hard_Prototypical":
        train_hard_prototypical_network(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger)
        install_prototype_classifier(model, train_eval_dataset, class_mapping5, output_root, args, device)
    elif args.experiment == "SSL_ClassBalanced_FineTune":
        train_supervised_classifier(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger)
    elif args.experiment == "SSL_MAML_ANIL":
        train_maml_anil(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger)
    elif args.experiment == "SSL_MetricLearning":
        train_metric_learning(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger)
    elif args.experiment == "SSL_Hybrid_FineTune_Episodic":
        train_supervised_classifier(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger, phase_name="blackice_base_supervised")
        train_prototypical_network(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger, phase_name="blackice_base_episodic")
        train_hard_prototypical_network(model, train_dataset, val_dataset, class_mapping5, output_root, args, device, logger)
        install_prototype_classifier(model, train_eval_dataset, class_mapping5, output_root, args, device)


def _blackice_datasets(base_records, black_records, args):
    train_records = split_records(base_records, "train")[: args.max_train_samples or None] + black_records
    val_records = split_records(base_records, "val")[: args.max_val_samples or None] + black_records
    train_dataset = RSCImageDataset(train_records, transform=get_eval_transform(args.image_size))
    val_dataset = RSCImageDataset(val_records, transform=get_eval_transform(args.image_size))
    return train_dataset, val_dataset


def _run_blackice_fewshot(model, train_dataset, val_dataset, class_mapping6, output_root: Path, args, device: torch.device) -> dict:
    fewshot_rows = []
    final_eval = None
    for shot_count in args.blackice_shots:
        probabilities, labels, paths = _fewshot_prototype_eval(model, train_dataset, val_dataset, class_mapping6, 5, int(shot_count), args.prototype_distance, device)
        metrics = classification_metrics(labels, probabilities, class_mapping6)
        per_class = metrics["per_class"]
        blackice_row = per_class[per_class["label_id"] == 5].iloc[0].to_dict() if len(per_class[per_class["label_id"] == 5]) else {}
        fewshot_rows.append(
            {
                "shot_count": int(shot_count),
                "six_class_accuracy": metrics["summary"].get("overall_accuracy", 0.0),
                "six_class_macro_f1": metrics["summary"].get("macro_f1", 0.0),
                "six_class_balanced_accuracy": metrics["summary"].get("balanced_accuracy", 0.0),
                "blackice_precision": blackice_row.get("precision", 0.0),
                "blackice_recall": blackice_row.get("recall", 0.0),
                "blackice_f1": blackice_row.get("f1", 0.0),
            }
        )
        final_eval = {"probabilities": probabilities, "labels": labels, "paths": paths, "logits": np.log(np.clip(probabilities, 1e-12, 1.0)), "loss": 0.0}
    write_dataframe(output_root / "metrics" / "fewshot_results.csv", pd.DataFrame(fewshot_rows))
    return final_eval or {"probabilities": np.empty((0, 0)), "labels": np.array([], dtype=int), "paths": [], "logits": np.empty((0, 0)), "loss": 0.0}


def _save_blackice_specific_outputs(eval_result: dict, class_mapping6, output_root: Path) -> None:
    metrics = classification_metrics(eval_result["labels"], eval_result["probabilities"], class_mapping6)
    per_class = metrics["per_class"]
    blackice = per_class[per_class["label_id"] == 5].copy()
    if len(blackice):
        total = int(blackice.iloc[0]["support"])
        recall = float(blackice.iloc[0]["recall"])
        blackice["blackice_false_negative_rate"] = 1.0 - recall if total else 0.0
    write_dataframe(output_root / "metrics" / "blackice_metrics.csv", blackice)
    _blackice_plots(output_root)


def _blackice_plots(output_root: Path) -> None:
    plots_dir = ensure_dir(output_root / "plots")
    fewshot_path = output_root / "metrics" / "fewshot_results.csv"
    frame = pd.read_csv(fewshot_path) if fewshot_path.exists() else pd.DataFrame()
    for filename, metric, title in [
        ("blackice_fewshot_accuracy_curve.png", "six_class_accuracy", "Black-Ice Few-Shot Accuracy"),
        ("blackice_fewshot_f1_curve.png", "blackice_f1", "Black-Ice Few-Shot F1"),
        ("blackice_recall_by_shot_count.png", "blackice_recall", "Black-Ice Recall by Shot Count"),
        ("blackice_adaptation_summary.png", "six_class_macro_f1", "Black-Ice Adaptation Summary"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        if len(frame) and metric in frame:
            ax.plot(frame["shot_count"], frame[metric], marker="o")
            ax.set_xlabel("Black-ice support shots")
            ax.set_ylabel(metric)
        else:
            ax.text(0.5, 0.5, "No few-shot results available", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180)
        plt.close(fig)
    for filename in [
        "six_class_confusion_matrix.png",
        "blackice_confusion_row.png",
        "blackice_gradcam_gallery.png",
        "blackice_embedding_umap.png",
        "blackice_confidence_histogram.png",
        "blackice_false_negative_gallery.png",
    ]:
        save_placeholder_figure(plots_dir / filename, filename.replace("_", " ").replace(".png", "").title(), "See confusion_matrices/, gradcam/, embeddings/, and confidence/ for generated source artifacts.")


if __name__ == "__main__":
    main()
