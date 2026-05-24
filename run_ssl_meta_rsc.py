from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import torch

from src.data.datasets import scan_rsc_dataset
from src.experiments import run_single_experiment
from src.utils.io import EXPERIMENT_KEYS, ensure_dir
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL + Meta-Learning for winter Road Surface Condition classification")
    parser.add_argument("--models", nargs="+", choices=["convnext", "dino"], default=None)
    parser.add_argument("--model", choices=["convnext", "dino"], default=None)
    parser.add_argument("--experiments", nargs="+", default=["all"], help="Use 'all' or a list of experiment keys")
    parser.add_argument("--experiment", choices=EXPERIMENT_KEYS, default=None)
    parser.add_argument("--support_per_class", type=int, default=60)
    parser.add_argument("--query_per_class", type=int, default=60)
    parser.add_argument("--warmup_dir", default="Warm-up Dataset")
    parser.add_argument("--data_dir", default="Dataset_classes/1 Defined")
    parser.add_argument("--output_dir", default="Output")
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
    parser.add_argument("--resume", default=None)
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
    parser.add_argument("--pseudo_novel_class", default="3 One Track - Partly")
    parser.add_argument("--fewshot_values", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--augmentation_strength", default="medium", choices=["light", "medium", "strong"])
    parser.add_argument("--ssl_temperature", type=float, default=0.2)
    parser.add_argument("--triplet_margin", type=float, default=0.2)
    parser.add_argument("--hard_episode_probability", type=float, default=0.5)
    parser.add_argument("--generate_gradcam", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--max_gradcam_images", type=int, default=0, help="0 means all validation/query images")
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


def resolve_experiments(args: argparse.Namespace) -> List[str]:
    if args.experiment:
        return [args.experiment]
    requested = args.experiments or ["all"]
    if "all" in requested:
        return list(EXPERIMENT_KEYS)
    invalid = [experiment for experiment in requested if experiment not in EXPERIMENT_KEYS]
    if invalid:
        raise ValueError(f"Unsupported experiment(s): {invalid}. Valid keys: {EXPERIMENT_KEYS}")
    return requested


def resolve_device(requested: str, logger: logging.Logger) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    root_logger = setup_logger("ssl_meta_rsc", Path(args.output_dir) / "logs", "train.log")
    device = resolve_device(args.device, root_logger)
    root_logger.info("Using device: %s", device)
    root_logger.info("Validating data directories")
    records = scan_rsc_dataset(args.data_dir)
    experiments = resolve_experiments(args)
    models = resolve_models(args)
    root_logger.info("Running experiments=%s models=%s", experiments, models)
    for experiment in experiments:
        for model_key in models:
            root_logger.info("Starting experiment=%s model=%s", experiment, model_key)
            run_single_experiment(experiment, model_key, records, args, device, root_logger)
            root_logger.info("Finished experiment=%s model=%s", experiment, model_key)


if __name__ == "__main__":
    main()
