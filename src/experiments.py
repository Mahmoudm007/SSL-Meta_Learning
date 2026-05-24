from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from src.augmentations.rsc_augmentations import get_eval_transform, get_train_transform
from src.data.class_mapping import ClassMapping, default_class_mapping, parse_class_name
from src.data.datasets import RSCImageDataset, ImageRecord, save_dataset_summaries, split_records
from src.evaluation.evaluate import evaluate_and_save_all
from src.evaluation.metrics import classification_metrics, prediction_dataframe
from src.meta.hard_episodes import train_hard_prototypical_network
from src.meta.maml_anil import train_maml_anil
from src.meta.metric_learning import train_metric_learning
from src.meta.prototypical import install_prototype_classifier, train_prototypical_network
from src.models.backbones import build_model
from src.ssl.ssl_pretrain import train_ssl_encoder
from src.supervised.finetune import train_supervised_classifier
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.config import namespace_to_dict
from src.utils.io import create_experiment_tree, initialize_epoch_csvs, save_run_config, write_dataframe
from src.visualization.plots import save_augmentation_placeholders, save_dataset_figures


def make_datasets(records: Sequence[ImageRecord], args, include_classes: Sequence[int] | None = None, exclude_classes: Sequence[int] | None = None):
    train_records = _limit_records(split_records(records, "train"), args.max_train_samples)
    val_records = _limit_records(split_records(records, "val"), args.max_val_samples)
    train_transform = get_train_transform(args.image_size, args.augmentation_strength)
    eval_transform = get_eval_transform(args.image_size)
    train_dataset = RSCImageDataset(train_records, transform=train_transform, include_classes=include_classes, exclude_classes=exclude_classes)
    train_eval_dataset = RSCImageDataset(train_records, transform=eval_transform, include_classes=include_classes, exclude_classes=exclude_classes)
    val_dataset = RSCImageDataset(val_records, transform=eval_transform, include_classes=include_classes, exclude_classes=exclude_classes)
    return train_dataset, train_eval_dataset, val_dataset


def run_single_experiment(
    experiment: str,
    model_key: str,
    records: Sequence[ImageRecord],
    args,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    class_mapping = default_class_mapping()
    output_root = Path(args.output_dir) / experiment / model_key
    create_experiment_tree(output_root)
    initialize_epoch_csvs(output_root)
    save_run_config(output_root, namespace_to_dict(args), {"experiment": experiment, "model": model_key}, class_mapping.to_json_dict())
    save_dataset_summaries(list(records), output_root)
    save_dataset_figures(records, args.warmup_dir, output_root)
    save_augmentation_placeholders(output_root)

    run_logger = logging.getLogger(logger.name + f".{experiment}.{model_key}")
    model = build_model(model_key, num_classes=len(class_mapping.class_names), convnext_name=args.convnext_name, dino_name=args.dino_name, logger=run_logger).to(device)
    if args.resume:
        load_checkpoint(args.resume, model, map_location=device)
        run_logger.info("Loaded resume checkpoint: %s", args.resume)
    train_dataset, train_eval_dataset, val_dataset = make_datasets(records, args)
    train_ssl_encoder(model, args.warmup_dir, output_root, args, device, run_logger)

    eval_result = None
    if experiment == "SSL_Prototypical":
        train_prototypical_network(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger, hard_probability=0.0, phase_name="prototypical")
        install_prototype_classifier(model, train_eval_dataset, class_mapping, output_root, args, device)
    elif experiment == "SSL_Hard_Prototypical":
        train_hard_prototypical_network(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger)
        install_prototype_classifier(model, train_eval_dataset, class_mapping, output_root, args, device)
    elif experiment == "SSL_ClassBalanced_FineTune":
        eval_result = train_supervised_classifier(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger)
    elif experiment == "SSL_MAML_ANIL":
        train_maml_anil(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger)
    elif experiment == "SSL_MetricLearning":
        result = train_metric_learning(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger)
        eval_result = result.get("classifier_result")
    elif experiment == "SSL_Hybrid_FineTune_Episodic":
        eval_result = train_supervised_classifier(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger, phase_name="hybrid_supervised")
        save_checkpoint(Path(output_root) / "checkpoints" / "supervised_finetuned.pt", model, phase="hybrid_supervised")
        _copy_metric_if_exists(output_root, "supervised_epoch_metrics.csv", "after_supervised_finetune.csv")
        train_prototypical_network(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger, hard_probability=0.0, phase_name="episodic_balanced")
        save_checkpoint(Path(output_root) / "checkpoints" / "episodic_balanced.pt", model, phase="episodic_balanced")
        _copy_metric_if_exists(output_root, "meta_epoch_metrics.csv", "after_balanced_episodic.csv")
        train_hard_prototypical_network(model, train_dataset, val_dataset, class_mapping, output_root, args, device, run_logger)
        save_checkpoint(Path(output_root) / "checkpoints" / "episodic_hard_final.pt", model, phase="episodic_hard_final")
        install_prototype_classifier(model, train_eval_dataset, class_mapping, output_root, args, device)
        _copy_metric_if_exists(output_root, "meta_epoch_metrics.csv", "after_hard_episodic.csv")
    elif experiment == "SSL_Simulated_FutureClass":
        eval_result = run_simulated_future_class(model, records, class_mapping, output_root, args, device, run_logger)
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    evaluate_and_save_all(model, val_dataset, class_mapping, output_root, args, device, run_logger, experiment, model_key, eval_result=eval_result)
    summary_path = Path(output_root) / "metrics" / "metrics_summary.csv"
    write_dataframe(Path(output_root) / "metrics" / "final_metrics.csv", pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame())
    save_checkpoint(Path(output_root) / "checkpoints" / "best_model.pt", model, phase="final")
    save_checkpoint(Path(output_root) / "checkpoints" / "final_model.pt", model, phase="final")


def run_simulated_future_class(
    model: torch.nn.Module,
    records: Sequence[ImageRecord],
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    pseudo_name = parse_class_name(args.pseudo_novel_class, class_mapping.class_names)
    pseudo_id = class_mapping.name_to_id[pseudo_name]
    train_dataset, train_eval_dataset, val_dataset = make_datasets(records, args, exclude_classes=[pseudo_id])
    if len(train_dataset) > 0 and args.epochs_finetune > 0:
        train_supervised_classifier(model, train_dataset, val_dataset, class_mapping, output_root, args, device, logger, phase_name="future_base_supervised")
    full_train_eval = make_datasets(records, args)[1]
    full_val_dataset = make_datasets(records, args)[2]
    fewshot_rows = []
    best_eval = None
    for shot_count in args.fewshot_values:
        probabilities, labels, paths = _fewshot_prototype_eval(model, full_train_eval, full_val_dataset, class_mapping, pseudo_id, int(shot_count), args.prototype_distance, device)
        metrics = classification_metrics(labels, probabilities, class_mapping)
        one_track = metrics["one_track"].iloc[0].to_dict() if len(metrics["one_track"]) else {}
        fewshot_rows.append(
            {
                "shot_count": int(shot_count),
                "accuracy": metrics["summary"].get("overall_accuracy", 0.0),
                "macro_f1": metrics["summary"].get("macro_f1", 0.0),
                "balanced_accuracy": metrics["summary"].get("balanced_accuracy", 0.0),
                "pseudo_novel_recall": one_track.get("one_track_recall", 0.0) if pseudo_id == 3 else "",
                "pseudo_novel_f1": one_track.get("one_track_f1", 0.0) if pseudo_id == 3 else "",
            }
        )
        best_eval = {"probabilities": probabilities, "labels": labels, "paths": paths, "logits": np.log(np.clip(probabilities, 1e-12, 1.0)), "loss": 0.0}
    write_dataframe(Path(output_root) / "metrics" / "fewshot_results.csv", pd.DataFrame(fewshot_rows))
    install_prototype_classifier(model, full_train_eval, class_mapping, output_root, args, device)
    return best_eval or {"probabilities": np.empty((0, 0)), "labels": np.array([], dtype=int), "paths": [], "logits": np.empty((0, 0)), "loss": 0.0}


def _fewshot_prototype_eval(model, train_dataset, val_dataset, class_mapping: ClassMapping, pseudo_id: int, shot_count: int, distance: str, device: torch.device):
    model.eval()
    prototypes = []
    with torch.no_grad():
        for class_id in range(len(class_mapping.class_names)):
            embeddings = []
            used = 0
            for index, record in enumerate(train_dataset.records):
                if record.label_id != class_id:
                    continue
                if class_id == pseudo_id and used >= shot_count:
                    break
                image, _, _, _ = train_dataset[index]
                embeddings.append(model.forward_features(image.unsqueeze(0).to(device)).squeeze(0))
                used += 1
            if embeddings:
                prototypes.append(torch.stack(embeddings).mean(dim=0))
            else:
                prototypes.append(torch.zeros(model.embedding_dim, device=device))
        prototypes_tensor = torch.stack(prototypes)
        probabilities = []
        labels = []
        paths = []
        for index in range(len(val_dataset)):
            image, label, path, _ = val_dataset[index]
            embedding = model.forward_features(image.unsqueeze(0).to(device))
            if distance == "cosine":
                logits = torch.matmul(torch.nn.functional.normalize(embedding, dim=1), torch.nn.functional.normalize(prototypes_tensor, dim=1).T)
            else:
                logits = -torch.cdist(embedding, prototypes_tensor)
            probabilities.append(torch.softmax(logits, dim=1).squeeze(0).cpu().numpy())
            labels.append(label)
            paths.append(path)
    return np.stack(probabilities), np.asarray(labels, dtype=int), paths


def _limit_records(records: Sequence[ImageRecord], max_samples: int | None) -> List[ImageRecord]:
    if max_samples is None or max_samples <= 0:
        return list(records)
    by_class: dict[int, list[ImageRecord]] = {}
    for record in records:
        by_class.setdefault(record.label_id, []).append(record)
    per_class = max(1, int(np.ceil(max_samples / max(len(by_class), 1))))
    limited: List[ImageRecord] = []
    for class_id in sorted(by_class):
        limited.extend(by_class[class_id][:per_class])
    return limited[:max_samples]


def _copy_metric_if_exists(output_root: str | Path, source_name: str, destination_name: str) -> None:
    source = Path(output_root) / "epoch_progress" / source_name
    destination = Path(output_root) / "metrics" / destination_name
    if source.exists():
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
