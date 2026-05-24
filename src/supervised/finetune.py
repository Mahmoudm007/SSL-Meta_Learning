from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.class_mapping import ClassMapping
from src.evaluation.metrics import classification_metrics, softmax_numpy
from src.supervised.losses import build_loss
from src.supervised.samplers import make_balanced_sampler
from src.utils.checkpointing import save_checkpoint
from src.utils.io import append_dataframe
from src.utils.reproducibility import gpu_memory_mb


def train_supervised_classifier(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
    logger: logging.Logger,
    phase_name: str = "supervised",
) -> Dict[str, object]:
    labels = train_dataset.labels()
    class_counts = [max(labels.count(class_id), 1) for class_id in range(len(class_mapping.class_names))]
    criterion = build_loss(args.loss, class_counts, device)
    sampler = make_balanced_sampler(labels) if args.sampler == "balanced" else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    best_macro_f1 = -1.0
    best_checkpoint = Path(output_root) / "checkpoints" / "best_model.pt"
    final_checkpoint = Path(output_root) / "checkpoints" / "final_model.pt"
    epochs = max(int(args.epochs_finetune), 0)
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        losses = []
        gradient_norms = []
        progress = tqdm(train_loader, desc=f"{phase_name} epoch {epoch}/{epochs}", leave=False)
        for images, labels_tensor, _, _ in progress:
            images = images.to(device, non_blocking=True)
            labels_tensor = labels_tensor.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            gradient_norms.append(_gradient_norm(model))
            optimizer.step()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
        train_eval = evaluate_loader(model, train_loader, criterion, device, max_batches=args.train_eval_batches)
        val_eval = evaluate_loader(model, val_loader, criterion, device)
        train_metrics = classification_metrics(train_eval["labels"], train_eval["probabilities"], class_mapping)
        val_metrics = classification_metrics(val_eval["labels"], val_eval["probabilities"], class_mapping)
        val_summary = val_metrics["summary"]
        one_track = val_metrics["one_track"].iloc[0].to_dict() if not val_metrics["one_track"].empty else {}
        allocated, reserved = gpu_memory_mb()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else train_eval["loss"],
            "val_loss": val_eval["loss"],
            "train_accuracy": train_metrics["summary"].get("overall_accuracy", 0.0),
            "val_accuracy": val_summary.get("overall_accuracy", 0.0),
            "train_balanced_accuracy": train_metrics["summary"].get("balanced_accuracy", 0.0),
            "val_balanced_accuracy": val_summary.get("balanced_accuracy", 0.0),
            "train_macro_f1": train_metrics["summary"].get("macro_f1", 0.0),
            "val_macro_f1": val_summary.get("macro_f1", 0.0),
            "train_weighted_f1": train_metrics["summary"].get("weighted_f1", 0.0),
            "val_weighted_f1": val_summary.get("weighted_f1", 0.0),
            "one_track_precision": one_track.get("one_track_precision", 0.0),
            "one_track_recall": one_track.get("one_track_recall", 0.0),
            "one_track_f1": one_track.get("one_track_f1", 0.0),
            "one_track_false_negative_rate": one_track.get("one_track_false_negative_rate", 0.0),
            "top2_accuracy": val_summary.get("top2_accuracy", 0.0),
            "ece": val_summary.get("ece", 0.0),
            "brier_score": val_summary.get("brier_score", 0.0),
            "nll": val_summary.get("nll", 0.0),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "epoch_time_seconds": time.time() - start,
            "gpu_memory_allocated_mb": allocated,
            "gpu_memory_reserved_mb": reserved,
        }
        append_dataframe(Path(output_root) / "epoch_progress" / "supervised_epoch_metrics.csv", row)
        append_dataframe(Path(output_root) / "epoch_progress" / "learning_rate_schedule.csv", {"epoch": epoch, "phase": phase_name, "learning_rate": optimizer.param_groups[0]["lr"]})
        append_dataframe(Path(output_root) / "epoch_progress" / "gpu_memory_log.csv", {"epoch": epoch, "phase": phase_name, "allocated_mb": allocated, "reserved_mb": reserved})
        append_dataframe(Path(output_root) / "epoch_progress" / "epoch_time_log.csv", {"epoch": epoch, "phase": phase_name, "epoch_time_seconds": row["epoch_time_seconds"]})
        logger.info("%s epoch %s val macro-F1 %.5f one-track recall %.5f", phase_name, epoch, row["val_macro_f1"], row["one_track_recall"])
        if row["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = row["val_macro_f1"]
            save_checkpoint(best_checkpoint, model, optimizer, phase=phase_name, epoch=epoch, best_macro_f1=best_macro_f1)
    save_checkpoint(final_checkpoint, model, optimizer, phase=phase_name, epochs=epochs)
    if epochs == 0:
        logger.info("epochs_finetune=0; evaluating current classifier without supervised updates.")
    return evaluate_loader(model, val_loader, criterion, device)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, object]:
    model.eval()
    logits_list = []
    labels_list = []
    paths = []
    losses = []
    with torch.no_grad():
        for batch_index, (images, labels_tensor, batch_paths, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels_tensor = labels_tensor.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels_tensor)
            losses.append(float(loss.item()))
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels_tensor.detach().cpu().numpy())
            paths.extend(list(batch_paths))
    if not logits_list:
        return {"logits": np.empty((0, 0)), "probabilities": np.empty((0, 0)), "labels": np.array([], dtype=int), "paths": [], "loss": 0.0}
    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0).astype(int)
    return {
        "logits": logits_np,
        "probabilities": softmax_numpy(logits_np),
        "labels": labels_np,
        "paths": paths,
        "loss": float(np.mean(losses)) if losses else 0.0,
    }


def _gradient_norm(model: torch.nn.Module) -> float:
    norms = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            norms.append(parameter.grad.detach().norm(2).item())
    return float(np.linalg.norm(norms)) if norms else 0.0
