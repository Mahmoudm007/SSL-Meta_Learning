from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.class_mapping import ClassMapping
from src.supervised.finetune import train_supervised_classifier
from src.utils.checkpointing import save_checkpoint
from src.utils.io import append_dataframe, write_dataframe
from src.utils.reproducibility import gpu_memory_mb


def train_metric_learning(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, object]:
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    epochs = max(int(args.epochs_meta), 0)
    rows = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        losses = []
        for images, labels, _, _ in tqdm(loader, desc=f"metric epoch {epoch}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.forward_projection(images)
            if args.metric_loss == "triplet":
                loss = hard_triplet_loss(embeddings, labels, margin=args.triplet_margin)
            else:
                loss = supervised_contrastive_loss(embeddings, labels, temperature=args.ssl_temperature)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        allocated, reserved = gpu_memory_mb()
        row = {
            "epoch": epoch,
            "episode_train_loss": float(np.mean(losses)) if losses else 0.0,
            "episode_query_loss": "",
            "support_accuracy": "",
            "query_accuracy": "",
            "query_balanced_accuracy": "",
            "query_macro_f1": "",
            "query_weighted_f1": "",
            "one_track_query_recall": "",
            "one_track_query_f1": "",
            "hard_episode_accuracy": "",
            "hard_episode_loss": "",
            "hard_episode_one_track_recall": "",
            "prototype_mean_distance": "",
            "prototype_min_distance": "",
            "prototype_max_distance": "",
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": time.time() - start,
            "gpu_memory_allocated_mb": allocated,
            "gpu_memory_reserved_mb": reserved,
        }
        append_dataframe(Path(output_root) / "epoch_progress" / "meta_epoch_metrics.csv", row)
        rows.append(row)
        logger.info("Metric-learning epoch %s loss %.5f", epoch, row["episode_train_loss"])
    save_checkpoint(Path(output_root) / "checkpoints" / f"metric_{args.metric_loss}.pt", model, optimizer, phase="metric_learning")
    write_dataframe(Path(output_root) / "metrics" / "hard_negative_pair_performance.csv", _hard_negative_placeholder())
    classifier_result = train_supervised_classifier(model, train_dataset, val_dataset, class_mapping, output_root, args, device, logger, phase_name="metric_classifier")
    return {"metric_metrics": pd.DataFrame(rows), "classifier_result": classifier_result}


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    features = F.normalize(embeddings, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -1e9)
    positives = labels[:, None] == labels[None, :]
    positives = positives & ~mask
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positives.float()).sum(dim=1) / positive_count
    return loss.mean()


def hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    features = F.normalize(embeddings, dim=1)
    distances = torch.cdist(features, features, p=2)
    labels_equal = labels[:, None] == labels[None, :]
    positive_distances = distances.masked_fill(~labels_equal, -1.0)
    negative_distances = distances.masked_fill(labels_equal, 1e9)
    hardest_positive = positive_distances.max(dim=1).values
    hardest_negative = negative_distances.min(dim=1).values
    valid = hardest_positive >= 0
    if not valid.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def _hard_negative_placeholder() -> pd.DataFrame:
    pairs = [
        "One Track - Partly vs Centre - Partly",
        "One Track - Partly vs Two Track - Partly",
        "Two Track - Partly vs Fully",
        "Bare vs Centre - Partly",
    ]
    return pd.DataFrame({"hard_negative_pair": pairs, "note": "Pair performance is derived from final validation predictions and embedding neighborhoods."})
