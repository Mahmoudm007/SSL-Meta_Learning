from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

from src.data.class_mapping import ClassMapping, ONE_TRACK_CLASS_ID
from src.data.episodic_sampler import EpisodicSampler
from src.meta.meta_utils import batch_from_indices, remap_labels
from src.utils.checkpointing import save_checkpoint
from src.utils.io import append_dataframe, write_dataframe
from src.utils.reproducibility import gpu_memory_mb


def train_maml_anil(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, object]:
    algorithm = args.meta_algorithm.lower()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.outer_lr, weight_decay=args.weight_decay)
    sampler = EpisodicSampler(train_dataset.records, args.support_per_class, args.query_per_class, seed=args.seed)
    rows: List[dict] = []
    epochs = max(int(args.epochs_meta), 0)
    episodes_per_epoch = max(int(args.episodes_per_epoch), 1)
    for epoch in range(1, epochs + 1):
        start = time.time()
        outer_losses = []
        pre_accs = []
        post_accs = []
        pre_f1s = []
        post_f1s = []
        pre_recalls = []
        post_recalls = []
        inner_loss_steps: List[List[float]] = [[] for _ in range(max(args.inner_steps, 1))]
        for _ in tqdm(range(episodes_per_epoch), desc=f"{algorithm.upper()} epoch {epoch}/{epochs}", leave=False, ascii=True):
            episode = sampler.sample()
            result = _adapt_episode(model, train_dataset, episode, class_mapping, args, device)
            outer_losses.append(result["post_loss"])
            pre_accs.append(result["pre_accuracy"])
            post_accs.append(result["post_accuracy"])
            pre_f1s.append(result["pre_macro_f1"])
            post_f1s.append(result["post_macro_f1"])
            pre_recalls.append(result["pre_one_track_recall"])
            post_recalls.append(result["post_one_track_recall"])
            for step_index, value in enumerate(result["inner_losses"]):
                inner_loss_steps[step_index].append(value)
            if algorithm == "anil":
                optimizer.zero_grad(set_to_none=True)
                support_images, support_labels, _ = batch_from_indices(train_dataset, episode.support_indices, device)
                logits = model(support_images)
                loss = F.cross_entropy(logits, support_labels)
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                query_images, query_labels, _ = batch_from_indices(train_dataset, episode.query_indices, device)
                loss = F.cross_entropy(model(query_images), query_labels)
                loss.backward()
                optimizer.step()
        allocated, reserved = gpu_memory_mb()
        row = {
            "epoch": epoch,
            "outer_loss": float(np.mean(outer_losses)) if outer_losses else 0.0,
            "mean_inner_loss_step_1": _step_mean(inner_loss_steps, 0),
            "mean_inner_loss_step_2": _step_mean(inner_loss_steps, 1),
            "mean_inner_loss_step_3": _step_mean(inner_loss_steps, 2),
            "mean_inner_loss_step_4": _step_mean(inner_loss_steps, 3),
            "mean_inner_loss_step_5": _step_mean(inner_loss_steps, 4),
            "pre_adaptation_query_accuracy": float(np.mean(pre_accs)) if pre_accs else 0.0,
            "post_adaptation_query_accuracy": float(np.mean(post_accs)) if post_accs else 0.0,
            "adaptation_gain": float(np.mean(post_accs) - np.mean(pre_accs)) if post_accs and pre_accs else 0.0,
            "pre_adaptation_macro_f1": float(np.mean(pre_f1s)) if pre_f1s else 0.0,
            "post_adaptation_macro_f1": float(np.mean(post_f1s)) if post_f1s else 0.0,
            "one_track_pre_adaptation_recall": float(np.mean(pre_recalls)) if pre_recalls else 0.0,
            "one_track_post_adaptation_recall": float(np.mean(post_recalls)) if post_recalls else 0.0,
            "learning_rate_inner": args.inner_lr,
            "learning_rate_outer": args.outer_lr,
            "epoch_time_seconds": time.time() - start,
            "gpu_memory_allocated_mb": allocated,
            "gpu_memory_reserved_mb": reserved,
        }
        append_dataframe(Path(output_root) / "epoch_progress" / "maml_anil_epoch_metrics.csv", row)
        rows.append(row)
        logger.info("%s epoch %s adaptation gain %.5f", algorithm.upper(), epoch, row["adaptation_gain"])
    write_dataframe(Path(output_root) / "metrics" / "episode_metrics.csv", pd.DataFrame(rows))
    save_checkpoint(Path(output_root) / "checkpoints" / f"{algorithm}_{args.adapt_scope}.pt", model, optimizer, phase=algorithm, adapt_scope=args.adapt_scope)
    return {"maml_anil_metrics": pd.DataFrame(rows)}


def _adapt_episode(model, dataset, episode, class_mapping: ClassMapping, args, device: torch.device) -> dict:
    adapted = copy.deepcopy(model).to(device)
    if args.meta_algorithm.lower() == "anil" or args.adapt_scope == "head":
        for name, parameter in adapted.named_parameters():
            parameter.requires_grad = "classifier" in name
    elif args.adapt_scope == "last_block":
        for name, parameter in adapted.named_parameters():
            parameter.requires_grad = "classifier" in name or "blocks" in name or "stages.3" in name
    else:
        for parameter in adapted.parameters():
            parameter.requires_grad = True
    inner_optimizer = torch.optim.SGD([p for p in adapted.parameters() if p.requires_grad], lr=args.inner_lr)
    support_images, support_labels, _ = batch_from_indices(dataset, episode.support_indices, device)
    query_images, query_labels, _ = batch_from_indices(dataset, episode.query_indices, device)
    pre_logits = adapted(query_images)
    pre_result = _query_metrics(pre_logits, query_labels, episode.class_ids)
    inner_losses = []
    for _ in range(args.inner_steps):
        inner_optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(adapted(support_images), support_labels)
        loss.backward()
        inner_optimizer.step()
        inner_losses.append(float(loss.item()))
    post_logits = adapted(query_images)
    post_loss = F.cross_entropy(post_logits, query_labels)
    post_result = _query_metrics(post_logits, query_labels, episode.class_ids)
    return {
        "post_loss": float(post_loss.item()),
        "inner_losses": inner_losses,
        **{f"pre_{key}": value for key, value in pre_result.items()},
        **{f"post_{key}": value for key, value in post_result.items()},
    }


def _query_metrics(logits: torch.Tensor, labels: torch.Tensor, class_ids) -> dict:
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    one_track_mask = labels_np == ONE_TRACK_CLASS_ID
    return {
        "accuracy": float(accuracy_score(labels_np, pred)),
        "macro_f1": float(f1_score(labels_np, pred, labels=list(class_ids), average="macro", zero_division=0)),
        "one_track_recall": float(recall_score(labels_np == ONE_TRACK_CLASS_ID, pred == ONE_TRACK_CLASS_ID, zero_division=0)) if np.any(one_track_mask) else 0.0,
    }


def _step_mean(steps: List[List[float]], index: int) -> float | str:
    if index >= len(steps) or not steps[index]:
        return ""
    return float(np.mean(steps[index]))
