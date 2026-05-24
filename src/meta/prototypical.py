from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from src.data.class_mapping import ClassMapping, ONE_TRACK_CLASS_ID, sanitize_class_name
from src.data.episodic_sampler import EpisodicSampler
from src.evaluation.metrics import prediction_dataframe
from src.meta.meta_utils import batch_from_indices, pairwise_logits, remap_labels
from src.utils.checkpointing import save_checkpoint
from src.utils.io import append_dataframe, write_dataframe
from src.utils.reproducibility import gpu_memory_mb


def train_prototypical_network(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
    logger: logging.Logger,
    hard_probability: float = 0.0,
    phase_name: str = "prototypical",
) -> Dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = EpisodicSampler(
        train_dataset.records,
        support_per_class=args.support_per_class,
        query_per_class=args.query_per_class,
        hard_episode_probability=hard_probability,
        seed=args.seed,
    )
    epoch_rows: List[dict] = []
    prediction_rows: List[pd.DataFrame] = []
    support_query_rows: List[dict] = []
    last_prototypes = None
    last_class_ids = None
    epochs = max(int(args.epochs_meta), 0)
    episodes_per_epoch = max(int(args.episodes_per_epoch), 1)
    for epoch in range(1, epochs + 1):
        start = time.time()
        losses = []
        query_accuracies = []
        support_accuracies = []
        query_macro_f1s = []
        query_balanced = []
        hard_accuracies = []
        hard_one_track_recalls = []
        prototype_stats = []
        progress = tqdm(range(episodes_per_epoch), desc=f"{phase_name} epoch {epoch}/{epochs}", leave=False)
        for episode_index in progress:
            episode = sampler.sample()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, details = _episode_loss(model, train_dataset, episode, args.prototype_distance, device)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            support_accuracies.append(details["support_accuracy"])
            query_accuracies.append(details["query_accuracy"])
            query_macro_f1s.append(details["query_macro_f1"])
            query_balanced.append(details["query_balanced_accuracy"])
            if episode.hard:
                hard_accuracies.append(details["query_accuracy"])
                hard_one_track_recalls.append(details["one_track_recall"])
            prototype_stats.append(details["prototype_distances"])
            last_prototypes = details["prototypes"].detach().cpu().numpy()
            last_class_ids = episode.class_ids
            support_query_rows.append(
                {
                    "epoch": epoch,
                    "episode": episode_index,
                    "hard": episode.hard,
                    "hard_pair": str(episode.hard_pair),
                    "support_indices": " ".join(map(str, episode.support_indices)),
                    "query_indices": " ".join(map(str, episode.query_indices)),
                }
            )
            prediction_rows.append(_episode_prediction_frame(details, class_mapping, epoch, episode_index, episode.hard))
            progress.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{query_accuracies[-1]:.3f}")
        allocated, reserved = gpu_memory_mb()
        distances = np.concatenate([values.reshape(-1) for values in prototype_stats if values.size]) if prototype_stats else np.array([0.0])
        row = {
            "epoch": epoch,
            "episode_train_loss": float(np.mean(losses)) if losses else 0.0,
            "episode_query_loss": float(np.mean(losses)) if losses else 0.0,
            "support_accuracy": float(np.mean(support_accuracies)) if support_accuracies else 0.0,
            "query_accuracy": float(np.mean(query_accuracies)) if query_accuracies else 0.0,
            "query_balanced_accuracy": float(np.mean(query_balanced)) if query_balanced else 0.0,
            "query_macro_f1": float(np.mean(query_macro_f1s)) if query_macro_f1s else 0.0,
            "query_weighted_f1": float(np.mean(query_macro_f1s)) if query_macro_f1s else 0.0,
            "one_track_query_recall": float(np.mean(hard_one_track_recalls)) if hard_one_track_recalls else 0.0,
            "one_track_query_f1": "",
            "hard_episode_accuracy": float(np.mean(hard_accuracies)) if hard_accuracies else "",
            "hard_episode_loss": "",
            "hard_episode_one_track_recall": float(np.mean(hard_one_track_recalls)) if hard_one_track_recalls else "",
            "prototype_mean_distance": float(np.mean(distances)),
            "prototype_min_distance": float(np.min(distances)),
            "prototype_max_distance": float(np.max(distances)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": time.time() - start,
            "gpu_memory_allocated_mb": allocated,
            "gpu_memory_reserved_mb": reserved,
        }
        epoch_rows.append(row)
        append_dataframe(Path(output_root) / "epoch_progress" / "meta_epoch_metrics.csv", row)
        append_dataframe(Path(output_root) / "epoch_progress" / "learning_rate_schedule.csv", {"epoch": epoch, "phase": phase_name, "learning_rate": optimizer.param_groups[0]["lr"]})
        append_dataframe(Path(output_root) / "epoch_progress" / "gpu_memory_log.csv", {"epoch": epoch, "phase": phase_name, "allocated_mb": allocated, "reserved_mb": reserved})
        append_dataframe(Path(output_root) / "epoch_progress" / "epoch_time_log.csv", {"epoch": epoch, "phase": phase_name, "epoch_time_seconds": row["epoch_time_seconds"]})
        logger.info("%s epoch %s query acc %.5f macro-F1 %.5f", phase_name, epoch, row["query_accuracy"], row["query_macro_f1"])
    if last_prototypes is not None and last_class_ids is not None:
        prototype_frame = pd.DataFrame(last_prototypes, columns=[f"emb_{index}" for index in range(last_prototypes.shape[1])])
        prototype_frame.insert(0, "label_id", last_class_ids)
        prototype_frame.insert(1, "label_name", [class_mapping.id_to_name[int(class_id)] for class_id in last_class_ids])
        write_dataframe(Path(output_root) / "metrics" / "prototype_vectors.csv", prototype_frame)
        matrix = np.linalg.norm(last_prototypes[:, None, :] - last_prototypes[None, :, :], axis=-1)
        write_dataframe(Path(output_root) / "metrics" / "prototype_distance_matrix.csv", pd.DataFrame(matrix, columns=prototype_frame["label_name"]).assign(label_name=prototype_frame["label_name"]))
    episode_predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    write_dataframe(Path(output_root) / "metrics" / "episode_metrics.csv", pd.DataFrame(epoch_rows))
    write_dataframe(Path(output_root) / "metrics" / "query_metrics.csv", episode_predictions)
    write_dataframe(Path(output_root) / "predictions" / "query_predictions.csv", episode_predictions)
    write_dataframe(Path(output_root) / "predictions" / "support_query_indices.csv", pd.DataFrame(support_query_rows))
    save_checkpoint(Path(output_root) / "checkpoints" / f"{phase_name}.pt", model, optimizer, phase=phase_name, epochs=epochs)
    return {"episode_predictions": episode_predictions, "epoch_metrics": pd.DataFrame(epoch_rows)}


def install_prototype_classifier(
    model: torch.nn.Module,
    dataset,
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
) -> None:
    model.eval()
    sums = {}
    counts = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            image, label, _, _ = dataset[index]
            embedding = model.forward_features(image.unsqueeze(0).to(device)).squeeze(0)
            sums[label] = sums.get(label, torch.zeros_like(embedding)) + embedding
            counts[label] = counts.get(label, 0) + 1
    prototypes = []
    for class_id in range(len(class_mapping.class_names)):
        if class_id not in sums:
            prototypes.append(torch.zeros(model.embedding_dim, device=device))
        else:
            prototypes.append(sums[class_id] / max(counts[class_id], 1))
    prototypes_tensor = torch.stack(prototypes).to(device)
    linear = None
    if hasattr(model.classifier, "__iter__"):
        for module in model.classifier:
            if isinstance(module, torch.nn.Linear):
                linear = module
    if linear is None and isinstance(model.classifier, torch.nn.Linear):
        linear = model.classifier
    if linear is not None:
        with torch.no_grad():
            if args.prototype_distance == "cosine":
                linear.weight.copy_(torch.nn.functional.normalize(prototypes_tensor, dim=1))
                linear.bias.zero_()
            else:
                linear.weight.copy_(2.0 * prototypes_tensor)
                linear.bias.copy_(-(prototypes_tensor**2).sum(dim=1))
    prototype_frame = pd.DataFrame(prototypes_tensor.detach().cpu().numpy(), columns=[f"emb_{i}" for i in range(prototypes_tensor.shape[1])])
    prototype_frame.insert(0, "label_id", list(range(len(class_mapping.class_names))))
    prototype_frame.insert(1, "label_name", class_mapping.class_names)
    write_dataframe(Path(output_root) / "metrics" / "final_train_prototype_vectors.csv", prototype_frame)


def _episode_loss(model, dataset, episode, distance: str, device: torch.device):
    support_images, support_labels, support_paths = batch_from_indices(dataset, episode.support_indices, device)
    query_images, query_labels, query_paths = batch_from_indices(dataset, episode.query_indices, device)
    support_embeddings = model.forward_features(support_images)
    query_embeddings = model.forward_features(query_images)
    prototypes = []
    for class_id in episode.class_ids:
        mask = support_labels == class_id
        prototypes.append(support_embeddings[mask].mean(dim=0))
    prototypes_tensor = torch.stack(prototypes)
    logits = pairwise_logits(query_embeddings, prototypes_tensor, distance)
    query_targets = remap_labels(query_labels, episode.class_ids)
    loss = F.cross_entropy(logits, query_targets)
    with torch.no_grad():
        query_pred_local = logits.argmax(dim=1)
        query_pred = torch.tensor([episode.class_ids[index] for index in query_pred_local.cpu().numpy()], device=device)
        support_logits = pairwise_logits(support_embeddings, prototypes_tensor, distance)
        support_pred_local = support_logits.argmax(dim=1)
        support_pred = torch.tensor([episode.class_ids[index] for index in support_pred_local.cpu().numpy()], device=device)
        labels_np = query_labels.detach().cpu().numpy()
        pred_np = query_pred.detach().cpu().numpy()
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        one_track_mask = labels_np == ONE_TRACK_CLASS_ID
        one_track_recall = float(((pred_np == ONE_TRACK_CLASS_ID) & one_track_mask).sum() / max(one_track_mask.sum(), 1))
        prototype_distances = torch.cdist(prototypes_tensor, prototypes_tensor).detach().cpu().numpy()
    return loss, {
        "prototypes": prototypes_tensor,
        "prototype_distances": prototype_distances,
        "query_logits": logits.detach().cpu().numpy(),
        "query_probabilities_local": probabilities,
        "query_labels": labels_np,
        "query_predictions": pred_np,
        "query_paths": query_paths,
        "class_ids": episode.class_ids,
        "support_accuracy": float((support_pred == support_labels).float().mean().item()),
        "query_accuracy": float(accuracy_score(labels_np, pred_np)),
        "query_balanced_accuracy": float(balanced_accuracy_score(labels_np, pred_np)),
        "query_macro_f1": float(f1_score(labels_np, pred_np, labels=episode.class_ids, average="macro", zero_division=0)),
        "one_track_recall": one_track_recall,
    }


def _episode_prediction_frame(details: Dict[str, object], class_mapping: ClassMapping, epoch: int, episode_index: int, hard: bool) -> pd.DataFrame:
    local_probs = details["query_probabilities_local"]
    full_probs = np.zeros((local_probs.shape[0], len(class_mapping.class_names)), dtype=float)
    for local_index, class_id in enumerate(details["class_ids"]):
        full_probs[:, int(class_id)] = local_probs[:, local_index]
    frame = prediction_dataframe(details["query_paths"], details["query_labels"], full_probs, class_mapping)
    frame.insert(0, "epoch", epoch)
    frame.insert(1, "episode", episode_index)
    frame.insert(2, "hard_episode", hard)
    return frame
