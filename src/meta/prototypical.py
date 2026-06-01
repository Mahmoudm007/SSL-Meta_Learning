from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
from src.utils.checkpointing import load_checkpoint, save_checkpoint
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
    episode_batch_size = _resolve_episode_batch_size(args)
    checkpoint_path = Path(output_root) / "checkpoints" / f"{phase_name}.pt"
    start_epoch = _resume_prototypical_phase(checkpoint_path, model, optimizer, epochs, args, device, logger, phase_name)
    if start_epoch > epochs:
        logger.info("%s already completed through epoch %s; skipping phase.", phase_name, epochs)
        return _load_existing_episode_outputs(output_root)
    if epochs == 0:
        save_checkpoint(checkpoint_path, model, optimizer, phase=phase_name, epoch=0, epochs=0, episode_batch_size=episode_batch_size)
        return {"episode_predictions": pd.DataFrame(), "epoch_metrics": pd.DataFrame()}
    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        losses = []
        query_accuracies = []
        support_accuracies = []
        query_macro_f1s = []
        query_balanced = []
        hard_accuracies = []
        hard_one_track_recalls = []
        prototype_stats = []
        progress = tqdm(range(episodes_per_epoch), desc=f"{phase_name} epoch {epoch}/{epochs}", leave=False, ascii=True)
        for episode_index in progress:
            episode = sampler.sample()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value, details = _episode_step(model, train_dataset, episode, args.prototype_distance, device, episode_batch_size)
            optimizer.step()
            losses.append(float(loss_value))
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
        save_checkpoint(checkpoint_path, model, optimizer, phase=phase_name, epoch=epoch, epochs=epochs, episode_batch_size=episode_batch_size)
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
    save_checkpoint(checkpoint_path, model, optimizer, phase=phase_name, epoch=epochs, epochs=epochs, episode_batch_size=episode_batch_size)
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


def _resolve_episode_batch_size(args) -> int:
    configured = int(getattr(args, "episode_batch_size", 0) or 0)
    fallback = int(getattr(args, "batch_size", 1) or 1)
    return max(configured if configured > 0 else fallback, 1)


def _resume_prototypical_phase(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    args,
    device: torch.device,
    logger: logging.Logger,
    phase_name: str,
) -> int:
    if bool(getattr(args, "force_rerun", False)) or not checkpoint_path.exists():
        return 1
    metadata = load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    completed_epoch = _completed_epoch(metadata)
    if completed_epoch >= epochs:
        return epochs + 1
    logger.info("Resuming %s from checkpoint %s at epoch %s/%s", phase_name, checkpoint_path, completed_epoch + 1, epochs)
    return completed_epoch + 1


def _completed_epoch(metadata: Dict[str, object]) -> int:
    for key in ("epoch", "epochs"):
        value = metadata.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _load_existing_episode_outputs(output_root: str | Path) -> Dict[str, object]:
    root = Path(output_root)
    predictions_path = root / "predictions" / "query_predictions.csv"
    metrics_path = root / "metrics" / "episode_metrics.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() and predictions_path.stat().st_size > 0 else pd.DataFrame()
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() and metrics_path.stat().st_size > 0 else pd.DataFrame()
    return {"episode_predictions": predictions, "epoch_metrics": metrics}


def _episode_step(model, dataset, episode, distance: str, device: torch.device, episode_batch_size: int):
    support_embeddings, support_labels, _ = _embed_indices_no_grad(model, dataset, episode.support_indices, device, episode_batch_size)
    prototypes_tensor = _build_prototypes(support_embeddings, support_labels, episode.class_ids)
    query_count = len(episode.query_indices)
    if query_count <= 0:
        raise ValueError("Episodes must contain at least one query image.")

    loss_value = 0.0
    query_labels_all: List[np.ndarray] = []
    query_predictions_all: List[np.ndarray] = []
    query_probabilities_all: List[np.ndarray] = []
    query_paths: List[str] = []
    for query_indices in _chunks(episode.query_indices, episode_batch_size):
        query_images, query_labels, batch_paths = batch_from_indices(dataset, query_indices, device)
        query_embeddings = model.forward_features(query_images)
        logits = pairwise_logits(query_embeddings, prototypes_tensor, distance)
        query_targets = remap_labels(query_labels, episode.class_ids)
        loss = F.cross_entropy(logits, query_targets, reduction="sum") / query_count
        loss.backward()
        loss_value += float(loss.detach().item())
        with torch.no_grad():
            query_pred_local = logits.detach().argmax(dim=1)
            query_pred = _local_predictions_to_class_ids(query_pred_local, episode.class_ids)
            query_labels_all.append(query_labels.detach().cpu().numpy())
            query_predictions_all.append(query_pred)
            query_probabilities_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
            query_paths.extend(list(batch_paths))
        del query_images, query_labels, query_embeddings, logits, loss

    labels_np = np.concatenate(query_labels_all).astype(int)
    pred_np = np.concatenate(query_predictions_all).astype(int)
    probabilities = np.concatenate(query_probabilities_all, axis=0)
    with torch.no_grad():
        support_logits = pairwise_logits(support_embeddings, prototypes_tensor, distance)
        support_pred_local = support_logits.argmax(dim=1)
        support_pred_np = _local_predictions_to_class_ids(support_pred_local, episode.class_ids)
        support_labels_np = support_labels.detach().cpu().numpy().astype(int)
        one_track_mask = labels_np == ONE_TRACK_CLASS_ID
        one_track_recall = float(((pred_np == ONE_TRACK_CLASS_ID) & one_track_mask).sum() / max(one_track_mask.sum(), 1))
        prototype_distances = torch.cdist(prototypes_tensor, prototypes_tensor).detach().cpu().numpy()
    return loss_value, {
        "prototypes": prototypes_tensor.detach(),
        "prototype_distances": prototype_distances,
        "query_logits": np.log(np.clip(probabilities, 1e-12, 1.0)),
        "query_probabilities_local": probabilities,
        "query_labels": labels_np,
        "query_predictions": pred_np,
        "query_paths": query_paths,
        "class_ids": episode.class_ids,
        "support_accuracy": float((support_pred_np == support_labels_np).mean()),
        "query_accuracy": float(accuracy_score(labels_np, pred_np)),
        "query_balanced_accuracy": float(balanced_accuracy_score(labels_np, pred_np)),
        "query_macro_f1": float(f1_score(labels_np, pred_np, labels=episode.class_ids, average="macro", zero_division=0)),
        "one_track_recall": one_track_recall,
    }


def _embed_indices_no_grad(model, dataset, indices: Sequence[int], device: torch.device, batch_size: int):
    embeddings = []
    labels = []
    paths: List[str] = []
    with torch.no_grad():
        for chunk_indices in _chunks(indices, batch_size):
            images, batch_labels, batch_paths = batch_from_indices(dataset, chunk_indices, device)
            embeddings.append(model.forward_features(images).detach())
            labels.append(batch_labels.detach())
            paths.extend(list(batch_paths))
            del images, batch_labels
    if not embeddings:
        raise ValueError("Cannot embed an empty index list.")
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0), paths


def _build_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    prototypes = []
    for class_id in class_ids:
        mask = labels == int(class_id)
        if not bool(mask.any()):
            raise ValueError(f"Episode support set has no samples for class id {class_id}.")
        prototypes.append(embeddings[mask].mean(dim=0))
    return torch.stack(prototypes)


def _local_predictions_to_class_ids(local_predictions: torch.Tensor, class_ids: Sequence[int]) -> np.ndarray:
    class_id_array = np.asarray([int(class_id) for class_id in class_ids], dtype=int)
    return class_id_array[local_predictions.detach().cpu().numpy().astype(int)]


def _chunks(indices: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


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
