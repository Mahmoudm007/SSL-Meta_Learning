from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def batch_from_indices(dataset, indices: Sequence[int], device: torch.device):
    images = []
    labels = []
    paths = []
    for index in indices:
        image, label, path, _ = dataset[index]
        images.append(image)
        labels.append(label)
        paths.append(path)
    return torch.stack(images).to(device), torch.tensor(labels, dtype=torch.long, device=device), paths


def pairwise_logits(query_embeddings: torch.Tensor, prototypes: torch.Tensor, distance: str = "euclidean") -> torch.Tensor:
    if distance == "cosine":
        return torch.matmul(F.normalize(query_embeddings, dim=1), F.normalize(prototypes, dim=1).T)
    if distance == "euclidean":
        return -torch.cdist(query_embeddings, prototypes, p=2)
    raise ValueError("distance must be 'euclidean' or 'cosine'")


def remap_labels(labels: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    mapping = {int(class_id): index for index, class_id in enumerate(class_ids)}
    return torch.tensor([mapping[int(label.item())] for label in labels], dtype=torch.long, device=labels.device)


def gpu_safe_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return float(sum(values_list) / max(len(values_list), 1))
