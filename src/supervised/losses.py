from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        probability = torch.exp(-cross_entropy)
        return ((1.0 - probability) ** self.gamma * cross_entropy).mean()


def class_weights_from_counts(counts: list[int], device: torch.device) -> torch.Tensor:
    counts_tensor = torch.tensor(counts, dtype=torch.float32, device=device).clamp_min(1.0)
    weights = counts_tensor.sum() / (len(counts) * counts_tensor)
    return weights / weights.mean()


def build_loss(loss_name: str, class_counts: list[int], device: torch.device) -> nn.Module:
    name = loss_name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    weights = class_weights_from_counts(class_counts, device)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "focal":
        return FocalLoss(gamma=2.0, weight=weights)
    raise ValueError(f"Unsupported loss '{loss_name}'. Expected ce, weighted_ce, or focal.")
