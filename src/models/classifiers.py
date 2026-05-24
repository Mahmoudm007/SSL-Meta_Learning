from __future__ import annotations

from torch import nn


def make_classifier(input_dim: int, num_classes: int, dropout: float = 0.2) -> nn.Module:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))
