from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
