from __future__ import annotations

import torch
from torch import nn


class IdentityMAEPlaceholder(nn.Module):
    """Explicit placeholder for future MAE-style SSL extensions.

    The implemented SSL path uses contrastive learning. This class exists so
    MAE-style reconstruction can be added without changing the runner API.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features
