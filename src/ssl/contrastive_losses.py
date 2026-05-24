from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("Contrastive views must have the same batch size")
    batch_size = z1.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
    similarity = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
    return F.cross_entropy(similarity, positives)


def alignment(z1: torch.Tensor, z2: torch.Tensor) -> float:
    with torch.no_grad():
        return float((F.normalize(z1, dim=1) - F.normalize(z2, dim=1)).pow(2).sum(dim=1).mean().item())


def uniformity(z: torch.Tensor) -> float:
    with torch.no_grad():
        normalized = F.normalize(z, dim=1)
        if normalized.shape[0] < 2:
            return 0.0
        distances = torch.pdist(normalized, p=2).pow(2)
        return float(torch.log(torch.exp(-2 * distances).mean()).item())
