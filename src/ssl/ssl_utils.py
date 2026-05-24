from __future__ import annotations

from pathlib import Path

import torch


def should_skip_ssl(checkpoint_path: str | Path, skip_if_exists: bool, force_rerun: bool) -> bool:
    return bool(skip_if_exists and not force_rerun and Path(checkpoint_path).exists())


def freeze_encoder(model: torch.nn.Module) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False


def unfreeze_encoder(model: torch.nn.Module) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = True
