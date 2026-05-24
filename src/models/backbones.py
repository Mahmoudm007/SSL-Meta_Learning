from __future__ import annotations

import logging

from torch import nn

from src.models.convnext_model import build_convnext
from src.models.dino_model import build_dino


def build_model(model_key: str, num_classes: int, convnext_name: str, dino_name: str, logger: logging.Logger | None = None) -> nn.Module:
    key = model_key.lower()
    if key == "convnext":
        return build_convnext(convnext_name, num_classes=num_classes, logger=logger)
    if key == "dino":
        return build_dino(dino_name, num_classes=num_classes, logger=logger)
    raise ValueError(f"Unsupported model key '{model_key}'. Expected 'convnext' or 'dino'.")
