from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

try:
    import timm
except Exception as exc:  # pragma: no cover
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None

from src.models.classifiers import make_classifier
from src.models.projection_heads import ProjectionHead


CONVNEXT_FALLBACK = "convnext_base"


class ConvNeXtRSCModel(nn.Module):
    def __init__(self, model_name: str = "convnext_base_in22k", num_classes: int = 5, projection_dim: int = 128, logger: logging.Logger | None = None) -> None:
        super().__init__()
        if timm is None:
            raise ImportError(f"timm is required for ConvNeXt models: {_TIMM_IMPORT_ERROR}")
        selected_name = model_name
        try:
            self.encoder = timm.create_model(selected_name, pretrained=False, num_classes=0, global_pool="avg")
        except Exception as exc:
            if selected_name == CONVNEXT_FALLBACK:
                raise
            if logger:
                logger.warning("ConvNeXt model '%s' is unavailable in this timm installation. Falling back to '%s'. Error: %s", model_name, CONVNEXT_FALLBACK, exc)
            selected_name = CONVNEXT_FALLBACK
            self.encoder = timm.create_model(selected_name, pretrained=False, num_classes=0, global_pool="avg")
        self.model_name = selected_name
        self.requested_model_name = model_name
        self.embedding_dim = int(getattr(self.encoder, "num_features", 1024))
        self.projection_head = ProjectionHead(self.embedding_dim, projection_dim=projection_dim)
        self.classifier = make_classifier(self.embedding_dim, num_classes)
        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if features.ndim > 2:
            features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)
        return features

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def replace_classifier(self, num_classes: int) -> None:
        self.classifier = make_classifier(self.embedding_dim, num_classes).to(next(self.parameters()).device)
        self.num_classes = num_classes

    def gradcam_target_layer(self) -> nn.Module | None:
        for path in [
            ("stages", -1, "blocks", -1),
            ("stages", -1),
        ]:
            try:
                module: Any = self.encoder
                for part in path:
                    module = module[part] if isinstance(part, int) else getattr(module, part)
                return module
            except Exception:
                continue
        return None


def build_convnext(model_name: str, num_classes: int, logger: logging.Logger | None = None) -> ConvNeXtRSCModel:
    if logger:
        logger.info("Building ConvNeXt backbone: %s", model_name)
    return ConvNeXtRSCModel(model_name=model_name, num_classes=num_classes, logger=logger)
