from __future__ import annotations

import logging

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


DINO_FALLBACK = "vit_base_patch16_224"


class DINORSCModel(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m", num_classes: int = 5, projection_dim: int = 128, logger: logging.Logger | None = None) -> None:
        super().__init__()
        if timm is None:
            raise ImportError(f"timm is required for DINO/ViT models: {_TIMM_IMPORT_ERROR}")
        selected_name = model_name
        try:
            self.encoder = timm.create_model(selected_name, pretrained=False, num_classes=0, global_pool="token")
        except Exception as exc:
            if logger:
                logger.warning("DINO model '%s' is unavailable in this timm installation. Falling back to '%s'. Error: %s", model_name, DINO_FALLBACK, exc)
            selected_name = DINO_FALLBACK
            self.encoder = timm.create_model(selected_name, pretrained=False, num_classes=0, global_pool="token")
        self.model_name = selected_name
        self.requested_model_name = model_name
        self.embedding_dim = int(getattr(self.encoder, "num_features", 768))
        self.projection_head = ProjectionHead(self.embedding_dim, projection_dim=projection_dim)
        self.classifier = make_classifier(self.embedding_dim, num_classes)
        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if features.ndim == 3:
            features = features[:, 0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def replace_classifier(self, num_classes: int) -> None:
        self.classifier = make_classifier(self.embedding_dim, num_classes).to(next(self.parameters()).device)
        self.num_classes = num_classes

    def gradcam_target_layer(self) -> nn.Module | None:
        blocks = getattr(self.encoder, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            return blocks[-1]
        return None


def build_dino(model_name: str, num_classes: int, logger: logging.Logger | None = None) -> DINORSCModel:
    if logger:
        logger.info("Building DINO/ViT backbone: %s", model_name)
    return DINORSCModel(model_name=model_name, num_classes=num_classes, logger=logger)
