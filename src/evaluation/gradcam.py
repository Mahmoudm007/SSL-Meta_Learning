from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.augmentations.rsc_augmentations import IMAGENET_MEAN, IMAGENET_STD
from src.data.class_mapping import ClassMapping, sanitize_class_name
from src.utils.io import ensure_dir, write_dataframe
from src.visualization.gradcam_layout import save_interpretability_panel


def generate_interpretability_outputs(
    model: torch.nn.Module,
    predictions: pd.DataFrame,
    output_root: str | Path,
    class_mapping: ClassMapping,
    experiment: str,
    model_key: str,
    image_size: int,
    device: torch.device,
    logger: logging.Logger,
    max_images: int = 0,
    include_black_ice: bool = False,
    primary_subdir: str = "all_validation",
) -> None:
    root = Path(output_root) / "gradcam"
    ensure_dir(root)
    if predictions.empty:
        write_dataframe(root / "gradcam_metadata.csv", pd.DataFrame())
        return
    rows = predictions if max_images in (None, 0) else predictions.head(max_images)
    metadata_rows = []
    for row_index, row in tqdm(rows.iterrows(), total=len(rows), desc=f"{model_key} visualizations", leave=False, ascii=True):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            logger.error("Grad-CAM image path is missing: %s", image_path)
            continue
        try:
            with Image.open(image_path) as image:
                original = image.convert("RGB")
            heatmap, method = _compute_heatmap(model, original, int(row["pred_label_id"]), image_size, device, model_key)
            filename = f"{row_index:06d}_{image_path.stem}.png"
            destination = root / primary_subdir / filename
            metadata = _metadata_from_row(row, class_mapping, experiment, model_key, method)
            save_interpretability_panel(original.resize((heatmap.shape[1], heatmap.shape[0])), heatmap, destination, metadata)
            _save_subset_copy(original, heatmap, root, filename, metadata, row)
            metadata["image_path"] = str(image_path)
            metadata["saved_gradcam_path"] = str(destination)
            metadata_rows.append(metadata)
        except Exception as exc:
            logger.error("Failed to generate Grad-CAM/attention figure for %s: %s", image_path, exc)
    metadata_path = root / "gradcam_metadata.csv"
    metadata_frame = pd.DataFrame(metadata_rows)
    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(metadata_path)
            metadata_frame = pd.concat([existing, metadata_frame], ignore_index=True)
        except Exception:
            pass
    write_dataframe(metadata_path, metadata_frame)


def _metadata_from_row(row: pd.Series, class_mapping: ClassMapping, experiment: str, model_key: str, method: str) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "experiment": experiment,
        "model": model_key,
        "true_label_id": int(row["true_label_id"]),
        "true_label_name": row["true_label_name"],
        "pred_label_id": int(row["pred_label_id"]),
        "pred_label_name": row["pred_label_name"],
        "correct": bool(row["correct"]),
        "confidence": float(row["confidence"]),
        "method": method,
    }
    for class_name in class_mapping.class_names:
        probability_name = f"p_{sanitize_class_name(class_name)}"
        metadata[probability_name] = float(row.get(probability_name, 0.0))
    return metadata


def _save_subset_copy(original: Image.Image, heatmap: np.ndarray, root: Path, filename: str, metadata: Dict[str, object], row: pd.Series) -> None:
    correctness_dir = "correct" if bool(row["correct"]) else "incorrect"
    save_interpretability_panel(original.resize((heatmap.shape[1], heatmap.shape[0])), heatmap, root / correctness_dir / filename, metadata)
    true_dir = root / "by_true_class" / sanitize_class_name(str(row["true_label_name"]))
    pred_dir = root / "by_predicted_class" / sanitize_class_name(str(row["pred_label_name"]))
    save_interpretability_panel(original.resize((heatmap.shape[1], heatmap.shape[0])), heatmap, true_dir / filename, metadata)
    save_interpretability_panel(original.resize((heatmap.shape[1], heatmap.shape[0])), heatmap, pred_dir / filename, metadata)
    if int(row["true_label_id"]) == 3 and int(row["pred_label_id"]) != 3:
        save_interpretability_panel(original.resize((heatmap.shape[1], heatmap.shape[0])), heatmap, root / "one_track_false_negatives" / filename, metadata)
    if int(row["true_label_id"]) == 5 or int(row["pred_label_id"]) == 5:
        save_interpretability_panel(original.resize((heatmap.shape[1], heatmap.shape[0])), heatmap, root / "blackice" / filename, metadata)


def _compute_heatmap(
    model: torch.nn.Module,
    original: Image.Image,
    pred_label: int,
    image_size: int,
    device: torch.device,
    model_key: str,
) -> tuple[np.ndarray, str]:
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tensor = transform(original).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    if model_key == "convnext":
        try:
            heatmap = _gradcam_feature_heatmap(model, tensor, pred_label)
            return _resize_heatmap(heatmap, original.size[::-1]), "convnext_gradcam"
        except Exception:
            pass
    heatmap = _input_gradient_heatmap(model, tensor, pred_label)
    method = "dino_input_gradient_saliency" if model_key == "dino" else "input_gradient_saliency_fallback"
    return _resize_heatmap(heatmap, original.size[::-1]), method


def _gradcam_feature_heatmap(model: torch.nn.Module, tensor: torch.Tensor, pred_label: int) -> np.ndarray:
    target_layer = model.gradcam_target_layer()
    if target_layer is None:
        raise RuntimeError("Model does not expose a Grad-CAM target layer")
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        score = logits[:, pred_label].sum()
        score.backward()
        activation = activations[-1]
        gradient = gradients[-1]
        if activation.ndim == 3:
            side = int(np.sqrt(activation.shape[1]))
            activation = activation[:, : side * side, :].transpose(1, 2).reshape(activation.shape[0], activation.shape[2], side, side)
            gradient = gradient[:, : side * side, :].transpose(1, 2).reshape(gradient.shape[0], gradient.shape[2], side, side)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()
        return _normalize(cam)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _input_gradient_heatmap(model: torch.nn.Module, tensor: torch.Tensor, pred_label: int) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    score = logits[:, pred_label].sum()
    score.backward()
    saliency = tensor.grad.detach().abs().max(dim=1).values.squeeze().cpu().numpy()
    return _normalize(saliency)


def _normalize(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array)
    if array.max() <= array.min():
        return np.zeros_like(array, dtype=float)
    return (array - array.min()) / (array.max() - array.min())


def _resize_heatmap(heatmap: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.uint8(_normalize(heatmap) * 255))
    image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image).astype(float) / 255.0
