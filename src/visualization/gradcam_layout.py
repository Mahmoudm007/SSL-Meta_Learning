from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils.io import ensure_dir


def save_interpretability_panel(
    original: Image.Image,
    heatmap: np.ndarray,
    output_path: str | Path,
    metadata: Dict[str, object],
) -> None:
    target = Path(output_path)
    ensure_dir(target.parent)
    original_rgb = original.convert("RGB")
    original_array = np.asarray(original_rgb).astype(float) / 255.0
    heatmap = np.clip(heatmap, 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    heatmap_rgb = cmap(heatmap)[..., :3]
    overlay = np.clip(0.55 * original_array + 0.45 * heatmap_rgb, 0.0, 1.0)
    fig = plt.figure(figsize=(13, 5.2), facecolor="white")
    axes = [fig.add_subplot(2, 3, index + 1) for index in range(3)]
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[1].imshow(heatmap, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title("Heatmap Only")
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay")
    for ax in axes:
        ax.axis("off")
    text_ax = fig.add_subplot(2, 1, 2)
    text_ax.axis("off")
    lines = [
        f"experiment={metadata.get('experiment')}    model={metadata.get('model')}",
        f"true={metadata.get('true_label_name')}    pred={metadata.get('pred_label_name')}    conf={float(metadata.get('confidence', 0.0)):.4f}",
    ]
    probability_items = [(key, value) for key, value in metadata.items() if str(key).startswith("p_")]
    if probability_items:
        lines.append("    ".join(f"{key}={float(value):.4f}" for key, value in probability_items))
    if metadata.get("method"):
        lines.append(f"visualization_method={metadata.get('method')}")
    text_ax.text(0.01, 0.70, "\n".join(lines), fontsize=9, va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
