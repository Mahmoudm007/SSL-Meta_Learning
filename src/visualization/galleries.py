from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.utils.io import ensure_dir


def save_error_gallery(predictions: pd.DataFrame, output_path: str | Path, title: str, max_images: int = 12) -> None:
    target = Path(output_path)
    ensure_dir(target.parent)
    rows = predictions.head(max_images)
    cols = 4
    rows_count = max(1, (len(rows) + cols - 1) // cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.2, rows_count * 3))
    axes_flat = list(getattr(axes, "flat", [axes]))
    for ax in axes_flat:
        ax.axis("off")
    if len(rows) == 0:
        axes_flat[0].text(0.5, 0.5, "No examples", ha="center", va="center")
    for ax, (_, row) in zip(axes_flat, rows.iterrows()):
        try:
            ax.imshow(Image.open(row["image_path"]).convert("RGB"))
            ax.set_title(f"T:{row['true_label_name']}\nP:{row['pred_label_name']}\nconf={float(row['confidence']):.2f}", fontsize=7)
        except Exception:
            ax.text(0.5, 0.5, "read error", ha="center", va="center")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)
