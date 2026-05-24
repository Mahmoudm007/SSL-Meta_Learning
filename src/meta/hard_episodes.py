from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from src.data.class_mapping import ClassMapping
from src.meta.prototypical import train_prototypical_network
from src.utils.io import write_dataframe


def train_hard_prototypical_network(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    class_mapping: ClassMapping,
    output_root: str | Path,
    args,
    device: torch.device,
    logger: logging.Logger,
):
    result = train_prototypical_network(
        model,
        train_dataset,
        val_dataset,
        class_mapping,
        output_root,
        args,
        device,
        logger,
        hard_probability=max(args.hard_episode_probability, 0.5),
        phase_name="hard_prototypical",
    )
    predictions = result.get("episode_predictions", pd.DataFrame())
    if len(predictions):
        hard = predictions[predictions["hard_episode"].astype(bool)]
        normal = predictions[~predictions["hard_episode"].astype(bool)]
        summary = pd.DataFrame(
            [
                {"subset": "hard", "accuracy": hard["correct"].mean() if len(hard) else 0.0},
                {"subset": "normal", "accuracy": normal["correct"].mean() if len(normal) else 0.0},
            ]
        )
        write_dataframe(Path(output_root) / "metrics" / "hard_query_subset_metrics.csv", summary)
        partly = predictions[predictions["true_label_name"].str.contains("Partly|Fully", regex=True)]
        write_dataframe(Path(output_root) / "metrics" / "hard_partly_query_predictions.csv", partly)
        one_track_fn = predictions[(predictions["true_label_id"] == 3) & (predictions["pred_label_id"] != 3)]
        write_dataframe(Path(output_root) / "predictions" / "one_track_false_negatives.csv", one_track_fn)
    return result
