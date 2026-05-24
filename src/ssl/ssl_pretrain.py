from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations.rsc_augmentations import get_ssl_transform
from src.data.warmup_dataset import WarmupPairDataset
from src.ssl.contrastive_losses import alignment, nt_xent_loss, uniformity
from src.ssl.ssl_utils import should_skip_ssl
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.io import append_dataframe, ensure_dir
from src.utils.reproducibility import gpu_memory_mb


def train_ssl_encoder(model: torch.nn.Module, warmup_dir: str, output_root: str | Path, args, device: torch.device, logger: logging.Logger) -> None:
    checkpoint_path = Path(output_root) / "checkpoints" / "ssl_encoder.pt"
    if should_skip_ssl(checkpoint_path, args.skip_ssl_if_checkpoint_exists, args.force_rerun):
        logger.info("Skipping SSL pretraining because checkpoint exists: %s", checkpoint_path)
        load_checkpoint(checkpoint_path, model, map_location=device)
        return
    ssl_logger = logging.getLogger(logger.name + ".ssl")
    transform = get_ssl_transform(args.image_size, args.augmentation_strength)
    dataset = WarmupPairDataset(warmup_dir, transform=transform, max_samples=args.max_warmup_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    model.train()
    ensure_dir(Path(output_root) / "epoch_progress")
    epochs = max(int(args.epochs_ssl), 0)
    if epochs == 0:
        logger.info("epochs_ssl=0; saving current encoder as SSL checkpoint without optimization.")
        save_checkpoint(checkpoint_path, model, optimizer, phase="ssl", epochs=0)
        return
    for epoch in range(1, epochs + 1):
        start = time.time()
        losses = []
        feature_norms = []
        alignments = []
        uniforms = []
        progress = tqdm(loader, desc=f"SSL epoch {epoch}/{epochs}", leave=False)
        for view_a, view_b, _ in progress:
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z1 = model.forward_projection(view_a)
            z2 = model.forward_projection(view_b)
            loss = nt_xent_loss(z1, z2, temperature=args.ssl_temperature)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            with torch.no_grad():
                features = torch.cat([z1, z2], dim=0)
                feature_norms.append(float(features.norm(dim=1).mean().item()))
                alignments.append(alignment(z1, z2))
                uniforms.append(uniformity(features))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
        allocated, reserved = gpu_memory_mb()
        row = {
            "epoch": epoch,
            "ssl_train_loss": float(pd.Series(losses).mean()) if losses else 0.0,
            "ssl_val_loss": "",
            "contrastive_loss": float(pd.Series(losses).mean()) if losses else 0.0,
            "reconstruction_loss": "",
            "feature_norm_mean": float(pd.Series(feature_norms).mean()) if feature_norms else 0.0,
            "feature_norm_std": float(pd.Series(feature_norms).std()) if len(feature_norms) > 1 else 0.0,
            "embedding_alignment": float(pd.Series(alignments).mean()) if alignments else 0.0,
            "embedding_uniformity": float(pd.Series(uniforms).mean()) if uniforms else 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": time.time() - start,
            "gpu_memory_allocated_mb": allocated,
            "gpu_memory_reserved_mb": reserved,
        }
        append_dataframe(Path(output_root) / "epoch_progress" / "ssl_epoch_metrics.csv", row)
        append_dataframe(Path(output_root) / "epoch_progress" / "learning_rate_schedule.csv", {"epoch": epoch, "phase": "ssl", "learning_rate": optimizer.param_groups[0]["lr"]})
        append_dataframe(Path(output_root) / "epoch_progress" / "gpu_memory_log.csv", {"epoch": epoch, "phase": "ssl", "allocated_mb": allocated, "reserved_mb": reserved})
        append_dataframe(Path(output_root) / "epoch_progress" / "epoch_time_log.csv", {"epoch": epoch, "phase": "ssl", "epoch_time_seconds": row["epoch_time_seconds"]})
        ssl_logger.info("SSL epoch %s loss %.5f", epoch, row["ssl_train_loss"])
        save_checkpoint(checkpoint_path, model, optimizer, phase="ssl", epoch=epoch)
