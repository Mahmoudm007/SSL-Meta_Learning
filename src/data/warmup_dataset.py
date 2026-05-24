from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from src.utils.io import list_image_files


def validate_warmup_dir(warmup_dir: str | Path) -> None:
    root = Path(warmup_dir)
    if not root.exists():
        raise FileNotFoundError(f"Warm-Up SSL image directory is missing: {root}")
    image_paths = list_image_files(root)
    if not image_paths:
        raise RuntimeError(f"No Warm-Up images were found in: {root}")


class WarmupPairDataset(Dataset):
    def __init__(self, warmup_dir: str | Path, transform: Callable, max_samples: int | None = None) -> None:
        validate_warmup_dir(warmup_dir)
        image_paths = list_image_files(warmup_dir)
        self.image_paths = image_paths[:max_samples] if max_samples else image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            view_a = self.transform(image)
            view_b = self.transform(image)
        return view_a, view_b, str(path)


class WarmupImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform: Callable | None = None) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, -1, str(path), index
