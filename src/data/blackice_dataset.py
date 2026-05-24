from __future__ import annotations

from pathlib import Path
from typing import Callable, List

from PIL import Image
from torch.utils.data import Dataset

from src.data.class_mapping import BLACK_ICE_CLASS_NAME
from src.data.datasets import ImageRecord
from src.utils.io import list_image_files


def validate_blackice_dir(blackice_dir: str | Path) -> None:
    root = Path(blackice_dir)
    if not root.exists():
        raise FileNotFoundError(f"Black-ice directory is missing: {root}")
    if not list_image_files(root):
        raise RuntimeError(f"No black-ice images were found in: {root}")


def blackice_records(blackice_dir: str | Path, split: str = "adapt") -> List[ImageRecord]:
    validate_blackice_dir(blackice_dir)
    return [ImageRecord(path, 5, BLACK_ICE_CLASS_NAME, split) for path in list_image_files(blackice_dir)]


class BlackIceDataset(Dataset):
    def __init__(self, blackice_dir: str | Path, transform: Callable | None = None) -> None:
        self.records = blackice_records(blackice_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, record.label_id, str(record.path), index
