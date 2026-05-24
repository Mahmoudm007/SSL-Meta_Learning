from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.class_mapping import ClassMapping, RSC_CLASS_NAMES, default_class_mapping
from src.utils.io import IMAGE_EXTENSIONS, list_image_files, write_dataframe


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label_id: int
    label_name: str
    split: str


def validate_rsc_structure(data_dir: str | Path, class_names: Sequence[str] | None = None) -> None:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"RSC data directory is missing: {root}")
    required_splits = ["train", "val"]
    names = list(class_names or RSC_CLASS_NAMES)
    missing: List[str] = []
    for split in required_splits:
        split_path = root / split
        if not split_path.exists():
            missing.append(str(split_path))
            continue
        for class_name in names:
            class_path = split_path / class_name
            if not class_path.exists():
                missing.append(str(class_path))
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Required RSC folders are missing:\n  - {joined}")


def scan_rsc_dataset(data_dir: str | Path, class_mapping: ClassMapping | None = None) -> List[ImageRecord]:
    mapping = class_mapping or default_class_mapping()
    validate_rsc_structure(data_dir, mapping.class_names)
    root = Path(data_dir)
    records: List[ImageRecord] = []
    for split in ["train", "val"]:
        for label_id, label_name in enumerate(mapping.class_names):
            class_dir = root / split / label_name
            for image_path in list_image_files(class_dir):
                records.append(ImageRecord(image_path, label_id, label_name, split))
    if not records:
        raise RuntimeError(f"No images found under RSC data directory: {root}")
    return records


def distribution_dataframe(records: Iterable[ImageRecord]) -> pd.DataFrame:
    rows = [
        {"split": record.split, "label_id": record.label_id, "label_name": record.label_name, "count": 1}
        for record in records
    ]
    if not rows:
        return pd.DataFrame(columns=["split", "label_id", "label_name", "count"])
    return (
        pd.DataFrame(rows)
        .groupby(["split", "label_id", "label_name"], as_index=False)["count"]
        .sum()
        .sort_values(["split", "label_id"])
    )


def dataset_summary_dataframe(records: Iterable[ImageRecord]) -> pd.DataFrame:
    frame = distribution_dataframe(records)
    totals = frame.groupby("split", as_index=False)["count"].sum().rename(columns={"count": "split_total"})
    frame = frame.merge(totals, on="split", how="left")
    frame["percentage"] = frame["count"] / frame["split_total"].clip(lower=1) * 100.0
    return frame


def save_dataset_summaries(records: List[ImageRecord], output_root: str | Path) -> None:
    root = Path(output_root) / "configs"
    summary = dataset_summary_dataframe(records)
    write_dataframe(root / "dataset_summary.csv", summary)
    for split in ["train", "val"]:
        write_dataframe(root / f"{split}_distribution.csv", summary[summary["split"] == split])


class RSCImageDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable | None = None,
        include_classes: Sequence[int] | None = None,
        exclude_classes: Sequence[int] | None = None,
    ) -> None:
        allowed = set(include_classes) if include_classes is not None else None
        excluded = set(exclude_classes or [])
        self.records = [
            record
            for record in records
            if (allowed is None or record.label_id in allowed) and record.label_id not in excluded
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, record.label_id, str(record.path), index

    def labels(self) -> List[int]:
        return [record.label_id for record in self.records]

    def paths(self) -> List[str]:
        return [str(record.path) for record in self.records]


def split_records(records: Sequence[ImageRecord], split: str) -> List[ImageRecord]:
    return [record for record in records if record.split == split]


def count_images_by_folder(root: str | Path) -> Dict[str, int]:
    target = Path(root)
    if not target.exists():
        return {}
    counts: Dict[str, int] = {}
    for child in sorted(path for path in target.iterdir() if path.is_dir()):
        counts[child.name] = sum(1 for path in child.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return counts
