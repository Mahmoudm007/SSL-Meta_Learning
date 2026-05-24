from __future__ import annotations

import io
import random
from typing import Callable

from PIL import Image, ImageFilter
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression:
    def __init__(self, probability: float = 0.25, quality: tuple[int, int] = (65, 95)) -> None:
        self.probability = probability
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomMildBlur:
    def __init__(self, probability: float = 0.15, radius: tuple[float, float] = (0.1, 0.8)) -> None:
        self.probability = probability
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        return image.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))


def _jitter(strength: str) -> transforms.ColorJitter:
    if strength == "light":
        return transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.01)
    if strength == "strong":
        return transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.12, hue=0.02)
    return transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.08, hue=0.015)


def _crop_scale(strength: str) -> tuple[float, float]:
    if strength == "light":
        return (0.88, 1.0)
    if strength == "strong":
        return (0.70, 1.0)
    return (0.78, 1.0)


def get_train_transform(image_size: int = 512, augmentation_strength: str = "medium") -> Callable:
    if augmentation_strength not in {"light", "medium", "strong"}:
        raise ValueError("augmentation_strength must be one of: light, medium, strong")
    rotation = {"light": 2, "medium": 4, "strong": 6}[augmentation_strength]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=_crop_scale(augmentation_strength), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([_jitter(augmentation_strength)], p=0.65),
            transforms.RandomRotation(degrees=rotation, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            RandomMildBlur(probability={"light": 0.05, "medium": 0.15, "strong": 0.25}[augmentation_strength]),
            RandomJPEGCompression(probability={"light": 0.05, "medium": 0.15, "strong": 0.25}[augmentation_strength]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_ssl_transform(image_size: int = 512, augmentation_strength: str = "medium") -> Callable:
    return get_train_transform(image_size=image_size, augmentation_strength=augmentation_strength)


def get_eval_transform(image_size: int = 512) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_visual_transform(image_size: int = 512) -> Callable:
    return transforms.Compose([transforms.Resize(int(image_size * 1.14)), transforms.CenterCrop(image_size)])
