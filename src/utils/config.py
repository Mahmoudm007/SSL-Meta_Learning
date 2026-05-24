from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml


def namespace_to_dict(args: Namespace) -> Dict[str, Any]:
    return {key: value for key, value in vars(args).items()}


def load_yaml_if_exists(path: str | Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a mapping: {target}")
    return payload


def merge_config(*configs: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_config(merged[key], value)
            else:
                merged[key] = value
    return merged
