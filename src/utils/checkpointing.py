from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from src.utils.io import ensure_dir


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, **metadata: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    payload: Dict[str, Any] = {"model_state": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    state = payload.get("model_state", payload)
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and tuple(current[key].shape) == tuple(value.shape)}
    model.load_state_dict(compatible, strict=False)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("metadata", {})
