from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.io import ensure_dir


def setup_logger(name: str, log_dir: str | Path, filename: str = "train.log") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(log_dir) / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    error_handler = logging.FileHandler(Path(log_dir) / "errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    return logger


def get_phase_logger(base_name: str, log_dir: str | Path, phase: str) -> logging.Logger:
    return setup_logger(f"{base_name}.{phase}", log_dir, f"{phase}.log")
