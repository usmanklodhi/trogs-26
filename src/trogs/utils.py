"""Small helpers shared across training and inference."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_device(device: str = "auto") -> str:
    """Map ``"auto"`` to ``"cuda:0"`` when a GPU is available, else ``"cpu"``."""
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy and torch RNGs. ``None`` leaves them unseeded (thesis behaviour)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def setup_logging(log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=handlers, force=True)
