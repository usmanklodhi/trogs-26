"""Train the YOLO11-OBB letter detector on the tiled dataset."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass

import yaml


@dataclass
class DetectorTrainConfig:
    model: str = "yolo11n-obb.pt"
    data: str = "configs/yolo_dataset.yaml"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    # Geometric augmentation; flips are off because letters are not symmetric.
    degrees: float = 15.0
    fliplr: float = 0.0
    flipud: float = 0.0
    scale: float = 0.5
    shear: float = 2.0
    # Photometric augmentation; strong brightness shift for light/dark stone.
    hsv_h: float = 0.015
    hsv_s: float = 0.2
    hsv_v: float = 0.5
    amp: bool = False


def resolve_data_yaml(path: str) -> str:
    """Ultralytics resolves a relative dataset ``path`` against its own datasets dir,
    so write a copy of the data file with ``path`` made absolute (relative to the cwd)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "path" not in data or os.path.isabs(data["path"]):
        return path
    data["path"] = os.path.abspath(data["path"])
    fd, resolved = tempfile.mkstemp(suffix=".yaml", prefix="yolo_dataset_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return resolved


def train(cfg: DetectorTrainConfig):
    from ultralytics import YOLO

    params = asdict(cfg)
    model = YOLO(params.pop("model"))
    params["data"] = resolve_data_yaml(params["data"])
    return model.train(**params)
