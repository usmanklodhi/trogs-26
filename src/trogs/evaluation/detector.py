"""Validate the YOLO-OBB detector and save predictions for a random sample of tiles."""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass


@dataclass
class DetectorEvalConfig:
    weights: str = "runs/obb/train3/weights/best.pt"
    data: str = "configs/yolo_dataset.yaml"
    imgsz: int = 640
    batch: int = 16
    device: str = "cpu"
    # Sample predictions
    source_dir: str = "data/processed/greek_letters_obb/images/val"
    num_samples: int = 10
    conf: float = 0.25


def validate(cfg: DetectorEvalConfig):
    from ultralytics import YOLO

    from trogs.training.detector import resolve_data_yaml

    metrics = YOLO(cfg.weights).val(
        data=resolve_data_yaml(cfg.data), split="val", imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device
    )
    print(f"mAP50 (Oriented): {metrics.box.map50}")
    print(f"mAP50-95 (Oriented): {metrics.box.map}")
    return metrics


def predict_samples(cfg: DetectorEvalConfig):
    """Run the detector on ``num_samples`` random images and save the annotated results."""
    from ultralytics import YOLO

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(glob.glob(os.path.join(cfg.source_dir, ext)))
    if len(images) > cfg.num_samples:
        images = random.sample(images, cfg.num_samples)
    else:
        print(f"Warning: only found {len(images)} images, using all of them.")

    print(f"Running inference on {len(images)} images...")
    results = YOLO(cfg.weights).predict(
        source=images, save=True, imgsz=cfg.imgsz, conf=cfg.conf, line_width=2
    )
    print("Inference complete. Results are under runs/obb/predict*/")
    return results
