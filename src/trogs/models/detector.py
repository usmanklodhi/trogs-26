"""YOLO-OBB letter detector, run with SAHI sliced inference on full-size squeezes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SliceParams:
    slice_size: int = 640
    overlap_ratio: float = 0.2


def load_detector(weights: str, confidence: float, device: str):
    from sahi import AutoDetectionModel

    # SAHI's 'yolov8' wrapper also handles Ultralytics YOLO11 OBB weights.
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weights,
        confidence_threshold=confidence,
        device=device,
    )


def detect(image_path: str, detector, params: SliceParams | None = None, verbose: int = 1):
    """Sliced prediction over a full image; returns SAHI ``ObjectPrediction`` objects."""
    from sahi.predict import get_sliced_prediction

    params = params or SliceParams()

    result = get_sliced_prediction(
        image_path,
        detector,
        slice_height=params.slice_size,
        slice_width=params.slice_size,
        overlap_height_ratio=params.overlap_ratio,
        overlap_width_ratio=params.overlap_ratio,
        postprocess_type="NMS",
        verbose=verbose,
    )
    return result.object_prediction_list


def prediction_corners(pred) -> np.ndarray:
    """Oriented corners of a prediction (from its mask polygon), else its axis-aligned box."""
    if hasattr(pred, "mask") and pred.mask is not None and hasattr(pred.mask, "polygon"):
        return np.array(pred.mask.polygon[0]).reshape(-1, 2).astype("float32")
    x1, y1, x2, y2 = map(float, pred.bbox.to_xyxy())
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
