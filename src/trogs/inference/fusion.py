"""Pathway 1 letter recognition: classify each detected letter, fusing both squeeze rotations.

A squeeze is scanned under two lighting rotations. Letters are detected in
both images; each Rotation1 box is matched to the nearest Rotation2 box
(centre distance below ``proximity_threshold`` px), and the classifier logits
of both crops are averaged before the softmax (late fusion). Unmatched boxes
are classified from Rotation1 alone.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from trogs.models.detector import SliceParams, detect

log = logging.getLogger(__name__)


def _box_xyxy(pred) -> tuple[int, int, int, int]:
    return tuple(map(int, pred.bbox.to_xyxy()))


def _safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]


def _to_tensor(crop: np.ndarray, transform, device) -> torch.Tensor:
    return transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)


def _nearest(pred, candidates) -> tuple[object | None, float]:
    x1, y1, x2, y2 = _box_xyxy(pred)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    best, best_dist = None, float("inf")
    for cand in candidates:
        a1, b1, a2, b2 = _box_xyxy(cand)
        dist = math.hypot(cx - (a1 + a2) / 2, cy - (b1 + b2) / 2)
        if dist < best_dist:
            best, best_dist = cand, dist
    return best, best_dist


def classify_letters(
    img1_path: str,
    img2_path: str | None,
    detector,
    classifier,
    transform,
    class_names: list[str],
    device,
    proximity_threshold: float = 50,
    slice_params: SliceParams | None = None,
) -> list[dict]:
    """Detect and classify every letter of a squeeze.

    Returns dicts with the Rotation1 box (``x1, y1, x2, y2``), ``label``,
    classifier confidence ``conf`` and detector confidence ``det_conf``.
    """
    slice_params = slice_params or SliceParams()
    log.info("Detecting letters in %s", img1_path)
    preds1 = detect(img1_path, detector, slice_params)
    img1 = cv2.imread(img1_path)

    if img2_path is None:
        log.info("No Rotation2 image, classifying a single image")
        preds2, img2 = [], None
    else:
        log.info("Detecting letters in %s", img2_path)
        preds2 = detect(img2_path, detector, slice_params)
        img2 = cv2.imread(img2_path)

    results = []
    for p1 in preds1:
        x1, y1, x2, y2 = _box_xyxy(p1)
        crop1 = _safe_crop(img1, x1, y1, x2, y2)
        p2, dist = _nearest(p1, preds2)

        if p2 is not None and dist < proximity_threshold:
            crop2 = _safe_crop(img2, *_box_xyxy(p2))
            if crop1.size == 0 or crop2.size == 0:
                continue
            with torch.no_grad():
                logits = (
                    classifier(_to_tensor(crop1, transform, device))
                    + classifier(_to_tensor(crop2, transform, device))
                ) / 2.0
                conf, idx = torch.max(F.softmax(logits, dim=1), 1)
            det_conf = (p1.score.value + p2.score.value) / 2.0
        else:
            if crop1.size == 0:
                continue
            with torch.no_grad():
                conf, idx = torch.max(F.softmax(classifier(_to_tensor(crop1, transform, device)), dim=1), 1)
            det_conf = p1.score.value

        results.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": class_names[idx.item()],
                "conf": conf.item(),
                "det_conf": det_conf,
            }
        )
    return results
