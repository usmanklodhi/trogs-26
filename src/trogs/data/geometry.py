"""Crop and rectification helpers for oriented letter and line boxes."""

from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _upright(width: float, height: float) -> np.ndarray:
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")


def warp_box(img: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Rectify an oriented box to its own size, then resize to ``size`` x ``size`` (stretches)."""
    rect = order_points(box)
    tl, tr, br, bl = rect
    max_width = max(int(np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)),
                    int(np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)))
    max_height = max(int(np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)),
                     int(np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)))
    m = cv2.getPerspectiveTransform(rect, _upright(max_width, max_height))
    warped = cv2.warpPerspective(img, m, (max_width, max_height))
    return cv2.resize(warped, (size, size))


def warp_box_square(img: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving crop: grow the box to a square around its centre, then warp to ``size``.

    Regions outside the image are filled with black.
    """
    rect = order_points(box)
    tl, tr, br, bl = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_side = max(int(max(width_a, width_b)), int(max(height_a, height_b)))

    center = np.mean(rect, axis=0)
    vec_w = tr - tl
    unit_w = vec_w / (np.linalg.norm(vec_w) + 1e-6)
    vec_h = bl - tl
    unit_h = vec_h / (np.linalg.norm(vec_h) + 1e-6)

    half = max_side / 2.0
    src = np.array([
        center - (unit_w * half) - (unit_h * half),
        center + (unit_w * half) - (unit_h * half),
        center + (unit_w * half) + (unit_h * half),
        center - (unit_w * half) + (unit_h * half),
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(src, _upright(size, size))
    return cv2.warpPerspective(img, m, (size, size), borderValue=(0, 0, 0))


def line_rect(points: np.ndarray) -> tuple[tuple[float, float], float, float, float]:
    """Minimum-area rectangle around all points of a line, oriented so width >= height.

    Returns ``(center, width, height, angle)``.
    """
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    if h > w:
        w, h = h, w
        angle -= 90.0
    return (cx, cy), w, h, angle


def warp_line(img: np.ndarray, center, width: float, height: float, angle: float,
              target_h: int = 32, border_value=0) -> np.ndarray:
    """Rectify a rotated line rectangle to height ``target_h``, keeping its aspect ratio."""
    src = order_points(cv2.boxPoints((center, (width, height), angle)))
    target_w = max(1, int(width * (target_h / max(1, height))))
    m = cv2.getPerspectiveTransform(src, _upright(target_w, target_h))
    return cv2.warpPerspective(img, m, (target_w, target_h), borderValue=border_value)
