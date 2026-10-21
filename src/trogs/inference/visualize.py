"""Debug overlays for pipeline outputs."""

from __future__ import annotations

import cv2
import numpy as np


def draw_letters(image_path: str, boxes: list[dict], output_path: str) -> None:
    """Draw classified letter boxes with ``label [C:<cls conf> D:<det conf>]`` tags."""
    img = cv2.imread(image_path)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{b['label']} [C:{b['conf']:.2f} D:{b.get('det_conf', 0.0):.2f}]"
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        # Solid background keeps the tag readable over the stone texture.
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1, y1 - 5), font, scale, (0, 0, 0), thickness)
    cv2.imwrite(output_path, img)


def draw_lines(img: np.ndarray, lines: list[list[dict]], transcript: list[str], output_path: str) -> None:
    """Draw each detected line (chain, boxes and recognised text) in its own colour."""
    vis = img.copy()
    rng = np.random.RandomState(42)
    for i, line in enumerate(lines):
        color = tuple(rng.randint(50, 255, 3).tolist())
        for a, b in zip(line, line[1:], strict=False):
            cv2.line(vis, (int(a["cx"]), int(a["cy"])), (int(b["cx"]), int(b["cy"])), color, 3)
        for box in line:
            cv2.polylines(vis, [np.int32(box["pts"])], True, color, 2)
        cv2.putText(
            vis,
            f"L{i + 1}: {transcript[i]}",
            (int(line[0]["cx"]) - 50, int(line[0]["cy"]) - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )
    cv2.imwrite(output_path, vis)
