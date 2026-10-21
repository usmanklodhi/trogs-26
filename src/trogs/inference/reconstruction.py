"""Line reconstruction from letter boxes, following Howe et al.

Each box gets a tolerance ``r`` (half its diagonal). Box ``j`` is a horizontal
neighbour of box ``i`` when the centre of ``j`` lies within ``r(j)`` of the
horizontal line through the centre of ``i``; the nearest neighbour on each side
is kept. Rows are grown greedily from unassigned seed boxes by following these
links, then rows whose bounding boxes overlap strongly are merged.
"""

from __future__ import annotations

import math


def chain_rows(boxes: list[dict]) -> list[list[dict]]:
    """Group boxes into rows by following nearest left/right neighbours.

    Every box needs ``cx``, ``cy`` and ``r``; ``id`` (its index), ``left`` and
    ``right`` are written onto the dicts.
    """
    for i, box in enumerate(boxes):
        box["id"] = i
    for i, box_i in enumerate(boxes):
        left, right = [], []
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            if abs(box_j["cy"] - box_i["cy"]) <= box_j["r"]:
                if box_j["cx"] < box_i["cx"]:
                    left.append(box_j)
                elif box_j["cx"] > box_i["cx"]:
                    right.append(box_j)
        box_i["left"] = max(left, key=lambda b: b["cx"])["id"] if left else None
        box_i["right"] = min(right, key=lambda b: b["cx"])["id"] if right else None

    unassigned = set(range(len(boxes)))
    rows = []
    while unassigned:
        seed = unassigned.pop()
        row = [seed]
        for direction, insert_front in (("left", True), ("right", False)):
            current = seed
            while True:
                nxt = boxes[current][direction]
                if nxt is None or nxt not in unassigned:
                    break
                if insert_front:
                    row.insert(0, nxt)
                else:
                    row.append(nxt)
                unassigned.remove(nxt)
                current = nxt
        rows.append([boxes[i] for i in row])
    return rows


def sort_rows(rows: list[list[dict]]) -> list[list[dict]]:
    """Sort letters left-to-right within each row and rows top-to-bottom (in place)."""
    for row in rows:
        row.sort(key=lambda b: b["cx"])
    rows.sort(key=lambda row: sum(b["cy"] for b in row) / len(row))
    return rows


def row_bbox(row: list[dict]) -> tuple[float, float, float, float]:
    return (
        min(b["x1"] for b in row),
        min(b["y1"] for b in row),
        max(b["x2"] for b in row),
        max(b["y2"] for b in row),
    )


def row_ioa(row1: list[dict], row2: list[dict]) -> tuple[float, float]:
    """Intersection over area of each row's bounding box: ``(I / A1, I / A2)``."""
    r1_x1, r1_y1, r1_x2, r1_y2 = row_bbox(row1)
    r2_x1, r2_y1, r2_x2, r2_y2 = row_bbox(row2)
    area1 = max(0, r1_x2 - r1_x1) * max(0, r1_y2 - r1_y1)
    area2 = max(0, r2_x2 - r2_x1) * max(0, r2_y2 - r2_y1)
    if area1 == 0 or area2 == 0:
        return 0, 0
    inter_w = max(0, min(r1_x2, r2_x2) - max(r1_x1, r2_x1))
    inter_h = max(0, min(r1_y2, r2_y2) - max(r1_y1, r2_y1))
    inter = inter_w * inter_h
    return inter / area1, inter / area2


def merge_overlapping_rows(
    rows: list[list[dict]], mutual: float = 0.5, one_way: float = 0.8
) -> list[list[dict]]:
    """Merge rows with mutual IoA > ``mutual`` or one-way IoA > ``one_way``, until stable."""
    merged = True
    while merged:
        merged = False
        new_rows = []
        skip = set()
        for i in range(len(rows)):
            if i in skip:
                continue
            current = rows[i]
            for j in range(i + 1, len(rows)):
                if j in skip:
                    continue
                ioa1, ioa2 = row_ioa(current, rows[j])
                if (ioa1 > mutual and ioa2 > mutual) or ioa1 > one_way or ioa2 > one_way:
                    current.extend(rows[j])
                    skip.add(j)
                    merged = True
            new_rows.append(current)
        rows = new_rows
    return rows


def reconstruct_transcript(results: list[dict]) -> str:
    """Turn classified axis-aligned letter boxes (``x1, y1, x2, y2, label``) into text lines."""
    if not results:
        return ""
    for box in results:
        box["cx"] = (box["x1"] + box["x2"]) / 2.0
        box["cy"] = (box["y1"] + box["y2"]) / 2.0
        box["r"] = 0.5 * math.hypot(box["x2"] - box["x1"], box["y2"] - box["y1"])

    rows = merge_overlapping_rows(chain_rows(results))
    sort_rows(rows)
    return "\n".join("".join(b["label"] for b in row) for row in rows)
