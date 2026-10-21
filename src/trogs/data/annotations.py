"""Parser for the Howe et al. squeeze annotation files (``*_letters.txt``).

The files are split into sections introduced by ``#`` header lines::

    # Angle      one float (global text angle)
    # Boxes      one letter box per line: x1 y1 x2 y2 x3 y3 x4 y4
    # Transcript the text, one inscription line per row; '*' marks a letter without a box
    # Lines      optional baseline segments

Boxes are listed in transcript order, so the n-th box belongs to the n-th
transcript character that is not a '*'.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Characters that never become a letter label.
_LETTER_SKIP = ("*", " ", "\n", "\r", "\t")


@dataclass
class TextLine:
    text: str  # transcript of the line, '*' placeholders kept
    points: np.ndarray  # (4 * n_boxes, 2) float32 corners of all boxes in the line


def _read_lines(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def _parse_boxes(lines: list[str]) -> list[np.ndarray]:
    """Return the ``# Boxes`` section as float64 arrays of shape (4, 2)."""
    boxes: list[np.ndarray] = []
    in_boxes = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if in_boxes:
                break
            in_boxes = stripped.startswith("# Boxes")
            continue
        if not in_boxes:
            continue
        parts = stripped.split()
        if len(parts) != 8:
            continue
        try:
            coords = [float(p) for p in parts]
        except ValueError:
            continue
        boxes.append(np.array(coords, dtype=np.float64).reshape(4, 2))
    return boxes


def _transcript_rows(lines: list[str]) -> list[str]:
    """Return the raw (unstripped) rows of the ``# Transcript`` section."""
    for i, line in enumerate(lines):
        if line.strip().startswith("# Transcript"):
            rows = []
            for row in lines[i + 1 :]:
                if row.strip().startswith("# Lines"):
                    break
                rows.append(row)
            return rows
    return []


def parse_boxes(path: str | Path) -> list[np.ndarray]:
    """All letter boxes of a file, as float64 (4, 2) corner arrays. Missing file -> []."""
    if not Path(path).exists():
        return []
    return _parse_boxes(_read_lines(path))


def parse_letters(path: str | Path) -> tuple[list[np.ndarray], list[str]]:
    """Pair every box with its letter (used for the glyph classification dataset).

    '*' and whitespace are dropped from the transcript, then both lists are
    truncated to the shorter one. Boxes are float32 (4, 2) arrays.
    """
    if not Path(path).exists():
        return [], []
    lines = _read_lines(path)
    boxes = [b.astype(np.float32) for b in _parse_boxes(lines)]
    labels = [c for row in _transcript_rows(lines) for c in row if c not in _LETTER_SKIP]
    n = min(len(boxes), len(labels))
    return boxes[:n], labels[:n]


def parse_text_lines(path: str | Path) -> list[TextLine]:
    """Group boxes into inscription lines (used for the CRNN line dataset).

    Each transcript row consumes as many boxes as it has non-'*' characters;
    the '*' placeholders stay in the text label.
    """
    if not Path(path).exists():
        return []
    lines = _read_lines(path)
    boxes = [b.astype(np.float32) for b in _parse_boxes(lines)]

    result: list[TextLine] = []
    box_index = 0
    for row in _transcript_rows(lines):
        text = "".join(c for c in row.strip() if c not in (" ", "\t"))
        n_boxed = sum(1 for c in text if c != "*")
        if n_boxed == 0:
            continue
        line_boxes = boxes[box_index : box_index + n_boxed]
        box_index += n_boxed
        if not line_boxes:
            continue
        result.append(TextLine(text=text, points=np.vstack(line_boxes)))
    return result
