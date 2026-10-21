"""Finding squeeze images and annotation files on disk.

File names follow ``<ID>_Rotation{1,2}_300dpi.<ext>`` for images and
``<ID>_Rotation{1,2}_300dpi_letters.txt`` for annotations.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

ROTATION1_SUFFIX = "_Rotation1_300dpi"
ROTATION2_SUFFIX = "_Rotation2_300dpi"
ANNOTATION_SUFFIX = "_letters.txt"


def annotation_files(ann_dir: str, pattern: str) -> list[str]:
    """Annotation files matching ``pattern`` (in ``glob`` order, which the splits depend on)."""
    return glob.glob(os.path.join(ann_dir, pattern))


def image_for_annotation(ann_path: str, img_dir: str) -> tuple[str, str]:
    """Return ``(base_name, image_path)`` for an annotation file."""
    base_name = os.path.basename(ann_path).replace(ANNOTATION_SUFFIX, "")
    return base_name, os.path.join(img_dir, base_name + ".png")


@dataclass
class SqueezePair:
    base: str  # squeeze id, e.g. "IAS_27288"
    rotation1: str
    rotation2: str | None


def find_squeeze_pairs(img_dir: str, extensions: tuple[str, ...] = ("png", "tif")) -> list[SqueezePair]:
    """Pair every Rotation1 image with its Rotation2 counterpart, if one exists.

    Rotation2 is looked up as PNG first, then TIF, regardless of the Rotation1 extension.
    """
    pairs = []
    for name in os.listdir(img_dir):
        for ext in extensions:
            suffix = f"{ROTATION1_SUFFIX}.{ext}"
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        else:
            continue
        rotation2 = None
        for ext in extensions:
            candidate = os.path.join(img_dir, f"{base}{ROTATION2_SUFFIX}.{ext}")
            if os.path.exists(candidate):
                rotation2 = candidate
                break
        pairs.append(SqueezePair(base, os.path.join(img_dir, name), rotation2))
    return pairs


def find_images(img_dir: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> list[str]:
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(extensions)]
