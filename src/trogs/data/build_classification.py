"""Build the glyph classification dataset: one rectified crop per annotated letter.

Output layout (``torchvision.datasets.ImageFolder``)::

    <output_dir>/{train,val}/<letter>/<squeeze>_<i>.jpg

Two crop modes are available:

* ``stretch`` - rectify the box and resize to ``img_size`` (original dataset)
* ``square``  - grow the box to a square first, preserving aspect ratio ("stratified" dataset)
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import dataclass

import cv2
from tqdm import tqdm

from trogs.data.annotations import parse_letters
from trogs.data.discovery import annotation_files, image_for_annotation
from trogs.data.geometry import warp_box, warp_box_square
from trogs.utils import seed_everything

log = logging.getLogger(__name__)

CROP_MODES = {"stretch": warp_box, "square": warp_box_square}


@dataclass
class ClassificationDataConfig:
    img_dir: str = "data/raw/Images"
    ann_dir: str = "data/raw/Annotations"
    output_dir: str = "data/processed/greek_letters_classification_stratified"
    crop_mode: str = "square"
    img_size: int = 128
    split_ratio: float = 0.8
    annotation_glob: str = "*_Rotation1_300dpi_letters.txt"
    clean_output: bool = True
    seed: int | None = None


def build(cfg: ClassificationDataConfig) -> None:
    if cfg.crop_mode not in CROP_MODES:
        raise ValueError(f"crop_mode must be one of {sorted(CROP_MODES)}, got {cfg.crop_mode!r}")
    crop_fn = CROP_MODES[cfg.crop_mode]
    seed_everything(cfg.seed)

    if cfg.clean_output and os.path.exists(cfg.output_dir):
        log.info("Removing old %s", cfg.output_dir)
        shutil.rmtree(cfg.output_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(cfg.output_dir, split), exist_ok=True)

    files = annotation_files(cfg.ann_dir, cfg.annotation_glob)
    random.shuffle(files)
    split_idx = int(len(files) * cfg.split_ratio)
    splits = {"train": files[:split_idx], "val": files[split_idx:]}

    log.info("Processing %d annotation files into %s", len(files), cfg.output_dir)
    skipped = 0
    for split, file_list in splits.items():
        for ann_path in tqdm(file_list, desc=split):
            base_name, img_path = image_for_annotation(ann_path, cfg.img_dir)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            boxes, labels = parse_letters(ann_path)
            for i, (box, label) in enumerate(zip(boxes, labels, strict=True)):
                try:
                    crop = crop_fn(img, box, cfg.img_size)
                    label_dir = os.path.join(cfg.output_dir, split, label)
                    os.makedirs(label_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(label_dir, f"{base_name}_{i}.jpg"), crop)
                except Exception:
                    # Degenerate (near zero-size) boxes make the perspective warp fail.
                    skipped += 1

    log.info("Done. Skipped %d degenerate boxes.", skipped)
