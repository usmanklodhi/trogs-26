"""Build the CRNN line recognition dataset: one rectified crop per inscription line.

Output::

    <output_dir>/images/<squeeze>_line<NNN>.jpg
    <output_dir>/labels.txt          # "<file name>\\t<line text>" per crop

Lines are rectified to height ``target_h``. As augmentation ("+T padding") each
crop is randomly extended left and right by 0, h/2 or h.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import dataclass

import cv2
from tqdm import tqdm

from trogs.data.annotations import parse_text_lines
from trogs.data.discovery import annotation_files, image_for_annotation
from trogs.data.geometry import line_rect, warp_line
from trogs.utils import seed_everything

log = logging.getLogger(__name__)


@dataclass
class CrnnLinesConfig:
    img_dir: str = "data/raw/Images"
    ann_dir: str = "data/raw/Annotations"
    output_dir: str = "data/processed/crnn_training_data"
    annotation_glob: str = "*_letters.txt"
    target_h: int = 32
    seed: int | None = None


def build(cfg: CrnnLinesConfig) -> None:
    seed_everything(cfg.seed)
    if os.path.exists(cfg.output_dir):
        log.info("Removing old %s", cfg.output_dir)
        shutil.rmtree(cfg.output_dir)
    images_dir = os.path.join(cfg.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    ann_files = annotation_files(cfg.ann_dir, cfg.annotation_glob)
    log.info("Found %d annotation files", len(ann_files))

    total_lines = 0
    missing = 0
    with open(os.path.join(cfg.output_dir, "labels.txt"), "w", encoding="utf-8") as labels_file:
        for ann_path in tqdm(ann_files):
            base_name, img_path = image_for_annotation(ann_path, cfg.img_dir)
            img = cv2.imread(img_path) if os.path.exists(img_path) else None
            if img is None:
                missing += 1
                continue

            for idx, line in enumerate(parse_text_lines(ann_path)):
                center, w, h, angle = line_rect(line.points)
                h = max(1.0, h)
                extension = random.choice([0, int(h // 2), int(h)])
                crop = warp_line(img, center, w + 2 * extension, h, angle,
                                 target_h=cfg.target_h, border_value=(0, 0, 0))

                file_name = f"{base_name}_line{idx:03d}.jpg"
                cv2.imwrite(os.path.join(images_dir, file_name), crop)
                labels_file.write(f"{file_name}\t{line.text}\n")
                total_lines += 1

    log.info("Line crops written: %d | files skipped (missing/unreadable image): %d", total_lines, missing)
    log.info("Dataset ready at %s", cfg.output_dir)
