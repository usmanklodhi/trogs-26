"""Build the YOLO-OBB letter detection dataset by tiling full squeeze images.

Each image is cut into ``tile_size`` tiles with ``stride`` step. A letter box is
kept in a tile when at least ``ioa_threshold`` of its area lies inside; it is
then clipped to the tile and replaced by its minimum rotated rectangle.
Only tiles containing letters are written. Labels use the YOLO OBB format
``0 x1 y1 x2 y2 x3 y3 x4 y4`` normalised to the tile size.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from tqdm import tqdm

from trogs.data.annotations import parse_boxes
from trogs.data.discovery import annotation_files, image_for_annotation
from trogs.utils import seed_everything

log = logging.getLogger(__name__)


@dataclass
class YoloTilesConfig:
    img_dir: str = "data/raw/Images"
    ann_dir: str = "data/raw/Annotations"
    output_dir: str = "data/processed/greek_letters_obb"
    tile_size: int = 640
    stride: int = 512
    ioa_threshold: float = 0.7
    split_ratio: float = 0.9
    annotation_glob: str = "*_Rotation1_300dpi_letters.txt"
    seed: int | None = None


def write_yolo_label(path: str, polygons, width: int, height: int) -> None:
    with open(path, "w") as f:
        for poly in polygons:
            coords = []
            for x, y in poly:
                coords.append(f"{max(0, min(x / width, 1.0)):.6f}")
                coords.append(f"{max(0, min(y / height, 1.0)):.6f}")
            f.write(f"0 {' '.join(coords)}\n")


def tile_boxes(boxes: list[Polygon], x: int, y: int, x_end: int, y_end: int, ioa_threshold: float):
    """Boxes visible enough in the tile, in tile-local coordinates."""
    tile = box(x, y, x_end, y_end)
    kept = []
    for poly in boxes:
        if not poly.is_valid:
            continue
        try:
            inter = poly.intersection(tile)
            if inter.is_empty or inter.area / poly.area < ioa_threshold:
                continue
            if inter.geom_type == "Polygon":
                rect = inter.minimum_rotated_rectangle
                kept.append([(px - x, py - y) for px, py in list(rect.exterior.coords)[:-1]])
        except Exception:
            # Shapely can fail on degenerate geometry; such boxes are skipped.
            continue
    return kept


def build(cfg: YoloTilesConfig) -> None:
    seed_everything(cfg.seed)
    for split in ("train", "val"):
        os.makedirs(os.path.join(cfg.output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "labels", split), exist_ok=True)

    files = annotation_files(cfg.ann_dir, cfg.annotation_glob)
    np.random.shuffle(files)
    split_idx = int(len(files) * cfg.split_ratio)
    splits = {"train": files[:split_idx], "val": files[split_idx:]}

    log.info("Tiling %d images (%dx%d tiles, stride %d)", len(files), cfg.tile_size, cfg.tile_size, cfg.stride)
    for split, file_list in splits.items():
        for ann_path in tqdm(file_list, desc=split):
            base_name, img_path = image_for_annotation(ann_path, cfg.img_dir)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img, _ = img.shape

            polygons = [Polygon([tuple(p) for p in b]) for b in parse_boxes(ann_path)]

            for y in range(0, h_img, cfg.stride):
                for x in range(0, w_img, cfg.stride):
                    x_end = min(x + cfg.tile_size, w_img)
                    y_end = min(y + cfg.tile_size, h_img)
                    kept = tile_boxes(polygons, x, y, x_end, y_end, cfg.ioa_threshold)
                    if not kept:
                        continue

                    tile = img[y:y_end, x:x_end]
                    cur_h, cur_w = y_end - y, x_end - x
                    if cur_h < cfg.tile_size or cur_w < cfg.tile_size:
                        padded = np.zeros((cfg.tile_size, cfg.tile_size, 3), dtype=np.uint8)
                        padded[:cur_h, :cur_w] = tile
                        tile = padded

                    name = f"{base_name}_{x}_{y}"
                    cv2.imwrite(os.path.join(cfg.output_dir, "images", split, name + ".jpg"), tile)
                    write_yolo_label(os.path.join(cfg.output_dir, "labels", split, name + ".txt"),
                                     kept, cfg.tile_size, cfg.tile_size)

    log.info("Done. Point configs/yolo_dataset.yaml at %s", os.path.abspath(cfg.output_dir))
