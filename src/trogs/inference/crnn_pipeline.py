"""Pathway 2: YOLO-OBB detection -> line grouping -> CRNN line recognition."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from trogs.config import dump_config
from trogs.data.discovery import find_images
from trogs.data.geometry import line_rect, warp_line
from trogs.inference.reconstruction import chain_rows, sort_rows
from trogs.inference.visualize import draw_lines
from trogs.models.crnn import Vocab, greedy_decode, load_crnn
from trogs.models.detector import SliceParams, detect, load_detector, prediction_corners
from trogs.utils import resolve_device

log = logging.getLogger(__name__)


@dataclass
class CrnnPipelineConfig:
    detector_weights: str = "runs/obb/train3/weights/best.pt"
    crnn_weights: str = "models/crnn_greek_20260420_155707_best.pth"
    vocab_file: str = "data/processed/crnn_training_data/labels.txt"  # training labels rebuild the vocab
    confidence: float = 0.25
    slice_size: int = 640
    overlap_ratio: float = 0.2
    target_h: int = 32
    device: str = "auto"
    # Batch mode only
    input_dir: str = "data/raw/Images"
    output_dir: str = "outputs/crnn/batch_transcripts"
    vis_dir: str | None = "outputs/crnn/batch_visualizations"


def group_lines(predictions) -> list[list[dict]]:
    """Chain detected letters into text lines, sorted top-to-bottom and left-to-right."""
    boxes = []
    for pred in predictions:
        pts = prediction_corners(pred)
        cx, cy = np.mean(pts, axis=0)
        diag1 = np.linalg.norm(pts[0] - pts[2]) if len(pts) == 4 else 0
        diag2 = np.linalg.norm(pts[1] - pts[3]) if len(pts) == 4 else 0
        r = max(diag1, diag2) * 0.5
        boxes.append({"pts": pts, "cx": cx, "cy": cy, "r": r if r != 0 else 20})
    return sort_rows(chain_rows(boxes))


class CrnnPipeline:
    def __init__(self, cfg: CrnnPipelineConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        log.info("Loading models on %s", self.device)
        self.vocab = Vocab.from_labels_file(cfg.vocab_file)
        self.detector = load_detector(cfg.detector_weights, cfg.confidence, self.device)
        self.crnn = load_crnn(cfg.crnn_weights, len(self.vocab), self.device)
        self.slice_params = SliceParams(cfg.slice_size, cfg.overlap_ratio)

    def recognise_line(self, img_gray: np.ndarray, line: list[dict]) -> str:
        center, w, h, angle = line_rect(np.vstack([b["pts"] for b in line]))
        # "+T" padding: widen the crop by one line height, as during training.
        crop = warp_line(img_gray, center, w + h, h, angle, target_h=self.cfg.target_h, border_value=0)
        x = torch.FloatTensor((crop / 127.5) - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return greedy_decode(self.crnn(x), self.vocab.idx2char)[0]

    def run(self, img_path: str, verbose: int = 1) -> tuple[list[str], list[list[dict]], np.ndarray]:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image {img_path}")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        predictions = detect(img_path, self.detector, self.slice_params, verbose=verbose)
        if not predictions:
            return [], [], img
        lines = group_lines(predictions)
        log.info("%d letters grouped into %d lines", len(predictions), len(lines))
        return [self.recognise_line(img_gray, line) for line in lines], lines, img


def run_single(cfg: CrnnPipelineConfig, img: str, out: str, vis_out: str | None = None) -> None:
    transcript, lines, image = CrnnPipeline(cfg).run(img)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript))
    log.info("Transcript saved to %s", out)
    if vis_out:
        os.makedirs(os.path.dirname(os.path.abspath(vis_out)), exist_ok=True)
        draw_lines(image, lines, transcript, vis_out)
        log.info("Visualisation saved to %s", vis_out)


def run_batch(cfg: CrnnPipelineConfig) -> str:
    """Transcribe every image in ``input_dir``; writes ``<name>.txt`` and ``batch_diagnostics.tsv``."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.vis_dir:
        os.makedirs(cfg.vis_dir, exist_ok=True)
    dump_config(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    pipeline = CrnnPipeline(cfg)
    image_paths = find_images(cfg.input_dir)
    log.info("Found %d images", len(image_paths))

    rows = []
    for i, img_path in enumerate(image_paths, 1):
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        start = time.time()
        try:
            transcript, lines, image = pipeline.run(img_path, verbose=0)
            with open(os.path.join(cfg.output_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(transcript))
            if cfg.vis_dir and lines:
                draw_lines(image, lines, transcript, os.path.join(cfg.vis_dir, f"{stem}_vis.jpg"))
            elapsed = time.time() - start
            rows.append(
                f"{name}\tSUCCESS\t{elapsed:.2f}\t{len(lines)}\t{sum(len(t) for t in transcript)}\tNone"
            )
            log.info("[%d/%d] %s: %d lines in %.2f s", i, len(image_paths), name, len(lines), elapsed)
        except Exception as e:  # keep going: one bad image must not stop the batch
            elapsed = time.time() - start
            rows.append(f"{name}\tERROR\t{elapsed:.2f}\t0\t0\t{e}")
            log.error("[%d/%d] %s: %s", i, len(image_paths), name, e)

    diag_file = os.path.join(cfg.output_dir, "batch_diagnostics.tsv")
    with open(diag_file, "w", encoding="utf-8") as f:
        f.write("Image_Name\tStatus\tTime_Sec\tNum_Lines\tTotal_Chars\tError_Msg\n")
        f.write("\n".join(rows))
    log.info("Batch complete. Diagnostics: %s", diag_file)
    return cfg.output_dir
