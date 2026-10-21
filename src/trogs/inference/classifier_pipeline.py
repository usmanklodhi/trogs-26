"""Pathway 1: YOLO-OBB detection + ResNet18 classification + line reconstruction."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass

from trogs.config import dump_config
from trogs.data.discovery import find_squeeze_pairs
from trogs.inference.fusion import classify_letters
from trogs.inference.reconstruction import reconstruct_transcript
from trogs.inference.visualize import draw_letters
from trogs.models.classifier import class_names_from_dir, eval_transforms, load_classifier
from trogs.models.detector import SliceParams, load_detector
from trogs.utils import resolve_device, timestamp

log = logging.getLogger(__name__)


@dataclass
class ClassifierPipelineConfig:
    detector_weights: str = "runs/obb/train3/weights/best.pt"
    classifier_weights: str = "models/greek_classifier_resnet18_20260214_212940.pth"
    class_dir: str = "data/processed/greek_letters_classification/train"  # defines the label order
    confidence: float = 0.25
    proximity_threshold: float = 50
    slice_size: int = 640
    overlap_ratio: float = 0.2
    img_size: int = 128
    device: str = "auto"
    # Batch mode only
    input_dir: str = "data/raw/Images"
    results_dir: str = "results"


class ClassifierPipeline:
    def __init__(self, cfg: ClassifierPipelineConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        log.info("Loading models on %s", self.device)
        self.class_names = class_names_from_dir(cfg.class_dir)
        self.detector = load_detector(cfg.detector_weights, cfg.confidence, self.device)
        self.classifier = load_classifier(cfg.classifier_weights, len(self.class_names), self.device)
        self.transform = eval_transforms(cfg.img_size)
        self.slice_params = SliceParams(cfg.slice_size, cfg.overlap_ratio)

    def run(self, img1: str, img2: str | None = None) -> tuple[str, list[dict]]:
        boxes = classify_letters(
            img1,
            img2,
            self.detector,
            self.classifier,
            self.transform,
            self.class_names,
            self.device,
            self.cfg.proximity_threshold,
            self.slice_params,
        )
        return reconstruct_transcript(boxes), boxes


def run_single(
    cfg: ClassifierPipelineConfig, img1: str, img2: str | None, out: str, vis_out: str | None = None
) -> None:
    transcript, boxes = ClassifierPipeline(cfg).run(img1, img2)
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(transcript)
    log.info("Transcript saved to %s", out)
    if vis_out:
        draw_letters(img1, boxes, vis_out)
        log.info("Visualisation saved to %s", vis_out)


DIAGNOSTIC_COLUMNS = [
    "base",
    "img1",
    "img2",
    "status",
    "time_sec",
    "error",
    "num_boxes",
    "transcript_length",
    "num_rows",
    "avg_conf",
]


def run_batch(cfg: ClassifierPipelineConfig) -> str:
    """Transcribe every Rotation1 image in ``input_dir`` (fused with Rotation2 when present).

    Writes ``<results_dir>/batch_run_<timestamp>/{transcripts,visualizations,diagnostics.txt}``
    and returns the run directory.
    """
    run_dir = os.path.join(cfg.results_dir, f"batch_run_{timestamp()}")
    out_dir = os.path.join(run_dir, "transcripts")
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))

    pipeline = ClassifierPipeline(cfg)
    diagnostics = []
    for pair in find_squeeze_pairs(cfg.input_dir):
        out_txt1 = os.path.join(out_dir, f"{pair.base}_Rotation1_300dpi_transcript.txt")
        out_txt2 = os.path.join(out_dir, f"{pair.base}_Rotation2_300dpi_transcript.txt")
        row = {
            "base": pair.base,
            "img1": pair.rotation1,
            "img2": pair.rotation2,
            "error": None,
            "num_boxes": 0,
            "transcript_length": 0,
            "num_rows": 0,
            "avg_conf": 0.0,
        }
        log.info("Processing %s", pair.base)
        start = time.time()
        try:
            transcript, boxes = pipeline.run(pair.rotation1, pair.rotation2)
            with open(out_txt1, "w", encoding="utf-8") as f:
                f.write(transcript)
            if pair.rotation2:
                # Both rotations show the same inscription, so they share one transcript.
                shutil.copy(out_txt1, out_txt2)
            draw_letters(pair.rotation1, boxes, os.path.join(vis_dir, f"{pair.base}_Visualized.png"))
            row.update(
                status="success" if pair.rotation2 else "success_single",
                num_boxes=len(boxes),
                transcript_length=len(transcript),
                num_rows=len(transcript.split("\n")) if transcript else 0,
                avg_conf=sum(b["conf"] for b in boxes) / len(boxes) if boxes else 0.0,
            )
            log.info("Done %s in %.2f s", pair.base, time.time() - start)
        except Exception as e:  # keep going: one bad squeeze must not stop the batch
            row.update(status="error", error=str(e))
            log.error("Error processing %s: %s", pair.base, e)
        row["time_sec"] = time.time() - start
        diagnostics.append(row)

    with open(os.path.join(run_dir, "diagnostics.txt"), "w", encoding="utf-8") as f:
        f.write("\t".join(DIAGNOSTIC_COLUMNS) + "\n")
        for row in diagnostics:
            f.write("\t".join(str(row[c]) for c in DIAGNOSTIC_COLUMNS) + "\n")
    log.info("Batch complete: %s", run_dir)
    return run_dir
