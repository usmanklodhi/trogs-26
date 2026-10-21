"""Command-line interface: ``trogs <stage> <target> [--config FILE] [--set key=value ...]``.

Stages and targets::

    trogs prepare  classification | yolo | crnn
    trogs train    detector | classifier | crnn
    trogs predict  classifier | crnn            (single image, or --batch)
    trogs evaluate compare | detector | detector-samples

Every target has a dataclass config whose defaults reproduce the thesis setup;
``--config`` loads a YAML file on top and ``--set`` overrides single keys.
"""

from __future__ import annotations

import argparse
import logging

from trogs.config import load_config, parse_overrides
from trogs.utils import setup_logging

log = logging.getLogger(__name__)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a config value (repeatable)",
    )


def _cfg(cls, args):
    return load_config(cls, args.config, parse_overrides(args.overrides))


# --- prepare -----------------------------------------------------------------


def _prepare_classification(args):
    from trogs.data.build_classification import ClassificationDataConfig, build

    build(_cfg(ClassificationDataConfig, args))


def _prepare_yolo(args):
    from trogs.data.build_yolo_tiles import YoloTilesConfig, build

    build(_cfg(YoloTilesConfig, args))


def _prepare_crnn(args):
    from trogs.data.build_crnn_lines import CrnnLinesConfig, build

    build(_cfg(CrnnLinesConfig, args))


# --- train -------------------------------------------------------------------


def _train_detector(args):
    from trogs.training.detector import DetectorTrainConfig, train

    train(_cfg(DetectorTrainConfig, args))


def _train_classifier(args):
    from trogs.training.classifier import ClassifierTrainConfig, train

    train(_cfg(ClassifierTrainConfig, args))


def _train_crnn(args):
    from trogs.training.crnn import CrnnTrainConfig, train

    train(_cfg(CrnnTrainConfig, args))


# --- predict -----------------------------------------------------------------


def _predict_classifier(args):
    from trogs.inference import classifier_pipeline as p

    cfg = _cfg(p.ClassifierPipelineConfig, args)
    if args.batch:
        p.run_batch(cfg)
    else:
        _require(args, "image", "out")
        p.run_single(cfg, args.image, args.image2, args.out, args.vis_out)


def _predict_crnn(args):
    from trogs.inference import crnn_pipeline as p

    cfg = _cfg(p.CrnnPipelineConfig, args)
    if args.batch:
        p.run_batch(cfg)
    else:
        _require(args, "image", "out")
        p.run_single(cfg, args.image, args.out, args.vis_out)


def _require(args, *names):
    missing = [f"--{n.replace('_', '-')}" for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"error: single-image mode needs {', '.join(missing)} (or use --batch)")


# --- evaluate ----------------------------------------------------------------


def _evaluate_compare(args):
    from trogs.evaluation.compare_systems import CompareConfig, compare

    compare(_cfg(CompareConfig, args))


def _evaluate_detector(args):
    from trogs.evaluation.detector import DetectorEvalConfig, validate

    validate(_cfg(DetectorEvalConfig, args))


def _evaluate_detector_samples(args):
    from trogs.evaluation.detector import DetectorEvalConfig, predict_samples

    predict_samples(_cfg(DetectorEvalConfig, args))


COMMANDS = {
    "prepare": {
        "classification": (
            _prepare_classification,
            "crop one image per annotated letter (ImageFolder layout)",
        ),
        "yolo": (_prepare_yolo, "tile full squeezes into a YOLO-OBB detection dataset"),
        "crnn": (_prepare_crnn, "crop one image per inscription line, with labels.txt"),
    },
    "train": {
        "detector": (_train_detector, "train the YOLO11-OBB letter detector"),
        "classifier": (_train_classifier, "train the ResNet18 letter classifier"),
        "crnn": (_train_crnn, "train the CRNN line recogniser (CTC)"),
    },
    "predict": {
        "classifier": (_predict_classifier, "pathway 1: detection + classification + line reconstruction"),
        "crnn": (_predict_crnn, "pathway 2: detection + line grouping + CRNN"),
    },
    "evaluate": {
        "compare": (_evaluate_compare, "compare both pathways on the test set (CER, edit ops, speed)"),
        "detector": (_evaluate_detector, "YOLO validation metrics (mAP50, mAP50-95)"),
        "detector-samples": (_evaluate_detector_samples, "save detector predictions for random tiles"),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trogs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, targets in COMMANDS.items():
        stage_parser = stages.add_parser(stage, help=f"{stage} step")
        sub = stage_parser.add_subparsers(dest="target", required=True)
        for target, (func, help_text) in targets.items():
            p = sub.add_parser(target, help=help_text, description=help_text)
            _common(p)
            if stage == "predict":
                p.add_argument("--batch", action="store_true", help="process every image in input_dir")
                p.add_argument("--image", help="input image (Rotation1 for the classifier pathway)")
                if target == "classifier":
                    p.add_argument("--image2", help="optional Rotation2 image, fused with --image")
                p.add_argument("--out", help="output transcript .txt")
                p.add_argument("--vis-out", dest="vis_out", help="optional visualisation image")
            p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
