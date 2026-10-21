"""Compare both pathways on the test squeezes: CER, edit operations and inference time."""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import pandas as pd

from trogs.evaluation.contest_metrics import edit_operations, evaluate, get_row_transcript

log = logging.getLogger(__name__)


@dataclass
class CompareConfig:
    test_set: str = "data/raw/test_set.txt"
    gt_dir: str = "data/raw/Annotations"
    crnn_pred_dir: str = "outputs/crnn/batch_transcripts"
    crnn_diagnostics: str = "outputs/crnn/batch_transcripts/batch_diagnostics.tsv"
    classifier_pred_dir: str = "results"  # searched recursively for */transcripts/
    classifier_diagnostics: str | None = None  # a run's diagnostics.txt


def load_test_squeezes(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_ground_truth(gt_dir: str, squeeze: str) -> str | None:
    gt_file = os.path.join(gt_dir, f"{squeeze}_Rotation1_300dpi_letters.txt")
    if not os.path.exists(gt_file):
        log.warning("GT file not found: %s", gt_file)
        return None
    return get_row_transcript(gt_file)


def crnn_prediction_file(pred_dir: str, squeeze: str) -> str | None:
    return os.path.join(pred_dir, f"{squeeze}_Rotation1_300dpi.txt")


def classifier_prediction_file(pred_dir: str, squeeze: str) -> str | None:
    pattern = os.path.join(pred_dir, "**", "transcripts", f"{squeeze}_Rotation1_300dpi_transcript.txt")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def read_prediction(path: str | None) -> str:
    if not path or not os.path.exists(path):
        log.warning("Prediction file not found: %s", path)
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_times(diag_file: str | None) -> list[float]:
    """Per-image inference times from either pipeline's diagnostics file."""
    if not diag_file or not os.path.exists(diag_file):
        return []
    df = pd.read_csv(diag_file, sep="\t")
    column = "time_sec" if "time_sec" in df.columns else "Time_Sec"
    return df[column].dropna().tolist()


def compare(cfg: CompareConfig) -> dict:
    systems = {
        "CRNN": lambda s: crnn_prediction_file(cfg.crnn_pred_dir, s),
        "YOLO": lambda s: classifier_prediction_file(cfg.classifier_pred_dir, s),
    }
    cers = {name: [] for name in systems}
    ops = {name: [0, 0, 0] for name in systems}

    squeezes = load_test_squeezes(cfg.test_set)
    print(f"Evaluating {len(squeezes)} test squeezes")
    for squeeze in squeezes:
        gt = load_ground_truth(cfg.gt_dir, squeeze)
        if not gt:
            print(f"No GT for {squeeze}")
            continue
        line = []
        for name, find in systems.items():
            pred = read_prediction(find(squeeze))
            cer = evaluate(pred, gt)[0]
            cers[name].append(cer)
            ops[name] = [a + b for a, b in zip(ops[name], edit_operations(pred, gt), strict=True)]
            line.append(f"{name} CER {cer:.4f}")
        print(f"{squeeze}: " + ", ".join(line))

    summary = {}
    print()
    for name in systems:
        mean = sum(cers[name]) / len(cers[name]) if cers[name] else 0
        summary[name] = {
            "mean_cer": mean,
            "insertions": ops[name][0],
            "deletions": ops[name][1],
            "substitutions": ops[name][2],
        }
        print(f"Mean {name} CER: {mean:.4f}")

    print("\n--- Error Analysis (Edit Operations) ---")
    for name in systems:
        s = summary[name]
        print(
            f"{name} -> Insertions (Hallucinations): {s['insertions']} | Deletions (Blindness): "
            f"{s['deletions']} | Substitutions: {s['substitutions']}"
        )

    times = {"CRNN": load_times(cfg.crnn_diagnostics), "YOLO": load_times(cfg.classifier_diagnostics)}
    print()
    for name, values in times.items():
        if values:
            summary[name]["mean_time_sec"] = sum(values) / len(values)
            print(f"Mean {name} inference time: {summary[name]['mean_time_sec']:.2f} seconds")
    if times["CRNN"] and times["YOLO"]:
        print(
            f"CRNN is {summary['YOLO']['mean_time_sec'] / summary['CRNN']['mean_time_sec']:.1f}x faster than YOLO"
        )
    return summary
