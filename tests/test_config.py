from dataclasses import dataclass

import pytest

from trogs.config import load_config, parse_overrides


@dataclass
class Cfg:
    lr: float = 0.1
    name: str = "a"
    seed: int | None = None


def test_yaml_then_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("lr: 0.5\nname: b\n")
    cfg = load_config(Cfg, path, parse_overrides(["name=c", "seed=3"]))
    assert cfg == Cfg(lr=0.5, name="c", seed=3)


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        load_config(Cfg, overrides={"typo": 1})


def test_repo_configs_match_dataclasses():
    """Every shipped YAML file must load into its dataclass."""
    from trogs.data.build_classification import ClassificationDataConfig
    from trogs.data.build_crnn_lines import CrnnLinesConfig
    from trogs.data.build_yolo_tiles import YoloTilesConfig
    from trogs.evaluation.compare_systems import CompareConfig
    from trogs.inference.classifier_pipeline import ClassifierPipelineConfig
    from trogs.inference.crnn_pipeline import CrnnPipelineConfig
    from trogs.training.classifier import ClassifierTrainConfig
    from trogs.training.crnn import CrnnTrainConfig
    from trogs.training.detector import DetectorTrainConfig

    pairs = {
        "configs/data/classification.yaml": ClassificationDataConfig,
        "configs/data/classification_stratified.yaml": ClassificationDataConfig,
        "configs/data/yolo_obb.yaml": YoloTilesConfig,
        "configs/data/crnn.yaml": CrnnLinesConfig,
        "configs/train/detector.yaml": DetectorTrainConfig,
        "configs/train/classifier.yaml": ClassifierTrainConfig,
        "configs/train/crnn.yaml": CrnnTrainConfig,
        "configs/pipeline/classifier.yaml": ClassifierPipelineConfig,
        "configs/pipeline/classifier_batch.yaml": ClassifierPipelineConfig,
        "configs/pipeline/crnn.yaml": CrnnPipelineConfig,
        "configs/evaluate.yaml": CompareConfig,
    }
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for rel, cls in pairs.items():
        load_config(cls, root / rel)
