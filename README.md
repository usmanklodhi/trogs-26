# TROGS-26: Transcribing Greek Inscription Squeezes

Thesis code for automatic transcription of paper squeezes of ancient Greek inscriptions.
Every squeeze is photographed under two lighting directions (`Rotation1`, `Rotation2`).
Two recognition pathways are implemented and compared:

| | Pathway 1: detect + classify | Pathway 2: detect + CRNN |
|---|---|---|
| Letter detection | YOLO11-OBB with SAHI sliced inference | same detector |
| Recognition | ResNet18 classifies each letter crop; logits of matching letters from both rotations are averaged | letters are chained into lines; each rectified line strip is read by a CNN-BiLSTM-CTC model |
| Reading order | line reconstruction after Howe et al. (neighbour chaining + row merging) | line chaining before recognition |
| Code | `trogs.inference.classifier_pipeline` | `trogs.inference.crnn_pipeline` |

Both are evaluated with the competition CER metric (`trogs.evaluation`).

## Repository layout

```
configs/                 YAML configs (defaults reproduce the thesis runs)
  data/                  dataset builders
  train/                 detector, classifier, CRNN training
  pipeline/              inference (weights, thresholds, device)
  evaluate.yaml          pathway comparison
  yolo_dataset.yaml      Ultralytics dataset file
src/trogs/
  data/                  annotation parser, crop geometry, dataset builders
  models/                ResNet18 classifier, GreekCRNN, SAHI detector wrapper
  training/              training loops
  inference/             rotation fusion, line reconstruction, both pipelines
  evaluation/            competition metrics, pathway comparison, YOLO validation
  cli.py                 the `trogs` command
scripts/slurm/job.sbatch generic SLURM job for the FAU HPC
notebooks/               annotation viewer, tile inspection, pipeline walkthrough, thesis charts
tests/                   unit tests (CPU only, synthetic data)
data/raw/                dataset metadata (split lists, README, make_xml.py)
```

## Setup

```bash
conda create -n yolo-obb python=3.12 -y && conda activate yolo-obb
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

`requirements.txt` pins the versions used for the thesis; `pyproject.toml` gives looser lower bounds.

## Data

The dataset accompanies *"Character Recognition for Greek Squeezes"* (Howe, Chang, Falbo, Brown,
Hershkowitz; see `data/raw/README.txt`). It isn't included in this repository. Unpack it like this:

```
data/raw/
  Images/                      <ID>_Rotation{1,2}_300dpi.png
  Annotations/                 <ID>_Rotation{1,2}_300dpi_letters.txt
  test_fullsized_images/       (optional) test squeezes
  {training,validation,test}_set.txt
```

An annotation file has `# Boxes` (one 8-number letter quadrilateral per line), `# Transcript` (one row
per inscription line, `*` marking a letter without a box) and `# Lines` sections.
`trogs.data.annotations` is the single parser for this format.

Everything that is generated goes to `data/processed/`, `runs/`, `models/`, `results/`, `outputs/` and
`logs/`, all git-ignored.

## Usage

All commands run from the repository root. Each one takes `--config <yaml>` and any number of
`--set key=value` overrides; `trogs <stage> <target> --help` lists the options.

### 1. Build datasets

```bash
trogs prepare yolo           --config configs/data/yolo_obb.yaml                      # 640px tiles for the detector
trogs prepare classification --config configs/data/classification.yaml                # stretched letter crops
trogs prepare classification --config configs/data/classification_stratified.yaml     # square, aspect-preserving crops
trogs prepare crnn           --config configs/data/crnn.yaml                          # line crops + labels.txt
```

The train/validation splits are random and unseeded by default, as in the thesis runs.
Pass `--set seed=42` to make them reproducible.

### 2. Train

```bash
trogs train detector   --config configs/train/detector.yaml     # -> runs/obb/train*/weights/best.pt
trogs train classifier --config configs/train/classifier.yaml   # -> models/greek_classifier_resnet18_v4_<ts>.pth
trogs train crnn       --config configs/train/crnn.yaml         # -> models/crnn_greek_<ts>_best.pth (+ logs/)
```

### 3. Transcribe

```bash
# Pathway 1, one squeeze (Rotation2 is optional and fused when given)
trogs predict classifier --config configs/pipeline/classifier.yaml \
    --image data/raw/Images/IAS_27288_Rotation1_300dpi.png \
    --image2 data/raw/Images/IAS_27288_Rotation2_300dpi.png \
    --out outputs/IAS_27288.txt --vis-out outputs/IAS_27288.png

# Pathway 1, every squeeze in input_dir -> results/batch_run_<ts>/
trogs predict classifier --config configs/pipeline/classifier_batch.yaml --batch \
    --set input_dir="data/raw/Test PNG Images"

# Pathway 2, one image / batch -> outputs/crnn/
trogs predict crnn --config configs/pipeline/crnn.yaml --image <img> --out <txt> [--vis-out <jpg>]
trogs predict crnn --config configs/pipeline/crnn.yaml --batch
```

The weight paths in `configs/pipeline/*.yaml` point to the checkpoints used in the thesis. The
classifier loader accepts checkpoints with either a plain `fc` head or the dropout head used by `train classifier`.

### 4. Evaluate

```bash
trogs evaluate compare --config configs/evaluate.yaml    # CER, insert/delete/substitute counts, speed
trogs evaluate detector --set weights=runs/obb/train3/weights/best.pt
trogs evaluate detector-samples --set num_samples=10
```

### Running on the FAU HPC

`scripts/slurm/job.sbatch` sets up the environment, prints a GPU check and runs whatever command it
is given. Defaults: one RTX 3080, 12 h, 8 CPUs. Flags passed to `sbatch` override them.

```bash
mkdir -p logs
sbatch scripts/slurm/job.sbatch trogs train classifier --config configs/train/classifier.yaml
sbatch --partition=a100 --gres=gpu:a100:1 --job-name=yolo-obb \
    scripts/slurm/job.sbatch trogs train detector --config configs/train/detector.yaml
sbatch --job-name=pathway1 scripts/slurm/job.sbatch \
    trogs predict classifier --config configs/pipeline/classifier_batch.yaml --batch
```

## Where the old scripts went

| Before | Now |
|---|---|
| `src/prepare_yolo_tiled.py` | `trogs prepare yolo` |
| `src/prepare_classification_dataset.py` | `trogs prepare classification --config configs/data/classification.yaml` |
| `src/prepare_dataset_stratified.py` | `trogs prepare classification --config configs/data/classification_stratified.yaml` |
| `src/prepare_crnn_training_data.py` | `trogs prepare crnn` |
| `src/train_detector.py` | `trogs train detector` |
| `src/train_classifier.py` | `trogs train classifier` |
| `src/train_crnn.py` | `trogs train crnn` |
| `src/run_pipeline.py`, `src/optimized_run_pipeline.py` | `trogs predict classifier --image ...` (`--set device=cpu` for the CPU variant) |
| `src/optimized_batch_run_pipeline.py`, `src/batch_run_pipeline.py` | `trogs predict classifier --batch` |
| `src/run_crnn_pipeline.py` | `trogs predict crnn --image ...` |
| `src/batch_run_crnn_pipeline.py` | `trogs predict crnn --batch` |
| `src/system_comparison.py` | `trogs evaluate compare` |
| `src/validate_yolo.py` | `trogs evaluate detector` |
| `src/predict_yolo_samples.py` | `trogs evaluate detector-samples` |
| `run_*.sh` (6 SLURM scripts) | `scripts/slurm/job.sbatch <command>` |

The restructure doesn't change behaviour. On synthetic data, the dataset builders write byte-identical
files, and both pipelines produce the same letters, lines and transcripts as the original scripts.
Batch output names and diagnostics columns are also unchanged, so `trogs evaluate compare` reads old
result folders. The originals are in the git history (commit `ffb872c`).

## Development

```bash
pytest          # unit tests
ruff check .    # lint
ruff format .   # format
```

## Citation

If you use the dataset, cite:

> N. R. Howe, F. Chang, I. Falbo, T. Brown, A. Hershkowitz. *Character Recognition for Greek Squeezes.*
> International Journal on Document Analysis and Recognition (in review).
