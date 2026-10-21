#!/bin/bash -l
# ============================================================
# Generic SLURM job for the FAU HPC (TinyGPU) running one `trogs` command.
#
# Usage (from the repository root; logs/ must exist before submitting):
#   mkdir -p logs
#   sbatch scripts/slurm/job.sbatch trogs train classifier --config configs/train/classifier.yaml
#   sbatch --partition=a100 --gres=gpu:a100:1 --job-name=yolo-obb \
#          scripts/slurm/job.sbatch trogs train detector --config configs/train/detector.yaml
#
# Flags given to sbatch override the #SBATCH defaults below.
# ============================================================

#SBATCH --job-name=trogs
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

set -euo pipefail

# Conda environment with the package installed (`pip install -e .`).
CONDA_ENV="yolo-obb"

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch scripts/slurm/job.sbatch <command> [args...]" >&2
    exit 1
fi

echo "========== Job started: $(date) =========="
echo "Job ID: ${SLURM_JOB_ID:-unknown} | Node: $(hostname) | Workdir: $(pwd)"
echo "Command: $*"

module load python/3.12-conda || echo "Module load skipped/failed, proceeding..."
conda activate "$CONDA_ENV"
echo "Python: $(which python) ($(python --version 2>&1))"

nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu[0]:", torch.cuda.get_device_name(0))
PY

# Ultralytics may import Polars, which crashes on some older node CPUs without this.
export POLARS_SKIP_CPU_CHECK=1

"$@"

echo "========== Job ended: $(date) =========="
