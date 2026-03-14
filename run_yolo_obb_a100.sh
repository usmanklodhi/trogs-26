#!/bin/bash -l
# ============================================================
# SLURM job script: Ultralytics YOLO-OBB training (A100)
# ============================================================

#SBATCH --job-name=yolo-obb-a100          # Job name shown in squeue
#SBATCH --partition=a100                  # Partition/queue
#SBATCH --gres=gpu:a100:1                 # Request 1x a100 GPU
#SBATCH --time=12:00:00                      # Max walltime (HH:MM:SS)

#SBATCH --cpus-per-task=8                    # DataLoader workers / CPU threads
#SBATCH --output=logs/%x-%j.out              # Stdout log  (jobname-jobid.out)
#SBATCH --error=logs/%x-%j.err               # Stderr log  (jobname-jobid.err)

#SBATCH --export=NONE                        # Do not inherit submission env
unset SLURM_EXPORT_ENV                       # Cluster-specific safety

# ---------------------------
# Safety / debugging settings
# ---------------------------
set -euo pipefail

echo "========== Job started: $(date) =========="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "=========================================="

# Create log dir if not present (safe if already exists)
mkdir -p logs

# ---------------------------
# Load modules + activate env
# ---------------------------
module load python/3.12-conda

# Some clusters need shell init for conda to work reliably:
# (If this line ever fails, replace ~/.bashrc with ~/.bash_profile)
# source ~/.bashrc

conda activate yolo-obb

echo "Python: $(which python)"
python --version

# ---------------------------
# GPU sanity check
# ---------------------------
echo "----- GPU status (nvidia-smi) -----"
nvidia-smi || true
echo "----------------------------------"

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu[0]:", torch.cuda.get_device_name(0))
PY

# ---------------------------
# Optional: avoid Polars CPU crash on some nodes
# (Better fix is: pip install "polars[rtcompat]"
# but this prevents hard-crash if Polars gets imported.)
# ---------------------------
export POLARS_SKIP_CPU_CHECK=1

# ---------------------------
# Run training
# ---------------------------
# If your script assumes a project root, cd there explicitly:
# cd /home/hpc/iwi5/iwi5340h/trogs-26

echo "----- Starting training -----"
python src/train.py
echo "----- Training finished -----"

echo "========== Job ended: $(date) =========="
