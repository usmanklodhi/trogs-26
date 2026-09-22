#!/bin/bash -l
# ============================================================
# SLURM job script: Greek Letters Classifier Training (RTX 3080)
# ============================================================

#SBATCH --job-name=greek-crnn-batch         # Job name shown in squeue
#SBATCH --partition=rtx3080                  # Partition/queue
#SBATCH --gres=gpu:rtx3080:1                 # Request 1x RTX3080 GPU
#SBATCH --time=12:00:00                      # Max walltime (HH:MM:SS) - Reduced to 4h for classifier

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
# Adjust this line if your cluster uses a different module name
module load python/3.12-conda || echo "Module load skipped/failed, proceeding..."

# Initialize Conda (ensure this points to your specific installation if needed)
# source ~/.bashrc

# Activate the environment we used for YOLO (since we added sklearn to it)
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
# Optional: avoid Polars CPU crash
# ---------------------------
export POLARS_SKIP_CPU_CHECK=1

# ---------------------------
# Dependency Check (Optional)
# ---------------------------
# Just to be safe, we ensure sklearn is installed before running
# pip install scikit-learn seaborn matplotlib --quiet

# ---------------------------
# Run training
# ---------------------------
# cd /home/hpc/iwi5/iwi5340h/trogs-26  # Uncomment if needed

echo "----- Starting CRNN Batch Script -----"
python src/batch_run_crnn_pipeline.py
echo "----- Training finished -----"

echo "========== Job ended: $(date) =========="