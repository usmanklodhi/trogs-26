#!/bin/bash -l
# ============================================================
# SLURM job script: Optimized Batch Pipeline (RTX 3080)
# ============================================================

#SBATCH --job-name=opt-batch-pipeline        # Job name shown in squeue
#SBATCH --partition=rtx3080                  # Partition/queue
#SBATCH --gres=gpu:rtx3080:1                 # Request 1x RTX3080 GPU
#SBATCH --time=12:00:00                      # Max walltime (HH:MM:SS)

#SBATCH --cpus-per-task=8                    # DataLoader workers / CPU threads
#SBATCH --output=logs/%x-%j.out              # Stdout log  (jobname-jobid.out)
#SBATCH --error=logs/%x-%j.err               # Stderr log  (jobname-jobid.err)

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

# Activate the environment we used for YOLO
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
# Run pipeline
# ---------------------------
# cd /home/hpc/iwi5/iwi5340h/trogs-26  # Uncomment if needed

echo "----- Starting Optimized Batch Pipeline -----"
python src/optimized_batch_run_pipeline.py
echo "----- Pipeline finished -----"

echo "========== Job ended: $(date) =========="