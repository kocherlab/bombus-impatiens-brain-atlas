#!/bin/bash
# bimp-sbatch: cpu_short  (submit: bash config/sbatch.sh viewer/tools/gene_projection/01_ref_gene_stats.sbatch)
#SBATCH --job-name=gp_ref_stats
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Step 1. Measured 2026-10-16: ~1 min.
# Submit from the repo root: bash config/sbatch.sh viewer/tools/gene_projection/01_ref_gene_stats.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=src:scripts
.venv/bin/python viewer/tools/gene_projection/ref_gene_stats.py "$@"
