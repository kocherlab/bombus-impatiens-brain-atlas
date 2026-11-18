#!/bin/bash
# bimp-sbatch: cpu_short  (submit: bash config/sbatch.sh viewer/tools/gene_projection/02_select_genes.sbatch)
#SBATCH --job-name=gp_select
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Step 2. Seconds; run under Slurm anyway (no work on the login node).
# Submit from the repo root: bash config/sbatch.sh viewer/tools/gene_projection/02_select_genes.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src:scripts
.venv/bin/python viewer/tools/gene_projection/select_genes.py "$@"
