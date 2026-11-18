#!/bin/bash
# bimp-sbatch: cpu_short  (submit: bash config/sbatch.sh viewer/tools/gene_projection/03_project_genes.sbatch)
#SBATCH --job-name=gp_project
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
# Step 3. CPU only (a 29-row matrix product, no Tangram training). Measured: 6.7 min, MaxRSS 52 GB.
# Submit from the repo root: bash config/sbatch.sh viewer/tools/gene_projection/03_project_genes.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=src:scripts
# args default to the checked-in gene list and paths.CANONICAL_TANGRAM's *_imputed_atlas_v2.h5ad
.venv/bin/python viewer/tools/gene_projection/project_genes.py \
    --genes "${1:-viewer/tools/gene_projection/genes_atlas_v2.txt}" "${@:2}"
