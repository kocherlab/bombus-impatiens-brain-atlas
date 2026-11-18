#!/bin/bash
# bimp-sbatch: cpu_day  (submit: bash config/sbatch.sh viewer/tools/gene_projection/04_export_pages.sbatch)
#SBATCH --job-name=gp_export
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Step 4. Measured: 39 min, MaxRSS 53 GB -- too close to a 1-hour limit, so the day class.
# Submit from the repo root: bash config/sbatch.sh viewer/tools/gene_projection/04_export_pages.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=src:scripts
# Writes viewer/pages/data/ (base, measured + inferred gene files, manifest).
# Never touches pages/data/histo*. Optional $1: the imputed h5ad (default: atlas v2).
IMP="${1:-$(.venv/bin/python -c "from bimp_st.paths import CANONICAL_TANGRAM as T; print(str(T).replace('_tangram.h5ad', '_imputed_atlas_v2.h5ad'))")}"
# the inferred gene list changes wholesale, so drop the old files and sidecar first
rm -f viewer/pages/data/genes/i*.b4 viewer/pages/data/inferred_meta.json
.venv/bin/python viewer/export_data.py --pages --genes 100 --imputed-genes \
    --imputed-h5ad "$IMP" --inferred-mapping canonical
