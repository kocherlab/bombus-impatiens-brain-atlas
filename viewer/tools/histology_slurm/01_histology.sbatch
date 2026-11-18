#!/bin/bash
# bimp-sbatch: cpu_short  (submit: bash config/sbatch.sh viewer/tools/histology_slurm/01_histology.sbatch)
# Step 1: DAPI overview layer -> viewer/pages/data/histo/
#   manifest.json, registration.json, {queen,worker}/NN.webp + NN-s.webp, preview/
# export_histology.py reads each brain's PASTE3-aligned DAPI stack page by page and
# registers it to the viewer frame; the worker's pages go through histo_refit.py's
# per-section fit (its stack predates its April re-alignment). ~16 min on 4 cpu.
#SBATCH --job-name=histo_dapi
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=viewer/tools/histology_slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source viewer/tools/histology_slurm/env.sh

"$PY" "$TOOLS/export_histology.py" \
    --repo "$DATA_REPO" --atlas "$ATLAS" --out "$DATA_OUT/histo"
