#!/bin/bash
# bimp-sbatch: cpu_short  (submit: bash config/sbatch.sh viewer/tools/histology_slurm/02_stains.sbatch)
# Step 2 (after step 1): the other three Xenium stains -> viewer/pages/data/histo_stains/
#   {caste}/NN-{boundary,rna18s,smavim}.webp (+ -s.webp), stains.json, stains_check.json
# export_stains.py samples each per-crop h5ad's uns/image through the crop's own cells
# onto every section of histo/ (same bounds, size and alpha), and re-renders DAPI the
# same way as a check against histo/ (stains_check.json: shift should be 0 px). ~16 min.
#SBATCH --job-name=histo_stains
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=viewer/tools/histology_slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source viewer/tools/histology_slurm/env.sh

"$PY" "$TOOLS/export_stains.py" \
    --repo "$DATA_REPO" --atlas "$ATLAS" \
    --histo "$DATA_OUT/histo" --out "$DATA_OUT/histo_stains"
