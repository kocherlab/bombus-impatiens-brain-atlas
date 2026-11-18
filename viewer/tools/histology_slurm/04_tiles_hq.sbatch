#!/bin/bash
# bimp-sbatch: cpu_day  (submit: bash config/sbatch.sh viewer/tools/histology_slurm/04_tiles_hq.sbatch)
# Step 4 (after step 1; optional): high-resolution DAPI tile pyramid -> pages/data/histo_tiles_hq/
#   {caste}/{ii}/dapi/{level}/{tx}_{ty}.webp, tiles.json, tiles_check.json
# 8x upscale (~0.44 um/px queen, ~0.34 worker), DAPI only: roughly 19-20k files across both brains,
# so it is uploaded as shards (tools/shard_hq_tiles.py, deploy/cloudflare.sh tiles-hq). Writes to
# its own folder; the shipped histo_tiles/ is untouched. One array task per brain (0 queen, 1 worker).
#SBATCH --job-name=histo_tiles_hq
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=viewer/tools/histology_slurm/logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source viewer/tools/histology_slurm/env.sh

CASTES=(queen worker)
CASTE="${CASTE:-${CASTES[${SLURM_ARRAY_TASK_ID:-0}]}}"
echo "caste $CASTE"

"$PY" "$TOOLS/export_tiles.py" \
    --repo "$DATA_REPO" --atlas "$ATLAS" \
    --histo "$DATA_OUT/histo" --out "$DATA_OUT/histo_tiles_hq" --url-dir histo_tiles_hq \
    --upscale 8 --stains dapi --castes "$CASTE"
