#!/bin/bash
# bimp-sbatch: cpu_short  (submit: bash config/sbatch.sh viewer/tools/histology_slurm/03_tiles.sbatch)
# Step 3 (after step 1): high-resolution tile pyramid -> viewer/pages/data/histo_tiles/
#   {caste}/{ii}/{stain}/{level}/{tx}_{ty}.webp, tiles.json, tiles_check.json
# One array task per brain (0 = queen, 1 = worker). export_tiles.py merges its brain into
# tiles.json under a file lock, so the two tasks can run at once, and re-running one
# brain keeps the other's entries. ~45 min per brain on 4 cpu, ~2 GB peak.
#SBATCH --job-name=histo_tiles
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
    --histo "$DATA_OUT/histo" --out "$DATA_OUT/histo_tiles" --castes "$CASTE"
