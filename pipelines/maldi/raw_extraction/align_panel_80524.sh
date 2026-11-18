#!/bin/bash
# bimp-sbatch: cpu  (submit: bash config/sbatch.sh pipelines/maldi/raw_extraction/align_panel_80524.sh)
#SBATCH --job-name=align_80524
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=pipelines/maldi/raw_extraction/ion_out/align_%j.log
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)

set -euo pipefail
M=$REPO/pipelines/maldi/raw_extraction
PY=$REPO/.venv/bin/python
ALIGN=$REPO/pipelines/maldi/maldi_alignment.py
SLICES=$M/ion_out/slices_princeton_80524
OUT=$M/ion_out/aligned_panel_princeton_80524.h5ad
VOL=$M/ion_out/volume_panel_princeton_80524.zarr
FIGS=$M/ion_out/align_figures_80524

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} -- PASTE3 align (prior recipe: method=both, seed=5)"
cd "$M"
# Prior recipe = pairwise -> center_align WITH spatial-heuristic pi_inits (method both)
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} $PY "$ALIGN" align \
  --slice-dir "$SLICES" --out "$OUT" --method both --seed 5
echo "=== voxelize ==="
$PY "$ALIGN" voxelize --aligned "$OUT" --out "$VOL"
echo "=== plot-align ==="
$PY "$ALIGN" plot-align --aligned "$OUT" --fig-dir "$FIGS" || echo "(plot-align optional step failed)"
echo "done"
