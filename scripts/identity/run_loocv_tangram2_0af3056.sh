#!/bin/bash
# bimp-sbatch: gpu_h100  (submit: bash config/sbatch.sh scripts/identity/run_loocv_tangram2_0af3056.sh)
# Leave-one-out gene-imputation cross-validation (LOO-CV), re-run at the
# tangram2 commit the canonical cell-type mapping uses (0af3056).
#
# A run at tangram2 d19f184 (not available upstream;
# results/tangram/combined_clustered_loocv_scores.csv, job 12517686) is the
# reference point. This is the like-for-like run: the same cross-validate entry
# point at its defaults
# (mode="cells", 100,000 spatial cells, 15,000 reference nuclei with an equal
# quota per cell type, 500 epochs per fold, density_prior rna_count_based,
# seed 42), the same spatial object (combined_clustered.h5ad) and the same
# reference (BIMP_atlas_qw_combined.h5ad). Only the tangram2 commit and
# the output directory differ.
#
# H100 only: a 100,000 x 14,564 mapping does not fit a 40 GB A100 (CUDA out of
# memory); the canonical run used an H100.
#
# --out-dir is LOAD-BEARING: without it cross-validate writes over the
# d19f184 scores and results/tangram_qc/tangram_loocv.png.
#
# Then the random-gene-pair null and the increment over it:
#   PYTHONPATH=src:scripts .venv/bin/python scripts/identity/loocv_null.py \
#       --loocv-csv results/tangram/loocv_0af3056_20261020/combined_clustered_loocv_scores.csv \
#       --out results/reframe/loocv_null_0af3056_20261020
#
# Usage (from the repo root; REPO defaults to the shared checkout):
#   BIMP_SBATCH_EXTRA=--output=logs/loocv_0af3056_%j.out bash config/sbatch.sh scripts/identity/run_loocv_tangram2_0af3056.sh [OUT_DIR]
#SBATCH --job-name=loocv_0af3056
#SBATCH --output=logs/loocv_0af3056_%j.out
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

REPO="${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)}"
VENV="${VENV:-$REPO/.venv}"
OUT_DIR="${1:-$REPO/results/tangram/loocv_0af3056_$(date +%Y%m%d)}"
export BIMP_SNRNA_REF="${BIMP_SNRNA_REF:-$REPO/data/sn/BIMP_atlas_qw_combined.h5ad}"
export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

# Check the tangram2 the venv actually imports (an editable --no-deps install;
# the .pth names its source tree), not whatever submodule REPO has checked out.
TG_SRC=$(cat "$VENV"/lib/python*/site-packages/__editable__.tangram2-*.pth)
TG_COMMIT=$(git -C "$TG_SRC" rev-parse --short HEAD)
if [ "$TG_COMMIT" != "0af3056" ]; then
    echo "tangram2 submodule is at $TG_COMMIT, expected 0af3056" >&2
    exit 1
fi

cd "$REPO/pipelines/combined_atlas"
echo "=== LOO-CV | tangram2=$TG_COMMIT | ref=$(basename "$BIMP_SNRNA_REF") | out=$OUT_DIR | host=$(hostname) | $(date -Is) ==="
echo "repo HEAD $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$VENV/bin/python" tangram_mapping.py cross-validate \
    --input-file combined_clustered \
    --max-cells 100000 --max-ref-cells 15000 \
    --num-epochs 500 --density-prior rna_count_based --seed 42 \
    --device cuda \
    --out-dir "$OUT_DIR"

echo "=== done | $(date -Is) ==="
