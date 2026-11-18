#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh scripts/identity/run_tangram_seed_replicate.sh)
# Ad hoc replicate of run_tangram_region.sh with an explicit --seed, to test
# whether independent Tangram training runs (same config, different seed)
# agree on the ambiguous per-cell KC-subtype/optic calls -- diagnostic only,
# does not touch the tracked script or the canonical results/tangram/*.h5ad
# output (writes under a distinct --out-tag).
#
# Usage: bash config/sbatch.sh scripts/identity/run_tangram_seed_replicate.sh <SEED> <OUT_TAG>
#SBATCH --job-name=tg2_seed_ens
#SBATCH --output=logs/tangram_seed_%j.out
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
set -euo pipefail

# Overridable, same form as pipelines/combined_atlas/run_tangram_region.sh:
#   BIMP_REPO=<path> bash config/sbatch.sh scripts/identity/run_tangram_seed_replicate.sh ...
# to run a worktree's copy.
REPO="${BIMP_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)}"
VENV="${BIMP_VENV:-$REPO/.venv}"
cd "$REPO"
mkdir -p logs

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

SEED="$1"; TAG="$2"
REF="data/sn/BIMP_atlas_qw_combined.h5ad"
echo "=== tangram2 seed-ensemble | seed=$SEED tag='$TAG' | $(date) ==="

BIMP_SNRNA_REF="$REF" "$VENV/bin/python" \
    pipelines/combined_atlas/tangram_mapping.py train \
    --reference "$REF" \
    --region all \
    --out-tag "$TAG" \
    --mode clusters \
    --cluster-label cell_type \
    --num-epochs 500 \
    --max-cells 1400000 \
    --max-ref-cells 15000 \
    --markers-per-cluster 200 \
    --seed "$SEED" \
    --device cuda

echo "=== done tag='$TAG' seed=$SEED | $(date) ==="
