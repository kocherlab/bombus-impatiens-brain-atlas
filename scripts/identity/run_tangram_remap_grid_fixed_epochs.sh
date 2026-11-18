#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh scripts/identity/run_tangram_remap_grid_fixed_epochs.sh)
# One cell of the Tangram re-map grid, through the PRODUCTION train path.
#
# Grid settings:
#   - the FULL reference, --max-ref-cells 0. Cluster mode collapses to 29
#     centroids regardless, so subsampling bought nothing and added a per-seed
#     random draw whose variance exceeded every between-arm difference.
#   - density_prior is a parameter, because "none" is the ONLY way to switch the
#     density term off (lambda_d=0 is coerced to 1, mapping_utils.py:225).
#   - no ref_sampling axis: with the full reference it does not exist.
#
# --out-tag is LOAD-BEARING: without it this writes the canonical object.
#
# Usage: bash config/sbatch.sh scripts/identity/run_tangram_remap_grid_fixed_epochs.sh <cell> <density_prior> <lambda_d> <source_density> <seed>
#SBATCH --job-name=remap2
#SBATCH --output=logs/remap2_%j.out
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

CELL="$1"; DP="$2"; LD="$3"; SD="$4"; SEED="$5"
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
REF="$REPO/data/sn/BIMP_atlas_qw_combined.h5ad"
TAG="g2_${CELL}_s${SEED}"

cd "$REPO/pipelines/combined_atlas"
mkdir -p "$REPO/logs"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$REPO/src"

echo "cell=$CELL density_prior=$DP lambda_d=$LD source_density=$SD seed=$SEED tag=$TAG"
echo "host=$(hostname)  tangram2=$(git -C "$REPO/submodules/tangram2" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "start $(date -Is)"

"$REPO/.venv/bin/python" tangram_mapping.py train \
    --reference "$REF" \
    --mode clusters --cluster-label cell_type \
    --out-tag "$TAG" \
    --num-epochs 500 --max-cells 1400000 \
    --max-ref-cells 0 --markers-per-cluster 200 \
    --density-prior "$DP" --lambda-d "$LD" --source-density "$SD" \
    --seed "$SEED" \
    --device cuda

echo "done  $(date -Is)"
