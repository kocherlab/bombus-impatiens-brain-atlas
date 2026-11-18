#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh scripts/identity/run_tangram_remap_no_optic_kc.sh)
# noolkc variant of run_tangram_remap_grid.sh (re-map grid, production train
# path), for the "delete the generic optic-lobe Kenyon cells entry, retrain"
# experiment (any-Kenyon-in-calyx share with and without that entry).
#
# Identical config to the canonical R2 run (same density_prior/lambda_d/
# source_density/seed/epochs/scale/max-ref-cells/markers-per-cluster), the
# ONLY difference is --reference pointing at the noOLKC reference instead of
# the full combined reference. --block-mass is left empty, which is the
# R2-equivalent default (mapping_utils falls back to the uniform prior's
# cluster-count split; --block-mass is opt-in).
#
# --out-tag is LOAD-BEARING and deliberately NOT "noolkc": that tag belongs to a
# separate pre-R2-config run (results/tangram/combined_clustered_noolkc_tangram.h5ad).
# This tag is g2_R2_noolkc_s<seed> so it cannot collide with the canonical
# object or that noolkc object.
#
# Usage: bash config/sbatch.sh scripts/identity/run_tangram_remap_no_optic_kc.sh <cell> <density_prior> <lambda_d> <source_density> <seed>
#   e.g.  bash config/sbatch.sh scripts/identity/run_tangram_remap_no_optic_kc.sh R2 rna_count_based 1 cluster_density_within 42
#SBATCH --job-name=remap2_noolkc
#SBATCH --output=logs/remap2_noolkc_%j.out
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

CELL="$1"; DP="$2"; LD="$3"; SD="$4"; SEED="$5"; BM="${6:-}"; EPOCHS="${7:-500}"
SCALE="${8:-linear}"
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
REF="$REPO/data/sn/BIMP_atlas_qw_combined_noOLKC.h5ad"
TAG="g2_${CELL}_noolkc_s${SEED}"

cd "$REPO/pipelines/combined_atlas"
mkdir -p "$REPO/logs"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$REPO/src"

echo "cell=$CELL density_prior=$DP lambda_d=$LD source_density=$SD seed=$SEED tag=$TAG"
echo "reference=$REF"
echo "host=$(hostname)  tangram2=$(git -C "$REPO/submodules/tangram2" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "start $(date -Is)"

"$REPO/.venv/bin/python" tangram_mapping.py train \
    --reference "$REF" \
    --mode clusters --cluster-label cell_type \
    --out-tag "$TAG" \
    --num-epochs "$EPOCHS" --max-cells 1400000 --data-scale "$SCALE" \
    --max-ref-cells 0 --markers-per-cluster 200 \
    --density-prior "$DP" --lambda-d "$LD" --source-density "$SD" \
    --seed "$SEED" ${BM:+--block-mass "$BM"} \
    --device cuda

echo "done  $(date -Is)"
