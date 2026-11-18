#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh pipelines/combined_atlas/run_tangram_region.sh)
# Region-specific tangram2 mapping (CB / OL / combined) against the
# annotated atlas. Bypasses `uv run` (broken: tangram2 pin gone) and uses the
# repaired .venv directly with tangram2 @ main (0af3056) installed --no-deps.
#
# Usage: bash config/sbatch.sh pipelines/combined_atlas/run_tangram_region.sh <REF_H5AD> <REGION: all|optic|central> <OUT_TAG>
#SBATCH --job-name=tg2_region
#SBATCH --output=logs/tangram_region_%j.out
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
set -euo pipefail

# Which CHECKOUT the code comes from. Defaults to the checkout you submit from;
# set BIMP_REPO to run another checkout's tangram_mapping.py
# (sbatch exports the environment by default, so `BIMP_REPO=... sbatch ...`
# is enough). VENV stays independent because worktrees share the main
# checkout's repaired .venv.
REPO="${BIMP_REPO:-$REPO}"
VENV="${BIMP_VENV:-$REPO/.venv}"
cd "$REPO"
mkdir -p pipelines/combined_atlas/logs
echo "=== code checkout: $REPO | venv: $VENV ==="

export PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

REF="$1"; REGION="$2"; TAG="$3"
echo "=== tangram2 mapping | ref=$(basename "$REF") region=$REGION tag='$TAG' | $(date) ==="

BIMP_SNRNA_REF="$REF" "$VENV/bin/python" \
    pipelines/combined_atlas/tangram_mapping.py train \
    --reference "$REF" \
    --region "$REGION" \
    --out-tag "$TAG" \
    --mode clusters \
    --cluster-label cell_type \
    --num-epochs 500 \
    --max-cells 1400000 \
    --max-ref-cells 15000 \
    --markers-per-cluster 200 \
    --device cuda

echo "=== done tag='$TAG' | $(date) ==="
