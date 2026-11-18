#!/bin/bash
# bimp-sbatch: none  (submit: bash config/sbatch.sh scripts/atlas/run_mosaicfield_alignment.sh)
#SBATCH --job-name=mosaic-all
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mosaicfield_noskip_%j.log


# BIMP_ROOT: repo root, derived from this script's own location
BIMP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIMP_PY="$BIMP_ROOT/.venv/bin/python"
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
cd "$REPO"
mkdir -p logs
export PYTHONUNBUFFERED=1

PYTHONPATH="$BIMP_ROOT/src" "$BIMP_PY" pipelines/wholebrain/queen/run_mosaicfield.py \
    --slices all \
    --device cuda:0 \
    --epochs 3000 \
    --output-dir results/mosaicfield_noskip \
    --debug \
    --no-skip
