#!/bin/bash
# bimp-sbatch: cpu  (submit: bash config/sbatch.sh pipelines/maldi/raw_extraction/extract_panel_80524.sh)
#SBATCH --job-name=panel_80524
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=pipelines/maldi/raw_extraction/ion_out/panel_%j.log
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)

set -euo pipefail
cd $REPO/pipelines/maldi/raw_extraction
mkdir -p ion_out
source .venv/bin/activate
STAGE="/tmp/${SLURM_JOB_ID:-$$}_panel"
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} (panel extract, stage+local)"
python extract_panel.py \
  cache/offsets_80524.parquet \
  data/Princeton_imzmls/princeton_80524.ibd \
  channel_spec.json \
  princeton_80524 \
  ion_out/panel_princeton_80524.parquet \
  "${SLURM_CPUS_PER_TASK:-64}" "$STAGE"
rm -rf "$STAGE" || true
echo "done"
