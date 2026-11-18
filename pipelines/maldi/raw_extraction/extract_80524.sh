#!/bin/bash
# bimp-sbatch: cpu  (submit: bash config/sbatch.sh pipelines/maldi/raw_extraction/extract_80524.sh)
#SBATCH --job-name=ion_80524
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=pipelines/maldi/raw_extraction/ion_out/slurm_%j.log
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)

set -euo pipefail
cd $REPO/pipelines/maldi/raw_extraction
mkdir -p ion_out
source .venv/bin/activate

PPM=${1:-15}
WORKERS=${SLURM_CPUS_PER_TASK:-64}
# SAN is ~125 MB/s sequential but only ~40 MB/s under scattered parallel reads.
# So stage the .ibd to node-local disk with ONE sequential copy, then let the
# 64 workers read from fast local SSD (where concurrency actually helps).
STAGE="/tmp/${SLURM_JOB_ID:-$$}_ion"

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} cpus=$WORKERS ppm=$PPM (stage+local read)"
python slurm_extract.py \
  cache/offsets_80524.parquet \
  data/Princeton_imzmls/princeton_80524.ibd \
  princeton_80524 "$PPM" "$WORKERS" "$STAGE"

rm -rf "$STAGE" || true
echo "done"
