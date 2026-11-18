#!/bin/bash
# bimp-sbatch: cpu  (submit: bash config/sbatch.sh scripts/regions/run_region_prediction_foldwise_leiden.sh)
# Re-run the fold-wise Leiden leakage control on a compute node.
#
# Outputs: results/reframe/foldwise_leiden_v2{,_res2.0}/ (neighbours drawn from
# every cell of a section, as region_prediction.py; about 6 min per resolution).
# results/reframe/foldwise_leiden{,_res2.0}/ hold the in-region-neighbours run.
#
# On the head node this dies inside joblib's ThreadPool once thread slots are
# exhausted by other users, surfacing as
#   AttributeError: 'DummyProcess' object has no attribute 'terminate'
# which is a thread-creation failure wearing a misleading traceback, not a code
# fault. The fits run on one BLAS thread (threadpool_limits), so the cores mainly
# serve scanpy and the neighbour search.
#SBATCH --job-name=foldwise
#SBATCH --output=logs/foldwise_%j.out
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16

set -uo pipefail

REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
cd "$REPO"
mkdir -p logs

export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$REPO/src"

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  start $(date -Is)"

echo "=== default resolution (matches the pooled vocabulary's grain)"
timeout 18000 "$REPO/.venv/bin/python" scripts/regions/region_prediction_foldwise_leiden.py \
    && echo "    OK default" || echo "    FAILED default (exit $?)"

echo "=== resolution 2.0 (matches the projected vocabulary's width)"
BIMP_LEIDEN_RES=2.0 timeout 18000 "$REPO/.venv/bin/python" \
    scripts/regions/region_prediction_foldwise_leiden.py \
    && echo "    OK res2.0" || echo "    FAILED res2.0 (exit $?)"

echo "done  $(date -Is)"
