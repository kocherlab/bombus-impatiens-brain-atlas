#!/bin/bash
# Submit the whole histology export as a dependency chain. Run from the repo root on
# the cluster login node (it only submits):
#     bash viewer/tools/histology_slurm/submit_all.sh
# Steps 2 and 3 start once step 1 has finished successfully; they run side by side.
# Pass DATA_REPO=... (and any other env.sh override) in the environment; sbatch
# exports it to the jobs.
set -euo pipefail
here=viewer/tools/histology_slurm
[ -f "$here/01_histology.sbatch" ] || { echo "run from the repo root" >&2; exit 1; }
mkdir -p "$here/logs"

# partition/QOS per script class: config/sbatch.sh + config/profiles/slurm/sbatch.env
sb() { BIMP_SBATCH_EXTRA="--parsable ${1:-}" bash config/sbatch.sh "$2"; }
j1=$(sb "" "$here/01_histology.sbatch")
j2=$(sb "--dependency=afterok:$j1" "$here/02_stains.sbatch")
j3=$(sb "--dependency=afterok:$j1" "$here/03_tiles.sbatch")
echo "submitted: dapi $j1, stains $j2 (after $j1), tiles $j3 [array 0-1] (after $j1)"
echo "logs: $here/logs/   watch: squeue -j $j1,$j2,$j3"
