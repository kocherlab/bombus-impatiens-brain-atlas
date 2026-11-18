# Shared settings for the histology sbatch scripts. Sourced from the repo root
# (each script cd's to $SLURM_SUBMIT_DIR first), so submit from the repo root.
#
# Override any of these in the environment before submitting, e.g.
#   DATA_REPO=/path/to/data/checkout bash config/sbatch.sh ...

# Code: the checkout you submitted from.
REPO="${REPO:-$(pwd)}"
# Data: the checkout that holds the gitignored inputs -- results/ (atlas h5ad),
# pipelines/<brain>/data/*.h5ad (per-crop h5ads) and
# pipelines/<brain>/figures/alignment_validation/aligned_images.tif (DAPI stacks).
# A worktree usually lacks them; point this at the main checkout then.
DATA_REPO="${DATA_REPO:-$REPO}"
# Python: the project venv (never `uv run` against it: it would re-sync it).
PY="${PY:-$REPO/.venv/bin/python}"
# Atlas object the viewer is exported from: export_data.py's TANGRAM_REL.
TANGRAM_REL="$(sed -n 's/^TANGRAM_REL = "\(.*\)"/\1/p' "$REPO/viewer/export_data.py")"
ATLAS="${ATLAS:-$DATA_REPO/$TANGRAM_REL}"
# Where the viewer reads its data; outputs go to histo/, histo_stains/, histo_tiles/ here.
DATA_OUT="${DATA_OUT:-$REPO/viewer/pages/data}"
TOOLS="$REPO/viewer/tools"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# scipy's OpenBLAS segfaults when it fans out over a 128-core node: cap it to the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

for f in "$TOOLS/export_histology.py" "$PY" "$ATLAS"; do
    [ -e "$f" ] || { echo "missing: $f (submit from the repo root; set DATA_REPO for a worktree)" >&2; exit 1; }
done
echo "repo $REPO | data $DATA_REPO | atlas $ATLAS | out $DATA_OUT | host $(hostname) | job ${SLURM_JOB_ID:-none}"
