#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh scripts/kenyon/run_project_kc_populations_optic_control.sh)
#SBATCH --job-name=kc5pop_neg
#SBATCH --output=results/reframe/kc5pop_negative_control/slurm_negctrl_%j.out
#SBATCH --error=results/reframe/kc5pop_negative_control/slurm_negctrl_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Optic-neuropil negative control for the doubly-restricted 5-KC-population
# projection.  Identical config to run_project_kc_populations.sh, query swapped.
WT=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
cd "$WT" || exit 1
export OPENBLAS_NUM_THREADS=1
PYTHONPATH=src:scripts \
    "$WT/.venv/bin/python" \
    scripts/kenyon/project_kc_populations_optic_control.py
