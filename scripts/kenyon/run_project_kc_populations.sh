#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh scripts/kenyon/run_project_kc_populations.sh)
#SBATCH --job-name=kc5pop_dsrc
#SBATCH --output=results/tangram_ltKC/slurm_kc5pop_%j.out
#SBATCH --error=results/tangram_ltKC/slurm_kc5pop_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# Restricted 5-KC-population Tangram reclassification, same query restriction
# and classic-tangram d_source-restored config as
# tangram_ltKC_subclusters_allkc_dsource.py. See script docstring.
WT=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
cd "$WT" || exit 1
PYTHONPATH=src:scripts \
    "$WT/.venv/bin/python" \
    scripts/kenyon/project_kc_populations.py
