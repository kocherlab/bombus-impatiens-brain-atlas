#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh scripts/kenyon/run_project_large_type_subclusters.sh)
#SBATCH --job-name=ltKCcb_dsrc
#SBATCH --output=results/tangram_ltKC_cb/slurm_dsource_%j.out
#SBATCH --error=results/tangram_ltKC_cb/slurm_dsource_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# CB-reference ltKC subcluster projection onto every Kenyon-compartment cell,
# classic tangram d_source abundance prior. See the script docstring.
WT=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
cd "$WT" || exit 1
mkdir -p "$WT/results/tangram_ltKC_cb"
PYTHONPATH=src:scripts \
    "$WT/.venv/bin/python" \
    scripts/kenyon/project_large_type_subclusters.py
