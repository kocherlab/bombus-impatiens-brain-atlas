#!/bin/bash
# bimp-sbatch: gpu  (submit: bash config/sbatch.sh scripts/kenyon/run_project_kc_populations_class_gated.sh)
#SBATCH --job-name=kc5pop_cg
#SBATCH --output=results/tangram_ltKC/slurm_kc5pop_classgated_%j.out
#SBATCH --error=results/tangram_ltKC/slurm_kc5pop_classgated_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# Class-gated KC re-projection (scripts/kenyon/project_kc_populations_class_gated.py),
# run against CANONICAL_TANGRAM (g2_R2_s42). Same
# invocation pattern as scripts/kenyon/run_project_kc_populations.sh.
WT=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
cd "$WT" || exit 1
PYTHONPATH=src:scripts \
    "$WT/.venv/bin/python" \
    scripts/kenyon/project_kc_populations_class_gated.py
