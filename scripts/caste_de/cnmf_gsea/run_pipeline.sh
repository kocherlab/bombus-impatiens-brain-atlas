#!/bin/bash
# Slurm driver for the in-repo port of the joint cNMF / GSEA caste analysis
# (analysis design: I. Traniello; fig:caste_de panel d). Run from the repo (or
# worktree) root; every stage submits
# Slurm jobs and prints their ids. One command runs everything, in order:
#
#   bash scripts/caste_de/cnmf_gsea/run_pipeline.sh all
#
# Stages (each takes DEP="--dependency=afterok:<id>" to chain by hand):
#   freeze          freeze_inputs.py      copy the inputs + saved outputs (the ONLY
#                                         step that reads the snRNA-seq tree)
#   inputs          check_inputs.py       repo atlas == the atlas the analysis read
#   prepare         cnmf_prepare.py       cell 21: cnmf prepare, writes run_joint.sh
#   factorize       run_joint.sh          cell 21's script: 16 workers, combine
#   caste SRC       cnmf_caste.py         cells 23, 25 (SRC = repo | saved)
#   map             program_map.py        saved program numbers -> the repo run's
#   null SRC        label_swap_null.py    original cells 111, 113
#   go SRC          program_go.py         cell 27 + original cell 102 (GSEA)
#   gsea SRC        gsea_celltypes.py     cell 15 (GSEA)
#   fig3inputs      fig3_inputs.py        panel d tables (cell 30 labels)
#   parity          parity.py             ported vs verbatim vs the saved outputs
#   figure          fig_caste_de.py       fig:caste_de (repo .venv)
#
# Env: .venv-cnmf at the repo root, built from config/envs/cnmf.txt (exact versions of
# the analysis's sn-tools env; build recipe in that file):
#   uv venv --python 3.10 .venv-cnmf
#   uv pip install --python .venv-cnmf -r config/envs/cnmf.txt
# CNMF_ENV=<venv dir> points at another build of the same file (the env that
# produced results/caste_programs was a separate build whose `uv pip freeze`
# equals config/envs/cnmf.txt). Never the shared .venv (that runs only the figure).
# Partition and QOS: config/profiles/slurm/sbatch.env (or $BIMP_SBATCH_ENV).
set -euo pipefail
ENV="${CNMF_ENV:-$(git rev-parse --show-toplevel)/.venv-cnmf}/bin"
[ -x "$ENV/python" ] || { echo "no python in $ENV: build .venv-cnmf (config/envs/cnmf.txt) or set CNMF_ENV" >&2; exit 1; }
P=scripts/caste_de/cnmf_gsea
R9=results/caste_programs
L=$R9/pipeline/logs
mkdir -p "$L"
export PYTHONPATH=src:scripts MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# shellcheck disable=SC1090
source "${BIMP_SBATCH_ENV:-config/profiles/slurm/sbatch.env}"
SB="sbatch --parsable $BIMP_SBATCH_CPU --export=ALL"

stage() {  # stage NAME [SRC] -> prints the job id
  local s=$1 src=${2:-}
  case "$s" in
  freeze) $SB $BIMP_SBATCH_SHORT -c 2 --mem=64G -t 01:00:00 -J cnmf_freeze -o $L/freeze.log ${DEP:-} \
      --wrap="$ENV/python $P/freeze_inputs.py" ;;
  inputs) $SB $BIMP_SBATCH_SHORT -c 2 --mem=64G -t 01:00:00 -J cnmf_inputs -o $L/check_inputs.log ${DEP:-} \
      --wrap="$ENV/python $P/check_inputs.py" ;;
  prepare) $SB $BIMP_SBATCH_SHORT -c 2 --mem=64G -t 01:00:00 -J cnmf_prepare -o $L/cnmf_prepare.log ${DEP:-} \
      --wrap="$ENV/python $P/cnmf_prepare.py" ;;
  factorize) $SB $BIMP_SBATCH_DAY -c 16 --mem=96G -t 24:00:00 -J cnmf_factorize -o $L/factorize.log ${DEP:-} \
      --wrap="bash $R9/pipeline/cnmf_joint/run_joint.sh" ;;
  caste) $SB $BIMP_SBATCH_DAY -c 4 --mem=64G -t 06:00:00 -J cnmf_caste_$src -o $L/caste_$src.log ${DEP:-} \
      --wrap="$ENV/python $P/cnmf_caste.py --source $src" ;;
  map) $SB $BIMP_SBATCH_SHORT -c 2 --mem=32G -t 01:00:00 -J cnmf_map -o $L/program_map.log ${DEP:-} \
      --wrap="$ENV/python $P/program_map.py" ;;
  null) $SB $BIMP_SBATCH_SHORT -c 2 --mem=64G -t 01:00:00 -J cnmf_null_$src -o $L/null_$src.log ${DEP:-} \
      --wrap="$ENV/python $P/label_swap_null.py --source $src" ;;
  go) $SB $BIMP_SBATCH_DAY -c 4 --mem=64G -t 12:00:00 -J cnmf_go_$src -o $L/program_go_$src.log ${DEP:-} \
      --wrap="$ENV/python $P/program_go.py --source $src" ;;
  gsea) $SB $BIMP_SBATCH_DAY -c 2 --mem=64G -t 12:00:00 -J cnmf_gsea_$src -o $L/gsea_$src.log ${DEP:-} \
      --wrap="$ENV/python $P/gsea_celltypes.py --source $src" ;;
  fig3inputs) $SB $BIMP_SBATCH_SHORT -c 2 --mem=32G -t 01:00:00 -J cnmf_fig3in -o $L/fig3_inputs.log ${DEP:-} \
      --wrap="$ENV/python $P/fig3_inputs.py" ;;
  parity) $SB $BIMP_SBATCH_SHORT -c 2 --mem=32G -t 01:00:00 -J cnmf_parity -o $L/parity.log ${DEP:-} \
      --wrap="$ENV/python $P/parity.py" ;;
  figure) $SB $BIMP_SBATCH_SHORT -c 2 --mem=32G -t 01:00:00 -J cnmf_fig3 -o $L/fig3.log ${DEP:-} \
      --wrap=".venv/bin/python scripts/caste_de/fig_caste_de.py ${FIG_ARGS:-}" ;;
  *) echo "unknown stage $s" >&2; exit 1 ;;
  esac
}

after() { echo "--dependency=afterok:$(IFS=:; echo "$*")"; }

if [ "$1" = all ]; then
  F=$(DEP= stage freeze)
  I=$(DEP=$(after $F) stage inputs)
  PR=$(DEP=$(after $I) stage prepare)
  FA=$(DEP=$(after $PR) stage factorize)
  CR=$(DEP=$(after $FA) stage caste repo)
  M=$(DEP=$(after $CR) stage map)
  NR=$(DEP=$(after $M) stage null repo)
  GR=$(DEP=$(after $M) stage go repo)
  SR=$(DEP=$(after $I) stage gsea repo)
  CI=$(DEP=$(after $I) stage caste saved)
  NI=$(DEP=$(after $CI) stage null saved)
  GI=$(DEP=$(after $CI) stage go saved)
  SI=$(DEP=$(after $I) stage gsea saved)
  FI=$(DEP=$(after $GR $NR) stage fig3inputs)
  PA=$(DEP=$(after $FI $SR $NI $GI $SI) stage parity)
  FG=$(DEP=$(after $FI) stage figure)
  echo "freeze $F inputs $I prepare $PR factorize $FA caste_repo $CR map $M"
  echo "null_repo $NR go_repo $GR gsea_repo $SR caste_saved $CI null_saved $NI"
  echo "go_saved $GI gsea_saved $SI fig3inputs $FI parity $PA figure $FG"
else
  stage "$@"
fi
