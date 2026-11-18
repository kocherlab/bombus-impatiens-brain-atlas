#!/bin/bash
# Single entry point for the region relabel (region_relabel.smk).
#
#   ./run_region_relabel.sh                  everything: relabel, then every dependent analysis and figure
#   ./run_region_relabel.sh relabel          labels only, about two minutes
#   ./run_region_relabel.sh -n               dry run, show what would happen
#   ./run_region_relabel.sh --local          run here instead of submitting to SLURM
#
# Anything else is passed straight through to snakemake.
#
# Snakemake is not installed in the project venv (nor on PATH), so we fetch it
# as an isolated uvx tool. That deliberately does not touch the project's
# dependencies -- the analysis scripts themselves run under the shared .venv
# interpreter, which the Snakefile calls by absolute path. Not `uv run`: it
# re-syncs the hand-built .venv that every worktree shares and drops the
# --no-deps tangram2 install.
set -euo pipefail

cd "$(dirname "$0")"

UVX="${UVX:-$(command -v uvx || echo "$HOME/.local/bin/uvx")}"
PROFILE="../../config/profiles/slurm"

args=()
for a in "$@"; do
  case "$a" in
    --local) PROFILE="../../config/profiles/local" ;;
    *) args+=("$a") ;;
  esac
done

exec "$UVX" --from snakemake --with snakemake-executor-plugin-slurm \
  snakemake -s region_relabel.smk --profile "$PROFILE" "${args[@]+"${args[@]}"}"
