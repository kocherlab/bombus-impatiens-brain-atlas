#!/usr/bin/env bash
# Regenerate manuscript figures by slug: a thin wrapper around the `generator`
# rules of scripts/paper/Snakefile (manuscript_figure_registry.py names them).
# Only stale generators run, after their analysis steps; --force reruns the named
# figures and everything upstream of them.
#
# Usage:
#   bash scripts/paper/generate_all_figures.sh                    # every registered figure
#   bash scripts/paper/generate_all_figures.sh main               # every main-text figure
#   bash scripts/paper/generate_all_figures.sh si                 # every Supplementary figure
#   bash scripts/paper/generate_all_figures.sh caste_de mosaicfield   # figures by slug
#   bash scripts/paper/generate_all_figures.sh --list             # slug -> generator
#   bash scripts/paper/generate_all_figures.sh --plan [targets]   # dry run: what would run
#   bash scripts/paper/generate_all_figures.sh --force [targets]  # rerun even if up to date
#
# Environment: BIMP_PY (interpreter, default .venv/bin/python), SNAKEMAKE
# (default: snakemake on PATH, else uvx), CORES (default 4). Then copy into the
# manuscript with scripts/paper/sync_manuscript_figures.py --apply.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# CPU pins, as in the Snakefile's shell prefix (which keeps these values):
# numpy's AVX-512 kernels for log/log1p/exp round differently in the last bit
# from the AVX2 ones, so an Intel node and an AMD node wrote different CSVs from
# the same code and seed. Pinning numpy to its AVX2 kernels, OpenBLAS to the
# Haswell kernels (core type ZEN) and jax/XLA (scCODA) to AVX2 code makes every
# node match the AMD main partition. BIMP_FIG_SUBSAMPLE_FRACTION defaults to 1,
# the shipped artwork (the scripts' own default, 0.1, is a preview).
NPY_AVX2_ONLY="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
export NPY_DISABLE_CPU_FEATURES="${NPY_DISABLE_CPU_FEATURES-$NPY_AVX2_ONLY}"
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-ZEN}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_max_isa=AVX2}"
export BIMP_FIG_SUBSAMPLE_FRACTION="${BIMP_FIG_SUBSAMPLE_FRACTION:-1}"
export MPLBACKEND=Agg
export BIMP_PY="${BIMP_PY:-$PWD/.venv/bin/python}"

if [ -n "${SNAKEMAKE:-}" ]; then
    read -r -a SM <<<"$SNAKEMAKE"
elif command -v snakemake >/dev/null; then
    SM=(snakemake)
else
    SM=(uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake)
fi

registry() {
    env PYTHONPATH=src:scripts "$BIMP_PY" - "$@" <<'EOF'
import sys
from pathlib import Path

from paper.manuscript_figure_registry import FIGURES, REPO

mode, *args = sys.argv[1:]
if mode == "list":
    for f in FIGURES:
        print(f"{f.slug:28s} {f.kind:5s} {f.generator}")
    sys.exit()
args = args or ["all"]
slugs = {f.slug for f in FIGURES}
want = []
for a in args:
    if a == "all":
        want += FIGURES
    elif a in ("main", "si"):
        want += [f for f in FIGURES if f.kind == a]
    elif a in slugs:
        want += [f for f in FIGURES if f.slug == a]
    else:
        sys.exit(f"unknown target {a!r}; valid: all main si " + " ".join(sorted(slugs)))
# the Snakefile's generator stamp for each figure (absolute: snakemake matches
# rule outputs by absolute path)
keys = sorted({Path(f.generator).stem for f in want})
print("\n".join(str(REPO / ".stamps" / "generator" / f"{k}.done") for k in keys))
EOF
}

case "${1:-}" in
    --list)
        registry list
        exit 0
        ;;
    --plan)
        shift
        out=$(registry targets "$@")  # an unknown slug exits here, not later
        mapfile -t targets <<<"$out"
        exec "${SM[@]}" -s scripts/paper/Snakefile -n --cores 1 "${targets[@]}"
        ;;
    --force)
        shift
        force=(--forceall)
        ;;
    *)
        force=()
        ;;
esac

out=$(registry targets "$@")
mapfile -t targets <<<"$out"
"${SM[@]}" -s scripts/paper/Snakefile --cores "${CORES:-4}" "${force[@]}" "${targets[@]}"
echo "Copy into the manuscript: PYTHONPATH=src:scripts $BIMP_PY scripts/paper/sync_manuscript_figures.py --apply"
