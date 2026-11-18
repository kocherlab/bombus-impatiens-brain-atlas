#!/usr/bin/env python
"""GO GSEA of every cNMF program, for the program GO Supplementary Data table.

Sources (``<snrna_source>/out/
fig4_supp/scripts/``, the scripts behind the saved ``Supp_program_GO_terms.csv``):

- ``prog_gsea.py`` (md5 3eeb7b3b30b02507308df83c4b2b7c9b), last block: one
  GSEApy prerank per program on its consensus gene-spectra score, eggNOG GO
  library with every namespace (``cnmf_gsea_common.go_library``), min_size 10,
  max_size 500, 1,000 permutations, seed 0, 1 thread. Raw ``res2d`` + ``program`` ->
  ``si_tables/prog_gsea/prog_<p>.csv.gz`` (source: ``scratchpad/prog_gsea/``).
- the BP-only library test, RECONSTRUCTED: the script that wrote the saved
  ``scratchpad/bp_out/prog_go.csv.gz`` (merged in by ``supp_full.py`` as
  ``FDR_BP_only_library``) is not on disk; ``rerun_gsea_bp.py`` (md5
  7ecfcba8efed5941f49f71579825e07a) is its cell-type variant with the same
  library cut (lines 189-190: non-obsolete biological-process terms only) and the
  same ``PROGRAM_GO_CSV`` path. Here that library goes through clean-notebook
  cell 27's ``program_bp_table`` (``program_go.py``; 1,000 permutations, seed 0,
  4 threads) for the figure programs 18, 20, 6, 7, 12 ->
  ``si_tables/prog_go_bp_only.csv.gz``. At 4 dp it matches the saved column except 3
  rows off by 1e-4 and one tie between GO:0044106 / GO:0009308 (same name after
  alt_id lookup; ``drop_duplicates`` keeps a different one).

Residual vs the saved table: NES and nominal p match to 3 / 4 dp everywhere, but
15 all-library FDRs differ by 1e-4 at 4 dp. The saved unrounded values
(``prog_go_six.csv.gz`` in ``cnmf_gsea_paths.SAVED``) show 4 terms in programs
6, 7, 12 with NES off by <= 0.0015, which moves those programs' FDRs (max
0.0033 unrounded);
programs 18, 20, 26 are identical. A rerun of program 6 here reproduced the first
run exactly, so it is not run-to-run noise in this env; the saved run used a
different thread count and machine.

``sd_program_go.py`` turns both into the table. The factorization is the repo's
(``--source repo``, byte-identical to the saved run: ``pipeline/parity``);
program numbers are the repo run's, the saved numbers mapped through
``program_map.csv`` for the BP-only
figure programs. Writes only ``results/caste_programs/si_tables/``; an
existing output is kept unless ``--force``.

Run (5-7 min per program; BP-only about 6 min), from the repo root::

    export PYTHONPATH=src:scripts MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 \\
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    PY=${CNMF_ENV:-.venv-cnmf}/bin/python  # config/envs/cnmf.txt
    L=results/caste_programs/si_tables/logs; mkdir -p $L
    sbatch $BIMP_SBATCH_CPU $BIMP_SBATCH_SHORT --export=ALL \\
      -c 1 --mem=16G -t 01:00:00 \\
        --array=0-49%25 -o $L/prog_gsea_%a.log --wrap="$PY \\
        scripts/caste_de/cnmf_gsea/program_go_all.py --index \\$SLURM_ARRAY_TASK_ID"
    sbatch $BIMP_SBATCH_CPU $BIMP_SBATCH_SHORT --export=ALL \\
      -c 4 --mem=16G -t 01:00:00 \\
        -o $L/prog_go_bp_only.log --wrap="$PY \\
        scripts/caste_de/cnmf_gsea/program_go_all.py --bp-only"
"""

import argparse

import gseapy as gp
import pandas as pd
from cnmf import cNMF
from cnmf_gsea_common import go_library, parse_obo, setup
from cnmf_gsea_paths import CNMF_NAME, R9, source
from program_go import figure_programs, program_bp_table

SI = R9 / "si_tables"
PROG_GSEA = SI / "prog_gsea"
BP_ONLY = SI / "prog_go_bp_only.csv.gz"


def bp_only_library(libs, go_meta) -> dict[str, dict[str, list[str]]]:
    """``rerun_gsea_bp.py`` 189-190: keep non-obsolete biological-process terms."""
    keep = {
        k: v
        for k, v in libs["GO_BP"].items()
        if go_meta.get(k, (None, None))[1] == "biological_process"
        and not go_meta[k][0].startswith("obsolete")
    }
    return {"GO_BP": keep}


def program_gsea(sscore, p, libs) -> pd.DataFrame:
    """``prog_gsea.py``: raw prerank table of program p, all-GO library."""
    rnk = sscore[p].astype(float).sort_values(ascending=False)
    rnk = rnk[~rnk.index.duplicated()]
    r = gp.prerank(
        rnk=rnk,
        gene_sets=libs["GO_BP"],
        min_size=10,
        max_size=500,
        permutation_num=1000,
        threads=1,
        seed=0,
        outdir=None,
        verbose=False,
    ).res2d.copy()
    r["program"] = p
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", type=int, help="program position (0-49); else all")
    ap.add_argument("--bp-only", action="store_true", help="the BP-only test only")
    ap.add_argument("--force", action="store_true", help="overwrite outputs")
    args = ap.parse_args()
    src = source("repo")
    setup()
    go_meta = parse_obo()
    libs = go_library()
    oj = cNMF(output_dir=str(src.cnmf_dir), name=CNMF_NAME)
    usage, sscore, _, _ = oj.load_results(K=50, density_threshold=2.0)
    if not set(usage.columns) <= set(sscore.columns):
        sscore = sscore.T  # genes x programs
    SI.mkdir(parents=True, exist_ok=True)

    if args.bp_only:
        if BP_ONLY.exists() and not args.force:
            print("exists:", BP_ONLY)
            return
        bp = bp_only_library(libs, go_meta)
        print(f"BP-only library: {len(bp['GO_BP'])} terms")
        progs = figure_programs("repo")
        tabs = [program_bp_table(sscore, q, bp, go_meta) for q in progs.values()]
        pd.concat(tabs).to_csv(BP_ONLY, index=False)
        print("wrote", BP_ONLY)
        return

    PROG_GSEA.mkdir(parents=True, exist_ok=True)
    progs = sorted(sscore.columns)
    todo = progs if args.index is None else [progs[args.index]]
    for p in todo:
        out = PROG_GSEA / f"prog_{p}.csv.gz"
        if out.exists() and not args.force:
            print("exists:", out)
            continue
        r = program_gsea(sscore, p, libs)
        r.to_csv(out, index=False)
        print("DONE program", p, len(r))


if __name__ == "__main__":
    main()
