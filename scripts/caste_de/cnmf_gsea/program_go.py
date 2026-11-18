#!/usr/bin/env python
"""Step 5: GO labels of the cNMF programs (GSEApy prerank on gene scores).

Sources:

- clean notebook 01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb
  (md5 184efd9af8192fd09f7329f1eeaa9f17) cell 27, ``program_bp_table``: for
  the five figure programs, every positive-NES, non-obsolete GO biological
  process term, sorted by FDR -> ``joint_k50_figure_programs_go_bp.csv.gz``.
  Panel d's row label is the top term at FDR < 0.05 (cell 30, applied in
  ``fig3_inputs.py``). The notebook's copy of this file was not saved.
- original notebook 01_QC_BIMP-Atlas_DifferentialExpression.ipynb
  (md5 5d120475e946d3953bc106195cad70ce) cell 109, ``bp_tabs``: the same BP
  tables at the permutation counts of that notebook, where its panel d
  labels come from: 500 for programs 18 and 12 (``full_bp(p, perms=500)``), and
  1,000 for 6, 7 and 20 (reused from an earlier 1,000-permutation run) ->
  ``joint_k50_figure_programs_go_bp_cell109.csv.gz``, computed with cell 27's
  function, only with ``--cell109`` (comparison). Its program 12 term is the
  label in the original notebook's figure; panel d uses cell 27's 1,000
  permutations (same term).
- original notebook cell 102 (= archive cell 74), part 1, "GO labels for every
  program": the top 5 positive-NES terms (all namespaces) of each of the 50
  programs -> ``joint_k50_go_labels.csv``. The cell is commented out in the
  notebook. The saved ``joint_k50_go_labels.csv`` is NOT this cell's output
  but cell 104's (programs 6, 7, 20 seeded from a 1,000-permutation run; 18,
  12, 11, 10, 15 at 500), so it matches this table only where the counts
  agree. Parts 2-4 of cell 102 build a summary table nothing downstream reads
  and are not ported.

All: ranks = the program's consensus gene-spectra scores; library = eggNOG GO
(all namespaces, ``cnmf_gsea_common.go_library``); min_size 10, max_size 500, 1,000
permutations unless stated, seed 0, one thread (notebook: 4). The outputs are
caches, as in the notebook: an existing file is reused (cell 102 resumes
program by program) unless ``--force``. For ``--source repo`` the figure programs
are the saved run's numbers mapped through ``program_map.csv``.

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/program_go.py --source repo
"""

import argparse

import gseapy as gp
import pandas as pd
from cnmf import cNMF
from cnmf_gsea_common import go_library, go_lookup, parse_obo, setup
from cnmf_gsea_paths import (
    CNMF_NAME,
    FIG_PROGS,
    PROGRAM_MAP,
    SAVED_PROGRAM_COL,
    add_source_arg,
    source,
)

BP = "biological_process"
PERMS = 1000
# One thread (the notebook used 4): GSEApy returns terms with tied NES (GO
# terms over the same genes) in an order that can vary between runs, so ties
# are also broken explicitly below (FDR, then NES, then GO id / name).
THREADS = 1
# gzip without a header timestamp, so a rerun writes the same bytes
GZ = {"method": "gzip", "mtime": 0}
# original cell 109: full_bp(p, perms=500) for 18 and 12; 6, 7, 20 at 1,000
CELL109_PERMS = {18: 500, 12: 500, 20: 1000, 6: 1000, 7: 1000}


def program_bp_table(sscore, p, libs, go_meta, perms=PERMS) -> pd.DataFrame:
    """Cell 27: all positive-NES GO BP terms for program p, sorted by FDR."""
    rnk = sscore[p].astype(float).sort_values(ascending=False)
    rnk = rnk[~rnk.index.duplicated()]
    pre = gp.prerank(
        rnk=rnk,
        gene_sets=libs["GO_BP"],
        min_size=10,
        max_size=500,
        permutation_num=perms,
        threads=THREADS,
        seed=0,
        outdir=None,
        verbose=False,
    )
    r = pre.res2d.copy()
    r["go_id"] = r["Term"].astype(str).str.extract(r"(GO:\d{7})", expand=False)
    r["name"] = r["go_id"].map(lambda g: go_lookup(go_meta, g, 0))
    r["ns"] = r["go_id"].map(lambda g: go_lookup(go_meta, g, 1))
    r["FDR q-val"] = r["FDR q-val"].astype(float)
    r["NES"] = r["NES"].astype(float)
    obsolete = r["name"].fillna("").str.contains("obsolete", case=False)
    r = r[(r["ns"] == BP) & (r["NES"] > 0) & ~obsolete]
    r["program"] = p
    cols = ["program", "go_id", "name", "NES", "FDR q-val"]
    r = r.sort_values(
        ["FDR q-val", "NES", "go_id"], ascending=[True, False, True], kind="mergesort"
    )
    return r.drop_duplicates("name")[cols]


def program_top_terms(sscore, p, libs, go_meta) -> pd.DataFrame:
    """Original cell 102, part 1: top 5 positive-NES terms of program p."""
    rnk = sscore[p].sort_values(ascending=False)
    pre = gp.prerank(
        rnk=rnk,
        gene_sets=libs["GO_BP"],
        min_size=10,
        max_size=500,
        permutation_num=PERMS,
        threads=THREADS,
        seed=0,
        verbose=False,
    )
    r = pre.res2d.copy()
    r["name"] = r["Term"].map(lambda t: go_meta[t][0] if t in go_meta else t)
    r = r[(r["NES"] > 0) & ~r["name"].str.contains("obsolete", case=False)]
    r = r.sort_values(
        ["FDR q-val", "NES", "Term"], ascending=[True, False, True], kind="mergesort"
    )
    r = r.drop_duplicates("name").head(5)
    r["program"] = p
    return r[["program", "name", "NES", "FDR q-val"]]


def figure_programs(src_name: str) -> dict[int, int]:
    """Figure programs (cell 27 FIG_PROGS, saved numbering) -> this run's numbers."""
    progs = {p: p for p in FIG_PROGS}
    if src_name == "repo":
        m = pd.read_csv(PROGRAM_MAP).set_index(SAVED_PROGRAM_COL)["repo_program"]
        progs = {p: int(m[p]) for p in FIG_PROGS}
    print("FIG_PROGS", progs)
    return progs


def report(table, progs) -> None:
    """Print each figure program's count of BP terms at FDR < 0.05 and < 0.25."""
    for p in progs:
        t = table[table.program == p]
        print(
            f"program {p}: {(t['FDR q-val'] < 0.05).sum()} BP terms at FDR < 0.05, "
            f"{(t['FDR q-val'] < 0.25).sum()} at FDR < 0.25"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arg(ap)
    ap.add_argument("--force", action="store_true", help="ignore cached tables")
    ap.add_argument(
        "--cell109",
        action="store_true",
        help="also the original-cell-109 (500-permutation) BP tables (comparison)",
    )
    args = ap.parse_args()
    src = source(args.source)
    setup()
    go_meta = parse_obo()
    libs = go_library()
    progs = figure_programs(src.name)

    oj = cNMF(output_dir=str(src.cnmf_dir), name=CNMF_NAME)
    usage, sscore, _, _ = oj.load_results(K=50, density_threshold=2.0)
    if not set(usage.columns) <= set(sscore.columns):
        sscore = sscore.T  # genes x programs

    # cell 27: figure programs, BP only, 1,000 permutations
    if args.force or not src.program_go_csv.exists():
        tabs = [program_bp_table(sscore, p, libs, go_meta) for p in progs.values()]
        pd.concat(tabs).to_csv(src.program_go_csv, index=False, compression=GZ)
    print("cell 27 (1,000 permutations):")
    report(pd.read_csv(src.program_go_csv), progs.values())

    # original cell 109: the same tables at that notebook's permutation counts
    if args.cell109:
        if args.force or not src.program_go_cell109_csv.exists():
            tabs = [
                program_bp_table(sscore, q, libs, go_meta, perms=CELL109_PERMS[p])
                for p, q in progs.items()
            ]
            pd.concat(tabs).to_csv(
                src.program_go_cell109_csv, index=False, compression=GZ
            )
        print("original cell 109 (permutations", CELL109_PERMS, "):")
        report(pd.read_csv(src.program_go_cell109_csv), progs.values())

    # original cell 102, part 1: every program, saved as it goes
    cache = src.go_labels_csv
    if cache.exists() and not args.force:
        done = pd.read_csv(cache)
    else:
        done = pd.DataFrame(columns=["program", "name", "NES", "FDR q-val"])
    todo = [p for p in sorted(set(sscore.columns)) if p not in set(done["program"])]
    rows = []
    for j, p in enumerate(todo, 1):
        rows.append(program_top_terms(sscore, p, libs, go_meta))
        print(f"{j}/{len(todo)} program {p}")
        pd.concat([done] + rows).to_csv(cache, index=False)


if __name__ == "__main__":
    main()
