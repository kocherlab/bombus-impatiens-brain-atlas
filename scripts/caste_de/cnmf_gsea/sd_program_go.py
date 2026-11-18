#!/usr/bin/env python
"""Supplementary Data: GO terms of the 50 joint cNMF programs (k = 50).

Port of the program half of ``out/fig4_supp/scripts/supp_full.py`` (table
design: I. Traniello; md5 70ce9f8c44e4af04ef773d8910976545), which wrote
``Supp_program_GO_terms.csv``:
every positive-NES GO term at FDR < 0.25 from the all-GO-library prerank of each
program (``program_go_all.py``; source ``prog_gsea.py``), obsolete and unnamed terms
removed (``annotate``), with

- ``FDR_BP_only_library``: the same term's FDR from the BP-only-library run of
  the five figure programs 18, 20, 6, 7, 12 (``program_go_all.py --bp-only``, a
  reconstruction: the source script for that run is not on disk);
  blank for other programs and for MF / CC terms;
- ``replicated_caste_effect``: the cell types where the program meets the
  replicated caste call (clean notebook cell 25's ``joint_k50_caste.csv``: one sign
  in all three trials, |mean| >= 0.5 SD, weakest trial >= 0.25 SD, usage >= 2%);
- ``in_figure``: programs 18, 20, 6, 7, 12, 26 (``supp_full.py`` ``FIG``);
- ``tier`` (FDR < 0.05 / < 0.25), ``n_leading_edge`` and the leading-edge genes
  by gene name, else the BIMP_v4.3 id without its prefix (``named``).

Rows sorted by program, FDR, then NES descending; NES rounded to 3 decimals and
the FDRs and nominal p to 4, as the source script does. Program numbers are the
repo factorization's (identical to the saved run's: ``program_map.csv``,
identity at r = 1.0). Gene names come from the DE table's genes
(``source="repo"``: the repo fit; ``"saved"``: the saved ``de_all.csv.gz``), the
program caste table likewise.

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/caste_de/cnmf_gsea/sd_program_go.py [--source saved]
"""

import argparse

import numpy as np
import pandas as pd
from cnmf_gsea_paths import PROGRAM_MAP, SAVED_PROGRAM_COL
from sd_common import (
    SI,
    SI_SAVED,
    add_source_arg,
    caste_csv,
    clean_ct,
    de_csv,
    go_annotate,
    name_map,
    named,
    parse_obo,
)
from sd_common import write as write_csv

OUT = SI / "Supp_program_GO_terms.csv"
PROG_GSEA = SI / "prog_gsea"  # program_go_all.py
BP_ONLY = SI / "prog_go_bp_only.csv.gz"  # program_go_all.py --bp-only
FIG = [18, 20, 6, 7, 12, 26]  # supp_full.py FIG, saved program numbers
COLUMNS = [
    "program",
    "in_figure",
    "replicated_caste_effect",
    "namespace",
    "go_id",
    "term",
    "NES",
    "nominal_p",
    "FDR_all_GO_library",
    "FDR_BP_only_library",
    "tier",
    "tag_pct",
    "gene_pct",
    "n_leading_edge",
    "leading_edge_genes",
]


def program_gsea() -> pd.DataFrame:
    """The 50 raw prerank tables of ``program_go_all.py``."""
    files = sorted(PROG_GSEA.glob("prog_*.csv.gz"))
    if len(files) != 50:
        raise FileNotFoundError(f"{len(files)}/50 program tables in {PROG_GSEA}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def replicated(source: str, programs) -> dict[int, str]:
    """``supp_full.py`` ``rep``: replicated caste calls of each program."""
    j = pd.read_csv(caste_csv(source))
    j["ct"] = clean_ct(j.cell_type)
    jj = j[
        j.same & (j["mean"].abs() >= 0.5) & (j.min_abs >= 0.25) & (j.usage_pct >= 2.0)
    ]
    rep = {}
    for p in sorted(programs):
        calls = [
            f"{r.ct} ({'queen' if r['mean'] > 0 else 'worker'}-up, {r['mean']:+.2f} SD)"
            for _, r in jj[jj.program == p].iterrows()
        ]
        rep[p] = "; ".join(calls)
    return rep


def build(source: str = "repo") -> pd.DataFrame:
    """The program GO table, the source table's column names and order."""
    nmap = name_map(pd.read_csv(de_csv(source), usecols=["gene"]))
    P = go_annotate(program_gsea(), parse_obo())
    pmap = pd.read_csv(PROGRAM_MAP).set_index(SAVED_PROGRAM_COL)["repo_program"]
    fig = [int(pmap[p]) for p in FIG]

    Pp = P[P.NES > 0].copy()
    bp5 = pd.read_csv(BP_ONLY)[["program", "go_id", "FDR q-val"]]
    bp5 = bp5.rename(columns={"FDR q-val": "FDR_BP_only_library"})
    Pp = Pp.merge(bp5, on=["program", "go_id"], how="left")
    Pp["n_leading_edge"] = Pp.Lead_genes.map(
        lambda s: len([x for x in str(s).split(";") if x])
    )
    Pp["leading_edge_genes"] = Pp.Lead_genes.map(lambda s: named(s, nmap))
    fdr = Pp["FDR q-val"]
    Pp["tier"] = np.where(
        fdr < 0.05, "FDR<0.05", np.where(fdr < 0.25, "0.05<=FDR<0.25", "n.s.")
    )
    rep = replicated(source, P.program.unique())
    Pp["replicated_caste_effect"] = Pp.program.map(rep).replace("", np.nan)
    Pp["in_figure"] = Pp.program.isin(fig)
    PT = (
        Pp[Pp["FDR q-val"] < 0.25]
        .rename(
            columns={
                "NOM p-val": "nominal_p",
                "FDR q-val": "FDR_all_GO_library",
                "Tag %": "tag_pct",
                "Gene %": "gene_pct",
            }
        )[COLUMNS]
        .sort_values(
            ["program", "FDR_all_GO_library", "NES"], ascending=[True, True, False]
        )
    )
    for c, d in [
        ("NES", 3),
        ("FDR_all_GO_library", 4),
        ("FDR_BP_only_library", 4),
        ("nominal_p", 4),
    ]:
        PT[c] = PT[c].round(d)
    return PT.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arg(ap)
    args = ap.parse_args()
    out = args.out or (OUT if args.source == "repo" else SI_SAVED / OUT.name)
    write_csv(build(args.source), out)


if __name__ == "__main__":
    main()
