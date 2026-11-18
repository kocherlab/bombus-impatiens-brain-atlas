"""Bento localization by section and by Kenyon compartment.

Usage: bento_regroup.py section | compartments
Descriptive only: no caste p-values. Outputs under results/bento/by_section/ or
results/bento/kc_compartments/.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regroup_common import (  # noqa: E402
    COMP_NAME,
    COUNTS,
    DIST_BINS,
    DIST_LABELS,
    PATTERNS,
    REPO,
    aggregate_lp,
    cell_map,
    fractions,
)


def run_section():
    out = REPO / "results/bento/by_section"
    out.mkdir(parents=True, exist_ok=True)
    tabs = []
    for brain in ("queen", "worker"):
        cells, n_amb = cell_map(brain)
        print(
            brain,
            "cells mapped",
            len(cells),
            "ambiguous barcodes dropped",
            n_amb,
            flush=True,
        )
        sec, _, _ = aggregate_lp(brain, cells)
        s = sec.groupby(["brain", "slice_name"], as_index=False)[COUNTS].sum()
        s["region"] = s.slice_name.str.extract(
            r"^(output-XETG\S+?__\d{8}_\d+|output-XETG\S+?__Region_\d+__\d+__\d+)"
        )[0]
        s["caste"] = brain
        tabs.append(fractions(s))
    t = pd.concat(tabs, ignore_index=True)
    t.to_csv(out / "section_composition.csv", index=False)
    summ = t.groupby("caste")[[f"frac_{p}" for p in PATTERNS]].agg(
        ["mean", "std", "min", "max"]
    )
    summ.to_csv(out / "section_composition_summary.csv")
    print(summ.T, flush=True)


def run_compartments():
    out = REPO / "results/bento/kc_compartments"
    out.mkdir(parents=True, exist_ok=True)
    comp_rows, dist_rows, cell_rows, gene_rows = [], [], [], []
    for brain in ("queen", "worker"):
        cells, n_amb = cell_map(brain)
        print(brain, "cells mapped", len(cells), "ambiguous dropped", n_amb, flush=True)
        _, cell, gene = aggregate_lp(brain, cells)
        cell = cell[cell.comp != ""].copy()
        n_cls = cell["n_classified"].replace(0, float("nan"))
        cell["frac_nuclear"] = cell["nuclear"] / n_cls
        cell_rows.append(cell)
        cs = cell.groupby(["brain", "slice_name", "comp"], as_index=False)[
            COUNTS + ["frac_nuclear"]
        ].agg({**{p: "sum" for p in COUNTS}, "frac_nuclear": "mean"})
        cs["n_cells"] = cell.groupby(["brain", "slice_name", "comp"]).size().values
        comp_rows.append(fractions(cs))
        large = cell[cell.comp == "KC-NC"].dropna(subset=["dist"]).copy()
        large["bin"] = pd.cut(large.dist, DIST_BINS, labels=DIST_LABELS, right=False)
        db = large.groupby(
            ["brain", "slice_name", "bin"], observed=True, as_index=False
        )[COUNTS].sum()
        dist_rows.append(fractions(db))
        g = gene.copy()
        gene_rows.append(fractions(g))
    comp = pd.concat(comp_rows, ignore_index=True)
    comp["compartment"] = comp.comp.map(COMP_NAME)
    comp.to_csv(out / "compartment_composition_by_section.csv", index=False)
    pd.concat(dist_rows, ignore_index=True).to_csv(
        out / "distance_bins_by_section.csv", index=False
    )
    gr = []
    for (b, sl), g in pd.concat(cell_rows).groupby(["brain", "slice_name"]):
        lg = g[(g.comp == "KC-NC") & g.dist.notna()]
        if len(lg) > 10:
            gr.append(
                {
                    "brain": b,
                    "slice_name": sl,
                    "n_large_cells": len(lg),
                    "spearman_dist_vs_nuclear": lg[["dist", "frac_nuclear"]]
                    .corr(method="spearman")
                    .iloc[0, 1],
                }
            )
    pd.DataFrame(gr).to_csv(out / "gradient_spearman_by_section.csv", index=False)
    pd.concat(gene_rows, ignore_index=True).to_csv(
        out / "gene_pattern_by_section.csv", index=False
    )
    print("written", out, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    run_section() if mode == "section" else run_compartments()
