#!/usr/bin/env python
"""Step 6: per-cell-type GSEA of the caste Wald statistic (clean cell 15).

Source: 01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb cell 15
(md5 184efd9af8192fd09f7329f1eeaa9f17).

For each of the 11 annotated glial / neuronal types (cell 11), GSEApy prerank
on the DESeq2 Wald statistic (+ = queen-up) of genes with baseMean > 10,
eggNOG GO library (all namespaces), min_size 10, max_size 500, 1,000
permutations, seed 0, 1 thread. Writes ``pipeline/<source>/gsea_all.csv.gz``
(the notebook's ``gsea_all.csv.gz``) and prints the GO namespace counts.
``--source repo`` ranks by the repo's DESeq2 table (``caste_de_blocked_all.py``);
``--source saved`` by the saved ``de_all.csv.gz``.

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/gsea_celltypes.py --source repo
"""

import argparse

import gseapy as gp
import numpy as np
import pandas as pd
from cnmf_gsea_common import (
    annotated_de,
    go_library,
    go_lookup,
    load_de,
    parse_obo,
    setup,
)
from cnmf_gsea_paths import add_source_arg, source

MIN_BASE = 10  # same expression floor as the other analyses
RUN_LIBS = ["GO_BP"]


def rank_vector(de, ct, drop=frozenset()) -> pd.Series:
    """Wald statistic of one cell type's expressed genes, + = queen-up."""
    d = de[(de.cell_type == ct) & de.stat.notna() & (de.baseMean > MIN_BASE)]
    d = d[~d.gene.isin(drop)]
    return d.set_index("gene").stat.sort_values(ascending=False)


def run_gsea(
    de, libs, cell_types, drop=frozenset(), nperm=1000, seed=0, threads=1
) -> pd.DataFrame:
    """Cell 15: one prerank per cell type and library."""
    out = []
    for ct in cell_types:
        rnk = rank_vector(de, ct, drop)
        for name in RUN_LIBS:
            r = gp.prerank(
                rnk=rnk,
                gene_sets=libs[name],
                min_size=10,
                max_size=500,
                permutation_num=nperm,
                seed=seed,
                threads=threads,
                no_plot=True,
                verbose=False,
            )
            t = r.res2d.copy()
            t.insert(0, "library", name)
            t.insert(0, "cell_type", ct)
            t["n_ranked"] = len(rnk)
            out.append(t)
        print("done:", ct, f"({len(rnk)} genes ranked)")
    g = pd.concat(out, ignore_index=True)
    for c in ["ES", "NES", "NOM p-val", "FDR q-val"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g["dir"] = np.where(g.NES > 0, "queen-up", "worker-up")
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arg(ap)
    src = source(ap.parse_args().source)
    setup()
    de = load_de(src.de_csv)
    d0 = annotated_de(de)
    go_meta = parse_obo()
    libs = go_library()

    cell_types = sorted(d0.cell_type.unique())  # the 11 glial / neuronal types
    run_gsea(de, libs, cell_types).to_csv(src.gsea_csv, index=False)

    gsea = pd.read_csv(src.gsea_csv)
    # GO id / name / namespace; biological-process terms are used downstream
    gsea["go_id"] = gsea["Term"].astype(str).str.extract(r"(GO:\d{7})", expand=False)
    gsea["name"] = gsea.go_id.map(lambda g: go_lookup(go_meta, g, 0))
    gsea["ns"] = gsea.go_id.map(lambda g: go_lookup(go_meta, g, 1))
    assert (gsea.ns == "biological_process").sum() > 0
    print(gsea.ns.value_counts(dropna=False).to_dict())
    print(gsea.groupby("cell_type").n_ranked.first().to_dict())


if __name__ == "__main__":
    main()
