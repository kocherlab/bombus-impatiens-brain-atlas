#!/usr/bin/env python
"""Recompute the gene rule recorded for the viewer's imputed-gene projection
(atlas_v2) and compare gene sets.

``combined_clustered_g2_R2_s42_imputed_atlas_v2.h5ad`` records in
``uns['provenance']``::

    gene_rule: reference genes detected (counts>0) in >=1% of nuclei of at least
    one of the 29 cell_type groups (9,702), plus all
    BIMPv4.3_neuropeptides_MALDI.tsv precursors and Xenium panel genes present in
    the reference;  n_genes 9,710

The projection script it names (``imputed_v2/project_v2.py``) is not in the repo,
so the rule is re-derived here from ``data/sn/BIMP_atlas_qw_combined.h5ad`` and
compared with the object's gene list (var index, read with h5py). The 100 measured
Xenium genes are part of the 9,710, so inferred = 9,710 - 100.

Writes ``results/round5/viewer_gene_rule/gene_rule_check.csv``.

    PYTHONPATH=src:scripts .venv/bin/python scripts/identity/viewer_imputed_gene_rule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (
    PANEL_CROSSWALK,  # noqa: E402
    SN_COMBINED,  # noqa: E402
)

OUT = REPO / "results/round5/viewer_gene_rule"
V2 = REPO / "results/tangram/combined_clustered_g2_R2_s42_imputed_atlas_v2.h5ad"
CANON = REPO / "results/tangram/combined_clustered_g2_R2_s42_imputed.h5ad"
OLD = REPO / "results/tangram/combined_clustered_imputed.h5ad"
MALDI = REPO / "data/panel/BIMPv4.3_neuropeptides_MALDI.tsv"
MAP = PANEL_CROSSWALK


def var_names(p: Path) -> set[str]:
    with h5py.File(p, "r") as f:
        g = f["var"]
        k = g.attrs.get("_index", "_index")
        k = k.decode() if isinstance(k, bytes) else k
        return {
            (x.decode() if isinstance(x, bytes) else str(x)).lower() for x in g[k][:]
        }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(SN_COMBINED, backed="r")
    genes = np.array([g.lower() for g in a.var_names])
    ct = a.obs["cell_type"].astype(str).values
    groups = sorted(set(ct))
    X = a.layers["counts"] if "counts" in a.layers else a.X
    X = X.to_memory() if hasattr(X, "to_memory") else X
    X = sp.csr_matrix(X)
    G = sp.csr_matrix(
        (np.ones(len(ct)), (np.arange(len(ct)), [groups.index(c) for c in ct])),
        shape=(len(ct), len(groups)),
    )
    det = np.asarray(((X > 0).astype(np.float32).T @ G).todense())
    frac = det / np.asarray(G.sum(0)).ravel()
    rule = set(genes[(frac >= 0.01).any(axis=1)])
    maldi = pd.read_csv(MALDI, sep="\t")
    np_ids = {str(x).split("-")[0].lower() for x in maldi["gene_id"].dropna()}
    panel = {str(x).lower() for x in read_panel_crosswalk()["v43_id"].dropna()}
    inref = set(genes)
    extra = (np_ids | panel) & inref
    union = rule | extra
    v2 = var_names(V2)
    rows = [
        ("cell_type groups", len(groups)),
        ("reference genes", len(genes)),
        ("genes detected in >=1% of nuclei of >=1 group", len(rule)),
        ("MALDI precursors + panel genes present in the reference", len(extra)),
        ("  of which not already in the 1% set", len(extra - rule)),
        ("rule union", len(union)),
        ("atlas_v2 var", len(v2)),
        ("rule union == atlas_v2 var", union == v2),
        ("in union but not in atlas_v2", len(union - v2)),
        ("in atlas_v2 but not in union", len(v2 - union)),
        (
            "atlas_v2 genes in the 09-09 canonical imputed object",
            len(v2 & var_names(CANON)),
        ),
        (
            "atlas_v2 genes in the 07-01 old default imputed object",
            len(v2 & var_names(OLD)),
        ),
        ("measured panel genes in atlas_v2", len(v2 & panel)),
        ("inferred = atlas_v2 var - measured (100)", len(v2) - 100),
    ]
    df = pd.DataFrame(rows, columns=["quantity", "value"])
    df.to_csv(OUT / "gene_rule_check.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
