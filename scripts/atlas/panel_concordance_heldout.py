#!/usr/bin/env python3
"""Cross-modality concordance on the panel genes Tangram did not train on.

si_panel_concordance.py compares snRNA and Xenium expression per atlas type,
but the Xenium side is grouped by the Tangram labels and Tangram was trained on
85 of the 100 shared genes (``uns['training_genes']`` of the canonical run).
The other 15 were held out at random from the non-focal panel genes detected
in at least 15% of cells (tangram_mapping.py). This recomputes the same
statistics, from the same per-type mean tables, on those 15 and on the 85 for
comparison:

  bulk        Spearman of each gene's mean across types, snRNA vs Xenium
  per gene    median Spearman of a gene's profile across types
  per type    median Spearman of a type's expression vector over the genes
  class       genes whose top-expressing pooled class agrees, and Cohen's kappa

Reads results/reframe/panel_concordance/{sn,xenium}_mean_by_type.csv and
per_gene.csv; writes heldout_concordance.csv and heldout_genes.csv there.

    PYTHONPATH=src:scripts .venv/bin/python scripts/atlas/panel_concordance_heldout.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from si_panel_concordance import class_agreement  # noqa: E402

from bimp_st.paths import CANONICAL_TANGRAM, REPO  # noqa: E402

DATA = REPO / "results" / "reframe" / "panel_concordance"
OUT = Path(os.environ.get("BIMP_CONCORDANCE_HELDOUT_OUT", DATA))


def training_genes() -> set[str]:
    with h5py.File(CANONICAL_TANGRAM, "r") as f:
        return {str(g).lower() for g in read_elem(f["uns"]["training_genes"])}


def stats(A, B, pg, genes, tag):
    A, B = A.loc[genes], B.loc[genes]
    br, bp = spearmanr(A.mean(1), B.mean(1))
    sub = pg[pg.gene.isin(genes)]
    per_type = [spearmanr(A[c].values, B[c].values)[0] for c in A.columns]
    ca = class_agreement(sub)
    return dict(
        gene_set=tag,
        n_genes=len(genes),
        bulk_spearman=br,
        bulk_p=bp,
        per_gene_n=len(sub),
        per_gene_median_spearman=float(sub.spearman.median()),
        per_type_median_spearman=float(np.nanmedian(per_type)),
        n_types=len(A.columns),
        class_n_agree=ca["n_agree"],
        class_p_observed=ca["p_observed"],
        class_p_chance=ca["p_chance"],
        class_kappa=ca["kappa"],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(DATA / "sn_mean_by_type.csv", index_col=0)
    B = pd.read_csv(DATA / "xenium_mean_by_type.csv", index_col=0)
    pg = pd.read_csv(DATA / "per_gene.csv")
    train = training_genes()
    genes = list(A.index)
    held = [g for g in genes if g.lower() not in train]
    used = [g for g in genes if g.lower() in train]
    assert len(held) + len(used) == len(genes) == 100, (len(held), len(used))
    rows = [stats(A, B, pg, held, "held out"), stats(A, B, pg, used, "trained")]
    rows.append(stats(A, B, pg, genes, "all"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heldout_concordance.csv", index=False)
    pd.DataFrame({"gene": held}).to_csv(OUT / "heldout_genes.csv", index=False)
    print(df.T.to_string(), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
