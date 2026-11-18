#!/usr/bin/env python
"""Supplementary Data: per-cell-type caste GSEA, every term (unfiltered).

Rebuilds ``Supp_celltype_GO_terms_full_unfiltered.csv`` of the saved analysis
(``out/fig4_supp/``; table design: I. Traniello). It is the
clean-notebook cell 15 table (``01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb``,
md5 184efd9af8192fd09f7329f1eeaa9f17; ``gsea_celltypes.py`` here) with every row
kept, in that table's row order, and GO names attached the way ``supp_full.py``
(md5 70ce9f8c44e4af04ef773d8910976545) ``annotate`` does: GSEApy prerank of the
DESeq2 Wald statistic (``~ trial + caste``, + = queen-up, baseMean > 10) in the
11 glial and neuronal types, eggNOG GO library of all namespaces, 10-500 genes,
1,000 permutations, seed 0. No term is dropped (obsolete terms stay); GO names
and namespaces come from primary OBO ids only, as in the saved file, so the 135
secondary (``alt_id``) GO ids in the library have an empty term and namespace.
Leading-edge genes are BIMP_v4.3 ids.

Nothing is recomputed: ``source="repo"`` reshapes
``results/caste_programs/pipeline/repo/gsea_all.csv.gz`` (ranked by the
repo's DESeq2 table); ``source="saved"`` the saved ``gsea_all.csv.gz`` (frozen
copy).

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/caste_de/cnmf_gsea/sd_celltype_go.py [--source saved]
"""

import argparse

import numpy as np
import pandas as pd
from sd_common import (
    SI,
    SI_SAVED,
    add_source_arg,
    clean_ct,
    go_annotate,
    gsea_csv,
    parse_obo,
)
from sd_common import write as write_csv

OUT = SI / "Supp_celltype_GO_terms_full_unfiltered.csv"
COLUMNS = [
    "cell_type",
    "direction",
    "go_id",
    "term",
    "namespace",
    "NES",
    "nominal_p",
    "FDR_q",
    "FWER_p",
    "tag_pct",
    "gene_pct",
    "n_ranked",
    "leading_edge_genes",
]


def build(source: str = "repo") -> pd.DataFrame:
    """The unfiltered per-cell-type GSEA table, the saved table's column names
    and order."""
    g = go_annotate(pd.read_csv(gsea_csv(source)), parse_obo(alt_ids=False), drop=False)
    g["cell_type"] = clean_ct(g.cell_type)
    g["direction"] = np.where(g.NES > 0, "queen-up", "worker-up")
    g["FWER p-val"] = pd.to_numeric(g["FWER p-val"], errors="coerce")
    g = g.rename(
        columns={
            "NOM p-val": "nominal_p",
            "FDR q-val": "FDR_q",
            "FWER p-val": "FWER_p",
            "Tag %": "tag_pct",
            "Gene %": "gene_pct",
            "Lead_genes": "leading_edge_genes",
        }
    )
    return g[COLUMNS].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arg(ap)
    args = ap.parse_args()
    out = args.out or (OUT if args.source == "repo" else SI_SAVED / OUT.name)
    write_csv(build(args.source), out)


if __name__ == "__main__":
    main()
