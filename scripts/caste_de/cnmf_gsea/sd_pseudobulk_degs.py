#!/usr/bin/env python
"""Supplementary Data: pseudobulk caste DEGs per fine cell type, annotated.

Rebuilds ``Supp_pseudobulk_DEGs_annotated.csv`` of the saved analysis
(``out/fig4_supp/``; table design: I. Traniello): every gene at
padj < 0.05 (clean-notebook cell 9 ``PADJ``) in the pseudobulk DESeq2 fit
``~ trial + caste`` of each fine cell type, with the gene annotation attached as
cell 9 (``01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb``, md5
184efd9af8192fd09f7329f1eeaa9f17) does from ``BIMP_lookup.csv`` (gene name,
description, biotype), plus the lookup's chromosome. ``display_name`` is the gene
name, else the description, else the id; ``direction`` is queen-up for
log2FoldChange > 0. Rows: cell types in alphabetical order, padj ascending.
``flag`` marks the photoreceptor rows (residual optic lobe contamination, left
out of the gene total) and the Vgn rows (no P value is reported for Vgn).

``source="repo"`` (ships) reads the repo's fit, ``bimp_st.paths.CASTE_DE_FINE``
(``../caste_de_blocked_all.py``, pyDESeq2 0.5.4); ``source="saved"`` the saved
``de_all.csv.gz`` (frozen copy). The two fits differ slightly
(``bimp_st.paths.CASTE_DE_PARITY_SAVED``).

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/caste_de/cnmf_gsea/sd_pseudobulk_degs.py [--source saved]
"""

import argparse

import numpy as np
import pandas as pd
from sd_common import SI, SI_SAVED, add_source_arg, clean_ct, de_csv, lookup
from sd_common import write as write_csv

from bimp_st.caste_de import focal_gene_ids

OUT = SI / "Supp_pseudobulk_DEGs_annotated.csv"
PADJ = 0.05
COLUMNS = [
    "cell_type",
    "gene_id",
    "gene_name",
    "description",
    "display_name",
    "gene_biotype",
    "chromosome",
    "direction",
    "baseMean",
    "log2FoldChange",
    "lfcSE",
    "stat",
    "pvalue",
    "padj",
    "flag",
]


def build(source: str = "repo") -> pd.DataFrame:
    """DEGs at padj < 0.05 with gene annotation, the saved table's column names
    and order."""
    de = pd.read_csv(de_csv(source))
    # the repo fit carries its own short gene_name; the lookup's is attached below
    de = de.drop(columns=[c for c in ("gene_name", "Unnamed: 0") if c in de])
    de["cell_type"] = clean_ct(de.cell_type)
    d = de[de.padj < PADJ].rename(columns={"gene": "gene_id"})
    lk = (
        lookup()
        .rename(columns={"gene": "gene_name", "Chromosome": "chromosome"})
        .replace({"gene_name": {"": np.nan}, "description": {"": np.nan}})
        .drop_duplicates("gene_id")
    )
    cols = ["gene_id", "gene_name", "description", "gene_biotype", "chromosome"]
    d = d.merge(lk[cols], on="gene_id", how="left")
    d["display_name"] = d.gene_name.fillna(d.description).fillna(d.gene_id)
    d["direction"] = np.where(d.log2FoldChange > 0, "queen-up", "worker-up")
    d["flag"] = np.select(
        [d.cell_type == "Photoreceptors", d.gene_id == focal_gene_ids()["Vgn"]],
        ["photoreceptors", "Vgn, no P value reported"],
        "",
    )
    d = d.sort_values(["cell_type", "padj"], kind="mergesort")
    return d[COLUMNS].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arg(ap)
    args = ap.parse_args()
    out = args.out or (OUT if args.source == "repo" else SI_SAVED / OUT.name)
    write_csv(build(args.source), out)


if __name__ == "__main__":
    main()
