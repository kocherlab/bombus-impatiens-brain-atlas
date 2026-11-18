#!/usr/bin/env python
"""Top 10 pooled-Glia caste DEGs (replicated snRNA DESeq2, queen vs worker,
``~ trial + caste``, FDR < 0.05).

Joins the ten smallest-padj pooled-Glia DEGs of the trial-blocked table
(``results/caste_de/de_coarse_blocked.csv``, cell_type "Glia"; positive log2FC =
queen-up) to what the repo holds per gene:

* short description and Drosophila call:
  ``results/round5/glia_labels/v43_dmel_rbh.csv.gz`` (``genome_dmel_orthologs.py``:
  ``BIMP_OGS_v4.3.gtf`` product description; DIAMOND reciprocal best hit to D.
  melanogaster, e <= 1e-10, >= 30% query coverage);
* LOC ID: ``data/panel/np31_to_v43_mapping.csv``, which covers the 101 panel genes only.
  There is no v4.3 -> B. impatiens LOC table for other genes in the repo (the GTF
  carries only the A. mellifera NCBI GeneID as ``NCBIOrtholog``), so ``loc_id`` is blank
  off the panel; ``apis_geneid`` is given instead.
* panel membership: v4.3 IDs of ``np31_to_v43_mapping.csv``.

A Drosophila ortholog is called "confirmed" only for a DIAMOND reciprocal best hit;
ambiguous and non-reciprocal best hits are shown as candidates, not as orthologs.

Writes NEW files only: ``results/caste_de/glia_top10/glia_top10_degs.csv``, which
``deposit/supplementary_data_csvs.py`` ships as Supplementary Data 6.

    PYTHONPATH=src:scripts .venv/bin/python scripts/caste_de/table_glia_top10_degs.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (  # noqa: E402
    CASTE_DE_COARSE,
    CASTE_DE_DIR,  # noqa: E402
    PANEL_CROSSWALK,
)

REPO = Path(__file__).resolve().parents[2]
DE = CASTE_DE_COARSE
RBH = REPO / "results/round5/glia_labels/v43_dmel_rbh.csv.gz"
MAP = PANEL_CROSSWALK
OUT = CASTE_DE_DIR / "glia_top10"
FDR = 0.05


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    de = pd.read_csv(DE, index_col=0)
    g = de[(de.cell_type == "Glia")].dropna(subset=["padj"])
    top = g[g.padj < FDR].sort_values("padj").head(10).reset_index(drop=True)
    rbh = pd.read_csv(RBH).set_index("gene")
    mp = read_panel_crosswalk().set_index("v43_id")

    rows = []
    for i, r in top.iterrows():
        b = rbh.loc[r.gene] if r.gene in rbh.index else None
        call = str(b["call"]) if b is not None else "no hit"
        sym = (
            str(b["dmel_symbol"]).replace("Dmel\\", "")
            if b is not None and pd.notna(b["dmel_symbol"])
            else ""
        )
        fbgn = str(b["dmel_FBgn"]) if b is not None and pd.notna(b["dmel_FBgn"]) else ""
        desc = (
            str(b["description"])
            if b is not None and pd.notna(b["description"])
            else ""
        )
        rows.append(
            dict(
                rank=i + 1,
                gene=sym if call == "reciprocal best hit" else desc,
                bimp_v43_id=r.gene,
                loc_id=mp.loc[r.gene, "loc_id"] if r.gene in mp.index else "",
                apis_geneid=(
                    ""
                    if b is None or pd.isna(b["apis_geneid"])
                    else str(int(b["apis_geneid"]))
                ),
                short_description=desc
                if desc
                else (
                    "lncRNA, no annotation"
                    if b is not None and b["gene_biotype"] == "lncRNA"
                    else "uncharacterized protein, no annotation"
                ),
                gene_biotype=(b["gene_biotype"] if b is not None else ""),
                dmel_ortholog=(
                    f"{sym} ({fbgn})" if call == "reciprocal best hit" else ""
                ),
                dmel_candidate_not_confirmed=(
                    f"{sym} ({fbgn}); {call}"
                    if call != "reciprocal best hit" and sym
                    else ""
                ),
                direction="queen-up" if r.log2FoldChange > 0 else "worker-up",
                log2FC_queen_over_worker=round(float(r.log2FoldChange), 3),
                padj=float(r.padj),
                on_101_gene_panel="yes" if r.gene in mp.index else "no",
            )
        )
    t = pd.DataFrame(rows)
    # the gene column: Dmel symbol when confirmed, else the product description, else
    # the ID number
    t["gene"] = [
        x if x else f"{v[-6:]} (unannotated)"
        for x, v in zip(t["gene"], t.bimp_v43_id, strict=True)
    ]
    p = OUT / "glia_top10_degs.csv"
    t.to_csv(p, index=False)
    print(
        t.drop(columns=["apis_geneid", "dmel_candidate_not_confirmed"]).to_string(
            index=False
        )
    )
    print("saved", p)


if __name__ == "__main__":
    main()
