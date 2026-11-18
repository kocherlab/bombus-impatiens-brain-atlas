#!/usr/bin/env python
"""Top-500 marker genes per snRNA cell type, in the canonical-17 vocabulary.

Same analysis as the top-50 table in tables_s1_s2.py (Table S2): Scanpy
rank_genes_groups, Wilcoxon, one population versus the rest, ranked by score, on
the combined central-brain + optic-lobe reference. Only the cut-off differs, and
the grouping is selectable:

  --vocab canonical17   (default) the canonical 17 cell types: the 16 named types
                        plus "Optic lobe", which pools the 13 unannotated
                        optic-lobe subclusters OL_1 to OL_13
                        (bimp_st.cluster_labels.CANONICAL_ORDER).
  --vocab reference29   the 29 raw reference populations, OL_1 to OL_13 separate
                        (the grouping behind Table S2).

The 16 named types score identically in both, since one-versus-rest is against all
other nuclei whichever way those are grouped; only "Optic lobe" is new.

Output, long format, one row per (cell type, rank):

    results/handoff/D_tables/snrna_markers_top500_<vocab>.csv

Columns: cell_type, pooled_group, rank, gene_id, symbol, description, score,
logfoldchange, pval_adj, cohens_d, on_panel (gene is one of the 101 Xenium
panel probes).

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/gene_annotation/snrna_markers_top500.py [--vocab canonical17]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import tables_s1_s2 as t  # noqa: E402

from bimp_st.cluster_labels import (  # noqa: E402
    CANONICAL_ORDER,
    CELL_TYPE_ORDER,
    get_canonical,
    get_pooled,
)

N = 500

ap = argparse.ArgumentParser()
ap.add_argument("--atlas", type=Path, default=t.SN, help="combined snRNA reference")
ap.add_argument(
    "--vocab", choices=["canonical17", "reference29"], default="canonical17"
)
args = ap.parse_args()

adata = ad.read_h5ad(args.atlas)
raw = adata.obs["cell_type"].astype(str)
assert set(raw) == set(CELL_TYPE_ORDER) and raw.nunique() == 29, "29-type mismatch"
if args.vocab == "canonical17":
    ORDER = CANONICAL_ORDER
    labels = raw.map(get_canonical)
else:
    ORDER = CELL_TYPE_ORDER
    labels = raw
assert set(labels) == set(ORDER), sorted(set(labels) ^ set(ORDER))
adata.obs["cell_type"] = pd.Categorical(labels, categories=ORDER)
print(adata.obs.cell_type.value_counts().reindex(ORDER).to_string(), flush=True)

s1 = t.build_s1(set(adata.var_names))
panel_ids = set(s1.v43_id)
sym = {r.v43_id: r.symbol for r in s1.itertuples()}
lk = pd.read_csv(t.LOOKUP, sep="\t").set_index("gene_id")

sc.tl.rank_genes_groups(adata, "cell_type", method="wilcoxon")  # one-vs-rest

# Cohen's d (pooled SD) per type versus all other nuclei, on the same matrix. It is
# an effect size that does not grow with the number of nuclei; the Wilcoxon score
# above (a z statistic) does, so scores of types with different sizes are not
# comparable. d is reported beside the score and does not change the ranking.
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
X2 = X.multiply(X).tocsr()
tot_s = np.asarray(X.sum(0)).ravel()
tot_q = np.asarray(X2.sum(0)).ravel()
n_all = X.shape[0]
gene_idx = {g: i for i, g in enumerate(adata.var_names)}
cohen = {}
for ct in ORDER:
    m = (adata.obs["cell_type"] == ct).to_numpy()
    n1, n0 = int(m.sum()), n_all - int(m.sum())
    s1_ = np.asarray(X[m].sum(0)).ravel()
    q1_ = np.asarray(X2[m].sum(0)).ravel()
    m1_, m0_ = s1_ / n1, (tot_s - s1_) / n0
    v1_ = (q1_ - n1 * m1_**2) / (n1 - 1)
    v0_ = ((tot_q - q1_) - n0 * m0_**2) / (n0 - 1)
    pooled = np.sqrt(
        ((n1 - 1) * np.clip(v1_, 0, None) + (n0 - 1) * np.clip(v0_, 0, None))
        / (n1 + n0 - 2)
        + 1e-30
    )
    cohen[ct] = (m1_ - m0_) / pooled
rows = []
for ct in ORDER:
    df = sc.get.rank_genes_groups_df(adata, ct).head(N)
    for rank, r in enumerate(df.itertuples(), 1):
        g = r.names
        gene = sym.get(g) or lk["gene"].get(g)
        desc = lk["description"].get(g)
        rows.append(
            dict(
                cell_type=ct,
                pooled_group=get_pooled(ct),
                rank=rank,
                gene_id=g,
                symbol=gene if isinstance(gene, str) else "",
                description=desc if isinstance(desc, str) else "",
                score=r.scores,
                logfoldchange=r.logfoldchanges,
                pval_adj=r.pvals_adj,
                cohens_d=cohen[ct][gene_idx[g]],
                on_panel=g in panel_ids,
            )
        )
out = pd.DataFrame(rows)
t.OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(t.OUT / f"snrna_markers_top500_{args.vocab}.csv", index=False)
print(
    f"wrote {len(out):,} rows, {out.cell_type.nunique()} cell types, "
    f"{int(out.on_panel.sum())} panel-gene hits in total"
)
print(out.groupby("cell_type", sort=False).on_panel.sum().to_string())
