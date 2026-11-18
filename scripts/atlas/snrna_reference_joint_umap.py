"""Recompute a JOINT embedding on the combined CB+OL reference.

`anndata.concat` stacks CB and OL cells but keeps each file's own X_pca/X_umap,
which live in different coordinate spaces — a single UMAP plot of the raw
obsm['X_umap'] would show two disjoint blobs. This recomputes HVG -> PCA ->
(Harmony if available) -> neighbors -> UMAP on the concatenated object so the
snRNA reference-UMAP figure panels are valid. Overwrites obsm['X_umap'] in place.

Run: PYTHONPATH=src .venv/bin/python scripts/atlas/snrna_reference_joint_umap.py
"""

from pathlib import Path

import anndata as ad
import scanpy as sc

P = Path("data/sn/BIMP_atlas_qw_combined.h5ad")
a = ad.read_h5ad(P)
print(f"loaded {a.n_obs:,} x {a.n_vars:,}")

# Joint HVGs on the log-norm X, batch-aware by sample.
sc.pp.highly_variable_genes(a, n_top_genes=2000, batch_key="sample")
av = a[:, a.var["highly_variable"]].copy()
sc.pp.scale(av, max_value=10)
sc.pp.pca(av, n_comps=50)

rep = "X_pca"
try:
    import scanpy.external as sce

    sce.pp.harmony_integrate(av, "sample")
    rep = "X_pca_harmony"
    print("Harmony integration OK -> using X_pca_harmony")
except Exception as e:
    print(f"Harmony unavailable ({type(e).__name__}); using plain X_pca")

sc.pp.neighbors(av, use_rep=rep, n_neighbors=15)
sc.tl.umap(av)

a.obsm["X_umap"] = av.obsm["X_umap"]
a.obsm["X_pca_joint"] = av.obsm["X_pca"]
if "X_pca_harmony" in av.obsm:
    a.obsm["X_pca_harmony_joint"] = av.obsm["X_pca_harmony"]
a.uns["joint_umap"] = {"rep": rep, "n_hvg": int(a.var["highly_variable"].sum())}
a.write_h5ad(P)
print(f"joint UMAP recomputed on {rep}; wrote {P}")
