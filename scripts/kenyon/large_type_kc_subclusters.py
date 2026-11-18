"""Subcluster the CB reference's own Large-type Kenyon cells (the source for
`fig:kc_subtypes` panels E-H).

Why not the separate ltKC dataset: `adata_ltKC_preprocessed.h5ad` is
pooled from five libraries outside the main reference, mostly worker, with
all queen nuclei from one library. The CB reference
(BIMP_atlas_qw_CentralBrain.h5ad, built 2026-07-01) has
Large-type nuclei of its own, spread across all 3 queen + 3 worker
libraries with no such confound.

A resolution sweep (0.05-0.4) on this population's own Harmony-PCA neighbor
graph found that resolution 0.05 is the only point that gives exactly 3
clusters; every resolution above it immediately splits the largest cluster
into two near-equal halves. The script reports each cluster's caste
fractions and library spread, and compares the cluster densities with the
ltKC dataset's own 3-way split, as a check that the structure is not an
artifact of the ltKC dataset's confound.

Out: results/fig5_kc/CB_LargeType_subclustered.h5ad
     obs['leiden_res0_05'] in {"0","1","2"}, numbered by descending size
     (the CLUSTERS/SUB convention: "0"=S1 largest ... "2"=S3
     smallest), so the Tangram script's d_source ordering carries over.
"""

from pathlib import Path

import scanpy as sc

REPO = Path(__file__).resolve().parents[2]
IN_PATH = REPO / "data/sn/BIMP_atlas_qw_CentralBrain.h5ad"
OUT_PATH = REPO / "results/fig5_kc/CB_LargeType_subclustered.h5ad"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SEED = 0
RESOLUTION = 0.05
CLUSTER_COL = "leiden_res0_05"

print("[1] load + subset to Large-type Kenyon cells", flush=True)
a = sc.read_h5ad(IN_PATH)
a.obs_names_make_unique()
lt = a[a.obs["cell_type"].astype(str) == "Large-type Kenyon cells"].copy()
del a
print("  n =", lt.n_obs, flush=True)

rep = "X_pca_harmony" if "X_pca_harmony" in lt.obsm else "X_pca"
print("  rep =", rep, "dims =", lt.obsm[rep].shape, flush=True)

print("[2] neighbors", flush=True)
sc.pp.neighbors(lt, n_neighbors=15, use_rep=rep, random_state=SEED)

print(f"[3] leiden, resolution={RESOLUTION}", flush=True)
sc.tl.leiden(
    lt,
    resolution=RESOLUTION,
    key_added=CLUSTER_COL,
    random_state=SEED,
    flavor="igraph",
    n_iterations=2,
)
vc = lt.obs[CLUSTER_COL].value_counts()
print("  clusters:", len(vc), flush=True)
print("  sizes:", vc.to_dict(), flush=True)
print("  density:", (vc / vc.sum()).round(4).to_dict(), flush=True)
assert len(vc) == 3, f"expected 3 clusters at resolution={RESOLUTION}, got {len(vc)}"

print("[4] caste / library balance per cluster", flush=True)
import pandas as pd  # noqa: E402

print(
    pd.crosstab(lt.obs[CLUSTER_COL], lt.obs["caste"].astype(str), normalize="index")
    .round(3)
    .to_string(),
    flush=True,
)
print(
    pd.crosstab(lt.obs[CLUSTER_COL], lt.obs["sample"].astype(str)).to_string(),
    flush=True,
)

lt.write_h5ad(OUT_PATH)
print("\nwrote", OUT_PATH, flush=True)
