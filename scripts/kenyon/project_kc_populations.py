"""Project panel A's 5 named KC populations onto every Kenyon-compartment cell,
using the restricted-query/restricted-reference recipe of the
3-subcluster panel (project_large_type_subclusters.py), with classic Tangram's
d_source abundance prior restored (tangram2's mode="clusters" drops the
reference-density term and uses cluster MEAN instead of cluster SUM
expression, which lets rare clusters collapse to near-zero weight).

Why: the raw whole-atlas 29-category Tangram calls send most
KC-compartment cells to the generic "Kenyon cells" bucket and "Other neurons"
rather than one of the 5 fine KC types. Renormalising the whole-atlas
soft probabilities onto the 5 types
(results/tangram_ltKC/kc5pop_soft_probs_summary.txt) gives a weak signal
that does not keep the reference population sizes (the summary compares
the renormalised confidence with the 1/5 chance level and the spatial with
the reference proportions). This script is the RESTRICTED
reclassification instead: query narrowed to Kenyon-compartment cells only
(removing competition from all 24 non-KC-related reference categories),
reference narrowed to the 5 KC populations panel A shows (no generic
"Kenyon cells" escape hatch: this reference file has none -- see below).

Reference: data/sn/BIMP_atlas_qw_CentralBrain.h5ad, filtered to the 5 KC
cell_type values. This file has NO generic "Kenyon cells" cluster (its 15
categories are the 5 KC subtypes + 10 non-KC types, no residual KC bucket),
so filtering to the 5 KC names recovers exactly panel A's nuclei with
no extra step -- the KC subset shown in `fig:kc_subtypes` panel A.

Query: every cell inside a Kenyon compartment (in_kc_ic | in_kc_nc |
in_kc_oc) of the current atlas, NO cell-type gate -- identical restriction
to the 3-subcluster script, for a fair comparison.

Run:  bash config/sbatch.sh scripts/kenyon/run_project_kc_populations.sh
Out:  results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tangram2 import mapping as tg
from tangram2.mapping import mapping_optimizer as mo
from tangram2.mapping import mapping_utils as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.paths import CANONICAL_TANGRAM, SN_CB  # noqa: E402

REF_PATH = SN_CB
SPATIAL_PATH = CANONICAL_TANGRAM
OUTDIR = REPO / "results/tangram_ltKC"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "kc5pop_spatial_allKC_dsource.h5ad"

CLUSTER_COL = "cell_type"
CLUSTERS = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
]
SHORT = {
    "Large-type Kenyon cells": "Large",
    "Middle-type Kenyon cells": "Middle",
    "Small-type Kenyon cells": "Small",
    "Class II Kenyon cells": "ClassII",
    "Other Kenyon cells": "Other",
}
NUM_EPOCHS = 500
LEARNING_RATE = 0.1
SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"device={DEVICE}", flush=True)

# ── Reference: panel A's own 5-population KC nuclei ─────────────────────────
print("Loading central-brain snRNA reference ...", flush=True)
ref = sc.read_h5ad(REF_PATH)
ref.obs[CLUSTER_COL] = ref.obs[CLUSTER_COL].astype(str)
ref = ref[ref.obs[CLUSTER_COL].isin(CLUSTERS)].copy()
print(f"  reference: {ref.n_obs:,} nuclei, {ref.n_vars:,} genes", flush=True)
print(ref.obs[CLUSTER_COL].value_counts(), flush=True)
assert ref.n_obs == 32672, f"expected panel A's 32,672 nuclei, got {ref.n_obs}"

# ── Spatial query: every Kenyon-compartment cell, NO cell-type gate ────────
print("\nLoading spatial atlas ...", flush=True)
spatial = sc.read_h5ad(SPATIAL_PATH)
spatial.obs_names_make_unique()
kc_mask = (
    spatial.obs["in_kc_ic"].astype(bool)
    | spatial.obs["in_kc_nc"].astype(bool)
    | spatial.obs["in_kc_oc"].astype(bool)
).values
print(f"  atlas cells: {spatial.n_obs:,}", flush=True)
print(f"  Kenyon-compartment cells (NO cell-type gate): {kc_mask.sum():,}", flush=True)
spatial_kc = spatial[kc_mask].copy()
del spatial

# ── Gene overlap (case-insensitive; spatial is lowercase, ref uppercase) ────
ref_lower = {g.lower(): g for g in ref.var_names}
sp_lower = {g.lower(): g for g in spatial_kc.var_names}
overlap = sorted(set(ref_lower) & set(sp_lower))
print(f"\n  gene overlap: {len(overlap)}", flush=True)
if len(overlap) < 10:
    raise SystemExit("Too few overlapping genes for Tangram mapping")

rename = {sp_lower[g]: ref_lower[g] for g in overlap}
spatial_kc.var_names = [rename.get(g, g) for g in spatial_kc.var_names]
ref_names = [ref_lower[g] for g in overlap]
ref_sub = ref[:, ref_names].copy()
sp_sub = spatial_kc[:, ref_names].copy()
del spatial_kc
print(f"  after subsetting: ref={ref_sub.shape}, spatial={sp_sub.shape}", flush=True)

# ── Marker genes (identical recipe to the 3-subcluster run) ────────────────
sc.tl.rank_genes_groups(ref_sub, groupby=CLUSTER_COL, method="wilcoxon")
markers = pd.DataFrame(ref_sub.uns["rank_genes_groups"]["names"]).head(50)
marker_genes = [g for g in set(markers.values.flatten()) if g in ref_names]
print(f"  using {len(marker_genes)} marker genes", flush=True)

tg.pp_adatas(ref_sub, sp_sub, genes=marker_genes)
training_genes = list(ref_sub.uns["training_genes"])

d = np.asarray(sp_sub.obs["rna_count_based_density"], dtype="float32")
print(f"  target density prior sum={d.sum():.6f}", flush=True)

G = sp_sub[:, training_genes].X
G = np.asarray(G.toarray() if hasattr(G, "toarray") else G, dtype="float32")
print(f"  G {G.shape}", flush=True)

nc = sp_sub.obs["in_kc_nc"].values.astype(bool)
ic = sp_sub.obs["in_kc_ic"].values.astype(bool)
comp = np.where(ic, "KC-IC", np.where(nc, "KC-NC", "KC-OC"))
brain = sp_sub.obs["brain"].astype(str).values


def build_S(scale):
    """Cluster-level reference matrix + its density vector, rows aligned to
    CLUSTERS. adata_to_cluster_expression orders rows by value_counts()
    (DESCENDING FREQUENCY), not by cluster name -- reindexed explicitly, same
    footgun the 3-subcluster script's own docstring warns about."""
    a = mu.adata_to_cluster_expression(
        ref_sub, CLUSTER_COL, scale=scale, add_density=True
    )
    order = [str(c) for c in a.obs[CLUSTER_COL]]
    X = a[:, training_genes].X
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype="float32")
    dens = np.asarray(a.obs["cluster_density"], dtype="float32")
    idx = [order.index(c) for c in CLUSTERS]
    X, dens, order = X[idx], dens[idx], [order[i] for i in idx]
    assert order == CLUSTERS, order
    assert abs(float(dens.sum()) - 1.0) < 1e-4, dens
    return X, dens


S, dens = build_S(scale=True)
print(
    f"\n  S {S.shape}; cluster order {CLUSTERS}; d_source {dens.round(4).tolist()}",
    flush=True,
)
print(f"  S row sums {np.round(S.sum(axis=1), 1).tolist()}", flush=True)

mapper = mo.Mapper(
    S=S,
    G=G,
    d=d,
    coeff=None,
    d_source=dens,
    lambda_d=1,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    device=DEVICE,
    random_state=SEED,
)
M, _hist = mapper.train(
    learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, print_each=100
)
probs = pd.DataFrame(M.T, index=sp_sub.obs_names, columns=CLUSTERS)
am = probs.idxmax(axis=1)

v = am.value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
print(
    "\n  argmax share, all Kenyon compartments (n={:,}): ".format(len(am))
    + ", ".join(f"{SHORT[c]} {100 * v[c]:.2f}%" for c in CLUSTERS),
    flush=True,
)
for c_ in ("KC-IC", "KC-NC", "KC-OC"):
    m = comp == c_
    vv = am[m].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
    print(
        f"    {c_} (n={m.sum():,}): "
        + ", ".join(f"{SHORT[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS),
        flush=True,
    )
for b in ("queen", "worker"):
    for c_ in ("KC-IC", "KC-NC", "KC-OC"):
        m = (comp == c_) & (brain == b)
        vv = am[m].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
        print(
            f"    {c_} {b} (n={m.sum():,}): "
            + ", ".join(f"{SHORT[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS),
            flush=True,
        )

rn = probs.div(probs.sum(axis=1), axis=0)
print(
    f"\n  median max row-norm prob={rn.max(axis=1).median():.4f}  "
    f"frac<0.5={(rn.max(axis=1) < 0.5).mean():.4f}",
    flush=True,
)
sp = probs.sample(min(50000, len(probs)), random_state=0).corr(method="spearman")
sp.index = sp.columns = [SHORT[c] for c in CLUSTERS]
print("  column spearman:\n" + str(sp.round(3)), flush=True)

# Sanity check: does the RESTRICTED reclassification reproduce the true
# reference proportions, or does it still invert them like the post-hoc
# renormalization of the whole-atlas run did (Other KC over-called and
# Large-type under-called against their reference shares)?
ref_frac = ref.obs[CLUSTER_COL].value_counts(normalize=True).reindex(CLUSTERS)
print("\n  reference proportions vs spatial argmax proportions:", flush=True)
for c in CLUSTERS:
    print(
        f"    {SHORT[c]:8s} reference={100 * ref_frac[c]:.1f}%  "
        f"spatial={100 * v[c]:.1f}%",
        flush=True,
    )

out = sp_sub.copy()
out.obsm["tangram_ct_pred"] = probs
out.obs["kc5pop_call"] = am.astype(str).values
out.uns["kc5pop_projection_config"] = dict(
    d_source=dens.astype(float).tolist(),
    coeff="disabled",
    cluster_expression_scale=True,
    cluster_order=CLUSTERS,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    note="classic tangram clusters mode, same config as ltKC subclusters",
)
out.write_h5ad(OUT)
print(f"\n  wrote {OUT}", flush=True)
print("\ndone", flush=True)
