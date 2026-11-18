"""Restricted KC re-projection gated by PREDICTED CLASS, not by anatomy.

Sibling of project_kc_populations.py, identical in every respect but
the query mask, and that one difference is the point.

kc5pop restricts the query to cells inside a Kenyon COMPARTMENT
(in_kc_ic | in_kc_nc | in_kc_oc). That is the right choice for `fig:kc_subtypes`
panel C, which asks how the compartments differ and may take the compartments
as given. It is the wrong choice for `fig:atlas_overview`, which plots cell-type
labels against
those same compartments: an anatomy-gated query makes the label depend on the
anatomy it is later shown beside, which is the circularity the anatomy-free
labelling avoids.

This gates on the label instead. The query is every cell whose whole-atlas
pooled call is Kenyon-lineage -- expression in, expression out, no polygon
anywhere -- and the reference is narrowed to the five named KC populations,
removing the escape-hatch categories (the unspecified "Kenyon cells" entry
and "Other neurons") that the whole-atlas run otherwise falls back on. That
fallback is the documented failure: raw 29-way argmax resolves few
Kenyon-compartment cells to a named type, and post-hoc renormalisation onto
the five columns inverts the true population sizes.

Because nothing anatomical enters, the compartment breakdown of the result
is a TEST rather than a construction, and can be compared directly against
kc5pop's anatomy-gated compartment x call table. Agreement would mean the
anatomy gate was never doing
the work; disagreement would mean it was.

Config is deliberately unchanged from kc5pop -- 500 epochs, lr 0.1, seed 42,
cluster SUM expression, reference-density prior -- so the only variable is
the query definition.

Writes results/tangram_ltKC/kc5pop_spatial_classgated.h5ad.
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
OUT = OUTDIR / "kc5pop_spatial_classgated.h5ad"

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
# The one substantive difference from kc5pop: gate on the whole-atlas
# pooled CALL, never on a polygon. cluster_vote is what `fig:atlas_overview`
# plots, so gating on it means this re-projection refines exactly the block it
# already shows as one undifferentiated "Kenyon cells".
import h5py as _h5py
from anndata.io import read_elem as _read_elem

from bimp_st.cluster_labels import cluster_vote as _cluster_vote
from bimp_st.paths import CANONICAL_TANGRAM as _CANON

with _h5py.File(_CANON, "r") as _g:
    spatial.obs["tangram_cell_type"] = (
        _read_elem(_g["obs"])["tangram_cell_type"].astype(str).values
    )
_cluster_vote(
    spatial,
    cluster_col="leiden",
    label_col="tangram_cell_type",
    pool=True,
    min_purity=0.5,
    dst="_cv",
)
kc_mask = (spatial.obs["_cv"].astype(str) == "Kenyon cells").values
print(f"  atlas cells: {spatial.n_obs:,}", flush=True)
print(
    f"  cells whose POOLED CALL is Kenyon (no anatomy): {kc_mask.sum():,}", flush=True
)
_in_kc = (
    spatial.obs["in_kc_ic"].fillna(False).astype(bool)
    | spatial.obs["in_kc_nc"].fillna(False).astype(bool)
    | spatial.obs["in_kc_oc"].fillna(False).astype(bool)
).values
print(
    f"  of those, inside a Kenyon compartment: "
    f"{int((kc_mask & _in_kc).sum()):,} "
    f"({(kc_mask & _in_kc).sum() / max(kc_mask.sum(), 1):.1%}) "
    f"-- reported for interest only, NOT used to select",
    flush=True,
)
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

# Reporting only. Unlike kc5pop, the query is NOT anatomy-selected, so a
# queried cell can lie outside every compartment; those are labelled "none"
# rather than silently folded into KC-OC by the fall-through.
nc = sp_sub.obs["in_kc_nc"].fillna(False).values.astype(bool)
ic = sp_sub.obs["in_kc_ic"].fillna(False).values.astype(bool)
oc = sp_sub.obs["in_kc_oc"].fillna(False).values.astype(bool)
comp = np.where(ic, "KC-IC", np.where(nc, "KC-NC", np.where(oc, "KC-OC", "none")))
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
for c_ in ("KC-IC", "KC-NC", "KC-OC", "none"):
    m = comp == c_
    vv = am[m].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
    print(
        f"    {c_} (n={m.sum():,}): "
        + ", ".join(f"{SHORT[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS),
        flush=True,
    )
for b in ("queen", "worker"):
    for c_ in ("KC-IC", "KC-NC", "KC-OC", "none"):
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
    note=(
        "class-gated query (pooled cluster_vote == Kenyon cells); no anatomy used to "
        "select cells"
    ),
)
out.write_h5ad(OUT)
print(f"\n  wrote {OUT}", flush=True)
print("\ndone", flush=True)
