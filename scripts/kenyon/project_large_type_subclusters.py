"""Project the CB reference's own 3 Large-type-KC subclusters onto every
Kenyon-compartment cell, classic Tangram `d_source` abundance prior.

This is the source for `fig:kc_subtypes` panels E-H. The separate ltKC dataset
(`data/sn/adata_ltKC_preprocessed.h5ad`) is not used: it is pooled from
five libraries outside the main reference, mostly worker, with ALL queen
nuclei from one library. `large_type_kc_subclusters.py` finds the same 3-way
Leiden split (0.05 resolution) directly on the CB reference's Large-type
nuclei -- both castes in every cluster, spread across all 3 queen + 3 worker
libraries, and with cluster densities matching the ltKC split, which is
independent
evidence the 3-way structure is real and not an artifact of the ltKC dataset's
confound.

The mapping restores classic Tangram's `d_source` abundance prior with
`scale=True` (tangram2's defaults do not fit this use); the query set is every
Kenyon-compartment cell, with no cell-type gate.

CAUTION: the S1/S2 split downstream of this mapping is a soft, radial
expression axis, not a validated subtype partition -- distance to the nearest
KC-IC cell predicts the S1-vs-S2 call (see kc_subtypes_data).
Do not assume the cleaner reference alone earns a stronger subtype claim.

Run:  bash config/sbatch.sh scripts/kenyon/run_project_large_type_subclusters.sh
Out:  results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad  (read)
      results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource_meanS.h5ad
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
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

LTKC_PATH = REPO / "results/fig5_kc/CB_LargeType_subclustered.h5ad"
SPATIAL_PATH = CANONICAL_TANGRAM
OUTDIR = REPO / "results/tangram_ltKC_cb"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_C = OUTDIR / "ltKC_CB_spatial_subclusters_dsource.h5ad"
OUT_B = OUTDIR / "ltKC_CB_spatial_subclusters_dsource_meanS.h5ad"

CLUSTER_COL = "leiden_res0_05"
CLUSTERS = ["0", "1", "2"]
SUB = {"0": "S1", "1": "S2", "2": "S3"}
NUM_EPOCHS = 500
LEARNING_RATE = 0.1
SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"device={DEVICE}", flush=True)

# ── Reference ──────────────────────────────────────────────────────────────
print("Loading CB Large-type subcluster reference ...", flush=True)
ref = sc.read_h5ad(LTKC_PATH)
ref.obs[CLUSTER_COL] = ref.obs[CLUSTER_COL].astype(str)
ref = ref[ref.obs[CLUSTER_COL].isin(CLUSTERS)].copy()
print(f"  reference: {ref.n_obs:,} nuclei, {ref.n_vars:,} genes", flush=True)
print(ref.obs[CLUSTER_COL].value_counts(), flush=True)

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

# ── Marker genes ────────────────────────────────────────────────────────────
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
    (DESCENDING FREQUENCY), not by cluster name -- reindexed explicitly so
    S1/S2/S3 don't get silently permuted.
    """
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
    # measured by large_type_kc_subclusters.py's Leiden run; loose atol
    # since this is a re-derived, not hardcoded-from-elsewhere, density
    exp = np.array([0.4961, 0.3691, 0.1348], dtype="float32")
    assert np.allclose(dens, exp, atol=0.02), (dens, exp)
    return X, dens


def run(name, scale, out_path):
    S, dens = build_S(scale)
    print(
        f"\n===== {name}: d_source ON, coeff OFF, "
        f"adata_to_cluster_expression(scale={scale})",
        flush=True,
    )
    print(
        f"  S {S.shape}; cluster order {CLUSTERS}; d_source {dens.round(4).tolist()}",
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
        "  argmax share, all Kenyon compartments (n={:,}): ".format(len(am))
        + ", ".join(f"{SUB[c]} {100 * v[c]:.2f}%" for c in CLUSTERS),
        flush=True,
    )
    for c_ in ("KC-IC", "KC-NC", "KC-OC"):
        m = comp == c_
        vv = am[m].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
        print(
            f"    {c_} (n={m.sum():,}): "
            + ", ".join(f"{SUB[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS),
            flush=True,
        )
    for b in ("queen", "worker"):
        m = nc & (brain == b)
        vv = am[m].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
        print(
            f"    KC-NC {b} (n={m.sum():,}): "
            + ", ".join(f"{SUB[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS),
            flush=True,
        )
    rn = probs.div(probs.sum(axis=1), axis=0)
    print(
        f"  median max row-norm prob={rn.max(axis=1).median():.4f}  "
        f"frac<0.5={(rn.max(axis=1) < 0.5).mean():.4f}",
        flush=True,
    )
    sp = probs.sample(50000, random_state=0).corr(method="spearman")
    sp.index = sp.columns = [SUB[c] for c in CLUSTERS]
    print("  column spearman:\n" + str(sp.round(3)), flush=True)

    out = sp_sub.copy()
    out.obsm["tangram_ct_pred"] = probs
    out.obs["ltKC_subcluster"] = am.astype(str).values
    out.uns["ltKC_projection_config"] = dict(
        d_source=dens.astype(float).tolist(),
        coeff="disabled",
        cluster_expression_scale=bool(scale),
        cluster_order=CLUSTERS,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        reference="CB (BIMP_atlas_qw_CentralBrain.h5ad, caste-balanced)",
        note=(
            "classic tangram clusters mode"
            if scale
            else "d_source restored but cluster MEAN expression (hybrid)"
        ),
    )
    out.write_h5ad(out_path)
    print(f"  wrote {out_path}", flush=True)
    return probs


# Config C first: it is the one fig_kc_subtypes.py reads.
run("CONFIG C (classic-equivalent, SHIPPED)", True, OUT_C)
run("CONFIG B (d_source only, cluster MEAN -- for the record)", False, OUT_B)
print("\ndone", flush=True)
