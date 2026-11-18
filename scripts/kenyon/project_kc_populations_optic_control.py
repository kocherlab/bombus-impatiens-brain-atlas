"""NEGATIVE CONTROL for the doubly-restricted 5-KC-population projection.

The projection (scripts/kenyon/project_kc_populations.py, `fig:kc_subtypes` C)
restricts BOTH sides: the
reference is filtered to panel A's five named Kenyon-cell populations, and the
spatial query is restricted to Kenyon-compartment cells.  By construction every
queried cell must land on one of five KC labels, so compartment-structured
output is not self-evidently informative.

This script runs the IDENTICAL projection (same reference file, same five
clusters, same marker recipe, same classic-Tangram config: reference-density
prior restored, cluster-sum reference expression, 500 epochs, lr 0.1, seed 42)
on a query that contains NO Kenyon cells: the three optic neuropils
(in_lamina | in_lobula | in_medulla; verified mutually exclusive and disjoint
from the KC compartments).

The decisive readout is NOT confidence (a restricted 5-way simplex is
confident by construction) and NOT whether structure exists (lamina, lobula
and medulla are transcriptomically distinct tissue, so SOME structure is
expected).  It is whether the compartment x call table is a near one-to-one
permutation matrix at comparable effect size to the Kenyon run's.  Compared with a
matched statistic in kc_populations_optic_control_stats.py.

Run:  bash config/sbatch.sh scripts/kenyon/run_project_kc_populations_optic_control.sh
Out:  results/reframe/kc5pop_negative_control/
"""

import faulthandler
from pathlib import Path

faulthandler.enable()

import sys

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
OUTDIR = REPO / "results/reframe/kc5pop_negative_control"
OUTDIR.mkdir(parents=True, exist_ok=True)

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

# -- Reference: byte-identical to the project_kc_populations.py run ---------
print("Loading central-brain snRNA reference ...", flush=True)
ref = sc.read_h5ad(REF_PATH)
ref.obs[CLUSTER_COL] = ref.obs[CLUSTER_COL].astype(str)
ref = ref[ref.obs[CLUSTER_COL].isin(CLUSTERS)].copy()
print(f"  reference: {ref.n_obs:,} nuclei, {ref.n_vars:,} genes", flush=True)
assert ref.n_obs == 32672, f"expected panel A's 32,672 nuclei, got {ref.n_obs}"

# -- Spatial query: OPTIC NEUROPILS ONLY (no Kenyon cells present) ----------
print("\nLoading spatial atlas ...", flush=True)
spatial = sc.read_h5ad(SPATIAL_PATH)
spatial.obs_names_make_unique()
la = spatial.obs["in_lamina"].astype(bool).values
lo = spatial.obs["in_lobula"].astype(bool).values
me = spatial.obs["in_medulla"].astype(bool).values
kc = (
    spatial.obs["in_kc_ic"].astype(bool)
    | spatial.obs["in_kc_nc"].astype(bool)
    | spatial.obs["in_kc_oc"].astype(bool)
).values
assert (la.astype(int) + lo + me).max() <= 1, "optic flags overlap"
ol_mask = la | lo | me
assert (ol_mask & kc).sum() == 0, "optic query overlaps a Kenyon compartment"
print(f"  atlas cells: {spatial.n_obs:,}", flush=True)
print(
    f"  optic-neuropil cells (NO cell-type gate): {ol_mask.sum():,} "
    f"[lamina {la.sum():,} lobula {lo.sum():,} medulla {me.sum():,}]",
    flush=True,
)
spatial_ol = spatial[ol_mask].copy()
comp = np.where(la[ol_mask], "Lamina", np.where(lo[ol_mask], "Lobula", "Medulla"))
del spatial

# -- Gene overlap (identical recipe) ---------------------------------------
ref_lower = {g.lower(): g for g in ref.var_names}
sp_lower = {g.lower(): g for g in spatial_ol.var_names}
overlap = sorted(set(ref_lower) & set(sp_lower))
print(f"\n  gene overlap: {len(overlap)}", flush=True)
if len(overlap) < 10:
    raise SystemExit("Too few overlapping genes for Tangram mapping")

rename = {sp_lower[g]: ref_lower[g] for g in overlap}
spatial_ol.var_names = [rename.get(g, g) for g in spatial_ol.var_names]
ref_names = [ref_lower[g] for g in overlap]
ref_sub = ref[:, ref_names].copy()
sp_sub = spatial_ol[:, ref_names].copy()
del spatial_ol
print(f"  after subsetting: ref={ref_sub.shape}, spatial={sp_sub.shape}", flush=True)

sc.tl.rank_genes_groups(ref_sub, groupby=CLUSTER_COL, method="wilcoxon")
markers = pd.DataFrame(ref_sub.uns["rank_genes_groups"]["names"]).head(50)
marker_genes = [g for g in set(markers.values.flatten()) if g in ref_names]
print(f"  using {len(marker_genes)} marker genes", flush=True)

tg.pp_adatas(ref_sub, sp_sub, genes=marker_genes)
training_genes = list(ref_sub.uns["training_genes"])

d = np.asarray(sp_sub.obs["rna_count_based_density"], dtype="float32")
print(f"  target density prior sum={d.sum():.6f}", flush=True)
assert abs(float(d.sum()) - 1.0) < 1e-3, "density prior not renormalised"

G = sp_sub[:, training_genes].X
G = np.asarray(G.toarray() if hasattr(G, "toarray") else G, dtype="float32")
print(f"  G {G.shape}", flush=True)
brain = sp_sub.obs["brain"].astype(str).values


def build_S(scale):
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
print(f"\n  S {S.shape}; d_source {dens.round(4).tolist()}", flush=True)

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
    "\n  argmax share, all optic neuropils (n={:,}): ".format(len(am))
    + ", ".join(f"{SHORT[c]} {100 * v[c]:.2f}%" for c in CLUSTERS),
    flush=True,
)
for c_ in ("Lamina", "Lobula", "Medulla"):
    m = comp == c_
    vv = am[m].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
    print(
        f"    {c_} (n={m.sum():,}): "
        + ", ".join(f"{SHORT[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS),
        flush=True,
    )
for b in ("queen", "worker"):
    for c_ in ("Lamina", "Lobula", "Medulla"):
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

ref_frac = ref.obs[CLUSTER_COL].value_counts(normalize=True).reindex(CLUSTERS)
print("\n  reference proportions vs optic argmax proportions:", flush=True)
for c in CLUSTERS:
    print(
        f"    {SHORT[c]:8s} reference={100 * ref_frac[c]:.1f}%  "
        f"optic={100 * v[c]:.1f}%",
        flush=True,
    )

out = pd.DataFrame({"compartment": comp, "brain": brain}, index=sp_sub.obs_names)
out["call"] = am.values
for c in CLUSTERS:
    out["p_" + SHORT[c]] = probs[c].values
out["maxprob_renorm"] = rn.max(axis=1).values
out.to_parquet(OUTDIR / "optic_negcontrol_calls.parquet")
print(f"\n  wrote {OUTDIR / 'optic_negcontrol_calls.parquet'}", flush=True)
print("\ndone", flush=True)
