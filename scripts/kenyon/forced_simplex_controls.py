"""Controls on the queen-worker gap in the forced three-subcluster projection.

Tests whether the caste gap in the Large-type compartment's subcluster shares is
technical rather than Kenyon subtype biology. This writes the numbers of three
checks:

  confidence    the gap within quintiles of per-cell mapping confidence
                (max renormalized subcluster probability, as in
                fig:kc_restricted_confidence), quintiles over the compartment's
                cells with both castes pooled
  corr_control  a Tangram-free projection: Pearson correlation of each cell's
                expression (the object's .X over the projection's
                uns['overlap_genes']) with each subcluster's mean reference
                expression (results/fig5_kc/CB_LargeType_subclustered.h5ad,
                leiden_res0_05); argmax gives the call
  restrictions  the gap among cells the whole-atlas run (canonical seed 42,
                29 fine types, joined on (slice_name, cell_id)) calls
                Large-type Kenyon cells, or any Kenyon type, and what share of
                the compartment each restriction keeps, per caste

The gap is the largest absolute worker-minus-queen difference in a
subcluster's argmax share, in percentage points. Cells and
compartments are those of the projection object
(results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad) with its
stored region flags: Large-type compartment = in_kc_nc, all three = any of
in_kc_{ic,nc,oc}.

Output: results/handoff/forced_simplex_controls/summary.json (flat name -> value)

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/kenyon/forced_simplex_controls.py
"""

import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

from bimp_st.cluster_labels import get_canonical
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import RESULTS as RES

OUT = RES / "handoff" / "forced_simplex_controls"
LT = RES / "tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad"
REF = RES / "fig5_kc/CB_LargeType_subclustered.h5ad"
CAN = RES / "tangram/combined_clustered_g2_R2_s42_tangram.h5ad"
SUBS = np.array(["0", "1", "2"])
KC6 = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
    "Kenyon cells",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(LT, "r") as f:
        obs = read_elem(f["obs"])
        P = read_elem(f["obsm"]["tangram_ct_pred"])[list(SUBS)].to_numpy(float)
        X = read_elem(f["X"])
        var = pd.Index(read_elem(f["var"]).index.astype(str))
        overlap = [str(g) for g in read_elem(f["uns"]["overlap_genes"])]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    brain = obs["brain"].astype(str).to_numpy()
    sub = obs["ltKC_subcluster"].astype(str).to_numpy()
    assert (SUBS[P.argmax(1)] == sub).all()
    f = {
        r: obs[f"in_{r}"].fillna(False).to_numpy(bool)
        for r in ("kc_ic", "kc_nc", "kc_oc")
    }
    masks = {"kcnc": f["kc_nc"], "allkc": f["kc_ic"] | f["kc_nc"] | f["kc_oc"]}
    S = {}

    def gap(m, lab):
        sh = pd.crosstab(brain[m], lab[m], normalize="index") * 100
        return float((sh.loc["worker"] - sh.loc["queen"]).abs().max())

    conf = P.max(1) / P.sum(1)
    for mn, m in masks.items():
        S[f"n_{mn}"] = int(m.sum())
        S[f"gap_forced_{mn}_pp"] = gap(m, sub)
        b = pd.qcut(conf[m], 5, labels=False)
        idx = np.flatnonzero(m)
        for k in range(5):
            mm = np.zeros(len(m), bool)
            mm[idx[b == k]] = True
            S[f"gap_conf_q{k + 1}_{mn}_pp"] = gap(mm, sub)

    # Tangram-free correlation control.
    ref = ad.read_h5ad(REF)
    rs = ref.obs["leiden_res0_05"].astype(str).to_numpy()
    rlow = {g.lower(): i for i, g in enumerate(ref.var_names)}
    genes = [g for g in overlap if g in set(var) and g.lower() in rlow]
    S["corr_n_genes"] = len(genes)
    RX = ref.X[:, [rlow[g.lower()] for g in genes]]
    RX = RX.toarray() if sp.issparse(RX) else np.asarray(RX)
    C = np.vstack([RX[rs == k].mean(0) for k in SUBS])
    C = C - C.mean(1, keepdims=True)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    Sx = X[:, [var.get_loc(g) for g in genes]]
    Sx = Sx - Sx.mean(1, keepdims=True)
    Sx /= np.linalg.norm(Sx, axis=1, keepdims=True) + 1e-12
    corr_call = SUBS[(Sx @ C.T).argmax(1)]
    for mn, m in masks.items():
        S[f"gap_corr_{mn}_pp"] = gap(m, corr_call)
        S[f"gap_corr_over_forced_{mn}"] = (
            S[f"gap_corr_{mn}_pp"] / S[f"gap_forced_{mn}_pp"]
        )

    # Whole-atlas-label restrictions.
    can = read_obs(CAN)[["slice_name", "cell_id", "tangram_cell_type"]]

    def key(d):
        return d["slice_name"].astype(str) + "||" + d["cell_id"].astype(str)

    fine = can.set_index(key(can))["tangram_cell_type"].astype(str)
    fine = fine[~fine.index.duplicated()].reindex(key(obs)).to_numpy()
    S["n_unmatched_fine"] = int(pd.isna(fine).sum())
    fine = fine.astype(str)
    canon = np.array([get_canonical(v) for v in fine])
    nc = masks["kcnc"]
    for rn, keep in (
        ("confirmed_lt", fine == "Large-type Kenyon cells"),
        ("kc_lineage", np.isin(canon, KC6)),
    ):
        S[f"{rn}_share_kcnc_pct"] = 100 * float(keep[nc].mean())
        for c in ("queen", "worker"):
            S[f"{rn}_share_kcnc_{c}_pct"] = 100 * float(keep[nc & (brain == c)].mean())
        S[f"gap_{rn}_kcnc_pp"] = gap(nc & keep, sub)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
