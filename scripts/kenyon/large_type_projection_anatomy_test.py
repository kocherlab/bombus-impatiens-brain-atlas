"""Anatomy test on the d_source-restored Large-type subcluster projections.

Reads the objects written by project_large_type_subclusters.py, so the AUROC
is measured on the object `fig:kc_subtypes` draws.

Question: is the S1-vs-S2 split in `fig:kc_subtypes` D/E position in the compartment
(anatomy) rather than transcriptional subtype?  Reported as effect sizes:
  * AUROC of distance-to-nearest-KC-IC for the S1-vs-S2 argmax
  * the same AUROC within quintiles of nucleus_area (is it just cell size?)
  * S2 share stratified by the atlas cell-type call (if the axis were subtype
    structure the share would differ by cell type; if it is position it will
    not)
  * argmax composition by decile of distance-to-nearest-KC-IC
  * per-column k-NN spatial autocorrelation and CV (a column that looks like
    softmax of unfitted N(0,1) -- autocorr ~0, CV ~1.31 -- was never fitted)

Out: results/ltkc_diag/p3_summary.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

# Only geometry is read off this object (in_kc_* masks, obsm["spatial"], brain,
# z_level), all of which the column-name correction left untouched -- so the
# cited d_kcic numbers are identical either way. Pinned to the corrected file
# anyway so the tangram_cell_type breakdown below prints real names.
ATLAS = CANONICAL_TANGRAM
LTKC = REPO / "results/tangram_ltKC_cb"
OUT = REPO / "results/ltkc_diag"
OUT.mkdir(parents=True, exist_ok=True)

CLUSTERS = ["0", "1", "2"]
SUB = {"0": "S1", "1": "S2", "2": "S3"}
LINES = []


def log(s):
    print(s, flush=True)
    LINES.append(str(s))


def auroc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


log("Reading atlas (backed) ...")
atl = sc.read_h5ad(ATLAS, backed="r")
ao = atl.obs
a_ic = ao["in_kc_ic"].values.astype(bool)
a_nc = ao["in_kc_nc"].values.astype(bool)
a_comp = np.where(a_ic, "KC-IC", np.where(a_nc, "KC-NC", ""))
a_comp[ao["in_kc_oc"].values.astype(bool) & (a_comp == "")] = "KC-OC"
a_xy = np.asarray(atl.obsm["spatial"], dtype=float)
a_brain = ao["brain"].astype(str).values
a_z = ao["z_level"].values.astype(float)

TREES = {}
for b in np.unique(a_brain):
    for z in np.unique(a_z[a_brain == b]):
        sel = (a_brain == b) & (a_z == z)
        comp = a_comp[sel]
        xy = a_xy[sel]
        ic, non = comp == "KC-IC", comp == ""
        TREES[(b, float(z))] = (
            cKDTree(xy[ic]) if ic.sum() else None,
            cKDTree(xy[non]) if non.sum() else None,
        )
log(f"  built {len(TREES)} per-section KD-trees")
del atl


def geom_for(xy, brain, z):
    d_ic = np.full(len(xy), np.nan)
    d_non = np.full(len(xy), np.nan)
    for key in np.unique(
        np.array([f"{b}|{zz}" for b, zz in zip(brain, z, strict=False)])
    ):
        b, zz = key.split("|")
        m = (brain == b) & (z == float(zz))
        t_ic, t_non = TREES.get((b, float(zz)), (None, None))
        if t_ic is not None:
            d_ic[m] = t_ic.query(xy[m])[0]
        if t_non is not None:
            d_non[m] = t_non.query(xy[m])[0]
    return d_ic, d_non


def battery(path, tag):
    log(f"\n########## {tag}  ({path.name})")
    ad = sc.read_h5ad(path)
    ad.obs_names_make_unique()
    p = ad.obsm["tangram_ct_pred"]
    cols = [str(c) for c in (p.columns if hasattr(p, "columns") else CLUSTERS)]
    p = pd.DataFrame(np.asarray(p), index=ad.obs_names, columns=cols)[CLUSTERS]
    xy = np.asarray(ad.obsm["spatial"], dtype=float)
    brain = ad.obs["brain"].astype(str).values
    z = ad.obs["z_level"].values.astype(float)

    # ---- column fitness ---------------------------------------------------
    log("  -- column fitness (spatial autocorrelation vs k-NN mean, CV) --")
    sizes = (
        pd.Series(1, index=pd.MultiIndex.from_arrays([brain, z]))
        .groupby(level=[0, 1])
        .size()
    )
    (bb, zz) = sizes.idxmax()
    m = (brain == bb) & (z == zz)
    tree = cKDTree(xy[m])
    _, nb = tree.query(xy[m], k=11)
    nb = nb[:, 1:]
    for c in CLUSTERS:
        v = p.loc[m, c].values
        r = np.corrcoef(v, v[nb].mean(axis=1))[0, 1]
        log(
            f"    {SUB[c]}: kNN autocorr={r:.3f}  CV={v.std() / v.mean():.3f}  "
            f"(section {bb} z={zz:.0f}, n={m.sum():,})"
        )
    log("    reference: unfitted softmax(N(0,1)) row has autocorr~0, CV~1.31")

    # ---- shares -----------------------------------------------------------
    ic = ad.obs["in_kc_ic"].values.astype(bool)
    nc = ad.obs["in_kc_nc"].values.astype(bool)
    comp = np.where(ic, "KC-IC", np.where(nc, "KC-NC", "KC-OC"))
    am_all = p.idxmax(axis=1)
    v = am_all.value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
    log(
        f"  ALL Kenyon compartments n={len(am_all):,}: "
        + ", ".join(f"{SUB[c]} {100 * v[c]:.2f}%" for c in CLUSTERS)
    )
    for c_ in ("KC-IC", "KC-NC", "KC-OC"):
        mm = comp == c_
        vv = am_all[mm].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
        log(
            f"    {c_} (n={mm.sum():,}): "
            + ", ".join(f"{SUB[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS)
        )
    for b in ("queen", "worker"):
        mm = nc & (brain == b)
        vv = am_all[mm].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
        log(
            f"    KC-NC {b} (n={mm.sum():,}): "
            + ", ".join(f"{SUB[c]} {100 * vv[c]:.1f}%" for c in CLUSTERS)
        )
    sp = p.sample(50000, random_state=0).corr(method="spearman")
    sp.index = sp.columns = [SUB[c] for c in CLUSTERS]
    log("  column spearman (50k sample):\n" + str(sp.round(3)))

    # ---- the shell test, KC-NC only ---------------------------------------
    obs = ad.obs.loc[nc]
    pk = p.loc[nc].reset_index(drop=True)
    d_ic, d_non = geom_for(xy[nc], brain[nc], z[nc])
    feats = pd.DataFrame(index=pk.index)
    feats["d_kcic"] = d_ic
    feats["d_nonkc"] = d_non
    feats["d_edge"] = np.nanmin(np.vstack([d_ic, d_non]), axis=0)
    for k in (
        "rna_count_based_density",
        "total_counts",
        "transcript_counts",
        "n_genes",
        "cell_area",
        "nucleus_area",
    ):
        if k in obs:
            feats[k] = pd.to_numeric(obs[k], errors="coerce").values
    feats["z_level"] = obs["z_level"].values.astype(float)

    am = pk.idxmax(axis=1)
    s12 = am.isin(["0", "1"]).values
    y = (am[s12] == "1").values  # 1 = S2 (outer)
    lr = np.log((pk["1"] + 1e-30) / (pk["0"] + 1e-30)).values
    ok = np.isfinite(lr)

    rows = []
    for k in feats.columns:
        x = feats[k].values
        xs = x[s12]
        gs = np.isfinite(xs)
        a = auroc(xs[gs], y[gs])
        good = np.isfinite(x)
        rho = spearmanr(x[good & ok], lr[good & ok]).statistic
        rows.append(
            dict(
                predictor=k,
                n=int(gs.sum()),
                AUROC_S2_vs_S1=a,
                absAUROC=abs(a - 0.5) + 0.5,
                spearman_logratio=rho,
            )
        )
    tab = pd.DataFrame(rows).sort_values("absAUROC", ascending=False)
    log(
        "  -- predictors of S2-vs-S1 among KC-NC cells called S1 or S2 "
        f"(n={int(s12.sum()):,}); AUROC>0.5 = predictor HIGH in S2/outer --"
    )
    log(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for b in ("queen", "worker"):
        mb = obs["brain"].astype(str).values == b
        sel = s12 & mb
        yb = (am[sel] == "1").values
        for k in ("d_kcic", "d_nonkc", "rna_count_based_density", "total_counts"):
            xb = feats[k].values[sel]
            gb = np.isfinite(xb)
            log(
                f"    {b:6s} {k:24s} AUROC={auroc(xb[gb], yb[gb]):.4f} "
                f"(n={int(gb.sum()):,})"
            )

    # ---- is it just cell size? -------------------------------------------
    if "nucleus_area" in feats:
        na = feats["nucleus_area"].values
        keep = s12 & np.isfinite(na) & np.isfinite(feats["d_kcic"].values)
        yk = (am[keep] == "1").values
        dk = feats["d_kcic"].values[keep]
        nak = na[keep]
        log(
            f"  -- d_kcic AUROC within quintiles of nucleus_area "
            f"(n={int(keep.sum()):,}) --"
        )
        q = pd.qcut(pd.Series(nak), 5, labels=False, duplicates="drop").values
        for i in sorted(set(q)):
            mm = q == i
            log(
                f"    nucleus_area Q{i + 1}: AUROC(d_kcic)="
                f"{auroc(dk[mm], yk[mm]):.4f} (n={int(mm.sum()):,}, "
                f"S2 share={yk[mm].mean():.3f})"
            )
        log("  -- nucleus_area AUROC within quintiles of d_kcic --")
        q = pd.qcut(pd.Series(dk), 5, labels=False, duplicates="drop").values
        for i in sorted(set(q)):
            mm = q == i
            log(
                f"    d_kcic Q{i + 1}: AUROC(nucleus_area)="
                f"{auroc(nak[mm], yk[mm]):.4f} (n={int(mm.sum()):,}, "
                f"S2 share={yk[mm].mean():.3f})"
            )

    # ---- flat across cell types? -----------------------------------------
    if "tangram_cell_type" in obs:
        ct = obs["tangram_cell_type"].astype(str).values[s12]
        dk = feats["d_kcic"].values[s12]
        ys = y
        log("  -- S1/S2 composition by atlas cell-type call (top 8 by n) --")
        for name in pd.Series(ct).value_counts().head(8).index:
            mm = ct == name
            log(
                f"    {name:34s} n={int(mm.sum()):7,}  "
                f"S2 share={ys[mm].mean():.3f}  "
                f"AUROC(d_kcic)={auroc(dk[mm], ys[mm]):.4f}"
            )

    # ---- decile table -----------------------------------------------------
    dd = feats["d_kcic"].values
    good = np.isfinite(dd)
    q = pd.qcut(pd.Series(dd[good]), 10, labels=False, duplicates="drop")
    ct2 = pd.crosstab(q.values, am.values[good], normalize="index").round(3)
    ct2.columns = [SUB.get(c, c) for c in ct2.columns]
    ct2.index.name = "dec"
    log(
        "  -- argmax composition by decile of distance-to-nearest-KC-IC "
        "(0 = closest to the inner core) --"
    )
    log(ct2.to_string())


battery(
    LTKC / "ltKC_CB_spatial_subclusters_dsource.h5ad",
    "SHIPPED (CB reference): config C (classic-equivalent: d_source + cluster SUM)",
)
battery(
    LTKC / "ltKC_CB_spatial_subclusters_dsource_meanS.h5ad",
    "config B (d_source + cluster MEAN; not shipped)",
)

(OUT / "p3_cb_summary.txt").write_text("\n".join(LINES) + "\n")
print("wrote", OUT / "p3_cb_summary.txt", flush=True)
