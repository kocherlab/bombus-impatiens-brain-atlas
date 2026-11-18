#!/usr/bin/env python3
"""Why does "Small-type Kenyon cells" behave pathologically?

"Small-type Kenyon cells" can act as a brain-wide attractor in the Tangram
labelling: this script measures its share of the medulla and of unannotated
tissue in the restricted-projection test, the share of its region-assigned
per-cell calls inside the OPTIC LOBE, its enrichment in the Mixed block and
its compartment peak. Two competing explanations imply opposite fixes:

  (A) MAPPING ARTEFACT -- the reference type is fine, but its panel-restricted
      profile is close to the panel-wide mean, so it wins argmax by default
      wherever the 101-gene panel carries little signal. Fix lives in the
      mapping / labelling.

  (B) REFERENCE MIS-ANNOTATION -- the snRNA cluster called "Small-type Kenyon
      cells" is not small-type Kenyon cells at all. Fix lives in the reference,
      and this is a reportable finding.

Stages (``--stage``):

  ref      reference-side structure. Pseudobulk correlation over the FULL
           transcriptome across all 29 reference types; panel-restricted
           near-mean / flatness statistics ranked over all 29; de-circularized
           Kenyon-marker test (markers recomputed with Small-type EXCLUDED from
           the positive class, so the test is not self-fulfilling); QC profile;
           library / trial / caste / leiden provenance; UMAP + PCA geometry.

  spatial  the direct test. Drop one cell-type COLUMN from ``tangram_ct_pred``,
           renormalize, re-argmax, and re-run the anatomy-guided cluster vote.
           Reported per anatomical compartment (which is the measurement that
           discriminates A from B: if the medulla flips to OL_*, Small-type was
           outcompeting the correct optic types; if it flips to another Kenyon
           type, the attractor is the whole Kenyon block and dropping one label
           fixes nothing) and as a leave-one-out sweep over every canonical
           type, so "removing Small-type improves the atlas" has a null.

IMPORTANT SCOPE LIMIT: the leave-one-out here is at the LABEL level -- the
Tangram mapping matrix is unchanged and we merely delete a column of the
cell-type posterior and renormalize. A true reference-ablation would retrain
Tangram on the GPU and is out of scope. Deleting a column cannot change which
reference nuclei the spatial cells were mapped onto; it only asks where the
posterior mass of the deleted type would otherwise have gone.

Run:
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/identity/small_kc_attractor_diagnostic.py --stage ref
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/identity/small_kc_attractor_diagnostic.py --stage spatial
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bimp_st.cluster_labels import (  # noqa: E402
    _CALYX_FLAGS,
    _OPTIC_FLAGS,
    CANONICAL_ORDER,
    POOLED_ORDER,
    get_canonical,
    get_pooled,
)
from bimp_st.paths import CANONICAL_TANGRAM, SN_COMBINED  # noqa: E402

warnings.filterwarnings("ignore")

TANGRAM_H5AD = CANONICAL_TANGRAM
SN_CB = REPO / "data/sn/BIMP_atlas_qw_CentralBrain.h5ad"
GENE_BRIDGE = REPO / "results/reframe/noncircular/gene_bridge.csv"
OUT = REPO / "results/smallkc_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

FOCAL = "Small-type Kenyon cells"
SUSPECT2 = "Photoreceptors"  # the systemic-vs-one-outlier control
KC_FINE = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
    "Kenyon cells",
]
MIXED = "Mixed"
REGION_THRESH = 0.5
MIN_PURITY = 0.50

REGION_FLAGS = [
    "in_kc_ic",
    "in_kc_nc",
    "in_kc_oc",
    "in_lamina",
    "in_medulla",
    "in_lobula",
]


# ═════════════════════════════ shared I/O ═══════════════════════════════════


def load_ref(path: Path = SN_COMBINED):
    """log1p-normalized X (csr), obs, var index of the combined reference."""
    print(f"loading {path.name} ...", flush=True)
    with h5py.File(path, "r") as h:
        X = read_elem(h["X"])
        obs = read_elem(h["obs"])
        var = read_elem(h["var"])
    print(f"  {X.shape[0]:,} nuclei x {X.shape[1]:,} genes", flush=True)
    return X, obs, var


def panel_genes(var_index) -> list[str]:
    """The v4.3 IDs of the Xenium panel genes present in the reference."""
    br = pd.read_csv(GENE_BRIDGE)
    ids = br.loc[br["in_snuc"].astype(bool), "v43"].astype(str).unique().tolist()
    present = [g for g in ids if g in set(var_index)]
    print(
        f"  panel bridge: {len(ids)} v4.3 IDs, {len(present)} in this reference",
        flush=True,
    )
    return present


def gene_descriptions() -> pd.Series:
    """v4.3 ID -> human-readable description (only the CB h5ad carries it)."""
    with h5py.File(SN_CB, "r") as h:
        v = read_elem(h["var"])
    if "description" not in v.columns:
        return pd.Series(dtype=str)
    return v["description"].astype(str)


def pseudobulk(X, labels: pd.Series) -> pd.DataFrame:
    """Mean log1p expression per cell type -> types x genes."""
    cats = sorted(labels.unique())
    rows = []
    lv = labels.to_numpy()
    for c in cats:
        m = lv == c
        rows.append(np.asarray(X[m].mean(axis=0)).ravel())
    return pd.DataFrame(np.vstack(rows), index=cats)


# ═════════════════════════════ stage: ref ═══════════════════════════════════


def stage_ref() -> None:
    X, obs, var = load_ref()
    ct = obs["cell_type"].astype(str)
    genes = pd.Index(var.index.astype(str))
    desc = gene_descriptions()

    # ── 0. provenance: is it "one random outlier"? ───────────────────────────
    print("\n0. PROVENANCE of the focal cluster", flush=True)
    prov_rows = []
    for col in [
        "sample",
        "trial",
        "caste",
        "region",
        "leiden",
        "leiden_res0_50",
        "leiden_res1",
    ]:
        if col not in obs.columns:
            continue
        tab = pd.crosstab(ct, obs[col].astype(str), normalize="index")
        for t in KC_FINE + [SUSPECT2]:
            if t not in tab.index:
                continue
            row = tab.loc[t].sort_values(ascending=False)
            prov_rows.append(
                {
                    "cell_type": t,
                    "obs_col": col,
                    "n_levels_used": int((row > 0.01).sum()),
                    "top_level": row.index[0],
                    "top_frac": float(row.iloc[0]),
                    "top2_level": row.index[1] if len(row) > 1 else None,
                    "top2_frac": float(row.iloc[1]) if len(row) > 1 else np.nan,
                    "shannon_evenness": float(
                        -(row[row > 0] * np.log(row[row > 0])).sum()
                        / np.log(max(len(row), 2))
                    ),
                }
            )
        tab.to_csv(OUT / f"ref_composition_by_{col}.csv")
    prov = pd.DataFrame(prov_rows)
    prov.to_csv(OUT / "ref_provenance_summary.csv", index=False)
    print(prov.to_string(index=False), flush=True)

    # ── 1. QC profile ────────────────────────────────────────────────────────
    print("\n1. QC PROFILE per reference type", flush=True)
    qc_cols = [
        c
        for c in [
            "total_counts",
            "n_genes_by_counts",
            "n_genes",
            "pct_counts_mt",
            "doublet_score",
            "pct_counts_in_top_20_genes",
            "predicted_doublet",
        ]
        if c in obs.columns
    ]
    qc = obs.groupby(ct, observed=True)[qc_cols].median()
    qc["n_nuclei"] = ct.value_counts().reindex(qc.index)
    if "predicted_doublet" in obs.columns:
        qc["frac_predicted_doublet"] = (
            obs.groupby(ct, observed=True)["predicted_doublet"].mean().reindex(qc.index)
        )
    qc = qc.sort_values("n_nuclei", ascending=False)
    qc.to_csv(OUT / "ref_qc_by_type.csv")
    print(
        qc.loc[[t for t in KC_FINE + [SUSPECT2] if t in qc.index]].to_string(),
        flush=True,
    )

    # ── 2. full-transcriptome pseudobulk correlation ─────────────────────────
    print("\n2. FULL-TRANSCRIPTOME pseudobulk correlation across 29 types", flush=True)
    pb = pseudobulk(X, ct)
    pb.columns = genes
    pb.to_csv(OUT / "ref_pseudobulk_full.csv.gz")
    corr = pb.T.corr(method="pearson")
    corr.to_csv(OUT / "ref_pseudobulk_corr_full.csv")

    # do it on HVGs too -- the full transcriptome is dominated by shared
    # housekeeping structure and everything correlates > 0.9 there
    hv = (
        var["highly_variable"].astype(bool).to_numpy()
        if "highly_variable" in var.columns
        else np.ones(len(genes), bool)
    )
    corr_hvg = pb.loc[:, genes[hv]].T.corr(method="pearson")
    corr_hvg.to_csv(OUT / "ref_pseudobulk_corr_hvg.csv")
    print(f"  {int(hv.sum())} HVGs used for the HVG correlation", flush=True)

    nb_rows = []
    for t in list(corr.index):
        for name, C in [("full", corr), ("hvg", corr_hvg)]:
            r = C.loc[t].drop(t).sort_values(ascending=False)
            kc_others = [k for k in KC_FINE if k != t and k in r.index]
            nb_rows.append(
                {
                    "cell_type": t,
                    "gene_space": name,
                    "nn1": r.index[0],
                    "nn1_r": float(r.iloc[0]),
                    "nn2": r.index[1],
                    "nn2_r": float(r.iloc[1]),
                    "nn3": r.index[2],
                    "nn3_r": float(r.iloc[2]),
                    "mean_r_to_other_KC": float(r[kc_others].mean())
                    if kc_others
                    else np.nan,
                    "mean_r_to_OL": float(
                        r[[i for i in r.index if i.startswith("OL_")]].mean()
                    ),
                    "mean_r_to_all": float(r.mean()),
                }
            )
    nb = pd.DataFrame(nb_rows)
    nb.to_csv(OUT / "ref_nearest_types.csv", index=False)
    print(
        nb[nb.cell_type.isin(KC_FINE + [SUSPECT2])].to_string(index=False), flush=True
    )

    # within-Kenyon vs between-block collinearity, on both gene spaces
    coll = []
    kc_present = [k for k in KC_FINE if k in corr.index]
    for name, C in [("full", corr), ("hvg", corr_hvg)]:
        v = C.values.copy()
        np.fill_diagonal(v, np.nan)
        iskc = np.array([i in kc_present for i in C.index])
        within = v[np.ix_(iskc, iskc)]
        between = v[np.ix_(~iskc, ~iskc)]
        cross = v[np.ix_(iskc, ~iskc)]
        coll.append(
            {
                "gene_space": name,
                "median_r_within_KC": float(np.nanmedian(within)),
                "median_r_between_nonKC": float(np.nanmedian(between)),
                "median_r_KC_vs_nonKC": float(np.nanmedian(cross)),
            }
        )
    pd.DataFrame(coll).to_csv(OUT / "ref_collinearity.csv", index=False)
    print(pd.DataFrame(coll).to_string(index=False), flush=True)

    # ── 3. panel-restricted flatness / near-mean statistic ───────────────────
    print(
        "\n3. PANEL-RESTRICTED near-mean / flatness, ranked over all types", flush=True
    )
    pgenes = panel_genes(genes)
    P = pb.loc[:, pgenes]
    grand = P.mean(axis=0)  # panel-wide mean profile
    stats = []
    for t in P.index:
        p = P.loc[t]
        stats.append(
            {
                "cell_type": t,
                "n_nuclei": int((ct == t).sum()),
                "pearson_to_panel_mean": float(np.corrcoef(p, grand)[0, 1]),
                "cosine_to_panel_mean": float(
                    p @ grand / (np.linalg.norm(p) * np.linalg.norm(grand))
                ),
                "l2_from_panel_mean": float(np.linalg.norm(p - grand)),
                "panel_profile_sd": float(p.std()),
                "panel_profile_cv": float(p.std() / p.mean()) if p.mean() else np.nan,
                "panel_mean_expr": float(p.mean()),
                # how peaked is the profile: share of the profile in its top 5 genes
                "top5_share": float(np.sort(p)[::-1][:5].sum() / p.sum()),
            }
        )
    st = pd.DataFrame(stats)
    for c in ["pearson_to_panel_mean", "cosine_to_panel_mean"]:
        st[f"rank_{c}"] = st[c].rank(ascending=False).astype(int)
    st["rank_l2_from_panel_mean"] = st["l2_from_panel_mean"].rank().astype(int)
    st["rank_panel_profile_sd"] = st["panel_profile_sd"].rank().astype(int)
    st = st.sort_values("cosine_to_panel_mean", ascending=False)
    st.to_csv(OUT / "ref_panel_flatness.csv", index=False)
    print(st.to_string(index=False), flush=True)

    # same statistic on the FULL transcriptome, as the control: if Small-type
    # is near-mean on the panel but not on the transcriptome, the panel is the
    # cause; if it is near-mean on both, the cluster itself is indistinct.
    grand_f = pb.mean(axis=0)
    fs = []
    for t in pb.index:
        p = pb.loc[t]
        fs.append(
            {
                "cell_type": t,
                "pearson_to_full_mean": float(np.corrcoef(p, grand_f)[0, 1]),
                "cosine_to_full_mean": float(
                    p @ grand_f / (np.linalg.norm(p) * np.linalg.norm(grand_f))
                ),
                "full_profile_sd": float(p.std()),
            }
        )
    fsd = pd.DataFrame(fs)
    fsd["rank_cosine_to_full_mean"] = (
        fsd["cosine_to_full_mean"].rank(ascending=False).astype(int)
    )
    fsd.to_csv(OUT / "ref_full_flatness.csv", index=False)
    print("\n  full-transcriptome control (top 6 nearest the grand mean):", flush=True)
    print(
        fsd.sort_values("cosine_to_full_mean", ascending=False)
        .head(6)
        .to_string(index=False),
        flush=True,
    )

    # ── 4. markers ───────────────────────────────────────────────────────────
    print("\n4. MARKERS (one-vs-rest AUROC, full transcriptome)", flush=True)
    marker_frames = []
    targets = [
        t
        for t in [
            FOCAL,
            "Large-type Kenyon cells",
            "Middle-type Kenyon cells",
            "Class II Kenyon cells",
            SUSPECT2,
        ]
        if t in set(ct)
    ]
    aucs = _auroc_multi(X, {t: (ct.to_numpy() == t) for t in targets})
    for t in targets:
        y = ct.to_numpy() == t
        auc = aucs[t]
        d = pd.DataFrame({"gene": genes, "auroc": auc})
        d["mean_in_type"] = np.asarray(X[y].mean(axis=0)).ravel()
        d["mean_in_rest"] = np.asarray(X[~y].mean(axis=0)).ravel()
        d["log2fc"] = np.log2((d.mean_in_type + 1e-9) / (d.mean_in_rest + 1e-9))
        d["pct_in_type"] = np.asarray((X[y] > 0).mean(axis=0)).ravel()
        d["pct_in_rest"] = np.asarray((X[~y] > 0).mean(axis=0)).ravel()
        d["description"] = d["gene"].map(desc)
        d["on_xenium_panel"] = d["gene"].isin(set(pgenes))
        d["cell_type"] = t
        d = d.sort_values("auroc", ascending=False)
        marker_frames.append(d)
        print(f"\n  --- {t}: top 15 markers by one-vs-rest AUROC ---", flush=True)
        print(
            d.head(15)[
                [
                    "gene",
                    "auroc",
                    "log2fc",
                    "pct_in_type",
                    "pct_in_rest",
                    "on_xenium_panel",
                    "description",
                ]
            ].to_string(index=False),
            flush=True,
        )
        print(
            f"    genes with AUROC >= 0.70: {(d.auroc >= 0.70).sum()}; "
            f">= 0.80: {(d.auroc >= 0.80).sum()}; "
            f">= 0.90: {(d.auroc >= 0.90).sum()}",
            flush=True,
        )
    pd.concat(marker_frames).to_csv(OUT / "ref_markers.csv.gz", index=False)

    # ── 5. DE-CIRCULARIZED Kenyon-marker test ────────────────────────────────
    # The Kenyon-marker set (scored by AUROC) was derived from
    # ``major_cell_type == "Kenyon cells"``, which INCLUDES Small-type nuclei
    # in the positive class -- so asking whether Small-type expresses those
    # markers is partly self-fulfilling. Recompute them with Small-type
    # excluded from BOTH classes, then score Small-type against them.
    print("\n5. DE-CIRCULARIZED Kenyon-marker test", flush=True)
    major = obs["major_cell_type"].astype(str)
    keep = (ct != FOCAL).to_numpy()
    y_kc = (major.to_numpy()[keep] == "Kenyon cells").astype(int)
    Xp = X[:, [genes.get_loc(g) for g in pgenes]]
    auc_dec = _fast_auroc(Xp[keep], y_kc.astype(bool))
    order = np.argsort(auc_dec)[::-1]
    top = [pgenes[i] for i in order[:10]]
    print(
        f"  KC positives (Small-type excluded): {y_kc.sum():,} / {len(y_kc):,}",
        flush=True,
    )
    print(
        f"  top-10 de-circularized KC markers, AUROC "
        f"{auc_dec[order[9]]:.3f}-{auc_dec[order[0]]:.3f}",
        flush=True,
    )

    # score every type on the mean z of those 10 markers
    Z = Xp[:, [pgenes.index(g) for g in top]]
    Z = np.asarray(Z.todense()) if sparse.issparse(Z) else np.asarray(Z)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    score = Z.mean(1)
    sc = pd.DataFrame({"cell_type": ct.to_numpy(), "score": score})
    agg = sc.groupby("cell_type")["score"].agg(["mean", "median", "count"])
    agg = agg.sort_values("mean", ascending=False)
    agg["is_kenyon"] = [i in KC_FINE for i in agg.index]
    agg.to_csv(OUT / "ref_decircularized_kc_score.csv")
    pd.DataFrame(
        {
            "gene": top,
            "auroc_kc_vs_rest_smalltype_excluded": [
                auc_dec[order[i]] for i in range(10)
            ],
            "description": [desc.get(g, "") for g in top],
        }
    ).to_csv(OUT / "ref_decircularized_kc_markers.csv", index=False)
    print(agg.to_string(), flush=True)

    # also: circular version, for the contrast
    y_kc_all = (major.to_numpy() == "Kenyon cells").astype(bool)
    auc_circ = _fast_auroc(Xp, y_kc_all)
    o2 = np.argsort(auc_circ)[::-1][:10]
    Z2 = Xp[:, o2]
    Z2 = np.asarray(Z2.todense()) if sparse.issparse(Z2) else np.asarray(Z2)
    Z2 = (Z2 - Z2.mean(0)) / (Z2.std(0) + 1e-9)
    sc2 = pd.DataFrame({"cell_type": ct.to_numpy(), "score": Z2.mean(1)})
    a2 = sc2.groupby("cell_type")["score"].mean()
    cmp = pd.DataFrame({"decircularized": agg["mean"], "circular": a2})
    cmp.to_csv(OUT / "ref_kc_score_circular_vs_not.csv")
    print("\n  circular vs de-circularized KC score:", flush=True)
    print(cmp.sort_values("decircularized", ascending=False).to_string(), flush=True)

    # ── 6. geometry in the reference's own embedding ─────────────────────────
    print("\n6. GEOMETRY in the reference UMAP / harmony PCA", flush=True)
    with h5py.File(SN_COMBINED, "r") as h:
        um = read_elem(h["obsm/X_umap"])
        pcs = read_elem(h["obsm/X_pca_harmony"])
    geo_rows = []
    cts = sorted(set(ct))
    cent_u = {t: um[(ct == t).to_numpy()].mean(0) for t in cts}
    cent_p = {t: pcs[(ct == t).to_numpy()].mean(0) for t in cts}
    for t in cts:
        du = {o: float(np.linalg.norm(cent_u[t] - cent_u[o])) for o in cts if o != t}
        dp = {o: float(np.linalg.norm(cent_p[t] - cent_p[o])) for o in cts if o != t}
        su, sp = pd.Series(du).sort_values(), pd.Series(dp).sort_values()
        kco = [k for k in KC_FINE if k != t and k in su.index]
        olo = [i for i in su.index if i.startswith("OL_")]
        geo_rows.append(
            {
                "cell_type": t,
                "umap_nn": su.index[0],
                "umap_nn_dist": su.iloc[0],
                "umap_mean_dist_to_KC": float(su[kco].mean()) if kco else np.nan,
                "umap_mean_dist_to_OL": float(su[olo].mean()),
                "pca_nn": sp.index[0],
                "pca_nn_dist": sp.iloc[0],
                "pca_mean_dist_to_KC": float(sp[kco].mean()) if kco else np.nan,
                "pca_mean_dist_to_OL": float(sp[olo].mean()),
                "umap_spread": float(
                    np.linalg.norm(um[(ct == t).to_numpy()] - cent_u[t], axis=1).mean()
                ),
            }
        )
    geo = pd.DataFrame(geo_rows)
    geo.to_csv(OUT / "ref_geometry.csv", index=False)
    print(
        geo[geo.cell_type.isin(KC_FINE + [SUSPECT2])].to_string(index=False), flush=True
    )

    print(f"\nwrote {OUT}/ref_*.csv", flush=True)


def _fast_auroc(X, y: np.ndarray, block: int = 256) -> np.ndarray:
    """One-vs-rest AUROC per gene via Mann-Whitney rank sums.

    Densifies only ``block`` genes at a time, so the 73k x 10.5k reference
    never has to exist as a dense array.
    """
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    Xc = X.tocsc() if sparse.issparse(X) else X
    n_genes = Xc.shape[1]
    out = np.empty(n_genes, np.float64)
    for s in range(0, n_genes, block):
        e = min(s + block, n_genes)
        blk = Xc[:, s:e]
        blk = np.asarray(blk.todense()) if sparse.issparse(blk) else np.asarray(blk)
        r = rankdata(blk, axis=0)
        out[s:e] = (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def _auroc_multi(X, groups: dict[str, np.ndarray], block: int = 256):
    """One-vs-rest AUROC for several groups in ONE pass over the genes.

    Gene ranks do not depend on the group, so ranking once and taking several
    rank sums is ~len(groups)x cheaper than calling ``_fast_auroc`` per group.
    """
    from scipy.stats import rankdata

    Xc = X.tocsc() if sparse.issparse(X) else X
    n_genes = Xc.shape[1]
    out = {k: np.empty(n_genes, np.float64) for k in groups}
    ns = {k: (int(v.sum()), int((~v).sum())) for k, v in groups.items()}
    for s in range(0, n_genes, block):
        e = min(s + block, n_genes)
        blk = Xc[:, s:e]
        blk = np.asarray(blk.todense()) if sparse.issparse(blk) else np.asarray(blk)
        r = rankdata(blk, axis=0)
        for k, y in groups.items():
            n1, n0 = ns[k]
            out[k][s:e] = (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        if (s // block) % 10 == 0:
            print(f"    AUROC genes {e}/{n_genes}", flush=True)
    return out


# ═════════════════════════════ stage: spatial ═══════════════════════════════


def load_spatial():
    print(f"loading {TANGRAM_H5AD.name} (obs + tangram_ct_pred) ...", flush=True)
    with h5py.File(TANGRAM_H5AD, "r") as h:
        obs = read_elem(h["obs"])
        pred = read_elem(h["obsm/tangram_ct_pred"])
    pred = pred[[c for c in pred.columns]].astype(np.float32)
    print(f"  {len(obs):,} cells, posterior over {pred.shape[1]} types", flush=True)
    return obs, pred


def region_of(obs: pd.DataFrame) -> pd.Series:
    """Coarse anatomical compartment per cell (mutually exclusive; 'none')."""
    reg = pd.Series("none", index=obs.index, dtype=object)
    for f in REGION_FLAGS:
        if f in obs.columns:
            reg[obs[f].astype(bool).to_numpy()] = f.replace("in_", "")
    return reg


def vote(
    obs_labels: pd.Series,
    cluster: pd.Series,
    calyx: np.ndarray,
    optic: np.ndarray,
    min_purity: float = MIN_PURITY,
) -> pd.Series:
    """anatomy_guided_vote's decision rule, applied to per-cell fine labels."""
    pooled = obs_labels.map(get_pooled)
    df = pd.DataFrame(
        {
            "cl": cluster.to_numpy(),
            "lab": pooled.to_numpy(),
            "calyx": calyx,
            "optic": optic,
        }
    )
    out = {}
    for c, sub in df.groupby("cl", sort=False):
        if sub["optic"].mean() > REGION_THRESH:
            out[c] = "Optic lobe"
        elif sub["calyx"].mean() > REGION_THRESH:
            out[c] = "Kenyon cells"
        else:
            vc = sub["lab"].value_counts(normalize=True)
            out[c] = vc.index[0] if vc.iloc[0] >= min_purity else MIXED
    return cluster.map(out)


def tvd(a: pd.Series, b: pd.Series) -> float:
    a = a.reindex(POOLED_ORDER).fillna(0.0)
    b = b.reindex(POOLED_ORDER).fillna(0.0)
    return float(0.5 * np.abs(a - b).sum())


def ref_pooled_composition(drop: str | None = None) -> pd.Series:
    with h5py.File(SN_COMBINED, "r") as h:
        ct = pd.Series(read_elem(h["obs/cell_type"]).astype(str))
    if drop is not None:
        ct = ct[ct != drop]
    return (
        ct.map(get_pooled)
        .value_counts(normalize=True)
        .reindex(POOLED_ORDER)
        .fillna(0.0)
    )


def stage_spatial() -> None:
    obs, pred = load_spatial()
    cols = list(pred.columns)
    A = pred.to_numpy()
    reg = region_of(obs)
    cluster = obs["leiden"].astype(str)
    calyx = np.zeros(len(obs), bool)
    optic = np.zeros(len(obs), bool)
    for f in _CALYX_FLAGS:
        if f in obs.columns:
            calyx |= obs[f].astype(bool).to_numpy()
    for f in _OPTIC_FLAGS:
        if f in obs.columns:
            optic |= obs[f].astype(bool).to_numpy()

    base_idx = A.argmax(1)
    base_lab = pd.Series(np.array(cols)[base_idx], index=obs.index)
    ref_full = ref_pooled_composition()

    # ── A. where does the focal type's posterior mass go? ────────────────────
    print(
        "\nA. RUNNER-UP IDENTITY -- drop one column, renormalize, re-argmax", flush=True
    )
    runner_frames = []
    for focal in [FOCAL, SUSPECT2]:
        j = cols.index(focal)
        keep = [i for i in range(len(cols)) if i != j]
        new_idx = A[:, keep].argmax(1)
        new_lab = pd.Series(np.array(cols)[[keep[i] for i in new_idx]], index=obs.index)
        was = (base_lab == focal).to_numpy()
        print(
            f"\n  {focal}: {was.sum():,} cells "
            f"({100 * was.mean():.1f}% of atlas) held it at argmax",
            flush=True,
        )
        # global runner-up
        g = new_lab[was].value_counts(normalize=True)
        print("   global runner-up:", flush=True)
        print(("     " + g.head(8).to_string().replace("\n", "\n     ")), flush=True)
        for r, sub in reg[was].groupby(reg[was]):
            vc = new_lab[sub.index].value_counts(normalize=True)
            for k in range(min(4, len(vc))):
                runner_frames.append(
                    {
                        "dropped": focal,
                        "region": r,
                        "n_cells_in_region_with_focal": len(sub),
                        "rank": k + 1,
                        "runner_up": vc.index[k],
                        "frac": float(vc.iloc[k]),
                        "runner_up_canonical": get_canonical(vc.index[k]),
                        "runner_up_pooled": get_pooled(vc.index[k]),
                    }
                )
        # full re-argmax composition per region (not only the focal cells)
        cross = pd.crosstab(reg, new_lab.map(get_canonical), normalize="index")
        cross.to_csv(
            OUT / f"spatial_region_composition_without_{focal.split()[0].lower()}.csv"
        )
    rf = pd.DataFrame(runner_frames)
    rf.to_csv(OUT / "spatial_runner_up_by_region.csv", index=False)
    print("\n  runner-up by region (top 3 each):", flush=True)
    print(rf[rf["rank"] <= 3].to_string(index=False), flush=True)

    # baseline region composition WITH the focal type, for contrast
    pd.crosstab(reg, base_lab.map(get_canonical), normalize="index").to_csv(
        OUT / "spatial_region_composition_baseline.csv"
    )

    # ── B. leave-one-out sweep over every canonical type ─────────────────────
    print(
        "\nB. LEAVE-ONE-OUT sweep -- Mixed% and TVD for every dropped type", flush=True
    )
    base_vote = vote(base_lab, cluster, calyx, optic)
    base_counts = base_vote.value_counts()
    base_mixed = 100 * base_counts.get(MIXED, 0) / len(obs)
    called = base_counts.drop(labels=[MIXED], errors="ignore")
    base_tvd = tvd(called / called.sum(), ref_full)
    print(f"  baseline: Mixed {base_mixed:.1f}%  TVD {base_tvd:.4f}", flush=True)

    rows = [
        {
            "dropped": "(none)",
            "n_dropped_cells": 0,
            "pct_mixed": base_mixed,
            "delta_pct_mixed": 0.0,
            "tvd_vs_full_ref": base_tvd,
            "delta_tvd_vs_full_ref": 0.0,
            "tvd_vs_reduced_ref": base_tvd,
            "delta_tvd_vs_reduced_ref": 0.0,
        }
    ]
    # sweep the canonical vocabulary: the 16 curated types, plus "Optic lobe"
    # meaning all 13 OL_* columns at once.
    sweeps: list[tuple[str, list[str]]] = [
        (t, [t]) for t in CANONICAL_ORDER if t in cols and t != "Optic lobe"
    ]
    sweeps.append(("Optic lobe (all OL_*)", [c for c in cols if c.startswith("OL_")]))
    for name, drops in sweeps:
        keep = [i for i, c in enumerate(cols) if c not in set(drops)]
        new_idx = A[:, keep].argmax(1)
        new_lab = pd.Series(np.array(cols)[[keep[i] for i in new_idx]], index=obs.index)
        v = vote(new_lab, cluster, calyx, optic)
        counts = v.value_counts()
        pm = 100 * counts.get(MIXED, 0) / len(obs)
        cal = counts.drop(labels=[MIXED], errors="ignore")
        comp = cal / cal.sum()
        red = ref_pooled_composition(drop=drops[0] if len(drops) == 1 else None)
        if len(drops) > 1:
            with h5py.File(SN_COMBINED, "r") as h:
                c2 = pd.Series(read_elem(h["obs/cell_type"]).astype(str))
            c2 = c2[~c2.isin(set(drops))]
            red = (
                c2.map(get_pooled)
                .value_counts(normalize=True)
                .reindex(POOLED_ORDER)
                .fillna(0.0)
            )
        n_drop = int((base_lab.isin(set(drops))).sum())
        rows.append(
            {
                "dropped": name,
                "n_dropped_cells": n_drop,
                "pct_dropped_of_atlas": 100 * n_drop / len(obs),
                "pct_mixed": pm,
                "delta_pct_mixed": pm - base_mixed,
                "tvd_vs_full_ref": tvd(comp, ref_full),
                "delta_tvd_vs_full_ref": tvd(comp, ref_full) - base_tvd,
                "tvd_vs_reduced_ref": tvd(comp, red),
                "delta_tvd_vs_reduced_ref": tvd(comp, red) - base_tvd,
            }
        )
        print(
            f"  drop {name:<34s} n={n_drop:>8,}  Mixed {pm:5.1f}% "
            f"({pm - base_mixed:+5.1f})  TVD {tvd(comp, ref_full):.4f} "
            f"({tvd(comp, ref_full) - base_tvd:+.4f})",
            flush=True,
        )
    loo = pd.DataFrame(rows)
    loo.to_csv(OUT / "spatial_loo_sweep.csv", index=False)

    # ── C. anatomical sanity, with and without the focal type ────────────────
    print("\nC. ANATOMICAL SANITY of the medulla / unannotated tissue", flush=True)
    sanity = []
    for name, drops in [("(none)", []), (FOCAL, [FOCAL]), (SUSPECT2, [SUSPECT2])]:
        keep = [i for i, c in enumerate(cols) if c not in set(drops)]
        new_idx = A[:, keep].argmax(1)
        new_lab = pd.Series(np.array(cols)[[keep[i] for i in new_idx]], index=obs.index)
        can = new_lab.map(get_canonical)
        for r in ["medulla", "lamina", "lobula", "kc_ic", "kc_nc", "kc_oc", "none"]:
            m = (reg == r).to_numpy()
            if m.sum() == 0:
                continue
            vc = can[m].value_counts(normalize=True)
            sanity.append(
                {
                    "dropped": name,
                    "region": r,
                    "n_cells": int(m.sum()),
                    "top_label": vc.index[0],
                    "top_frac": float(vc.iloc[0]),
                    "frac_optic_lobe": float(vc.get("Optic lobe", 0.0)),
                    "frac_kenyon_block": float(sum(vc.get(k, 0.0) for k in KC_FINE)),
                    "frac_smalltype": float(vc.get(FOCAL, 0.0)),
                }
            )
    sd = pd.DataFrame(sanity)
    sd.to_csv(OUT / "spatial_anatomical_sanity.csv", index=False)
    print(sd.to_string(index=False), flush=True)

    # ── D. per-cell posterior geometry of the focal calls ────────────────────
    print("\nD. POSTERIOR GEOMETRY of cells argmaxed to each type", flush=True)
    geo = []
    srt = np.sort(A, axis=1)
    top1, top2 = srt[:, -1], srt[:, -2]
    ent = -(A * np.log(A + 1e-12)).sum(1)
    for t in cols:
        m = (base_lab == t).to_numpy()
        if m.sum() < 50:
            continue
        geo.append(
            {
                "cell_type": t,
                "n_cells": int(m.sum()),
                "canonical": get_canonical(t),
                "median_top1": float(np.median(top1[m])),
                "median_margin": float(np.median(top1[m] - top2[m])),
                "median_entropy": float(np.median(ent[m])),
                "median_ratio_top1_top2": float(np.median(top1[m] / (top2[m] + 1e-12))),
            }
        )
    gd = pd.DataFrame(geo).sort_values("median_top1")
    gd.to_csv(OUT / "spatial_posterior_geometry.csv", index=False)
    print(gd.to_string(index=False), flush=True)

    print(f"\nwrote {OUT}/spatial_*.csv", flush=True)


# ═════════════════════════════ stage: depth ═════════════════════════════════


def stage_depth() -> None:
    """Is the attractor a LIBRARY-COMPLEXITY match rather than a cell-type call?

    The drop-one test (stage ``spatial``) shows Small-type's mass moving to
    Cortex glia / subperineurial glia / hemocytes, not to optic types and not
    to other Kenyon types. Those runners-up have one thing in common: they are
    the shallowest libraries in the reference. This stage asks whether the
    Tangram posterior in low-signal spatial tissue is tracking RNA content
    rather than identity -- which would make the pathology a general property
    of mapping a sparse panel onto a depth-heterogeneous reference, not one
    bad label.
    """
    with h5py.File(SN_COMBINED, "r") as h:
        rct = pd.Series(read_elem(h["obs/cell_type"]).astype(str))
        rtc = pd.Series(read_elem(h["obs/total_counts"]))
        rng = pd.Series(read_elem(h["obs/n_genes_by_counts"]))
    ref_depth = pd.DataFrame(
        {
            "ref_median_total_counts": rtc.groupby(rct).median(),
            "ref_median_n_genes": rng.groupby(rct).median(),
            "n_nuclei": rct.value_counts(),
        }
    )

    obs, pred = load_spatial()
    cols = list(pred.columns)
    A = pred.to_numpy()
    base = pd.Series(np.array(cols)[A.argmax(1)], index=obs.index)
    tc = obs["transcript_counts"].astype(float)
    ng = obs["n_genes"].astype(float)

    sp = pd.DataFrame(
        {
            "spatial_median_transcripts": tc.groupby(base).median(),
            "spatial_median_n_genes": ng.groupby(base).median(),
            "n_spatial_cells": base.value_counts(),
        }
    )
    tab = ref_depth.join(sp, how="outer")
    tab["spatial_share"] = tab["n_spatial_cells"] / len(obs)
    tab["ref_share"] = tab["n_nuclei"] / len(rct)
    tab["share_ratio_spatial_over_ref"] = tab["spatial_share"] / tab["ref_share"]
    tab = tab.sort_values("ref_median_total_counts")
    tab.to_csv(OUT / "depth_by_type.csv")
    ok = tab.dropna(subset=["ref_median_total_counts", "spatial_median_transcripts"])
    from scipy.stats import spearmanr

    rho, p = spearmanr(ok["ref_median_total_counts"], ok["spatial_median_transcripts"])
    print(
        "\nDEPTH MATCHING: reference library depth vs the depth of the "
        "spatial cells that type wins",
        flush=True,
    )
    print(tab.to_string(), flush=True)
    print(
        f"\n  Spearman rho = {rho:.3f} (p = {p:.2g}, n = {len(ok)} types)", flush=True
    )

    # composition of the atlas by spatial transcript-count decile
    dec = pd.qcut(tc, 10, labels=False, duplicates="drop")
    comp = pd.crosstab(dec, base, normalize="index")
    comp.to_csv(OUT / "depth_composition_by_decile.csv")
    overall = base.value_counts(normalize=True)
    enr = comp.div(overall, axis=1)
    enr.to_csv(OUT / "depth_enrichment_by_decile.csv")
    lo = enr.iloc[0].sort_values(ascending=False)
    print(
        "\n  enrichment in the LOWEST transcript-count decile (top 8 types):",
        flush=True,
    )
    print(("    " + lo.head(8).to_string().replace("\n", "\n    ")), flush=True)
    hi = enr.iloc[-1].sort_values(ascending=False)
    print(
        "\n  enrichment in the HIGHEST transcript-count decile (top 8 types):",
        flush=True,
    )
    print(("    " + hi.head(8).to_string().replace("\n", "\n    ")), flush=True)

    # mean posterior (not argmax) in the lowest decile -- what the mapping
    # falls back on when the panel carries almost no signal
    m = (dec == 0).to_numpy()
    post = pd.Series(A[m].mean(0), index=cols).sort_values(ascending=False)
    post_hi = pd.Series(A[(dec == dec.max()).to_numpy()].mean(0), index=cols)
    pd.DataFrame(
        {
            "mean_posterior_lowest_decile": post,
            "mean_posterior_highest_decile": post_hi,
            "ref_share": tab["ref_share"],
        }
    ).to_csv(OUT / "depth_mean_posterior_by_decile.csv")
    print(
        "\n  MEAN POSTERIOR in the lowest decile (top 8) -- the mapping's "
        "fallback when the panel is uninformative:",
        flush=True,
    )
    print(("    " + post.head(8).to_string().replace("\n", "\n    ")), flush=True)

    # per-region transcript counts, to connect depth to the anatomy story
    reg = region_of(obs)
    rr = pd.DataFrame(
        {
            "median_transcripts": tc.groupby(reg).median(),
            "median_n_genes": ng.groupby(reg).median(),
            "n_cells": reg.value_counts(),
        }
    )
    rr.to_csv(OUT / "depth_by_region.csv")
    print("\n  spatial depth by compartment:", flush=True)
    print(rr.to_string(), flush=True)
    print(f"\nwrote {OUT}/depth_*.csv", flush=True)


# ═════════════════════════════ stage: panel ═════════════════════════════════


def stage_panel() -> None:
    """Can the 101-gene panel SEE each reference type at all?

    For every one of the 29 reference types, the one-vs-rest AUROC of every
    panel gene. A type whose best panel gene barely clears chance has no
    handle for Tangram to grab; its posterior is then decided by whatever the
    other types' genes say it is NOT, which is exactly how a type becomes a
    default winner in weak-signal tissue. Reported next to the same statistic
    on the full transcriptome, so "the panel cannot see it" is separable from
    "nothing can see it".
    """
    X, obs, var = load_ref()
    ct = obs["cell_type"].astype(str)
    genes = pd.Index(var.index.astype(str))
    pg = panel_genes(genes)
    Xp = X[:, [genes.get_loc(g) for g in pg]]
    desc = gene_descriptions()

    types = sorted(set(ct))
    aucs = _auroc_multi(Xp, {t: (ct.to_numpy() == t) for t in types}, block=100)
    rows, long = [], []
    for t in types:
        a = np.abs(aucs[t] - 0.5) + 0.5  # two-sided: markers OR anti-markers
        order = np.argsort(a)[::-1]
        rows.append(
            {
                "cell_type": t,
                "n_nuclei": int((ct == t).sum()),
                "panel_best_auroc": float(a[order[0]]),
                "panel_best_gene": pg[order[0]],
                "panel_best_gene_desc": desc.get(pg[order[0]], ""),
                "panel_top5_mean_auroc": float(a[order[:5]].mean()),
                "n_panel_genes_auroc_ge_070": int((a >= 0.70).sum()),
                "n_panel_genes_auroc_ge_080": int((a >= 0.80).sum()),
                "n_panel_genes_auroc_ge_090": int((a >= 0.90).sum()),
            }
        )
        for i in order[:10]:
            long.append(
                {
                    "cell_type": t,
                    "gene": pg[i],
                    "auroc_signed": float(aucs[t][i]),
                    "auroc_twosided": float(a[i]),
                    "description": desc.get(pg[i], ""),
                }
            )
    sep = pd.DataFrame(rows).sort_values("panel_best_auroc")
    sep.to_csv(OUT / "panel_separability.csv", index=False)
    pd.DataFrame(long).to_csv(OUT / "panel_top_genes_per_type.csv", index=False)
    print(
        "\nPANEL SEPARABILITY -- best one-vs-rest AUROC available on the "
        "100 panel genes, per reference type (ascending)",
        flush=True,
    )
    print(sep.to_string(index=False), flush=True)
    print(f"\nwrote {OUT}/panel_*.csv", flush=True)


# ═════════════════════════════ stage: summary ═══════════════════════════════


def stage_summary() -> None:
    """Put every per-type statistic on one row and ask what predicts the
    pathology -- is Small-type an outlier, or one of a family?

    Reads the CSVs the other stages wrote; no heavy I/O.
    """
    from scipy.stats import spearmanr

    depth = pd.read_csv(OUT / "depth_by_type.csv", index_col=0)
    sep = pd.read_csv(OUT / "panel_separability.csv", index_col="cell_type")
    flat = pd.read_csv(OUT / "ref_panel_flatness.csv", index_col="cell_type")
    geom = pd.read_csv(OUT / "spatial_posterior_geometry.csv", index_col="cell_type")
    nb = pd.read_csv(OUT / "ref_nearest_types.csv")
    nb = nb[nb.gene_space == "hvg"].set_index("cell_type")

    t = pd.DataFrame(index=sorted(set(depth.index) | set(sep.index)))
    t["n_ref_nuclei"] = depth["n_nuclei"]
    t["ref_share"] = depth["ref_share"]
    t["spatial_share"] = depth["spatial_share"]
    t["share_ratio"] = depth["share_ratio_spatial_over_ref"]
    t["log2_share_ratio"] = np.log2(t["share_ratio"])
    t["panel_best_auroc"] = sep["panel_best_auroc"]
    t["panel_top5_mean_auroc"] = sep["panel_top5_mean_auroc"]
    t["cosine_to_panel_mean"] = flat["cosine_to_panel_mean"]
    t["rank_cosine_to_panel_mean"] = flat["rank_cosine_to_panel_mean"]
    t["median_top1_posterior"] = geom["median_top1"]
    t["median_margin"] = geom["median_margin"]
    t["mean_r_to_other_KC_hvg"] = nb["mean_r_to_other_KC"]
    t["mean_r_to_OL_hvg"] = nb["mean_r_to_OL"]
    t = t.sort_values("log2_share_ratio")
    t.to_csv(OUT / "summary_per_type.csv")
    print(
        "\nPER-TYPE SUMMARY (sorted by log2 spatial/reference share ratio)", flush=True
    )
    print(t.to_string(), flush=True)

    ok = t.dropna(subset=["log2_share_ratio", "n_ref_nuclei"])
    for x in [
        "n_ref_nuclei",
        "panel_best_auroc",
        "cosine_to_panel_mean",
        "median_top1_posterior",
    ]:
        s = ok.dropna(subset=[x])
        r, p = spearmanr(s[x], s["log2_share_ratio"])
        print(
            f"  Spearman(log2 share ratio, {x:<24s}) = {r:+.3f}  "
            f"p = {p:.3g}  n = {len(s)}",
            flush=True,
        )

    # where does the focal type sit on each statistic?
    print(
        "\n  RANK OF THE FOCAL TYPES (1 = most extreme in the pathological "
        "direction, out of 29):",
        flush=True,
    )
    ranks = pd.DataFrame(
        {
            "|log2 share ratio| (most distorted abundance)": t["log2_share_ratio"]
            .abs()
            .rank(ascending=False),
            "median top-1 posterior (least confident calls)": t[
                "median_top1_posterior"
            ].rank(ascending=True),
            "panel_best_auroc (least panel-separable)": t["panel_best_auroc"].rank(
                ascending=True
            ),
            "cosine to panel mean (most generic panel profile)": t[
                "cosine_to_panel_mean"
            ].rank(ascending=False),
        }
    )
    print(
        ranks.loc[
            [
                FOCAL,
                SUSPECT2,
                "Kenyon cells",
                "Large-type Kenyon cells",
                "Other glia",
                "OL_8",
            ]
        ].to_string(),
        flush=True,
    )
    ranks.to_csv(OUT / "summary_ranks.csv")
    print(f"\nwrote {OUT}/summary_*.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["ref", "spatial", "depth", "panel", "summary", "all"],
        default="all",
    )
    a = ap.parse_args()
    if a.stage in ("ref", "all"):
        stage_ref()
    if a.stage in ("spatial", "all"):
        stage_spatial()
    if a.stage in ("panel", "all"):
        stage_panel()
    if a.stage in ("depth", "all"):
        stage_depth()
    if a.stage in ("summary", "all"):
        stage_summary()


if __name__ == "__main__":
    main()
