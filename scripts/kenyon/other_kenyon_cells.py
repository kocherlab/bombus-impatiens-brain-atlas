#!/usr/bin/env python
"""Is "Other Kenyon cells" a distinct population or a catch-all bin?

Five central-brain Kenyon populations exist in the snRNA reference
(``data/sn/BIMP_atlas_qw_CentralBrain.h5ad``): Large-type, Middle-type,
Small-type, Class II and Other. They are projected onto every Kenyon-compartment cell
by the restricted reclassification
``scripts/kenyon/project_kc_populations.py``. This reads both and reports, for
Other against the four named types:

(a) transfer / similarity: Tangram probabilities per called type, and cosine similarity
    of each cell (nucleus) to the five reference centroids on the 100 panel genes;
    nearest-centroid and logistic-regression recovery of the reference nuclei, on the
    panel and on transcriptome-wide variable genes;
(b) markers: Wilcoxon one-vs-rest and Other-vs-each-named-type on all genes, positive
    markers counted with fixed thresholds, panel membership;
(c) spatial distribution: by compartment, by section, by depth (distance to the nearest
    KC-IC cell of the section), and local same-label enrichment (kNN) against the share
    expected from the compartment's composition;
(d) caste: spatial shares per caste (one brain each, descriptive) and snRNA proportions
    per library (three bees per caste; the libraries are listed because they are the n).

Writes ``results/round4/other_kc/``. Needs only the main environment::

    PYTHONPATH=src:scripts .venv/bin/python scripts/kenyon/other_kenyon_cells.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import SN_CB  # noqa: E402

OUT = REPO / "results/round4/other_kc"
# gzip without a header timestamp, so a rerun writes the same bytes
GZ = {"method": "gzip", "mtime": 0}
KC5H5 = REPO / "results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad"
TYPES = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
]
SHORT = dict(zip(TYPES, ["Large", "Middle", "Small", "ClassII", "Other"], strict=False))
S = list(SHORT.values())
OTHER = "Other"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def log(*a):
    print(*a, flush=True)


def reference() -> ad.AnnData:
    r = sc.read_h5ad(SN_CB)
    r.obs["cell_type"] = r.obs["cell_type"].astype(str)
    r = r[r.obs["cell_type"].isin(TYPES)].copy()
    assert r.n_obs == 32672, r.n_obs
    r.obs["t"] = r.obs["cell_type"].map(SHORT)
    r.layers["counts"] = r.layers["counts"].copy()
    r.X = r.layers["counts"].copy()
    sc.pp.normalize_total(r, target_sum=1e4)
    sc.pp.log1p(r)
    return r


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def cosine(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return a @ b.T


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_map = read_panel_crosswalk()
    panel = set(panel_map["v43_id"].dropna())
    r = reference()
    log("reference", r.shape, r.obs["t"].value_counts().to_dict())
    lo = {g.lower(): g for g in r.var_names}

    sp = ad.read_h5ad(KC5H5)
    sp.obs_names_make_unique()
    o = sp.obs
    genes = [g for g in sp.var_names if g.lower() in lo]
    rg = [lo[g.lower()] for g in genes]
    log("spatial", sp.shape, "panel genes shared", len(genes))

    pr = pd.DataFrame(np.asarray(sp.obsm["tangram_ct_pred"]), columns=S)
    raw_max = pr.max(axis=1).values
    pn = pr.div(pr.sum(axis=1), axis=0)
    call = pn.idxmax(axis=1).values
    maxp = pn.max(axis=1).values
    brain = o["brain"].astype(str).values
    sl = o["slice_name"].astype(str).values
    z = o["z_level"].to_numpy(float)
    comp = np.where(
        o["in_kc_ic"].to_numpy(bool),
        "KC-IC",
        np.where(o["in_kc_nc"].to_numpy(bool), "KC-NC", "KC-OC"),
    )
    xy = np.asarray(sp.obsm["spatial_aligned"], float)

    # ---------------------------------------------------------------- (a) similarity
    Xs = dense(sp[:, genes].X).astype(np.float32)
    Xr = dense(r[:, rg].X).astype(np.float32)
    zs = (Xs - Xs.mean(0)) / np.maximum(Xs.std(0), 1e-6)
    zr = (Xr - Xr.mean(0)) / np.maximum(Xr.std(0), 1e-6)
    tr = r.obs["t"].values
    cent = np.vstack([zr[tr == t].mean(0) for t in S])
    sim_s = cosine(zs, cent)
    sim_r = cosine(zr, cent)
    rows = []
    for t in S:
        m = call == t
        k = S.index(t)
        mr = tr == t
        rows.append(
            dict(
                type=t,
                n_spatial_called=int(m.sum()),
                share_spatial=m.mean(),
                n_reference=int(mr.sum()),
                share_reference=mr.mean(),
                mean_maxprob_norm=maxp[m].mean(),
                median_maxprob_norm=np.median(maxp[m]),
                frac_below_0p5=(maxp[m] < 0.5).mean(),
                mean_raw_tangram_max=raw_max[m].mean(),
                mean_p_this_type_all_cells=pn[t].mean(),
                spatial_mean_cos_to_own_centroid=sim_s[m, k].mean(),
                spatial_mean_max_cos=sim_s[m].max(1).mean(),
                spatial_frac_nearest_centroid_is_own=(sim_s[m].argmax(1) == k).mean(),
                ref_mean_cos_to_own_centroid=sim_r[mr, k].mean(),
                ref_mean_best_other_cos=np.delete(sim_r[mr], k, axis=1).max(1).mean(),
                ref_frac_nearest_centroid_is_own=(sim_r[mr].argmax(1) == k).mean(),
            )
        )
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "a_transfer_and_similarity_by_type.csv", index=False)
    log("\n(a) transfer / similarity\n", A.round(3).to_string(index=False))
    sec = pd.crosstab(
        pd.Series(call, name="called"),
        pd.Series(np.array(S)[sim_s.argmax(1)], name="nearest_centroid"),
        normalize="index",
    )
    sec.to_csv(OUT / "a_called_vs_nearest_centroid_spatial.csv")
    # where does the second-best probability of Other-called cells land
    p2 = pn.values.copy()
    p2[np.arange(len(p2)), pn.values.argmax(1)] = -1
    second = np.array(S)[p2.argmax(1)]
    pd.crosstab(
        pd.Series(call, name="called"),
        pd.Series(second, name="second_best"),
        normalize="index",
    ).to_csv(OUT / "a_second_best_by_called.csv")

    # classifier recovery of the reference nuclei: panel genes and transcriptome-wide
    # HVGs
    rr = r.copy()
    sc.pp.highly_variable_genes(
        rr, n_top_genes=2000, flavor="seurat_v3", layer="counts"
    )
    hv = rr.var_names[rr.var["highly_variable"].values]
    rec = []
    for tag, X in (
        ("panel_100", zr),
        ("hvg_2000", dense(r[:, hv].X).astype(np.float32)),
    ):
        Xz = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
        itr, ite = train_test_split(
            np.arange(len(tr)), test_size=0.3, stratify=tr, random_state=0
        )
        # one BLAS thread: lbfgs sums in a thread-count-dependent order otherwise
        with threadpool_limits(limits=1):
            clf = LogisticRegression(max_iter=2000, n_jobs=1).fit(Xz[itr], tr[itr])
            pred = clf.predict(Xz[ite])
        cm = confusion_matrix(tr[ite], pred, labels=S, normalize="true")
        pd.DataFrame(cm, index=S, columns=S).to_csv(
            OUT / f"a_reference_confusion_{tag}.csv"
        )
        rec.append(
            dict(
                features=tag,
                n_features=X.shape[1],
                balanced_acc=balanced_accuracy_score(tr[ite], pred),
                **{f"recall_{t}": cm[i, i] for i, t in enumerate(S)},
            )
        )
    R = pd.DataFrame(rec)
    R.to_csv(OUT / "a_reference_classifier_recall.csv", index=False)
    log("\n(a) reference nuclei, held-out recall\n", R.round(3).to_string(index=False))

    # ---------------------------------------------------------------- (b) markers
    sc.tl.rank_genes_groups(r, "t", method="wilcoxon", pts=True, tie_correct=True)
    mk = []
    for t in S:
        d = sc.get.rank_genes_groups_df(r, group=t)
        d["pct_in"] = d["names"].map(r.uns["rank_genes_groups"]["pts"][t])
        d["pct_rest"] = d["names"].map(r.uns["rank_genes_groups"]["pts_rest"][t])
        d["type"] = t
        mk.append(d)
    M = pd.concat(mk, ignore_index=True)
    M["on_panel"] = M["names"].isin(panel)
    M.to_csv(OUT / "b_markers_one_vs_rest_all.csv.gz", index=False, compression=GZ)
    pos = M[(M.logfoldchanges > 0.5) & (M.pvals_adj < 0.01) & (M.pct_in >= 0.2)]
    neg = M[(M.logfoldchanges < -0.5) & (M.pvals_adj < 0.01) & (M.pct_rest >= 0.2)]
    cnt = (
        pd.DataFrame(
            {
                "n_positive_markers": pos.groupby("type").size(),
                "n_negative_markers": neg.groupby("type").size(),
            }
        )
        .reindex(S)
        .fillna(0)
        .astype(int)
    )
    cnt["n_positive_on_panel"] = (
        pos[pos.on_panel].groupby("type").size().reindex(S).fillna(0).astype(int)
    )
    cnt.to_csv(OUT / "b_marker_counts_one_vs_rest.csv")
    log(
        "\n(b) one-vs-rest marker counts (logFC>0.5, padj<0.01, expressed in >=20% of "
        "the group)\n",
        cnt.to_string(),
    )
    topO = (
        M[M.type == OTHER]
        .sort_values("scores", ascending=False)
        .head(25)[
            ["names", "logfoldchanges", "pvals_adj", "pct_in", "pct_rest", "on_panel"]
        ]
    )
    desc = (
        r.var["description"]
        if "description" in r.var
        else pd.Series("", index=r.var_names)
    )
    topO["description"] = [str(desc.get(g, ""))[:50] for g in topO.names]
    topO.to_csv(OUT / "b_top25_markers_Other_vs_rest.csv", index=False)
    log("\ntop Other-vs-rest markers\n", topO.round(3).to_string(index=False))
    pair = []
    for t in S[:4]:
        sub = r[r.obs["t"].isin([t, OTHER])].copy()
        sc.tl.rank_genes_groups(
            sub, "t", groups=[OTHER], reference=t, method="wilcoxon", pts=True
        )
        d = sc.get.rank_genes_groups_df(sub, group=OTHER)
        d["pct_other"] = d["names"].map(sub.uns["rank_genes_groups"]["pts"][OTHER])
        d["pct_ref"] = (
            d["names"].map(sub.uns["rank_genes_groups"]["pts_rest"][OTHER])
            if "pts_rest" in sub.uns["rank_genes_groups"]
            else np.nan
        )
        d["vs"] = t
        d["on_panel"] = d["names"].isin(panel)
        pair.append(d)
    P = pd.concat(pair, ignore_index=True)
    P.to_csv(OUT / "b_markers_Other_vs_each_named.csv.gz", index=False, compression=GZ)
    pc = (
        P[(P.logfoldchanges > 0.5) & (P.pvals_adj < 0.01) & (P.pct_other >= 0.2)]
        .groupby("vs")
        .size()
    )
    pn_ = P[(P.logfoldchanges < -0.5) & (P.pvals_adj < 0.01)].groupby("vs").size()
    pcnt = (
        pd.DataFrame({"Other_up": pc, "Other_down": pn_})
        .reindex(S[:4])
        .fillna(0)
        .astype(int)
    )
    pcnt.to_csv(OUT / "b_marker_counts_Other_vs_each_named.csv")
    log("\nOther vs each named type: genes up/down\n", pcnt.to_string())
    # named types against each other, for scale
    named_pairs = []
    for i, t1 in enumerate(S[:4]):
        for t2 in S[i + 1 : 4]:
            sub = r[r.obs["t"].isin([t1, t2])].copy()
            sc.tl.rank_genes_groups(
                sub, "t", groups=[t1], reference=t2, method="wilcoxon", pts=True
            )
            d = sc.get.rank_genes_groups_df(sub, group=t1)
            d["pct1"] = d["names"].map(sub.uns["rank_genes_groups"]["pts"][t1])
            named_pairs.append(
                dict(
                    pair=f"{t1}|{t2}",
                    up=int(
                        (
                            (d.logfoldchanges > 0.5)
                            & (d.pvals_adj < 0.01)
                            & (d.pct1 >= 0.2)
                        ).sum()
                    ),
                    down=int(((d.logfoldchanges < -0.5) & (d.pvals_adj < 0.01)).sum()),
                )
            )
    pd.DataFrame(named_pairs).to_csv(
        OUT / "b_marker_counts_named_vs_named.csv", index=False
    )
    log("\nnamed vs named\n", pd.DataFrame(named_pairs).to_string(index=False))

    # -------------------------------------------------------- (c) spatial distribution
    df = pd.DataFrame(dict(sl=sl, brain=brain, z=z, comp=comp, call=call))
    ct = pd.crosstab(df.comp, df.call, normalize="index")[S] * 100
    ct.to_csv(OUT / "c_share_by_compartment.csv")
    log("\n(c) share of calls by compartment (%)\n", ct.round(1).to_string())
    ctb = pd.crosstab([df.brain, df.comp], df.call, normalize="index")[S] * 100
    ctb.to_csv(OUT / "c_share_by_compartment_and_caste.csv")
    per = (
        df.groupby(["brain", "sl", "z", "comp"])
        .call.value_counts()
        .unstack(fill_value=0)
        .reindex(columns=S, fill_value=0)
    )
    per["n"] = per.sum(1)
    for t in S:
        per[f"frac_{t}"] = per[t] / per["n"]
    per.reset_index().to_csv(
        OUT / "c_share_by_section_and_compartment.csv", index=False
    )
    srows = []
    for compn, g in per.reset_index().groupby("comp"):
        g = g[g.n >= 50]
        for t in S:
            f = g[f"frac_{t}"]
            srows.append(
                dict(
                    comp=compn,
                    type=t,
                    n_sections=len(g),
                    median=f.median(),
                    q25=f.quantile(0.25),
                    q75=f.quantile(0.75),
                    min=f.min(),
                    max=f.max(),
                    cv=f.std() / max(f.mean(), 1e-9),
                    spearman_z_rho=stats.spearmanr(g.z, f)[0] if len(g) > 4 else np.nan,
                    spearman_z_p=stats.spearmanr(g.z, f)[1] if len(g) > 4 else np.nan,
                )
            )
    SS = pd.DataFrame(srows)
    SS.to_csv(OUT / "c_section_summary.csv", index=False)
    log(
        "\nper-section share summary (sections with >=50 cells in the compartment)\n",
        SS[SS.type.isin([OTHER, "Large"])].round(3).to_string(index=False),
    )

    # depth: distance to the nearest KC-IC cell of the same section, in aligned-frame px
    depth = np.full(len(df), np.nan)
    enr = []
    for (b, s_), idx in df.groupby(["brain", "sl"]).indices.items():
        ic = idx[comp[idx] == "KC-IC"]
        out_ic = idx[comp[idx] != "KC-IC"]
        if len(ic) >= 20 and len(out_ic) >= 20:
            # NC / OC cells: distance to the nearest KC-IC cell. IC cells sit at 0 from
            # their own set, so they get the distance to the nearest non-IC cell instead
            # (how far inside).
            nc_oc = comp[idx] != "KC-IC"
            depth[idx[nc_oc]] = cKDTree(xy[ic]).query(xy[idx[nc_oc]], k=1)[0]
            depth[idx[~nc_oc]] = cKDTree(xy[out_ic]).query(xy[idx[~nc_oc]], k=1)[0]
        for cn in ("KC-IC", "KC-NC", "KC-OC"):
            j = idx[comp[idx] == cn]
            if len(j) < 100:
                continue
            tree = cKDTree(xy[j])
            nb = tree.query(xy[j], k=16)[1][:, 1:]
            lab = call[j]
            for t in S:
                share = (lab == t).mean()
                if share < 0.01:
                    continue
                m = lab == t
                obs_same = (lab[nb[m]] == t).mean()
                enr.append(
                    dict(
                        brain=b,
                        slice=s_,
                        comp=cn,
                        type=t,
                        share=share,
                        same_label_neighbours=obs_same,
                        enrichment=obs_same / share,
                        n=len(j),
                    )
                )
    E = pd.DataFrame(enr)
    E.to_csv(OUT / "c_knn_same_label_enrichment_by_section.csv", index=False)
    Es = (
        E.groupby(["comp", "type"])
        .agg(
            sections=("slice", "size"),
            mean_share=("share", "mean"),
            median_same_label_nb=("same_label_neighbours", "median"),
            median_enrichment=("enrichment", "median"),
        )
        .reset_index()
    )
    Es.to_csv(OUT / "c_knn_enrichment_summary.csv", index=False)
    log(
        "\nlocal same-label enrichment (15 nearest same-compartment cells; 1 = no "
        "clustering beyond composition)\n",
        Es.round(3).to_string(index=False),
    )
    df["depth_to_IC"] = depth
    dd = []
    for compn, g in df.dropna(subset=["depth_to_IC"]).groupby("comp"):
        # quartiles of the distance (KC-NC / KC-OC: to the nearest KC-IC cell; KC-IC: to
        # the nearest non-IC cell), ranked first so ties cannot make the bin edges
        # collide
        g = g.assign(
            q=pd.qcut(
                g.depth_to_IC.rank(method="first"),
                4,
                labels=["Q1 (smallest distance)", "Q2", "Q3", "Q4 (largest)"],
            )
        )
        t = pd.crosstab(g.q, g.call, normalize="index")[S] * 100
        t.insert(0, "comp", compn)
        dd.append(t.reset_index())
    pd.concat(dd).to_csv(OUT / "c_share_by_depth_quartile.csv", index=False)
    log(
        "\nshare of calls by distance quartile from the nearest KC-IC cell (%)\n",
        pd.concat(dd).round(1).to_string(index=False),
    )

    # ---------------------------------------------------------------- (d) caste
    sp_caste = pd.crosstab(df.brain, df.call, normalize="index")[S] * 100
    sp_caste.to_csv(OUT / "d_spatial_share_by_caste.csv")
    log(
        "\n(d) spatial share by caste (%, descriptive: one brain per caste)\n",
        sp_caste.round(2).to_string(),
    )
    secs = per.reset_index()
    qs = secs[secs.brain == "queen"].groupby("sl")[S].sum()
    ws = secs[secs.brain == "worker"].groupby("sl")[S].sum()
    fq = qs[OTHER] / qs.sum(1)
    fw = ws[OTHER] / ws.sum(1)
    mw = stats.mannwhitneyu(fq, fw)
    log(
        f"section-level Other share: queen median {fq.median():.4f} (n={len(fq)}), "
        f"worker median {fw.median():.4f} (n={len(fw)}), "
        f"Mann-Whitney P {mw.pvalue:.3g} (sections of one brain per caste: descriptive)"
    )
    lib = pd.crosstab(r.obs["sample"], r.obs["t"])[S]
    lib["caste"] = r.obs.groupby("sample")["caste"].first().astype(str)
    lib["trial"] = r.obs.groupby("sample")["trial"].first().astype(str)
    lib["n_kc"] = lib[S].sum(1)
    lib["frac_Other"] = lib[OTHER] / lib["n_kc"]
    lib.to_csv(OUT / "d_snrna_libraries_kc_composition.csv")
    log("\n(d) snRNA libraries\n", lib.round(4).to_string())
    q = lib[lib.caste.str.lower().str.startswith("q")].sort_values("trial")
    w = lib[lib.caste.str.lower().str.startswith("w")].sort_values("trial")
    if len(q) == len(w) and len(q) > 2:
        wt = stats.ttest_ind(q.frac_Other, w.frac_Other, equal_var=False)
        pt = stats.ttest_rel(q.frac_Other, w.frac_Other)
        log(
            f"snRNA Other fraction of KC nuclei: queen {q.frac_Other.mean():.4f} vs "
            f"worker {w.frac_Other.mean():.4f}; "
            f"Welch P {wt.pvalue:.3g}; trial-paired t P {pt.pvalue:.3g} (n=3 vs 3)"
        )
    log("\ndone")


if __name__ == "__main__":
    main()
