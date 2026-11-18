#!/usr/bin/env python3
"""Why is Tangram's per-cell abundance distorted, and is it fixable without a
GPU retrain?

Extends ``scripts/identity/small_kc_attractor_diagnostic.py``
(results/smallkc_diagnosis), which measures how far the per-cell argmax composition is
from the reference (TVD, against the cluster vote's) and whether reference cluster
SIZE correlates with the distortion. This script asks WHY, and whether
a posterior-only correction reproduces the vote.

Stages (``--stage``):

  columns   IDENTITY CHECK. ``tangram_mapping.py`` names the 29 posterior
            columns by mapping integer positions of ``ad_map.obs_names`` through
            ``ref_cats.cat.categories`` (alphabetical), but tangram2's
            ``adata_to_cluster_expression`` orders its rows by
            ``value_counts().index`` (descending frequency). Those two orders
            are only equal if cluster sizes are already alphabetically
            descending. This stage tests, from the data alone, whether the
            column names correspond to the reference profiles they claim to.

  law       IS IT SIZE? Regress log2(spatial share) on log2(reference share) --
            the non-tautological form of the question (a correlation of the
            spatial/reference ratio with ref_share has ref_share in the
            denominator of its own dependent variable). Slope 1 = abundance
            recovered; slope 0 = the
            output marginal is uninformed by reference abundance. Then regress
            the RESIDUAL around the flat-prior expectation on the candidate
            confounds (panel separability, near-mean-ness, dispersion,
            expression level, post-subsample n).

  correct   IS IT FIXABLE? Re-weight each posterior column by pi_t^alpha (the
            reference prior; alpha swept 0 -> 1.5) and re-argmax. Reports the
            composition distance to the reference (CIRCULAR -- the correction
            injects that marginal, so this is the objective, not evidence), the
            territorial enrichment against its baseline (NOT circular), the
            Mixed fraction after the anatomy-guided vote, and the fraction of
            cells whose label changes.

  mixed     DOES IT EXPLAIN MIXED? Are Mixed clusters disproportionately made
            of cells whose argmax went to an inflated class, against the
            baseline of each class's atlas-wide argmax share? And does the
            prior correction move the Mixed fraction?

Run:
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/identity/prior_distortion_diagnostic.py \
        --stage columns
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bimp_st.cluster_labels import (  # noqa: E402
    _CALYX_FLAGS,
    _OPTIC_FLAGS,
    POOLED_ORDER,
    get_pooled,
)
from bimp_st.paths import CANONICAL_TANGRAM, SN_COMBINED  # noqa: E402

warnings.filterwarnings("ignore")

TANGRAM_H5AD = CANONICAL_TANGRAM
OUT = REPO / "results/prior_distortion"
OUT.mkdir(parents=True, exist_ok=True)

MIXED = "Mixed"
CLUSTER_COL = "leiden"
REGION_THRESH = 0.5
MIN_PURITY = 0.50
CALYX, OPTIC = "calyx", "optic"
EXPECTED_TERRITORY = {
    "Kenyon cells": CALYX,
    "Optic lobe": OPTIC,
    "Photoreceptors": OPTIC,
}
MIN_REGION_CELLS = 200
# tangram_mapping.py: per_cluster = max(50, max_ref_cells // n_clusters)
MAX_REF_CELLS = 15000


# ═══════════════════════════════ shared I/O ════════════════════════════════


def load_spatial(with_X: bool = False):
    print(f"loading {TANGRAM_H5AD.name} ...", flush=True)
    with h5py.File(TANGRAM_H5AD, "r") as h:
        obs = read_elem(h["obs"])
        pred = read_elem(h["obsm/tangram_ct_pred"]).astype(np.float32)
        var = read_elem(h["var"])
        X = read_elem(h["X"]) if with_X else None
    print(f"  {len(obs):,} cells x {pred.shape[1]} posterior columns", flush=True)
    return obs, pred, var, X


def anatomy_masks(obs: pd.DataFrame):
    def _mask(flags):
        m = np.zeros(len(obs), bool)
        for f in flags:
            if f in obs.columns:
                m |= obs[f].astype(bool).to_numpy()
        return m

    return _mask(_CALYX_FLAGS), _mask(_OPTIC_FLAGS)


def ref_sizes() -> pd.Series:
    with h5py.File(SN_COMBINED, "r") as h:
        ct = pd.Series(read_elem(h["obs/cell_type"]).astype(str))
    return ct.value_counts()


def ref_pooled_composition() -> pd.Series:
    with h5py.File(SN_COMBINED, "r") as h:
        ct = pd.Series(read_elem(h["obs/cell_type"]).astype(str))
    return (
        ct.map(get_pooled)
        .value_counts(normalize=True)
        .reindex(POOLED_ORDER)
        .fillna(0.0)
    )


def tvd(a: pd.Series, b: pd.Series) -> float:
    a = a.reindex(POOLED_ORDER).fillna(0.0)
    b = b.reindex(POOLED_ORDER).fillna(0.0)
    return float(0.5 * np.abs(a - b).sum())


# ═══════════════════════════ stage: columns ════════════════════════════════


def predicted_permutation(
    sizes: pd.Series, categories: list[str], cap: int
) -> pd.DataFrame:
    """What tangram_mapping.py's code-to-name map would do, simulated.

    ``adata_to_cluster_expression`` orders its rows by ``value_counts().index``
    of the (subsampled) reference; ``tangram_mapping.py`` names position j with
    ``categories[j]``. Returns the implied true-vs-labelled table.
    """
    capped = {c: int(min(sizes.get(c, 0), cap)) for c in categories}
    vals = []
    for c in categories:
        vals += [c] * capped[c]
    s = pd.Series(pd.Categorical(vals, categories=categories))
    true_order = list(s.value_counts().index)
    return pd.DataFrame(
        {
            "position": range(len(categories)),
            "true_type_predicted": true_order,
            "labelled_as": categories,
            "n_after_cap": [capped[c] for c in true_order],
            "mismatch": [
                t != lab for t, lab in zip(true_order, categories, strict=False)
            ],
        }
    )


def stage_columns() -> None:
    """Do the posterior columns carry the reference profiles they are named for?

    Decisive test: the cells that win column j should express the reference
    profile of whatever type actually occupies row j of the mapping matrix.
    Correlate the mean panel profile of each column's argmax cells against
    every reference type's mean panel profile, and read off the assignment.
    """
    obs, pred, var, X = load_spatial(with_X=True)
    cols = list(pred.columns)
    lab = pd.Series(np.array(cols)[pred.to_numpy().argmax(1)], index=obs.index)

    # spatial gene ids are lower-cased copies of the v4.3 ids
    sp_genes = pd.Index(var.index.astype(str))
    with h5py.File(SN_COMBINED, "r") as h:
        rX = read_elem(h["X"])
        rct = pd.Series(read_elem(h["obs/cell_type"]).astype(str))
        rvar = read_elem(h["var"])
    ref_genes = pd.Index(rvar.index.astype(str))
    lut = {g.lower(): g for g in ref_genes}
    shared_sp = [g for g in sp_genes if g.lower() in lut]
    shared_ref = [lut[g.lower()] for g in shared_sp]
    print(
        f"  {len(shared_sp)} genes shared between spatial h5ad and reference",
        flush=True,
    )

    si = [sp_genes.get_loc(g) for g in shared_sp]
    ri = [ref_genes.get_loc(g) for g in shared_ref]

    # spatial: mean profile of the cells argmaxed to each column
    Xs = np.asarray(X[:, si], dtype=np.float64)
    sp_prof = {}
    for c in cols:
        m = (lab == c).to_numpy()
        sp_prof[c] = Xs[m].mean(0) if m.sum() else np.full(len(si), np.nan)
    SP = pd.DataFrame(sp_prof, index=shared_sp).T  # columns x genes
    del Xs

    # reference: mean profile per cell type on the same genes (log1p space,
    # matching the spatial h5ad's X)
    Xr = rX[:, ri]
    Xr = np.asarray(Xr.todense()) if hasattr(Xr, "todense") else np.asarray(Xr)
    RP = pd.DataFrame(
        np.vstack([Xr[(rct == t).to_numpy()].mean(0) for t in sorted(set(rct))]),
        index=sorted(set(rct)),
        columns=shared_sp,
    )

    # z-score genes before correlating so shared depth structure does not
    # swamp identity structure
    def _z(D):
        return (D - D.mean(0)) / (D.std(0) + 1e-9)

    C = pd.DataFrame(
        np.corrcoef(_z(SP).to_numpy(), _z(RP).to_numpy())[: len(SP), len(SP) :],
        index=SP.index,
        columns=RP.index,
    )
    C.to_csv(OUT / "column_identity_corr.csv")

    # Hungarian assignment: the best 1-1 matching of columns to reference types
    from scipy.optimize import linear_sum_assignment

    r, c_ = linear_sum_assignment(-C.to_numpy())
    assign = {C.index[i]: C.columns[j] for i, j in zip(r, c_, strict=False)}

    sizes = ref_sizes()
    cats = sorted(sizes.index)
    perm = predicted_permutation(sizes, cats, MAX_REF_CELLS // len(cats))
    perm_map = dict(zip(perm["labelled_as"], perm["true_type_predicted"], strict=True))
    perm.to_csv(OUT / "column_permutation_predicted.csv", index=False)

    rows = []
    for cl in cols:
        best = C.loc[cl].idxmax()
        rows.append(
            {
                "column_label": cl,
                "n_spatial_cells": int((lab == cl).sum()),
                "best_corr_type": best,
                "best_corr": float(C.loc[cl, best]),
                "corr_to_own_name": float(C.loc[cl, cl]) if cl in C.columns else np.nan,
                "hungarian_type": assign.get(cl),
                "corr_hungarian": float(C.loc[cl, assign[cl]]),
                "code_order_prediction": perm_map.get(cl),
                "corr_to_code_order_prediction": float(C.loc[cl, perm_map[cl]])
                if perm_map.get(cl) in C.columns
                else np.nan,
            }
        )
    res = pd.DataFrame(rows)
    res["hungarian_eq_name"] = res["hungarian_type"] == res["column_label"]
    res["hungarian_eq_code_pred"] = (
        res["hungarian_type"] == res["code_order_prediction"]
    )
    res.to_csv(OUT / "column_identity.csv", index=False)

    print(
        "\nCOLUMN IDENTITY -- does each posterior column carry the reference "
        "profile it is named for?",
        flush=True,
    )
    print(res.to_string(index=False), flush=True)

    n = len(res)
    print(
        f"\n  identity assignment (Hungarian == column name): "
        f"{int(res.hungarian_eq_name.sum())}/{n}",
        flush=True,
    )
    print(
        f"  matches the code-order permutation hypothesis:   "
        f"{int(res.hungarian_eq_code_pred.sum())}/{n}",
        flush=True,
    )
    print(
        f"  mean corr to own name          : {res.corr_to_own_name.mean():+.3f}",
        flush=True,
    )
    print(
        f"  mean corr to Hungarian partner : {res.corr_hungarian.mean():+.3f}",
        flush=True,
    )
    print(
        f"  mean corr to code-order pred   : "
        f"{res.corr_to_code_order_prediction.mean():+.3f}",
        flush=True,
    )
    # trace comparison: identity vs the permutation hypothesis
    tr_id = float(np.nansum([C.loc[c, c] for c in cols if c in C.columns]))
    tr_pm = float(
        np.nansum([C.loc[c, perm_map[c]] for c in cols if perm_map.get(c) in C.columns])
    )
    tr_hu = float(np.nansum([C.loc[i, assign[i]] for i in C.index]))
    print(f"\n  total correlation under IDENTITY naming    : {tr_id:.3f}", flush=True)
    print(f"  total correlation under CODE-ORDER naming  : {tr_pm:.3f}", flush=True)
    print(f"  total correlation under HUNGARIAN optimum  : {tr_hu:.3f}", flush=True)
    json.dump(
        {
            "n_columns": n,
            "hungarian_eq_name": int(res.hungarian_eq_name.sum()),
            "hungarian_eq_code_pred": int(res.hungarian_eq_code_pred.sum()),
            "trace_identity": tr_id,
            "trace_code_order": tr_pm,
            "trace_hungarian": tr_hu,
        },
        open(OUT / "column_identity_summary.json", "w"),
        indent=2,
    )
    print(f"\nwrote {OUT}/column_*.csv", flush=True)


# ═══════════════════════════ shared: vote + sanity ═════════════════════════


def vote(
    labels: pd.Series,
    cluster: pd.Series,
    calyx: np.ndarray,
    optic: np.ndarray,
    min_purity: float = MIN_PURITY,
):
    """anatomy_guided_vote's decision rule; returns per-cell label + provenance."""
    pooled = labels.map(get_pooled)
    df = pd.DataFrame(
        {
            "cl": cluster.to_numpy(),
            "lab": pooled.to_numpy(),
            "calyx": calyx,
            "optic": optic,
        }
    )
    out, prov, purity = {}, {}, {}
    for c, sub in df.groupby("cl", sort=False):
        vc = sub["lab"].value_counts(normalize=True)
        purity[c] = float(vc.iloc[0])
        if sub["optic"].mean() > REGION_THRESH:
            out[c], prov[c] = "Optic lobe", "anatomy"
        elif sub["calyx"].mean() > REGION_THRESH:
            out[c], prov[c] = "Kenyon cells", "anatomy"
        elif vc.iloc[0] >= min_purity:
            out[c], prov[c] = vc.index[0], "expression"
        else:
            out[c], prov[c] = MIXED, "none"
    return (cluster.map(out), cluster.map(prov), pd.Series(purity))


def territorial_table(
    voted: pd.Series, calyx: np.ndarray, optic: np.ndarray, provenance: pd.Series
) -> pd.DataFrame:
    """Enrichment of each voted label in its expected territory, vs the
    atlas-wide baseline share of region-assigned cells in that territory.

    An absolute fraction would just reproduce how much calyx vs optic tissue
    was hand-drawn, so the baseline is mandatory. Labels assigned BY the
    anatomy rule are flagged circular and excluded from the verdict.
    """
    reg = calyx | optic
    base = {
        CALYX: float(calyx.sum()) / reg.sum(),
        OPTIC: float(optic.sum()) / reg.sum(),
    }
    rows = []
    for lab in sorted(set(voted)):
        m = (voted == lab).to_numpy()
        n_reg = int(reg[m].sum())
        n_cal, n_opt = int(calyx[m].sum()), int(optic[m].sum())
        exp = EXPECTED_TERRITORY.get(lab)
        circ = bool((provenance[m] == "anatomy").mean() > 0.5)
        row = {
            "label": lab,
            "n_cells": int(m.sum()),
            "n_region_assigned": n_reg,
            "n_in_calyx": n_cal,
            "n_in_optic": n_opt,
            "baseline_calyx": base[CALYX],
            "baseline_optic": base[OPTIC],
            "expected_territory": exp,
            "mostly_assigned_by_anatomy_rule": circ,
        }
        if exp is None:
            row.update(
                {
                    "verdict": "NOT_TESTABLE",
                    "enrichment": np.nan,
                    "frac_in_expected": np.nan,
                    "frac_in_contradicting": np.nan,
                }
            )
        else:
            n_e, n_b = (n_cal, n_opt) if exp == CALYX else (n_opt, n_cal)
            f_e = n_e / n_reg if n_reg else np.nan
            f_b = n_b / n_reg if n_reg else np.nan
            enr = f_e / base[exp] if n_reg else np.nan
            if n_reg < MIN_REGION_CELLS:
                v = "UNDER_POWERED"
            elif circ:
                v = "CIRCULAR"
            elif enr > 1.0 and f_e > f_b:
                v = "PASS"
            else:
                v = "FAIL"
            row.update(
                {
                    "verdict": v,
                    "enrichment": enr,
                    "frac_in_expected": f_e,
                    "frac_in_contradicting": f_b,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def percell_territorial(
    labels: pd.Series, calyx: np.ndarray, optic: np.ndarray
) -> pd.DataFrame:
    """Same enrichment test on the RAW per-cell fine labels (no vote, so no
    anatomy-rule circularity at all). Fine types are mapped to the pooled
    vocabulary the territorial expectation is defined on."""
    pooled = labels.map(get_pooled)
    reg = calyx | optic
    base = {
        CALYX: float(calyx.sum()) / reg.sum(),
        OPTIC: float(optic.sum()) / reg.sum(),
    }
    rows = []
    for lab, exp in EXPECTED_TERRITORY.items():
        m = (pooled == lab).to_numpy()
        n_reg = int(reg[m].sum())
        n_cal, n_opt = int(calyx[m].sum()), int(optic[m].sum())
        n_e, n_b = (n_cal, n_opt) if exp == CALYX else (n_opt, n_cal)
        f_e = n_e / n_reg if n_reg else np.nan
        rows.append(
            {
                "label": lab,
                "expected_territory": exp,
                "n_cells": int(m.sum()),
                "n_region_assigned": n_reg,
                "frac_in_expected": f_e,
                "frac_in_contradicting": n_b / n_reg if n_reg else np.nan,
                "baseline": base[exp],
                "enrichment": f_e / base[exp] if n_reg else np.nan,
                "under_powered": n_reg < MIN_REGION_CELLS,
            }
        )
    return pd.DataFrame(rows)


def relabel_map(mode: str) -> dict[str, str] | None:
    """Column renaming to apply, or None for the stored naming."""
    if mode == "asis":
        return None
    p = pd.read_csv(OUT / "column_identity.csv")
    if mode == "hungarian":
        return dict(zip(p["column_label"], p["hungarian_type"], strict=True))
    if mode == "codeorder":
        return dict(zip(p["column_label"], p["code_order_prediction"], strict=True))
    raise ValueError(mode)


# ═════════════════════════════ stage: law ══════════════════════════════════


def stage_law(naming: str = "asis") -> None:
    obs, pred, _, _ = load_spatial()
    cols = list(pred.columns)
    A = pred.to_numpy()

    # degenerate rows: tangram_mapping.py replaces zero row-sums with 1.0, so
    # all-zero posteriors survive and argmax silently returns column 0.
    zero_rows = int((A.sum(1) == 0).sum())
    flat_rows = int((A.max(1) - A.min(1) < 1e-9).sum())
    print(
        f"\n  degenerate posterior rows: all-zero {zero_rows:,}, "
        f"flat {flat_rows:,} of {len(A):,}",
        flush=True,
    )

    lab = pd.Series(np.array(cols)[A.argmax(1)], index=obs.index)
    rm = relabel_map(naming)
    if rm:
        lab = lab.map(rm)
    sizes = ref_sizes()
    ref_share = sizes / sizes.sum()
    sp_share = lab.value_counts(normalize=True)

    cats = sorted(sizes.index)
    perm = predicted_permutation(sizes, cats, MAX_REF_CELLS // len(cats))
    n_cap = dict(zip(perm["true_type_predicted"], perm["n_after_cap"], strict=True))

    flat = pd.read_csv(
        REPO / "results/smallkc_diagnosis/ref_panel_flatness.csv", index_col="cell_type"
    )
    sep = pd.read_csv(
        REPO / "results/smallkc_diagnosis/panel_separability.csv", index_col="cell_type"
    )

    t = pd.DataFrame(index=cats)
    t["n_ref_nuclei"] = sizes
    t["n_after_subsample_cap"] = pd.Series(n_cap)
    t["ref_share"] = ref_share
    t["spatial_share"] = sp_share.reindex(cats).fillna(0.0)
    t["log2_ref_share"] = np.log2(t["ref_share"])
    t["log2_spatial_share"] = np.log2(t["spatial_share"].clip(lower=1e-7))
    t["log2_share_ratio"] = t["log2_spatial_share"] - t["log2_ref_share"]
    # residual around the flat-prior expectation: every type gets 1/29
    t["flat_prior_residual"] = t["log2_spatial_share"] - np.log2(1 / len(cats))
    t["panel_best_auroc"] = sep["panel_best_auroc"]
    t["panel_top5_mean_auroc"] = sep["panel_top5_mean_auroc"]
    t["cosine_to_panel_mean"] = flat["cosine_to_panel_mean"]
    t["panel_profile_sd"] = flat["panel_profile_sd"]
    t["panel_mean_expr"] = flat["panel_mean_expr"]
    t["top5_share"] = flat["top5_share"]
    t.to_csv(OUT / f"law_per_type_{naming}.csv")

    from scipy.stats import spearmanr

    def ols(y, X, names):
        X = np.column_stack([np.ones(len(y))] + list(X))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        s2 = resid @ resid / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        from scipy.stats import t as tdist

        tv = beta / se
        p = 2 * tdist.sf(np.abs(tv), dof)
        r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
        return pd.DataFrame(
            {"term": ["intercept"] + names, "beta": beta, "se": se, "t": tv, "p": p}
        ), r2

    print(
        "\nLAW 1. log2(spatial share) ~ log2(reference share) "
        "[slope 1 = abundance recovered; 0 = no prior]",
        flush=True,
    )
    tab, r2 = ols(
        t["log2_spatial_share"].to_numpy(),
        [t["log2_ref_share"].to_numpy()],
        ["log2_ref_share"],
    )
    print(tab.to_string(index=False), flush=True)
    print(f"  R2 = {r2:.3f}   n = {len(t)}", flush=True)
    b = tab.loc[1, "beta"]
    se = tab.loc[1, "se"]
    print(
        f"  slope 95% CI: [{b - 1.96 * se:+.3f}, {b + 1.96 * se:+.3f}]  "
        f"-- excludes 1 (abundance recovered)? "
        f"{'YES' if b + 1.96 * se < 1 else 'no'};  "
        f"excludes 0 (no prior)? "
        f"{'YES' if (b - 1.96 * se > 0 or b + 1.96 * se < 0) else 'no'}",
        flush=True,
    )
    tab.to_csv(OUT / f"law_slope_{naming}.csv", index=False)

    rho, p = spearmanr(t["n_ref_nuclei"], t["log2_share_ratio"])
    rho2, p2 = spearmanr(t["n_ref_nuclei"], t["log2_spatial_share"])
    print(
        f"\n  Spearman(n_ref_nuclei, log2 SHARE RATIO)   = {rho:+.3f} "
        f"p={p:.3g}   <- the share-ratio form (ref share in its own denominator)",
        flush=True,
    )
    print(
        f"  Spearman(n_ref_nuclei, log2 SPATIAL share) = {rho2:+.3f} "
        f"p={p2:.3g}   <- the non-tautological form",
        flush=True,
    )

    print(
        "\nLAW 2. what predicts the residual around the flat-prior expectation?",
        flush=True,
    )
    preds = [
        "log2_ref_share",
        "n_after_subsample_cap",
        "panel_best_auroc",
        "cosine_to_panel_mean",
        "panel_profile_sd",
        "panel_mean_expr",
        "top5_share",
    ]
    uni = []
    for c in preds:
        s = t.dropna(subset=[c, "flat_prior_residual"])
        r, pp = spearmanr(s[c], s["flat_prior_residual"])
        uni.append({"predictor": c, "spearman_rho": r, "p": pp, "n": len(s)})
    uni = pd.DataFrame(uni).sort_values("p")
    print(uni.to_string(index=False), flush=True)
    uni.to_csv(OUT / f"law_univariate_{naming}.csv", index=False)

    s = t.dropna(subset=preds + ["flat_prior_residual"])
    Z = [(s[c] - s[c].mean()) / s[c].std() for c in preds]
    tab2, r22 = ols(s["flat_prior_residual"].to_numpy(), Z, preds)
    print(
        f"\n  multiple regression (standardized predictors), "
        f"R2 = {r22:.3f}, n = {len(s)}",
        flush=True,
    )
    print(tab2.to_string(index=False), flush=True)
    tab2.to_csv(OUT / f"law_multiple_{naming}.csv", index=False)

    print("\n  per-type table (sorted by log2 share ratio):", flush=True)
    print(t.sort_values("log2_share_ratio").to_string(), flush=True)
    print(f"\nwrote {OUT}/law_*_{naming}.csv", flush=True)


# ═══════════════════════════ stage: correct ════════════════════════════════


def stage_correct(naming: str = "asis") -> None:
    obs, pred, _, _ = load_spatial()
    cols = list(pred.columns)
    A = pred.to_numpy()
    calyx, optic = anatomy_masks(obs)
    cluster = obs[CLUSTER_COL].astype(str)
    rm = relabel_map(naming)
    names = np.array([rm[c] for c in cols]) if rm else np.array(cols)

    sizes = ref_sizes()
    pi = np.array([sizes.get(n, 0) for n in names], float)
    pi = pi / pi.sum()
    ref_pool = ref_pooled_composition()

    base_lab = pd.Series(names[A.argmax(1)], index=obs.index)
    base_v, base_prov, _ = vote(base_lab, cluster, calyx, optic)

    rows, sanity_frames, percell_frames = [], [], []
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    for alpha in alphas:
        w = pi**alpha
        lab = pd.Series(names[(A * w).argmax(1)], index=obs.index)
        pooled = lab.map(get_pooled).value_counts(normalize=True)
        v, prov, _ = vote(lab, cluster, calyx, optic)
        vc = v.value_counts()
        pm = 100 * vc.get(MIXED, 0) / len(v)
        called = vc.drop(labels=[MIXED], errors="ignore")
        st = territorial_table(v, calyx, optic, prov)
        st["alpha"] = alpha
        sanity_frames.append(st)
        pc = percell_territorial(lab, calyx, optic)
        pc["alpha"] = alpha
        percell_frames.append(pc)
        testable = st[st.verdict.isin(["PASS", "FAIL"])]
        rows.append(
            {
                "alpha": alpha,
                "tvd_percell_vs_ref": tvd(pooled, ref_pool),
                "pct_mixed_after_vote": pm,
                "tvd_voted_vs_ref": tvd(called / called.sum(), ref_pool),
                "frac_cells_label_changed": float((lab != base_lab).mean()),
                "frac_voted_label_changed": float((v != base_v).mean()),
                "n_fine_types_used": int(lab.nunique()),
                "percell_KC_calyx_enrichment": float(
                    pc.loc[pc.label == "Kenyon cells", "enrichment"].iloc[0]
                ),
                "percell_OL_optic_enrichment": float(
                    pc.loc[pc.label == "Optic lobe", "enrichment"].iloc[0]
                ),
                "percell_PR_optic_enrichment": float(
                    pc.loc[pc.label == "Photoreceptors", "enrichment"].iloc[0]
                ),
                "n_voted_labels_PASS": int((st.verdict == "PASS").sum()),
                "n_voted_labels_FAIL": int((st.verdict == "FAIL").sum()),
                "mean_testable_enrichment": float(testable["enrichment"].mean())
                if len(testable)
                else np.nan,
            }
        )
        print(
            f"  alpha={alpha:<5} TVD/cell {rows[-1]['tvd_percell_vs_ref']:.4f} "
            f" Mixed {pm:5.1f}%  TVD/voted {rows[-1]['tvd_voted_vs_ref']:.4f} "
            f" changed {100 * rows[-1]['frac_cells_label_changed']:5.1f}%  "
            f"KC-calyx enr {rows[-1]['percell_KC_calyx_enrichment']:.2f}  "
            f"OL-optic enr {rows[-1]['percell_OL_optic_enrichment']:.2f}",
            flush=True,
        )

    # Sinkhorn: rescale columns until the SOFT marginal matches the reference
    q = A / A.sum(1, keepdims=True)
    w = np.ones(len(cols))
    for _ in range(200):
        m = q * w
        m = m / m.sum(1, keepdims=True)
        cur = m.mean(0)
        w = w * (pi / np.maximum(cur, 1e-12)) ** 0.5
    lab = pd.Series(names[(A * w).argmax(1)], index=obs.index)
    pooled = lab.map(get_pooled).value_counts(normalize=True)
    v, prov, _ = vote(lab, cluster, calyx, optic)
    vc = v.value_counts()
    called = vc.drop(labels=[MIXED], errors="ignore")
    st = territorial_table(v, calyx, optic, prov)
    st["alpha"] = "sinkhorn"
    sanity_frames.append(st)
    pc = percell_territorial(lab, calyx, optic)
    pc["alpha"] = "sinkhorn"
    percell_frames.append(pc)
    testable = st[st.verdict.isin(["PASS", "FAIL"])]
    rows.append(
        {
            "alpha": "sinkhorn",
            "tvd_percell_vs_ref": tvd(pooled, ref_pool),
            "pct_mixed_after_vote": 100 * vc.get(MIXED, 0) / len(v),
            "tvd_voted_vs_ref": tvd(called / called.sum(), ref_pool),
            "frac_cells_label_changed": float((lab != base_lab).mean()),
            "frac_voted_label_changed": float((v != base_v).mean()),
            "n_fine_types_used": int(lab.nunique()),
            "percell_KC_calyx_enrichment": float(
                pc.loc[pc.label == "Kenyon cells", "enrichment"].iloc[0]
            ),
            "percell_OL_optic_enrichment": float(
                pc.loc[pc.label == "Optic lobe", "enrichment"].iloc[0]
            ),
            "percell_PR_optic_enrichment": float(
                pc.loc[pc.label == "Photoreceptors", "enrichment"].iloc[0]
            ),
            "n_voted_labels_PASS": int((st.verdict == "PASS").sum()),
            "n_voted_labels_FAIL": int((st.verdict == "FAIL").sum()),
            "mean_testable_enrichment": float(testable["enrichment"].mean())
            if len(testable)
            else np.nan,
        }
    )
    np.save(OUT / f"sinkhorn_weights_{naming}.npy", w)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"correct_sweep_{naming}.csv", index=False)
    pd.concat(sanity_frames).to_csv(
        OUT / f"correct_territorial_voted_{naming}.csv", index=False
    )
    pd.concat(percell_frames).to_csv(
        OUT / f"correct_territorial_percell_{naming}.csv", index=False
    )
    print("\nPRIOR-CORRECTION SWEEP", flush=True)
    print(res.to_string(index=False), flush=True)
    print(
        "\n  per-cell territorial enrichment (no vote, so no anatomy-rule "
        "circularity):",
        flush=True,
    )
    print(pd.concat(percell_frames).to_string(index=False), flush=True)
    print(f"\nwrote {OUT}/correct_*_{naming}.csv", flush=True)


# ═══════════════════════════ stage: mixed ══════════════════════════════════


def stage_mixed(naming: str = "asis") -> None:
    """Does the flat-prior distortion predict WHICH cells end up Mixed?

    Prediction, stated before the test: if the flat prior drives Mixed,
    prior-correcting concentrates mass on high-pi classes, raises within-cluster
    top-1 purity, and lowers the Mixed fraction below its as-is value. Baseline
    for the "Mixed clusters are made of inflated classes" test is each class's
    ATLAS-WIDE argmax share -- the inflated classes are large in the spatial
    atlas by construction, so they dominate Mixed at chance.
    """
    obs, pred, _, _ = load_spatial()
    cols = list(pred.columns)
    A = pred.to_numpy()
    calyx, optic = anatomy_masks(obs)
    cluster = obs[CLUSTER_COL].astype(str)
    rm = relabel_map(naming)
    names = np.array([rm[c] for c in cols]) if rm else np.array(cols)

    sizes = ref_sizes()
    pi = np.array([sizes.get(n, 0) for n in names], float)
    pi = pi / pi.sum()
    ref_share = pd.Series(pi, index=names)

    lab = pd.Series(names[A.argmax(1)], index=obs.index)
    v, prov, purity = vote(lab, cluster, calyx, optic)
    is_mixed = (v == MIXED).to_numpy()
    print(f"  baseline Mixed: {100 * is_mixed.mean():.2f}% of cells", flush=True)

    atlas_share = lab.value_counts(normalize=True)
    infl = np.log2(atlas_share / ref_share.reindex(atlas_share.index))

    # per fine type: share inside Mixed vs its atlas-wide share (the baseline)
    inside = lab[is_mixed].value_counts(normalize=True)
    tab = pd.DataFrame(
        {
            "atlas_share": atlas_share,
            "share_within_mixed": inside.reindex(atlas_share.index).fillna(0.0),
            "ref_share": ref_share.reindex(atlas_share.index),
            "log2_inflation": infl,
        }
    )
    tab["mixed_enrichment_vs_atlas_baseline"] = (
        tab["share_within_mixed"] / tab["atlas_share"]
    )
    tab["frac_of_this_type_that_is_mixed"] = [
        float(is_mixed[(lab == t).to_numpy()].mean()) for t in tab.index
    ]
    tab = tab.sort_values("log2_inflation")
    tab.to_csv(OUT / f"mixed_by_inflation_{naming}.csv")
    from scipy.stats import spearmanr

    r, p = spearmanr(
        tab["log2_inflation"],
        np.log2(tab["mixed_enrichment_vs_atlas_baseline"].clip(lower=1e-6)),
    )
    print(
        "\nIS MIXED MADE OF INFLATED CLASSES? (baseline = each class's "
        "atlas-wide argmax share)",
        flush=True,
    )
    print(tab.to_string(), flush=True)
    print(
        f"\n  Spearman(log2 inflation, log2 Mixed enrichment) = {r:+.3f} "
        f"p = {p:.3g}, n = {len(tab)}",
        flush=True,
    )

    # cluster level, with cluster size as covariate
    cl = pd.DataFrame(
        {"cl": cluster.to_numpy(), "lab": lab.to_numpy(), "mixed": is_mixed}
    )
    rows = []
    for c, sub in cl.groupby("cl", sort=False):
        w = sub["lab"].map(infl)
        rows.append(
            {
                "cluster": c,
                "n_cells": len(sub),
                "is_mixed": bool(sub["mixed"].iloc[0]),
                "mean_log2_inflation": float(w.mean()),
                "frac_cells_in_inflated_class": float((w > 0).mean()),
                "top_purity": float(purity[c]),
            }
        )
    cd = pd.DataFrame(rows)
    cd.to_csv(OUT / f"mixed_cluster_inflation_{naming}.csv", index=False)
    a = cd[cd.is_mixed]
    b = cd[~cd.is_mixed]
    from scipy.stats import mannwhitneyu

    u, pu = mannwhitneyu(a["mean_log2_inflation"], b["mean_log2_inflation"])
    print(
        f"\n  cluster-level: mean log2 inflation of member cells, "
        f"Mixed clusters {a['mean_log2_inflation'].mean():+.3f} "
        f"(n={len(a)}) vs non-Mixed {b['mean_log2_inflation'].mean():+.3f} "
        f"(n={len(b)}); Mann-Whitney p = {pu:.3g}",
        flush=True,
    )
    rs, ps = spearmanr(cd["mean_log2_inflation"], cd["top_purity"])
    rsz, psz = spearmanr(cd["n_cells"], cd["top_purity"])
    print(
        f"  Spearman(mean log2 inflation, cluster top-1 purity) = {rs:+.3f} p={ps:.3g}",
        flush=True,
    )
    print(
        f"  Spearman(cluster size,        cluster top-1 purity) = "
        f"{rsz:+.3f} p={psz:.3g}  <- the size covariate",
        flush=True,
    )

    # does correcting the prior move Mixed?
    out = []
    for alpha in [0.0, 0.5, 1.0, 1.5]:
        w = pi**alpha
        l2 = pd.Series(names[(A * w).argmax(1)], index=obs.index)
        v2, _, pur2 = vote(l2, cluster, calyx, optic)
        out.append(
            {
                "alpha": alpha,
                "pct_mixed": 100 * float((v2 == MIXED).mean()),
                "median_cluster_purity": float(pur2.median()),
                "mean_cluster_purity": float(pur2.mean()),
            }
        )
    od = pd.DataFrame(out)
    od.to_csv(OUT / f"mixed_vs_alpha_{naming}.csv", index=False)
    print("\n  Mixed fraction under prior correction:", flush=True)
    print(od.to_string(index=False), flush=True)
    print(f"\nwrote {OUT}/mixed_*_{naming}.csv", flush=True)


def stage_territory() -> None:
    """Per-POSTERIOR-COLUMN territorial enrichment, stored name next to the
    identity the profile test recovered.

    This is the independent (non-composition) check on the column-naming
    finding: if the recovered identities are right, the columns that turn out
    to be optic types must be optic-enriched and the ones that turn out to be
    Kenyon types must be calyx-enriched, regardless of what they are called.
    Enrichment is against the baseline share of region-assigned cells in each
    territory, never a raw fraction.
    """
    obs, pred, _, _ = load_spatial()
    cols = list(pred.columns)
    A = pred.to_numpy()
    m = relabel_map("hungarian")
    lab = pd.Series(np.array(cols)[A.argmax(1)])
    calyx, optic = anatomy_masks(obs)
    reg = calyx | optic
    b_cal = float(calyx.sum()) / reg.sum()
    b_opt = float(optic.sum()) / reg.sum()
    rows = []
    for c in cols:
        s = (lab == c).to_numpy()
        nr = int(reg[s].sum())
        rows.append(
            {
                "shipped_name": c,
                "recovered_identity": m[c],
                "n_cells": int(s.sum()),
                "n_region_assigned": nr,
                "frac_calyx": calyx[s].sum() / nr if nr else np.nan,
                "frac_optic": optic[s].sum() / nr if nr else np.nan,
                "calyx_enrichment": (calyx[s].sum() / nr) / b_cal if nr else np.nan,
                "optic_enrichment": (optic[s].sum() / nr) / b_opt if nr else np.nan,
                "under_powered": nr < MIN_REGION_CELLS,
            }
        )
    d = pd.DataFrame(rows)
    d["name_changed"] = d.shipped_name != d.recovered_identity
    d.to_csv(OUT / "percell_territory_by_column.csv", index=False)
    print(
        f"\nbaselines among region-assigned cells: calyx {b_cal:.3f}, "
        f"optic {b_opt:.3f}",
        flush=True,
    )
    print(
        d.sort_values("optic_enrichment", ascending=False).to_string(index=False),
        flush=True,
    )


def stage_other(h5ad: str, ref: str, label: str, tag: str) -> None:
    """Run the column-identity test on ANY tangram output.

    Same logic as ``--stage columns`` but parameterised, because every other
    Tangram call in the repo (stitched, region-restricted, ltKC, CCC) is a
    separate run with its own cluster count and therefore its own potential
    permutation. The mapping recovered for the combined run does NOT transfer.
    """
    hp, rp = Path(h5ad), Path(ref)
    print(f"loading {hp.name} ...", flush=True)
    with h5py.File(hp, "r") as h:
        pred = read_elem(h["obsm/tangram_ct_pred"]).astype(np.float32)
        var = read_elem(h["var"])
        X = read_elem(h["X"])
    cols = [str(c) for c in pred.columns]
    lab = pd.Series(np.array(cols)[pred.to_numpy().argmax(1)])
    print(f"  {len(lab):,} cells x {len(cols)} columns", flush=True)

    sp_genes = pd.Index(var.index.astype(str))
    with h5py.File(rp, "r") as h:
        rX = read_elem(h["X"])
        rct = pd.Series(read_elem(h[f"obs/{label}"])).astype(str)
        rvar = read_elem(h["var"])
    ref_genes = pd.Index(rvar.index.astype(str))
    lut = {g.lower(): g for g in ref_genes}
    shared_sp = [g for g in sp_genes if g.lower() in lut]
    shared_ref = [lut[g.lower()] for g in shared_sp]
    print(f"  {len(shared_sp)} shared genes", flush=True)
    si = [sp_genes.get_loc(g) for g in shared_sp]
    ri = [ref_genes.get_loc(g) for g in shared_ref]

    Xs = np.asarray(X[:, si], dtype=np.float64)
    SP = pd.DataFrame(
        {c: Xs[(lab == c).to_numpy()].mean(0) for c in cols}, index=shared_sp
    ).T
    del Xs
    Xr = rX[:, ri]
    Xr = np.asarray(Xr.todense()) if hasattr(Xr, "todense") else np.asarray(Xr)
    types = sorted(set(rct))
    RP = pd.DataFrame(
        np.vstack([Xr[(rct == t).to_numpy()].mean(0) for t in types]),
        index=types,
        columns=shared_sp,
    )

    def _z(D):
        return (D - D.mean(0)) / (D.std(0) + 1e-9)

    C = pd.DataFrame(
        np.corrcoef(_z(SP).to_numpy(), _z(RP).to_numpy())[: len(SP), len(SP) :],
        index=SP.index,
        columns=RP.index,
    )
    from scipy.optimize import linear_sum_assignment

    r, c_ = linear_sum_assignment(-C.to_numpy())
    assign = {C.index[i]: C.columns[j] for i, j in zip(r, c_, strict=False)}
    res = pd.DataFrame(
        [
            {
                "column_label": c,
                "n_spatial_cells": int((lab == c).sum()),
                "corr_to_own_name": float(C.loc[c, c]) if c in C.columns else np.nan,
                "hungarian_type": assign[c],
                "corr_hungarian": float(C.loc[c, assign[c]]),
                "hungarian_eq_name": assign[c] == c,
            }
            for c in cols
        ]
    )
    res.to_csv(OUT / f"column_identity_{tag}.csv", index=False)
    C.to_csv(OUT / f"column_identity_corr_{tag}.csv")
    print(res.to_string(index=False), flush=True)
    tr_id = float(np.nansum([C.loc[c, c] for c in cols if c in C.columns]))
    tr_hu = float(np.nansum([C.loc[i, assign[i]] for i in C.index]))
    print(
        f"\n  {tag}: Hungarian == column name on "
        f"{int(res.hungarian_eq_name.sum())}/{len(res)} columns",
        flush=True,
    )
    print(
        f"  total correlation, IDENTITY naming {tr_id:.3f} vs "
        f"HUNGARIAN optimum {tr_hu:.3f}",
        flush=True,
    )
    print(
        f"  VERDICT: {'CLEAN' if bool(res.hungarian_eq_name.all()) else 'PERMUTED'}",
        flush=True,
    )


def stage_concordance() -> None:
    """Recompute the per-cell-type cross-modal concordance under both namings.

    ``scripts/atlas/si_panel_concordance.py`` correlates, per canonical cell
    type, the snRNA mean log1p-CPM profile against the Xenium one across the
    100 panel genes (results/reframe/panel_concordance/per_cell_type.csv).
    The Xenium side is grouped by ``tangram_cell_type``, so
    for a misnamed column the two sides are different cell types and the rho is
    meaningless. Here the Xenium per-type profile is rebuilt as the
    cell-count-weighted mean of its member POSTERIOR COLUMNS, which lets the
    same statistic be recomputed under the corrected naming.

    The stored per_cell_type.csv values are reproduced first as a fidelity
    check; if the
    ``asis`` column does not match per_cell_type.csv, do not trust the
    ``corrected`` column either.
    """
    from scipy.stats import spearmanr

    from bimp_st.cluster_labels import get_canonical

    obs, pred, var, X = load_spatial(with_X=True)
    cols = list(pred.columns)
    lab = pd.Series(np.array(cols)[pred.to_numpy().argmax(1)])
    sp_genes = pd.Index(var.index.astype(str))

    A = pd.read_csv(
        REPO / "results/reframe/panel_concordance/sn_mean_by_type.csv", index_col=0
    )
    shipped = pd.read_csv(REPO / "results/reframe/panel_concordance/per_cell_type.csv")
    lut = {g.lower(): g for g in A.index.astype(str)}
    keep_sp = [g for g in sp_genes if g.lower() in lut]
    genes_ref = [lut[g.lower()] for g in keep_sp]
    print(f"  {len(keep_sp)} panel genes shared with sn_mean_by_type", flush=True)
    si = [sp_genes.get_loc(g) for g in keep_sp]
    Xs = np.asarray(X[:, si], dtype=np.float64)
    n_by_col, prof_by_col = {}, {}
    for c in cols:
        m = (lab == c).to_numpy()
        n_by_col[c] = int(m.sum())
        prof_by_col[c] = Xs[m].mean(0) if m.sum() else np.zeros(len(si))
    del Xs

    rm = relabel_map("hungarian")
    rows = []
    for naming in ["asis", "corrected"]:
        groups: dict[str, list[str]] = {}
        for c in cols:
            true = c if naming == "asis" else rm[c]
            groups.setdefault(get_canonical(true), []).append(c)
        for ct, members in groups.items():
            n = sum(n_by_col[c] for c in members)
            if n < 100 or ct not in A.columns:
                continue
            prof = sum(prof_by_col[c] * n_by_col[c] for c in members) / n
            r, p = spearmanr(A.loc[genes_ref, ct].to_numpy(), prof)
            rows.append(
                {
                    "naming": naming,
                    "cell_type": ct,
                    "n_cells": n,
                    "spearman": r,
                    "p": p,
                    "n_genes": len(genes_ref),
                    "member_columns": "|".join(sorted(members)),
                }
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "concordance_per_cell_type.csv", index=False)
    piv = d.pivot(index="cell_type", columns="naming", values=["spearman", "n_cells"])
    piv[("spearman", "SHIPPED_FILE")] = shipped.set_index("cell_type")["spearman"]
    print(
        "\nPER-CELL-TYPE CROSS-MODAL CONCORDANCE (Spearman across the panel genes)",
        flush=True,
    )
    print(
        piv.round(3).sort_values(("spearman", "asis"), ascending=False).to_string(),
        flush=True,
    )
    for naming in ["asis", "corrected"]:
        s = d[d.naming == naming]["spearman"]
        print(
            f"  {naming:<10s} n = {len(s)}  median {s.median():.3f}  "
            f"range {s.min():.3f}-{s.max():.3f}",
            flush=True,
        )
    sh = shipped["spearman"]
    print(
        f"  SHIPPED    n = {len(sh)}  median {sh.median():.3f}  "
        f"range {sh.min():.3f}-{sh.max():.3f}",
        flush=True,
    )
    print(f"\nwrote {OUT}/concordance_per_cell_type.csv", flush=True)


def stage_arbitrate() -> None:
    """Which naming is right, judged WITHOUT the reference marginal.

    TVD-to-reference is a bad arbiter here: the snRNA reference's optic share
    is set by how much optic lobe the dissections happened to yield, not by
    brain anatomy, so forcing the spatial marginal to match it fits a
    dissection artefact. Three tests that do not use the reference marginal:

      1. ANATOMICAL OPTIC SHARE of the spatial atlas, from the hand-drawn
         compartments -- what the optic fraction should be.
      2. RANDOM-PERMUTATION NULL for the voted TVD, so whether the vote's TVD
         is luck is a measurement rather than a claim.
      3. TERRITORIAL ENRICHMENT of every pooled label under both namings,
         each against its baseline share of region-assigned cells.
    """
    obs, pred, _, _ = load_spatial()
    cols = list(pred.columns)
    A = pred.to_numpy()
    calyx, optic = anatomy_masks(obs)
    cluster = obs[CLUSTER_COL].astype(str)
    ref = ref_pooled_composition()
    idx = A.argmax(1)

    # ── 1. how much of the atlas is anatomically optic? ──────────────────────
    reg = calyx | optic
    lam = np.zeros(len(obs), bool)
    for f in _OPTIC_FLAGS:
        if f in obs.columns:
            lam |= obs[f].astype(bool).to_numpy()
    print("\n1. ANATOMICAL OPTIC SHARE of the spatial atlas", flush=True)
    print(
        f"   cells in a hand-drawn compartment : {int(reg.sum()):,} "
        f"({100 * reg.mean():.1f}% of the atlas)",
        flush=True,
    )
    print(
        f"   of those, optic                   : {int(optic.sum()):,} "
        f"({100 * optic.sum() / reg.sum():.1f}%)",
        flush=True,
    )
    print(
        f"   of those, calyx                   : {int(calyx.sum()):,} "
        f"({100 * calyx.sum() / reg.sum():.1f}%)",
        flush=True,
    )
    print(
        f"   optic as a share of the WHOLE atlas (lower bound, unannotated "
        f"tissue counted as non-optic): {100 * optic.mean():.1f}%",
        flush=True,
    )
    per_flag = {}
    for f in _OPTIC_FLAGS + _CALYX_FLAGS:
        if f in obs.columns:
            per_flag[f] = int(obs[f].astype(bool).sum())
    print(f"   per-compartment cell counts: {per_flag}", flush=True)
    print(
        f"   snRNA reference optic share (the marginal TVD is measured "
        f"against): {ref['Optic lobe']:.3f}",
        flush=True,
    )
    pd.Series(per_flag).to_csv(OUT / "arbitrate_compartment_counts.csv")

    # ── 2. permutation null for the voted TVD ────────────────────────────────
    print("\n2. RANDOM-PERMUTATION NULL for the voted TVD", flush=True)
    rng = np.random.default_rng(0)
    obs_tvd = {}
    for naming in ["asis", "hungarian"]:
        rm = relabel_map(naming)
        nm = np.array([rm[c] for c in cols]) if rm else np.array(cols)
        lab = pd.Series(nm[idx], index=obs.index)
        v, _, _ = vote(lab, cluster, calyx, optic)
        vc = v.value_counts().drop(labels=[MIXED], errors="ignore")
        obs_tvd[naming] = tvd(vc / vc.sum(), ref)
    null = []
    for _ in range(200):
        nm = np.array(cols)[rng.permutation(len(cols))]
        lab = pd.Series(nm[idx], index=obs.index)
        v, _, _ = vote(lab, cluster, calyx, optic)
        vc = v.value_counts().drop(labels=[MIXED], errors="ignore")
        null.append(tvd(vc / vc.sum(), ref))
    null = np.array(null)
    pd.Series(null).to_csv(OUT / "arbitrate_tvd_null.csv", index=False)
    for k, val in obs_tvd.items():
        pct = float((null <= val).mean())
        print(
            f"   {k:<10s} voted TVD {val:.4f}   percentile of the random-"
            f"permutation null: {100 * pct:.1f}%",
            flush=True,
        )
    print(
        f"   null: min {null.min():.4f}  q05 {np.quantile(null, 0.05):.4f}  "
        f"median {np.median(null):.4f}  q95 {np.quantile(null, 0.95):.4f}  "
        f"max {null.max():.4f}  (n = {len(null)} permutations)",
        flush=True,
    )

    # ── 3. territorial enrichment of every pooled label, both namings ────────
    print(
        "\n3. TERRITORIAL ENRICHMENT per pooled label (per-cell, no vote, so "
        "no anatomy-rule circularity)",
        flush=True,
    )
    b_cal = float(calyx.sum()) / reg.sum()
    b_opt = float(optic.sum()) / reg.sum()
    rows = []
    for naming in ["asis", "hungarian"]:
        rm = relabel_map(naming)
        nm = np.array([rm[c] for c in cols]) if rm else np.array(cols)
        pooled = pd.Series(nm[idx], index=obs.index).map(get_pooled)
        for lab in POOLED_ORDER:
            m = (pooled == lab).to_numpy()
            nr = int(reg[m].sum())
            rows.append(
                {
                    "naming": naming,
                    "pooled_label": lab,
                    "n_cells": int(m.sum()),
                    "n_region_assigned": nr,
                    "calyx_enrichment": (calyx[m].sum() / nr) / b_cal if nr else np.nan,
                    "optic_enrichment": (optic[m].sum() / nr) / b_opt if nr else np.nan,
                    "under_powered": nr < MIN_REGION_CELLS,
                }
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "arbitrate_pooled_territory.csv", index=False)
    piv = d.pivot(
        index="pooled_label",
        columns="naming",
        values=["calyx_enrichment", "optic_enrichment", "n_cells"],
    )
    print(f"   baselines: calyx {b_cal:.3f}, optic {b_opt:.3f}", flush=True)
    print(piv.round(3).to_string(), flush=True)
    print(f"\nwrote {OUT}/arbitrate_*.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=[
            "columns",
            "law",
            "correct",
            "mixed",
            "territory",
            "arbitrate",
            "other",
            "concordance",
        ],
        default="columns",
    )
    ap.add_argument("--h5ad", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--label", default="cell_type")
    ap.add_argument("--tag", default="other")
    ap.add_argument(
        "--naming",
        choices=["asis", "hungarian", "codeorder"],
        default="asis",
        help="which column->cell-type naming to analyse under",
    )
    a = ap.parse_args()
    if a.stage == "columns":
        stage_columns()
    elif a.stage == "law":
        stage_law(a.naming)
    elif a.stage == "correct":
        stage_correct(a.naming)
    elif a.stage == "mixed":
        stage_mixed(a.naming)
    elif a.stage == "territory":
        stage_territory()
    elif a.stage == "arbitrate":
        stage_arbitrate()
    elif a.stage == "other":
        stage_other(a.h5ad, a.ref, a.label, a.tag)
    elif a.stage == "concordance":
        stage_concordance()


if __name__ == "__main__":
    main()
