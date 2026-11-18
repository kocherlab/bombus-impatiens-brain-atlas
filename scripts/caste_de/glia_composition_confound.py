#!/usr/bin/env python
"""Is the pooled-glia (and pooled-KC) caste DE in the n=3 vs n=3 snRNA reference
explained by queen/worker differences in SUBTYPE composition?

Reads existing pseudobulks only (no h5ad):
  paths.CASTE_DE_PSEUDOBULK_COARSE counts_{Glia,Kenyon_cells}.csv  (pooled input)
  paths.CASTE_DE_PSEUDOBULK_FINE counts_<fine type>.csv  (per-subtype pseudobulks)
  paths.CASTE_DE_NUCLEI
  paths.CASTE_DE_FINE  (per-subtype DE, same recipe)
  paths.CASTE_DE_COARSE  (the pooled DE)

Recipe: src/bimp_st/caste_de.py (the ~ trial + caste fit:
all-zero genes dropped, Cook's refit, Wald Queen vs Worker, + = queen-up,
unshrunk LFC, BH, FDR 0.05). Every fit is trial-blocked. Every subtype, Other
glia included, enters the composition standardization.

Output: results/caste_de/glia_composition/ (survival.csv and diag_Glia.csv
are the summary tables)

Run (pyDESeq2 lives in its own environment, not .venv)
    PYTHONPATH=src:scripts <pydeseq2-0.5.4 env>/bin/python \
        scripts/caste_de/glia_composition_confound.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bimp_st.caste_de import fit as recipe_fit
from bimp_st.caste_de import fn
from bimp_st.paths import (
    CASTE_DE_COARSE,
    CASTE_DE_DIR,  # noqa: E402
    CASTE_DE_FINE,
    CASTE_DE_NUCLEI,
    CASTE_DE_PSEUDOBULK_COARSE,
    CASTE_DE_PSEUDOBULK_FINE,
)

warnings.filterwarnings("ignore")
S = Path(__file__).resolve().parents[2]
OUT = CASTE_DE_DIR / "glia_composition"
OUT.mkdir(parents=True, exist_ok=True)
FIG3PB = CASTE_DE_PSEUDOBULK_COARSE
FINEPB = CASTE_DE_PSEUDOBULK_FINE
NUC = pd.read_csv(CASTE_DE_NUCLEI, index_col=0)
FINE_DE = CASTE_DE_FINE
PUB_DE = CASTE_DE_COARSE
SAMPLES = [f"BIMP_{c}_t{i}" for c in ("Queen", "Worker") for i in (1, 2, 3)]
FDR = 0.05
BASE = "pooled ~trial+caste"

CLASSES = {
    "Glia": dict(
        pooled="Glia",
        subtypes=[
            "Astrocyte-like glia",
            "Cortex glia",
            "Other glia",
            "Perineurial glia",
            "Subperineurial glia",
        ],
        tested=[
            "Astrocyte-like glia",
            "Cortex glia",
            "Other glia",
            "Perineurial glia",
            "Subperineurial glia",
        ],  # Other glia: tested on trials 2 and 3 (8 nuclei in Queen_t1)
    ),
    "Kenyon_cells": dict(
        pooled="Kenyon_cells",
        subtypes=[
            "Class II Kenyon cells",
            "Large-type Kenyon cells",
            "Middle-type Kenyon cells",
            "Other Kenyon cells",
            "Small-type Kenyon cells",
        ],
        tested=[
            "Class II Kenyon cells",
            "Large-type Kenyon cells",
            "Middle-type Kenyon cells",
            "Other Kenyon cells",
            "Small-type Kenyon cells",
        ],
    ),
}


def meta_base():
    m = pd.DataFrame(index=SAMPLES)
    m["caste"] = ["Queen"] * 3 + ["Worker"] * 3
    m["trial"] = [s.rsplit("_", 1)[1] for s in SAMPLES]
    return m


def fit(counts: pd.DataFrame, meta: pd.DataFrame, design: str) -> pd.DataFrame:
    """The caste_de recipe on all six samples (pooled classes keep every sample)."""
    r, _ = recipe_fit(counts.loc[SAMPLES], None, design, metadata=meta.loc[SAMPLES])
    return r


def sig(r):
    return r[(r.padj < FDR)]


def nsum(r):
    s = sig(r)
    return dict(
        n_tested=int(r.padj.notna().sum()),
        n_sig=len(s),
        queen_up=int((s.log2FoldChange > 0).sum()),
        worker_up=int((s.log2FoldChange < 0).sum()),
    )


def survive(base: pd.DataFrame, alt: pd.DataFrame) -> dict:
    b = sig(base)
    a = alt.reindex(b.index)
    same = np.sign(a.log2FoldChange) == np.sign(b.log2FoldChange)
    return dict(
        n_base=len(b),
        survive_padj_same_sign=int(((a.padj < FDR) & same).sum()),
        same_sign_any=int(same.sum()),
        median_lfc_ratio_alt_over_base=float(
            np.nanmedian(a.log2FoldChange / b.log2FoldChange)
        ),
    )


def subtype_overlap(pub: pd.DataFrame, fine: pd.DataFrame) -> pd.DataFrame:
    """Per-subtype DEG counts and their overlap with the pooled DEGs."""
    rows = []
    for cls, cfg in CLASSES.items():
        pooled = pub[pub.cell_type == cfg["pooled"]].set_index("gene")
        pooled_sig = sig(pooled)
        sets = {}
        for k in cfg["tested"]:
            f = fine[fine.cell_type == k].set_index("gene")
            s = sets[k] = sig(f)
            inter = s.index.intersection(pooled_sig.index)
            same = np.sign(s.loc[inter].log2FoldChange) == np.sign(
                pooled_sig.loc[inter].log2FoldChange
            )
            rows.append(
                dict(
                    cls=cls,
                    subtype=k,
                    **nsum(f),
                    in_pooled_set=len(inter),
                    in_pooled_same_sign=int(same.sum()),
                    frac_of_subtype_degs_in_pooled=round(
                        len(inter) / max(len(s), 1), 3
                    ),
                )
            )
        union = set().union(*(set(v.index) for v in sets.values()))
        rows += [
            dict(
                cls=cls,
                subtype="UNION of subtypes",
                n_sig=len(union),
                in_pooled_set=len(union & set(pooled_sig.index)),
            ),
            dict(
                cls=cls,
                subtype="POOLED (trial-blocked)",
                **nsum(pooled),
                in_pooled_set=len(pooled_sig),
                frac_of_subtype_degs_in_pooled=None,
            ),
            dict(
                cls=cls,
                subtype="POOLED DEGs DE in >=1 subtype (any sign)",
                n_sig=len(set(pooled_sig.index) & union),
            ),
        ]
    return pd.DataFrame(rows)


def main():
    pub = pd.read_csv(PUB_DE, index_col=0)
    fine = pd.read_csv(FINE_DE, index_col=0)
    meta = meta_base()
    summary_rows, surv_rows, prop_rows = [], [], []

    for cls, cfg in CLASSES.items():
        print(f"\n######## {cls}", flush=True)
        pooled = pd.read_csv(FIG3PB / f"counts_{cfg['pooled']}.csv", index_col=0).loc[
            SAMPLES
        ]
        sub = {
            k: pd.read_csv(FINEPB / f"counts_{fn(k)}.csv", index_col=0).loc[SAMPLES]
            for k in cfg["subtypes"]
        }
        # gate (b): subtypes sum to the pooled pseudobulk
        tot = sum(sub.values())
        diff = int((tot - pooled).abs().to_numpy().max())
        print("max |sum(subtypes) - pooled| =", diff, flush=True)
        assert diff == 0, "subtype pseudobulks do not sum to pooled"

        n = NUC.loc[cfg["subtypes"], SAMPLES].astype(float)
        props = n / n.sum(0)
        for k in cfg["subtypes"]:
            prop_rows.append(
                dict(
                    cls=cls,
                    subtype=k,
                    **props.loc[k].round(4).to_dict(),
                    queen_mean=props.loc[k, SAMPLES[:3]].mean(),
                    worker_mean=props.loc[k, SAMPLES[3:]].mean(),
                    no_overlap=bool(
                        props.loc[k, SAMPLES[:3]].min()
                        > props.loc[k, SAMPLES[3:]].max()
                        or props.loc[k, SAMPLES[:3]].max()
                        < props.loc[k, SAMPLES[3:]].min()
                    ),
                )
            )

        res = {}
        # gate (a): the trial-blocked pooled fit (CASTE_DE_COARSE) reproduced
        res[BASE] = fit(pooled, meta, "~ trial + caste")
        p = pub[pub.cell_type == cfg["pooled"]].set_index("gene")
        r = res[BASE]
        print("pooled refit:", nsum(r), "round9:", nsum(p), flush=True)
        assert nsum(r)["n_sig"] == nsum(p)["n_sig"], (
            "pooled refit does not reproduce caste_de"
        )

        # pooled restricted to tested subtypes (drops Other glia for glia)
        tested = cfg["tested"]
        pooled_t = sum(sub[k] for k in tested)
        res["pooled(tested subtypes) ~trial+caste"] = (
            fit(pooled_t, meta, "~ trial + caste")
            if set(tested) != set(cfg["subtypes"])
            else res[BASE]
        )

        # composition-standardized pseudobulk: each sample's own per-subtype
        # per-nucleus profile, recombined at the 6-sample mean composition
        nt = n.loc[tested]
        pt = nt / nt.sum(0)
        pbar = pt.mean(1)
        Ns = nt.sum(0)
        std = sum(pbar[k] * (sub[k].div(nt.loc[k], axis=0)) for k in tested).mul(
            Ns, axis=0
        )
        std = std.round().astype(int)
        std.to_csv(OUT / f"standardized_pseudobulk_{cls}.csv")
        res["standardized ~trial+caste"] = fit(std, meta, "~ trial + caste")

        for k, v in res.items():
            v.to_csv(
                OUT / f"de_{cls}_{fn(k.replace('~', 'by_').replace('+', '_'))}.csv"
            )
            summary_rows.append(dict(cls=cls, model=k, **nsum(v)))
            print(k, nsum(v), flush=True)

        base = res[BASE]
        for k in [
            "pooled(tested subtypes) ~trial+caste",
            "standardized ~trial+caste",
        ]:
            surv_rows.append(dict(cls=cls, base=BASE, alt=k, **survive(base, res[k])))

        # ---- composition-only predicted LFC and subtype-marker status ----
        mu = pd.DataFrame(
            {k: sub[k].sum(0) / n.loc[k].sum() for k in cfg["subtypes"]}
        )  # genes x subtype, per nucleus
        P = n.T.to_numpy() @ mu.T.to_numpy()  # samples x genes
        P = pd.DataFrame(P, index=SAMPLES, columns=mu.index)
        cpm = P.div(P.sum(1), axis=0) * 1e6
        pred = np.log2(cpm.loc[SAMPLES[:3]].mean() + 1e-3) - np.log2(
            cpm.loc[SAMPLES[3:]].mean() + 1e-3
        )
        mu_cpm = mu[tested].div(mu[tested].sum(0), axis=1) * 1e6
        lrange = np.log2(mu_cpm.max(1) + 1) - np.log2(mu_cpm.min(1) + 1)
        top_sub = mu_cpm.idxmax(1)

        d = base.copy()
        desc = pd.read_csv(FINEPB / "gene_desc.csv", index_col=0).iloc[:, 0]
        d["gene_name"] = desc.reindex(d.index).astype(str).str[:40]
        d["pred_comp_lfc"] = pred.reindex(d.index)
        d["subtype_log2_range"] = lrange.reindex(d.index)
        d["marker_gt4x"] = d.subtype_log2_range > 2
        d["top_subtype"] = top_sub.reindex(d.index)
        d["std_lfc"] = res["standardized ~trial+caste"].log2FoldChange.reindex(d.index)
        d["std_padj"] = res["standardized ~trial+caste"].padj.reindex(d.index)
        # per-subtype LFCs / padj (the fine-type DE, same recipe)
        for k in tested:
            f = fine[fine.cell_type == k].set_index("gene")
            d[f"lfc[{k}]"] = f.log2FoldChange.reindex(d.index)
            d[f"padj[{k}]"] = f.padj.reindex(d.index)
        d["lfc_in_top_subtype"] = [
            d.at[g, f"lfc[{t}]"] if isinstance(t, str) else np.nan
            for g, t in zip(d.index, d.top_subtype, strict=True)
        ]
        d.to_csv(OUT / f"pooled_{cls}_with_composition_diagnostics.csv")

        s = d[d.padj < FDR].copy()
        sg = np.sign(s.log2FoldChange)
        sub_sig_same = np.zeros(len(s), bool)
        n_sub_sig_same = np.zeros(len(s), int)
        for k in tested:
            same = np.sign(s[f"lfc[{k}]"]) == sg
            sub_sig_same |= (s[f"padj[{k}]"] < FDR) & same
            n_sub_sig_same += ((s[f"padj[{k}]"] < FDR) & same).astype(int)
        diag = dict(
            cls=cls,
            n_pooled_deg=len(s),
            de_same_sign_in_ge1_subtype=int(sub_sig_same.sum()),
            de_same_sign_in_ge2_subtypes=int((n_sub_sig_same >= 2).sum()),
            top_subtype_lfc_same_sign=int((np.sign(s.lfc_in_top_subtype) == sg).sum()),
            top_subtype_lfc_ge_half_pooled=int(
                (
                    (np.sign(s.lfc_in_top_subtype) == sg)
                    & (s.lfc_in_top_subtype.abs() >= 0.5 * s.log2FoldChange.abs())
                ).sum()
            ),
            pred_comp_same_sign=int((np.sign(s.pred_comp_lfc) == sg).sum()),
            abs_obs_gt_abs_pred_x2=int(
                (s.log2FoldChange.abs() > 2 * s.pred_comp_lfc.abs()).sum()
            ),
            pred_abs_max=float(s.pred_comp_lfc.abs().max()),
            pred_abs_median=float(s.pred_comp_lfc.abs().median()),
            obs_abs_median=float(s.log2FoldChange.abs().median()),
            corr_pred_obs_all_genes=float(
                d[["pred_comp_lfc", "log2FoldChange"]].dropna().corr().iloc[0, 1]
            ),
            corr_pred_obs_degs=float(
                s[["pred_comp_lfc", "log2FoldChange"]].corr().iloc[0, 1]
            ),
            n_marker_gt4x=int(s.marker_gt4x.sum()),
            n_marker_and_abs_obs_lt_2x_pred=int(
                (
                    s.marker_gt4x
                    & (s.log2FoldChange.abs() <= 2 * s.pred_comp_lfc.abs())
                ).sum()
            ),
            corr_std_vs_pooled_lfc_degs=float(
                s[["std_lfc", "log2FoldChange"]].corr().iloc[0, 1]
            ),
        )
        print(diag, flush=True)
        pd.DataFrame([diag]).to_csv(OUT / f"diag_{cls}.csv", index=False)
        top = s.sort_values("padj").head(20)[
            [
                "gene_name",
                "log2FoldChange",
                "padj",
                "pred_comp_lfc",
                "top_subtype",
                "subtype_log2_range",
                "lfc_in_top_subtype",
                "std_lfc",
                "std_padj",
            ]
        ]
        top.to_csv(OUT / f"top20_{cls}.csv")
        print(top.round(3).to_string(), flush=True)

    pd.DataFrame(summary_rows).to_csv(OUT / "model_summary.csv", index=False)
    pd.DataFrame(surv_rows).to_csv(OUT / "survival.csv", index=False)
    pd.DataFrame(prop_rows).to_csv(OUT / "subtype_proportions.csv", index=False)
    overlap = subtype_overlap(pub, fine)
    overlap.to_csv(OUT / "subtype_deg_overlap.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(surv_rows).to_string(index=False))
    print(pd.DataFrame(prop_rows).round(3).to_string(index=False))
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
