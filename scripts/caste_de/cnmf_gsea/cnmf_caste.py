#!/usr/bin/env python
"""Step 3: k selection, consensus and per-program caste test (cells 23, 25).

Source: 01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb cells 23 and 25
(md5 184efd9af8192fd09f7329f1eeaa9f17).

Cell 23: the k-selection table (stability, prediction error) and its plot.
Cell 25: consensus at k = 40 and 50 (density threshold 2.0; skipped when the
consensus files exist, as in the notebook, so ``--source saved`` never rewrites
the saved run), then ``joint_test``: per cell type and trial, the queen - worker
difference in mean program usage (cell-type x trial x caste groups of >= 30
nuclei) divided by the SD of usage within the cell type. A program x cell type
is a candidate when the sign agrees in all trials, |mean| >= 0.8 SD, the
weakest trial >= 0.4 SD and usage >= 1% (printed only; the figure's stricter
replication rule is in ``label_swap_null.py``).

Writes, under ``pipeline/<source>/``: cnmf_k_selection.csv, cnmf_k_selection.png,
joint_k40_caste.csv and joint_k50_caste.csv.

    PYTHONPATH=src:scripts <pipeline env>/bin/python \\
        scripts/caste_de/cnmf_gsea/cnmf_caste.py --source repo
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnmf import cNMF, load_df_from_npz
from cnmf_gsea_common import CASTE, CT, TRIAL, load_atlas, setup
from cnmf_gsea_paths import CNMF_NAME, add_source_arg, source
from scipy import stats

from bimp_st.paths import SN_CB

K_J = [40, 50]
DENSITY = 2.0


def k_selection(oj: cNMF, out_dir) -> pd.DataFrame:
    """Cell 23: k-selection table and plot."""
    kst = load_df_from_npz(oj.paths["k_selection_stats"]).reset_index(drop=True)
    if "silhouette" in kst.columns:
        kst = kst.rename(columns={"silhouette": "stability"})
    print(kst[["k", "stability", "prediction_error"]].round(4).to_string(index=False))
    kst.to_csv(os.path.join(out_dir, "cnmf_k_selection.csv"), index=False)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(kst.k, kst.stability, "o-", color="#2A8C82")
    ax.set_xlabel("k")
    ax.set_ylabel("stability (silhouette)", color="#2A8C82")
    ax2 = ax.twinx()
    ax2.plot(kst.k, kst.prediction_error, "s--", color="#8C8C8C")
    ax2.set_ylabel("prediction error", color="#8C8C8C")
    for a_ in (ax, ax2):
        a_.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cnmf_k_selection.png"), dpi=200)
    plt.close()
    return kst


def consensus(oj: cNMF, cnmf_dir) -> None:
    """Cell 25: consensus at each k, skipped if its outputs already exist."""
    for k in K_J:
        f = f"{CNMF_NAME}.usages.k_{k}.dt_2_0.consensus.txt"
        if not os.path.exists(os.path.join(cnmf_dir, CNMF_NAME, f)):
            oj.consensus(
                k=k,
                density_threshold=DENSITY,
                show_clustering=False,
                close_clustergram_fig=True,
            )


def joint_test(oj: cNMF, obs, ql, wl, k, min_cells=30) -> pd.DataFrame:
    """Cell 25: per-trial caste effect of every program in every cell type."""
    usage, sscore, _, _ = oj.load_results(K=k, density_threshold=DENSITY)
    if not set(usage.columns) <= set(sscore.columns):
        sscore = sscore.T
    usage = usage.div(usage.sum(1), axis=0)
    ob = obs.loc[usage.index, [CT, CASTE, TRIAL]].astype(str).copy()
    ob[CT] = ob[CT].str.replace("\n", " ", regex=False)
    pc = list(usage.columns)
    mm = usage.join(ob)
    grp = mm.groupby([CT, TRIAL, CASTE])
    mean_u, n = grp[pc].mean(), grp.size()
    mean_u = mean_u[n.reindex(mean_u.index) >= min_cells]
    diff = mean_u.xs(ql, level=CASTE) - mean_u.xs(wl, level=CASTE)
    diff = diff.dropna(how="all")
    sd = mm.groupby(CT)[pc].std()
    eff = diff.div(sd.reindex(diff.index.get_level_values(0)).values)
    eff.columns.name = "program"
    long = eff.stack().rename("eff").reset_index()
    res = long.pivot_table(
        index=[CT, "program"], columns=TRIAL, values="eff"
    ).reset_index()
    tc = [c for c in res.columns if c not in (CT, "program")]
    res["mean"] = res[tc].mean(1)
    res["min_abs"] = res[tc].abs().min(1)
    res["same"] = res[tc].apply(
        lambda r: (np.sign(r) == np.sign(r.iloc[0])).all(), axis=1
    )
    res["t"] = res[tc].apply(lambda r: stats.ttest_1samp(r, 0)[0], axis=1)
    use = (mm.groupby(CT)[pc].mean() * 100).stack().rename("usage_pct").reset_index()
    use.columns = [CT, "program", "usage_pct"]
    res = res.merge(use, on=[CT, "program"])
    share = usage.groupby(ob[CT].values).sum()
    share = share / share.sum()
    home = pd.DataFrame(
        {
            "program": share.columns,
            "home": share.idxmax().values,
            "home_frac": share.max().values,
        }
    )
    res = res.merge(home, on="program")
    res["k"] = k
    return res


def report(res: pd.DataFrame, k: int) -> None:
    """Cell 25's printout: candidate program x cell-type caste differences."""
    cand = res[
        res.same
        & (res["mean"].abs() >= 0.8)
        & (res.min_abs >= 0.4)
        & (res.usage_pct >= 1)
    ]
    print(
        f"\n== k={k}: {res.program.nunique()} programs x {res[CT].nunique()} "
        f"cell types = {len(res)} tests | same sign in all trials: "
        f"{int(res.same.sum())} ({res.same.mean():.0%}, ~25% expected) | "
        f"candidates: {len(cand)}"
    )
    cols = [CT, "program", "mean", "min_abs", "usage_pct", "home", "home_frac"]
    top = cand.sort_values("mean", key=abs, ascending=False)[cols].head(25)
    print(top.round(2).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arg(ap)
    src = source(ap.parse_args().source)
    setup()
    adata, ql, wl = load_atlas(SN_CB)
    oj = cNMF(output_dir=str(src.cnmf_dir), name=CNMF_NAME)
    k_selection(oj, src.out_dir)
    consensus(oj, src.cnmf_dir)
    for k in K_J:
        res = joint_test(oj, adata.obs, ql, wl, k)
        res.to_csv(src.out_dir / f"joint_k{k}_caste.csv", index=False)
        report(res, k)


if __name__ == "__main__":
    main()
