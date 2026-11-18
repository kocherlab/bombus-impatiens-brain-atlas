"""Bento localization in Large-type (KC-NC) cells by projected snRNA subcluster x distance.

Subcluster = argmax of the `fig:kc_subtypes` Tangram projection (ltKC_CB_spatial_subclusters_dsource.h5ad,
obsm['tangram_ct_pred']). The projection is soft and, inside KC-NC, largely position, so the
analysis is stratified by distance from the Small-type core and adjusted for depth.

Outputs (results/bento/kc_subclusters/):
  lt_cells_subcluster.parquet   per Large-type cell: section, distance, subcluster, patterns
  summary.txt, tables, fig_s_bento_kc_subclusters.png
and results/bento/by_section/ltkc_subcluster_calls.parquet (all KC-compartment cells, for ELLA).
"""

import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regroup_common import REPO, aggregate_lp, cell_map  # noqa: E402

from bimp_st.si_common import LABEL_SIZE, QUEEN_COLOR, WORKER_COLOR, set_style  # noqa: E402

OUT = REPO / "results/bento/kc_subclusters"
OUT.mkdir(parents=True, exist_ok=True)
H5 = REPO / "results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad"
CAST = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}
EDGES = [0, 15, 30, 45, 60, 90, 120, np.inf]
LAB = ["0-15", "15-30", "30-45", "45-60", "60-90", "90-120", ">120"]
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


a = ad.read_h5ad(H5, backed="r")
pred = a.obsm["tangram_ct_pred"]
cols = list(pred.columns) if hasattr(pred, "columns") else [f"S{i+1}" for i in range(pred.shape[1])]
P = np.asarray(pred.values if hasattr(pred, "values") else pred, float)
log(f"projection columns: {cols}; cells {P.shape[0]}")
sub = pd.DataFrame(
    {
        "slice_name": a.obs["slice_name"].astype(str).to_numpy(),
        "cell_id": a.obs["cell_id"].astype(str).to_numpy(),
        "sub": np.array(cols)[P.argmax(1)],
        "pmax": P.max(1),
        "margin": np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2],
    }
).drop_duplicates(["slice_name", "cell_id"])
a.file.close()
log("subcluster call counts: " + str(sub["sub"].value_counts().to_dict()))

allc = []
for brain in CAST:
    cells, _ = cell_map(brain)
    _, cell, _ = aggregate_lp(brain, cells)
    cell = cell[cell.comp != ""].merge(sub, on=["slice_name", "cell_id"], how="left")
    cell["brain"] = brain
    allc.append(cell)
cell = pd.concat(allc, ignore_index=True)
cell[["brain", "slice_name", "cell_id", "comp", "sub", "pmax", "margin"]].to_parquet(
    REPO / "results/bento/by_section/ltkc_subcluster_calls.parquet"
)
lt = cell[(cell.comp == "KC-NC") & cell["dist"].notna() & cell["sub"].notna()].copy()
lt["bin"] = pd.cut(lt["dist"], EDGES, labels=LAB, right=False)
lt.to_parquet(OUT / "lt_cells_subcluster.parquet")
log(f"Large-type cells with distance and subcluster call: {len(lt)} "
    f"({lt.groupby('brain').size().to_dict()})")

# (1) subcluster composition across distance (the confound)
comp = lt.groupby(["brain", "bin", "sub"], observed=True).size().unstack("sub").fillna(0)
share = comp.div(comp.sum(1), axis=0)
share.to_csv(OUT / "subcluster_share_by_distance.csv")
log("\n== 1. Subcluster share of Large-type cells by distance bin")
log(share.round(3).to_string())

# (2) nuclear fraction by subcluster x distance (pooled counts; per-section means)
g = lt.groupby(["brain", "slice_name", "sub", "bin"], observed=True)[["nuclear", "n_calls"]].sum().reset_index()
g["frac"] = g.nuclear / g.n_calls
tab = g.groupby(["brain", "sub", "bin"], observed=True).apply(
    lambda d: pd.Series(dict(frac=d.nuclear.sum() / d.n_calls.sum(), sd=d.frac.std(), sections=len(d), cells=0)),
    include_groups=False,
).reset_index()
tab = tab.astype({"frac": float, "sd": float, "sections": float})
tab.to_csv(OUT / "nuclear_by_subcluster_distance.csv", index=False)
log("\n== 2. Nuclear fraction (pooled) by subcluster x distance")
log(tab.pivot_table(index=["brain", "sub"], columns="bin", values="frac", observed=True).round(3).to_string())
sub_tot = lt.groupby(["brain", "sub"])[["nuclear", "n_calls"]].sum()
sub_tot["frac"] = sub_tot.nuclear / sub_tot.n_calls
log("\nnuclear fraction by subcluster, all distances:\n" + sub_tot.frac.round(4).to_string())
# S1 vs S2 within distance bins, paired by section
subs = sorted(lt["sub"].unique())
log(f"\nWithin each distance bin, difference between subclusters ({subs[0]} minus {subs[1]}), paired by section:")
dd = g.pivot_table(index=["brain", "slice_name", "bin"], columns="sub", values="frac", observed=True).dropna()
dd["diff"] = dd[subs[0]] - dd[subs[1]]
rep = dd.groupby(["brain", "bin"], observed=True)["diff"].agg(["median", "count", lambda s: (s > 0).sum()])
rep.columns = ["median_diff", "n_sections", "n_pos"]
rep.to_csv(OUT / "subcluster_diff_within_distance.csv")
log(rep.round(4).to_string())

# (3) depth-adjusted GLM per brain, section fixed effects, section-clustered SE
log("\n== 3. Binomial GLM per brain: nuclear calls ~ subcluster + distance(z, per 100um) + log depth + section")
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = []
    for brain, d in lt.groupby("brain"):
        d = d.assign(nonnuc=d.n_calls - d.nuclear, ld=np.log(d.n_calls), dist100=d["dist"] / 100.0)
        d = d[d.n_calls > 0]
        d = d.assign(sub=pd.Categorical(d["sub"]))
        f = "nuclear + nonnuc ~ C(sub) + dist100 + ld + C(slice_name)"
        m = smf.glm(f, data=d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d.slice_name.astype("category").cat.codes}
        )
        for k in m.params.index:
            if k.startswith("C(sub)") or k in ("dist100", "ld"):
                lo, hi = m.conf_int().loc[k]
                rows.append(dict(brain=brain, term=k, odds_ratio=np.exp(m.params[k]), lo=np.exp(lo), hi=np.exp(hi)))
    glm = pd.DataFrame(rows)
    glm.to_csv(OUT / "glm_subcluster_distance_depth.csv", index=False)
    log(glm.round(3).to_string(index=False))
    log("(section-clustered CIs; sections of one brain are not independent replicates of caste; descriptive)")
except Exception as e:  # noqa: BLE001
    glm = None
    log(f"GLM skipped: {e!r}")

set_style()
fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
SUBCOL = dict(zip(subs, ["#1b9e77", "#d95f02", "#7570b3", "#999999"], strict=False))
for r, brain in enumerate(CAST):
    t = tab[tab.brain == brain]
    a_ = ax[0, r]
    for s in subs:
        q = t[t["sub"] == s].set_index("bin").reindex(LAB)
        a_.errorbar(range(len(LAB)), q.frac, yerr=q["sd"] / np.sqrt(q.sections.clip(lower=1)), fmt="-o", ms=4, color=SUBCOL[s], label=s, capsize=2)
    a_.set_xticks(range(len(LAB))); a_.set_xticklabels(LAB, rotation=30)
    a_.set_xlabel("distance from Small-type core (um)"); a_.set_ylabel("nuclear fraction")
    a_.set_title(f"{'AB'[r]}  {brain}: Large-type by projected subcluster", loc="left", fontsize=LABEL_SIZE, color=CAST[brain])
    if r == 0:
        a_.legend(frameon=False, fontsize=7)
a_ = ax[1, 0]
for brain in CAST:
    s_ = share.loc[brain].reindex(LAB)
    for s in subs:
        a_.plot(range(len(LAB)), s_[s], "-" if brain == "queen" else "--", marker="o", ms=3,
                color=SUBCOL[s], label=f"subcluster {s}" if brain == "queen" else None)
a_.set_xticks(range(len(LAB))); a_.set_xticklabels(LAB, rotation=30)
a_.set_ylabel("share of Large-type cells (solid queen, dashed worker)"); a_.set_xlabel("distance (um)"); a_.legend(frameon=False, fontsize=7)
a_.set_title("C  Subcluster call tracks position", loc="left", fontsize=LABEL_SIZE)
a_ = ax[1, 1]
if glm is not None and len(glm):
    y = 0
    labels = []
    for brain in CAST:
        for _, rw in glm[(glm.brain == brain) & (glm.term != "ld")].iterrows():
            a_.errorbar(rw.odds_ratio, y, xerr=[[rw.odds_ratio - rw.lo], [rw.hi - rw.odds_ratio]], fmt="o", color=CAST[brain], capsize=2)
            nm = "per 100 um from core" if rw.term == "dist100" else f"subcluster {rw.term.replace('C(sub)[T.', '').replace(']', '')} vs 0"
            labels.append(f"{brain}: {nm}")
            y += 1
    a_.axvline(1, color="grey", lw=0.6)
    a_.set_yticks(range(len(labels))); a_.set_yticklabels(labels, fontsize=7)
    a_.set_xlabel("odds ratio of nuclear call (95% CI)")
a_.set_title("D  Adjusted for depth and section", loc="left", fontsize=LABEL_SIZE)
for q in ax.ravel():
    q.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_s_bento_kc_subclusters.png", dpi=300)
(OUT / "summary.txt").write_text("\n".join(lines))
