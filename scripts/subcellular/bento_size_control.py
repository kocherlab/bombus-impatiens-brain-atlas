"""Is the KC vs optic/AL difference in bento nuclear fraction just cell size?

Uses per-cell cell area, nucleus area and LP depth (area_control_cells.parquet).
(1) compare optic+AL vs KC within bins of absolute cell area;
(2) per-brain binomial GLM: nuclear ~ region group + log cell area + log nucleus area + log depth + section.
Outputs: results/bento/regions_summary/size_control_*.csv, size_control.png
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from bimp_st.si_common import QUEEN_COLOR, WORKER_COLOR, set_style

R = Path(__file__).resolve().parents[2] / "results/bento"
OUT = R / "regions_summary"
KC = ["KC-IC", "KC-NC", "KC-OC"]
OPT = ["lamina", "medulla", "lobula", "al"]
CAST = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


A = pd.read_parquet(OUT / "area_control_cells.parquet")
A["grp"] = np.where(A.comp.isin(KC), "KC", "optic")
EDGES = [0, 12, 16, 20, 25, 30, 40, 55, 80, 1e9]
A["abin"] = pd.cut(A.cell_area, EDGES, right=False)
log("== optic+AL vs KC nuclear fraction within bins of absolute cell area (um^2; >=200 cells each side)")
rows = []
for brain, g in A.groupby("brain"):
    for b, d in g.groupby("abin", observed=True):
        k, o = d[d.grp == "KC"], d[d.grp == "optic"]
        if len(k) >= 200 and len(o) >= 200:
            kn, on = k.nuclear.sum() / k.n_classified.sum(), o.nuclear.sum() / o.n_classified.sum()
            rows.append(dict(brain=brain, cell_area_bin=str(b), kc_cells=len(k), optic_cells=len(o),
                             kc_nuc_area=k.nuc_area.median(), optic_nuc_area=o.nuc_area.median(),
                             kc_nuclear=kn, optic_nuclear=on, ratio=on / kn))
con = pd.DataFrame(rows)
con.to_csv(OUT / "size_control_matched.csv", index=False)
log(con.round(3).to_string(index=False))
if len(con):
    log(f"ratio range {con.ratio.min():.2f}-{con.ratio.max():.2f}; optic > KC in {(con.ratio > 1).sum()}/{len(con)} bins")

# Small-type vs Large-type at matched cell area
log("\n== Small-type vs Large-type KC within cell-area bins (>=200 cells each side)")
r2 = []
for brain, g in A[A.comp.isin(["KC-IC", "KC-NC"])].groupby("brain"):
    for b, d in g.groupby("abin", observed=True):
        s, lg = d[d.comp == "KC-IC"], d[d.comp == "KC-NC"]
        if len(s) >= 200 and len(lg) >= 200:
            r2.append(dict(brain=brain, cell_area_bin=str(b), small_cells=len(s), large_cells=len(lg),
                           small_nuclear=s.nuclear.sum() / s.n_classified.sum(),
                           large_nuclear=lg.nuclear.sum() / lg.n_classified.sum()))
r2 = pd.DataFrame(r2)
r2["ratio_large_over_small"] = r2.large_nuclear / r2.small_nuclear
r2.to_csv(OUT / "size_control_small_vs_large.csv", index=False)
log(r2.round(3).to_string(index=False))

# GLM
log("\n== Binomial GLM per brain: nuclear calls ~ group(optic vs KC) + log cell area + log nucleus area + log depth + section")
glm = []
for brain, g in A.groupby("brain"):
    # binomial over classified calls; read depth (ld) stays all calls per cell
    d = g[g.n_classified > 0].copy()
    d["nonnuc"] = d.n_classified - d.nuclear
    d["lca"], d["lna"], d["ld"] = np.log(d.cell_area), np.log(d.nuc_area), np.log(d.n_calls)
    d["optic"] = (d.grp == "optic").astype(int)
    d["sec"] = d.slice_name
    for name, f in {
        "group only": "nuclear + nonnuc ~ optic",
        "+ depth": "nuclear + nonnuc ~ optic + ld",
        "+ depth + cell area": "nuclear + nonnuc ~ optic + ld + lca",
        "+ depth + cell area + nucleus area": "nuclear + nonnuc ~ optic + ld + lca + lna",
    }.items():
        m = smf.glm(f + " + C(sec)", data=d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d.sec.astype("category").cat.codes})
        lo, hi = m.conf_int().loc["optic"]
        glm.append(dict(brain=brain, model=name, optic_vs_kc_OR=np.exp(m.params["optic"]), lo=np.exp(lo), hi=np.exp(hi)))
glm = pd.DataFrame(glm)
glm.to_csv(OUT / "size_control_glm.csv", index=False)
log(glm.round(3).to_string(index=False))
log("(section-clustered CIs; descriptive; sections of a brain are not independent)")
(OUT / "size_control_summary.txt").write_text("\n".join(lines))

set_style()
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for brain in CAST:
    g = A[A.brain == brain]
    for grp, ls in (("KC", "-"), ("optic", "--")):
        ax[0].hist(g[g.grp == grp].cell_area.clip(upper=100), bins=50, histtype="step", density=True,
                   color=CAST[brain], ls=ls, label=f"{brain} {grp}")
ax[0].set_xlabel("cell area (um^2)"); ax[0].legend(frameon=False, fontsize=7)
for brain in CAST:
    c = con[con.brain == brain]
    ax[1].plot(range(len(c)), c.ratio, "-o", color=CAST[brain], label=brain)
    ax[1].set_xticks(range(len(c))); ax[1].set_xticklabels(c.cell_area_bin, rotation=40, fontsize=6)
ax[1].axhline(1, color="grey", lw=0.6); ax[1].set_xlabel("cell-area bin (um^2)")
ax[1].set_ylabel("optic+AL / KC nuclear fraction"); ax[1].legend(frameon=False)
order = list(glm.model.unique())
for i, brain in enumerate(CAST):
    q = glm[glm.brain == brain].set_index("model").reindex(order)
    y = np.arange(len(order)) + (i - 0.5) * 0.25
    ax[2].errorbar(q.optic_vs_kc_OR, y, xerr=[q.optic_vs_kc_OR - q.lo, q.hi - q.optic_vs_kc_OR], fmt="o",
                   color=CAST[brain], capsize=2, label=brain)
ax[2].axvline(1, color="grey", lw=0.6); ax[2].set_yticks(range(len(order))); ax[2].set_yticklabels(order, fontsize=7)
ax[2].set_xlabel("odds ratio, optic+AL vs KC (nuclear call)"); ax[2].invert_yaxis()
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "size_control.png", dpi=300)
