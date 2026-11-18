#!/usr/bin/env python
"""Summarise ELLA per-gene subcellular localization across groups and castes, and
compare with bento's nuclear fraction for the same gene x group.

ELLA score = position of the peak of the fitted radial intensity (0 = nucleus centre,
1 = cell edge). Significance = BY-FDR <= 0.05 per (section, group). 150 cells per group.
One brain per caste: descriptive, no caste p-values.

usage: si_ella_summary.py kc|regions
Outputs: results/ella[_regions]/summary/{summary.txt, *.csv, fig_s_ella_<mode>.png}
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bimp_st.si_common import LABEL_SIZE, QUEEN_COLOR, WORKER_COLOR, set_style

MODE = sys.argv[1] if len(sys.argv) > 1 else "kc"
B = Path(__file__).resolve().parents[2] / "results"
E = B / ("ella_regions" if MODE == "regions" else "ella")
OUT = E / "summary"
OUT.mkdir(exist_ok=True)
CASTE = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}
if MODE == "regions":
    TYPES = ["Lamina", "Medulla", "Lobula", "Antennal lobe"]
    BENTO = {"lamina": "Lamina", "medulla": "Medulla", "lobula": "Lobula", "al": "Antennal lobe"}
    BENTO_F = B / "bento/by_section/region_gene_pattern_by_section.csv"
else:
    TYPES = ["Small-type", "Large-type", "Class II"]
    BENTO = {"KC-IC": "Small-type", "KC-NC": "Large-type", "KC-OC": "Class II"}
    BENTO_F = B / "bento/kc_compartments/gene_pattern_by_section.csv"
BINS = [f"Large-type {b} um" for b in ["0-30", "30-60", "60-120", ">120"]]
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


files = sorted(E.glob("*/*/ella_results.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["caste"] = df.brain
log(f"{len(files)} sections; {len(df)} gene x group x section fits; failed: "
    f"{len(list(E.glob('*/*/FAILED.txt')))}")
log(f"groups: {df.groupby('caste').type.unique().to_dict()}")

# thresholds on score: peak position of intensity along the radius
INNER, OUTER = 0.35, 0.65
sig = df[df.sig]
log("\n== 1. Per group: genes tested, fraction significant, and where significant genes peak")
rows = []
for (caste, t), d in df.groupby(["caste", "type"]):
    s = d[d.sig]
    rows.append(dict(caste=caste, type=t, n_fits=len(d), n_sections=d.slice_name.nunique(),
                     frac_sig=d.sig.mean(), n_sig=len(s), median_score_sig=s.score.median(),
                     frac_inner=(s.score < INNER).mean(), frac_mid=((s.score >= INNER) & (s.score <= OUTER)).mean(),
                     frac_outer=(s.score > OUTER).mean(), mean_cells=d.n_cells.mean()))
g1 = pd.DataFrame(rows)
g1.to_csv(OUT / "group_summary.csv", index=False)
log(g1.round(3).to_string(index=False))

# gene-level consensus per (caste, type)
log("\n== 2. Gene-level consensus (a gene is 'localised' in a group if significant in >= 1/3 of its sections)")
gl = (df.groupby(["caste", "type", "feature_name"])
        .agg(n_sec=("sig", "size"), n_sig=("sig", "sum"),
             score_all=("score", "median")).reset_index())
sg = df[df.sig].groupby(["caste", "type", "feature_name"]).score.median().rename("score_sig")
gl = gl.merge(sg, on=["caste", "type", "feature_name"], how="left")
gl["localised"] = (gl.n_sig / gl.n_sec) >= 1 / 3
gl.to_csv(OUT / "gene_consensus.csv", index=False)
log(gl[gl.n_sec >= 3].groupby(["caste", "type"]).agg(
    genes=("feature_name", "size"), localised=("localised", "sum"),
    median_score_localised=("score_sig", "median")).round(3).to_string())

# compare groups: genes localised in both, paired
log("\n== 3. Same gene, different group (genes with >=3 sections in both, significant in >=1 section in both)")
pr = []
for caste in CASTE:
    p = gl[(gl.caste == caste) & (gl.n_sec >= 3) & gl.score_sig.notna()].pivot(index="feature_name", columns="type", values="score_sig")
    for i, a in enumerate(TYPES):
        for b in TYPES[i + 1:]:
            if a in p and b in p:
                x = p[[a, b]].dropna()
                if len(x) >= 5:
                    pr.append(dict(caste=caste, a=a, b=b, n_genes=len(x), median_a=x[a].median(),
                                   median_b=x[b].median(), median_diff_a_minus_b=(x[a] - x[b]).median(),
                                   spearman=stats.spearmanr(x[a], x[b])[0]))
pr = pd.DataFrame(pr)
pr.to_csv(OUT / "group_pairwise_gene_scores.csv", index=False)
log(pr.round(3).to_string(index=False))

# distance bins (kc mode)
if MODE == "kc":
    log("\n== 4. Large-type cells by distance from the Small-type core (ELLA)")
    b = g1[g1.type.isin(BINS)].copy()
    log(b.round(3).to_string(index=False))
    pb = []
    for caste in CASTE:
        p = gl[(gl.caste == caste) & (gl.n_sec >= 3) & gl.score_sig.notna()].pivot(index="feature_name", columns="type", values="score_sig")
        if BINS[0] in p and BINS[-1] in p:
            x = p[[BINS[0], BINS[-1]]].dropna()
            log(f"{caste}: genes localised in both 0-30 and >120 um bins: {len(x)}; "
                f"median score diff (>120 minus 0-30) = {(x[BINS[-1]] - x[BINS[0]]).median():+.3f}; "
                f"higher in >120 for {(x[BINS[-1]] > x[BINS[0]]).sum()}/{len(x)}")

# vs bento
log("\n== 5. ELLA vs bento nuclear fraction (same gene, same group, pooled over sections)")
bg = pd.read_csv(BENTO_F)
bg["type"] = bg.comp.map(BENTO)
bg = bg.dropna(subset=["type"]).groupby(["brain", "type", "feature_name"])[["nuclear", "n_calls"]].sum().reset_index()
bg["caste"] = bg.brain
bg["bento_nuclear"] = bg.nuclear / bg.n_calls
mm = gl.merge(bg[["caste", "type", "feature_name", "bento_nuclear", "n_calls"]], on=["caste", "type", "feature_name"])
mm = mm[(mm.n_calls >= 100) & (mm.n_sec >= 3)]
rr = []
for (caste, t), d in mm.groupby(["caste", "type"]):
    for lab, col in (("all genes, median score", "score_all"), ("localised genes, score", "score_sig")):
        x = d.dropna(subset=[col])
        if len(x) >= 8:
            rr.append(dict(caste=caste, type=t, which=lab, n=len(x),
                           spearman=stats.spearmanr(x[col], x.bento_nuclear)[0]))
rr = pd.DataFrame(rr)
rr.to_csv(OUT / "ella_vs_bento.csv", index=False)
log("Spearman(ELLA peak position, bento nuclear fraction); negative = agree (nuclear-enriched genes peak near the centre)")
log(rr.round(3).to_string(index=False))

# figure
set_style()
nrow = 2
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
x = np.arange(len(TYPES))
a = ax[0, 0]
for i, caste in enumerate(CASTE):
    v = [g1[(g1.caste == caste) & (g1.type == t)].frac_sig.mean() for t in TYPES]
    a.bar(x + (i - .5) * .38, v, .38, color=CASTE[caste], label=caste)
a.set_xticks(x); a.set_xticklabels(TYPES, rotation=25, ha="right"); a.set_ylabel("fraction of fits significant (FDR<=0.05)")
a.legend(frameon=False); a.set_title("A  Localised genes", loc="left", fontsize=LABEL_SIZE)
a = ax[0, 1]
SHADE = {"frac_inner": "#2b2b2b", "frac_mid": "#9a9a9a", "frac_outer": "#e6e6e6"}
for i, caste in enumerate(CASTE):
    bot = np.zeros(len(TYPES))
    for col, lab in (("frac_inner", "nucleus side (<0.35)"), ("frac_mid", "mid"), ("frac_outer", "cell edge (>0.65)")):
        v = np.array([g1[(g1.caste == caste) & (g1.type == t)][col].mean() for t in TYPES])
        a.bar(x + (i - .5) * .38, v, .36, bottom=bot, color=SHADE[col], edgecolor=CASTE[caste], lw=1.2,
              label=lab if i == 0 else None)
        bot += v
a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
a.set_xticks(x); a.set_xticklabels(TYPES, rotation=25, ha="right")
a.set_ylabel("share of significant genes"); a.set_ylim(0, 1)
a.set_title("B  Where significant genes peak", loc="left", fontsize=LABEL_SIZE)
a = ax[1, 0]
if MODE == "kc":
    for caste in CASTE:
        v = [g1[(g1.caste == caste) & (g1.type == t)].frac_inner.mean() for t in BINS]
        a.plot(range(4), v, "o-", color=CASTE[caste], label=caste)
    a.set_xticks(range(4)); a.set_xticklabels(["0-30", "30-60", "60-120", ">120"])
    a.set_ylim(0, 1)
    a.set_xlabel("distance from Small-type core (um)"); a.set_ylabel("share of significant genes peaking nucleus-side")
    a.set_title("C  Large-type cells along the gradient", loc="left", fontsize=LABEL_SIZE)
else:
    p = gl[(gl.n_sec >= 3) & gl.score_sig.notna()].pivot(index=["feature_name", "type"], columns="caste", values="score_sig").dropna()
    a.scatter(p.queen, p.worker, s=8, color="#555555"); a.plot([0, 1], [0, 1], color="grey", lw=.6)
    a.set_xlabel("queen: peak position"); a.set_ylabel("worker: peak position")
    a.set_title(f"C  Caste concordance (rho {stats.spearmanr(p.queen, p.worker)[0]:.2f})", loc="left", fontsize=LABEL_SIZE)
a = ax[1, 1]
d = mm.dropna(subset=["score_all"])
for caste in CASTE:
    q = d[d.caste == caste]
    a.scatter(q.bento_nuclear, q.score_all, s=6, color=CASTE[caste], alpha=.5)
a.set_xlabel("bento nuclear fraction (gene x group)"); a.set_ylabel("ELLA median peak position")
a.set_title("D  ELLA vs bento", loc="left", fontsize=LABEL_SIZE)
for q in ax.ravel():
    q.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / f"fig_s_ella_{MODE}.png", dpi=300)
(OUT / "summary.txt").write_text("\n".join(lines))
