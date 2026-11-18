#!/usr/bin/env python
"""Bento localization across ANATOMICAL regions: is the regional difference real, or
a gene-composition / read-depth artefact? Descriptive; one brain per caste.

Fractions are among classified calls (regroup_common.fractions).
Inputs (results/bento/by_section/): region_composition_by_section.csv,
region_gene_pattern_by_section.csv, region_cell_depth_by_section.csv
Outputs: results/bento/regions_summary/{summary.txt, *.csv, fig_s_bento_regions.png}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bimp_st.si_common import LABEL_SIZE, QUEEN_COLOR, WORKER_COLOR, set_style

R = Path(__file__).resolve().parents[2] / "results/bento"
OUT = R / "regions_summary"
OUT.mkdir(exist_ok=True)
CLS = ["nuclear", "nuclear_edge", "cytoplasmic", "cell_edge", "none"]
ORDER = ["KC-IC", "KC-NC", "KC-OC", "lamina", "medulla", "lobula", "al"]
LAB = {"KC-IC": "Small-type", "KC-NC": "Large-type", "KC-OC": "Class II",
       "lamina": "Lamina", "medulla": "Medulla", "lobula": "Lobula", "al": "Antennal lobe"}
CASTE = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}
MARK = dict(zip(ORDER, "ovsP^DX", strict=False))
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


reg = pd.read_csv(R / "by_section/region_composition_by_section.csv")
gen = pd.read_csv(R / "by_section/region_gene_pattern_by_section.csv")
dep = pd.read_csv(R / "by_section/region_cell_depth_by_section.csv")
reg["caste"], gen["caste"], dep["caste"] = reg.brain, gen.brain, dep.brain

# (a) region profiles + PCA of section-level class fractions
fc = [f"frac_{c}" for c in CLS]
log("== a. Region profiles (mean per-section fraction of classified LP calls; "
    "frac_unclassified = share of all calls with no pattern)")
prof = reg.groupby(["caste", "comp"])[fc + ["frac_unclassified"]].mean().round(4)
prof.to_csv(OUT / "region_profiles.csv")
log(prof.to_string())
X = reg[fc].to_numpy()
Z = (X - X.mean(0)) / X.std(0)
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
pc = U[:, :2] * S[:2]
reg["pc1"], reg["pc2"] = pc[:, 0], pc[:, 1]
log(f"PCA variance explained: {np.round(S[:3]**2 / (S**2).sum(), 3)}; "
    f"PC1 loadings {dict(zip(CLS, np.round(Vt[0], 2), strict=False))}")
# separability: nearest-centroid accuracy of region from the class fractions, leave-one-section-out
cent = {r: Z[(reg.comp == r).to_numpy()].mean(0) for r in ORDER}
pred = [min(cent, key=lambda r: np.linalg.norm(z - cent[r])) for z in Z]
acc = (np.array(pred) == reg.comp.to_numpy()).mean()
log(f"nearest-centroid region assignment from the 5 class fractions alone: {acc:.2f} "
    f"(chance {1/len(ORDER):.2f}; centroids include the section, so optimistic)")
# region within-caste consistency: queen vs worker profile correlation across (region, class)
pq = prof.loc["queen"][fc].to_numpy().ravel()
pw = prof.loc["worker"][fc].to_numpy().ravel()
log(f"queen-worker correlation of the region x class profile: Spearman rho = "
    f"{stats.spearmanr(pq, pw)[0]:.3f} (35 cells)")

# (b) read depth confound
log("\n== b. Read depth (LP calls per cell) by region")
dd = dep.groupby(["caste", "comp"]).agg(cells=("n_cells", "sum"), calls_per_cell=("calls_per_cell", "mean")).round(1)
dd.to_csv(OUT / "depth_by_region.csv")
log(dd.to_string())
m = reg.merge(dep[["slice_name", "comp", "calls_per_cell"]], on=["slice_name", "comp"])
for cl in ("frac_none", "frac_nuclear"):
    rho = stats.spearmanr(m.calls_per_cell, m[cl])[0]
    # within-region (partial): correlate after centring by region/caste
    mm = m.assign(a=m[cl] - m.groupby(["caste", "comp"])[cl].transform("mean"),
                  b=m.calls_per_cell - m.groupby(["caste", "comp"]).calls_per_cell.transform("mean"))
    log(f"calls/cell vs {cl}: Spearman across all section-regions {rho:.2f}; "
        f"within region (centred) {stats.spearmanr(mm.a, mm.b)[0]:.2f}")

# (c) gene-composition standardisation
log("\n== c. Gene-composition-standardised fractions")
# classified calls are the denominator; "none" counts only rows with no other pattern
gen["none"] = gen["none_only"]
gc = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge", "n_classified"]
pool = gen.groupby(["caste", "comp", "feature_name"])[gc].sum().reset_index()
MINC = 200
rows, stdrows = [], []
perg = {}
for caste, g in pool.groupby("caste"):
    ok = g.groupby("feature_name").apply(
        lambda d: d.set_index("comp").reindex(ORDER).n_classified.fillna(0).min() >= MINC,
        include_groups=False)
    genes = ok[ok].index
    gg = g[g.feature_name.isin(genes)].copy()
    w = gg.groupby("feature_name").n_classified.sum()
    w = w / w.sum()
    log(f"{caste}: {len(genes)} genes with >= {MINC} classified calls in every one of the 7 regions")
    for comp, d in gg.groupby("comp"):
        d = d.set_index("feature_name")
        raw = {c: d[c].sum() / d.n_classified.sum() for c in CLS}
        std = {c: float(((d[c] / d.n_classified) * w.reindex(d.index)).sum()) for c in CLS}
        rows.append(dict(caste=caste, comp=comp, kind="own_mix", **raw))
        stdrows.append(dict(caste=caste, comp=comp, kind="standardised", **std))
    perg[caste] = gg.assign(frac_nuclear=gg.nuclear / gg.n_classified)
ss = pd.DataFrame(rows + stdrows)
ss.to_csv(OUT / "gene_standardised_region_fractions.csv", index=False)
for caste in CASTE:
    t = ss[ss.caste == caste].pivot(index="comp", columns="kind", values="nuclear").reindex(ORDER)
    t["shift"] = t.standardised - t.own_mix
    log(f"{caste} nuclear fraction, region own gene mix vs common-mix standard:")
    log(t.round(4).to_string())
    r_own = t.own_mix.max() - t.own_mix.min(); r_std = t.standardised.max() - t.standardised.min()
    log(f"  range across regions: own mix {r_own:.3f} -> standardised {r_std:.3f}")

# variance partition of per-gene x region nuclear fraction
log("\nVariance of gene x region nuclear fraction explained by gene identity vs region")
for caste, gg in perg.items():
    y = gg.frac_nuclear
    tot = ((y - y.mean()) ** 2).sum()
    gm = gg.groupby("feature_name").frac_nuclear.transform("mean")
    r2_gene = 1 - ((y - gm) ** 2).sum() / tot if False else ((gm - y.mean()) ** 2).sum() / tot
    res1 = y - gm
    rm = res1.groupby(gg.comp).transform("mean")
    r2_reg = ((rm) ** 2).sum() / tot
    log(f"  {caste}: gene identity {r2_gene:.2f}, region (after gene) {r2_reg:.2f}, residual {1 - r2_gene - r2_reg:.2f}")

# (d) per-gene contrast: optic vs KC, concordance between castes
log("\n== d. Per-gene nuclear-fraction contrast, optic lobe (lamina/medulla/lobula) minus Kenyon (3 compartments)")
con = {}
for caste, gg in perg.items():
    p = gg.pivot(index="feature_name", columns="comp", values="frac_nuclear")
    con[caste] = p[["lamina", "medulla", "lobula"]].mean(1) - p[["KC-IC", "KC-NC", "KC-OC"]].mean(1)
c = pd.DataFrame(con).dropna()
rho = stats.spearmanr(c.queen, c.worker)[0]
log(f"{len(c)} genes in both castes; queen-worker Spearman rho of the contrast = {rho:.2f}; "
    f"genes with optic > KC in both castes: {int(((c.queen > 0) & (c.worker > 0)).sum())}, "
    f"opposite sign: {int(((c.queen * c.worker) < 0).sum())}; median contrast {c.queen.median():.3f} / {c.worker.median():.3f}")
c["mean_contrast"] = c[["queen", "worker"]].mean(1)
c.sort_values("mean_contrast").to_csv(OUT / "gene_optic_minus_kc_nuclear.csv")
log("largest concordant positive (optic more nuclear) and negative contrasts:")
cc = c[(c.queen * c.worker) > 0].sort_values("mean_contrast")
log(cc.head(5).round(3).to_string()); log(cc.tail(5).round(3).to_string())

# figure
set_style()
fig, ax = plt.subplots(2, 2, figsize=(10, 8.5))
a = ax[0, 0]
for r in ORDER:
    for caste in CASTE:
        d = reg[(reg.comp == r) & (reg.caste == caste)]
        a.scatter(d.pc1, d.pc2, marker=MARK[r], s=22, facecolor="none" if caste == "worker" else CASTE[caste],
                  edgecolor=CASTE[caste], label=LAB[r] if caste == "queen" else None)
a.set_xlabel(f"PC1 ({S[0]**2/(S**2).sum():.0%})"); a.set_ylabel(f"PC2 ({S[1]**2/(S**2).sum():.0%})")
a.legend(frameon=False, fontsize=7, ncol=2); a.set_title("A  Sections by localization profile", loc="left", fontsize=LABEL_SIZE)
a = ax[0, 1]
x = np.arange(len(ORDER))
for i, caste in enumerate(CASTE):
    t = ss[(ss.caste == caste)].pivot(index="comp", columns="kind", values="nuclear").reindex(ORDER)
    a.plot(x + (i - .5) * .15, t.own_mix, "o", color=CASTE[caste], label=f"{caste} own gene mix")
    a.plot(x + (i - .5) * .15, t.standardised, "x", color=CASTE[caste], ms=7, label=f"{caste} common mix")
    for xi, (u, v) in enumerate(zip(t.own_mix, t.standardised, strict=False)):
        a.plot([xi + (i - .5) * .15] * 2, [u, v], color=CASTE[caste], lw=.6, alpha=.6)
a.set_xticks(x); a.set_xticklabels([LAB[r] for r in ORDER], rotation=40, ha="right")
a.set_ylabel("nuclear fraction"); a.legend(frameon=False, fontsize=7)
a.set_title("B  Gene-mix standardised", loc="left", fontsize=LABEL_SIZE)
a = ax[1, 0]
a.scatter(c.queen, c.worker, s=10, color="#555555")
lim = max(abs(c.queen).max(), abs(c.worker).max()) * 1.05
a.plot([-lim, lim], [-lim, lim], color="grey", lw=.6); a.axhline(0, color="grey", lw=.4); a.axvline(0, color="grey", lw=.4)
a.set_xlabel("queen: optic - KC nuclear fraction"); a.set_ylabel("worker: optic - KC nuclear fraction")
a.set_title(f"C  Per-gene contrast (rho {rho:.2f})", loc="left", fontsize=LABEL_SIZE)
a = ax[1, 1]
for r in ORDER:
    for caste in CASTE:
        d = m[(m.comp == r) & (m.caste == caste)]
        a.scatter(d.calls_per_cell, d.frac_none, marker=MARK[r], s=14, facecolor="none" if caste == "worker" else CASTE[caste],
                  edgecolor=CASTE[caste])
a.set_xlabel("LP calls per cell"); a.set_ylabel("fraction with no pattern")
a.set_title("D  Read depth vs no-pattern calls", loc="left", fontsize=LABEL_SIZE)
for q in ax.ravel():
    q.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_s_bento_regions.png", dpi=300)
(OUT / "summary.txt").write_text("\n".join(lines))
