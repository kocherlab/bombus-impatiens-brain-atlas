#!/usr/bin/env python
"""Supplementary figure: subcellular transcript localization (the one SI figure).

Regions (one brain per caste, descriptive):
A  mean ELLA radial profile of significant genes, both brains averaged: the three
   Kenyon compartments against the four optic and antennal-lobe regions pooled
B  bento nuclear fraction per region among classified calls, queen and worker
C  odds of a nuclear call, optic and antennal lobe vs Kenyon cells, as read depth, cell
   area and nucleus area are added to the model
Caste, per gene (15 queen vs 13 worker capture regions, run-confounded):
D, E  queen - worker difference in nuclear and cytoplasmic fraction, ranked
F  genes with a queen-higher or worker-higher fraction among classified calls
   and among called patterns
Caste contrasts are effect sizes (differences of region means) with no test:
the regions are sections of one brain per caste.

Every bento fraction is among CLASSIFIED calls (cell-gene rows with at least one
pattern flag; a pattern takes precedence over "none"); see
regroup_common.fractions and bento_pergene_stats.classified_region_gene_means.
Gold and blue mark caste only.
Inputs (results/): ella/<brain>/*/{ella_results.csv,output/lam_est.pkl},
ella_regions/...,
bento/regions_summary/{region_profiles,size_control_glm}.csv,
bento/differential/{pergene_caste_stats,region_gene_means}.csv
Outputs: results/figures/si/figS_subcellular_regions.{png,pdf};
bento/differential/{normalization_sign_counts,detection_by_run}.csv (sign counts
by normalisation, detection per run)
"""

import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bimp_st.paths import REPO  # noqa: E402
from bimp_st.si_common import QUEEN_COLOR, WORKER_COLOR, set_style

sys.path.insert(0, str(REPO / "scripts"))
import subcellular.si_bento_pergene as pg  # noqa: E402

RES = REPO / "results"
DIFF = RES / "bento/differential"
# BIMP_SUBCELL_OUT writes the figure and the count tables to a scratch dir
# (render-identity checks) instead of the shared results/.
SCRATCH = os.environ.get("BIMP_SUBCELL_OUT")
DIFF_OUT = Path(SCRATCH) if SCRATCH else DIFF
DIFF_OUT.mkdir(parents=True, exist_ok=True)
set_style()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
    }
)
INK, MUTED = "#1a1a1a", "#666666"
KC_COL = {"Small-type": "#4a1486", "Large-type": "#807dba", "Class II": "#b9b4d9"}
OPT_COL = "#1f8a4c"
X = np.linspace(0.001, 0.999, 100)
KC_GROUPS = [("ella", t) for t in KC_COL]
OPT_GROUPS = [
    ("ella_regions", t) for t in ("Lamina", "Medulla", "Lobula", "Antennal lobe")
]


def mean_profile(root, typ):
    """Mean over sections (per brain), then over the two brains, of the
    per-gene-normalised fitted intensity of significant genes."""
    per_brain = []
    for brain in ("queen", "worker"):
        secs = []
        for d in sorted((RES / root / brain).glob("*")):
            f, p = d / "ella_results.csv", d / "output/lam_est.pkl"
            if not (f.exists() and p.exists()):
                continue
            res = pd.read_csv(f)
            lam = pickle.load(open(p, "rb"))["weighted_lam_dict"]
            r = res[res.type == typ].reset_index(drop=True)
            if typ not in lam or len(r) != len(lam[typ]):
                continue
            L = np.asarray(lam[typ])[r.sig.to_numpy()]
            if len(L) >= 3:
                secs.append((L / np.maximum(L.mean(1, keepdims=True), 1e-12)).mean(0))
        per_brain.append(np.mean(secs, 0))
    return np.mean(per_brain, 0)


def header(letter, x, y, title):
    """Panel letter and bold title above a panel, in figure coordinates."""
    fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top")
    fig.text(x + 0.035, y - 0.003, title, fontsize=7.2, fontweight="bold", va="top")


fig = plt.figure(figsize=(7.2, 6.4))
TOP, BOT = (0.60, 0.33), (0.08, 0.36)  # (y0, height) of the two rows

# ---------------- A: ELLA profiles
axB = fig.add_axes([0.07, TOP[0], 0.25, TOP[1]])
for root, t in KC_GROUPS:
    axB.plot(X, mean_profile(root, t), color=KC_COL[t], lw=1.8, label=t)
opt = np.mean([mean_profile(r, t) for r, t in OPT_GROUPS], 0)
axB.plot(X, opt, color=OPT_COL, lw=2.4, label="optic and\nantennal lobe")
axB.set_xlim(0, 1)
axB.set_xticks([0, 0.5, 1])
axB.set_xticklabels(["0\nnucleus", "0.5", "1\nedge"])
axB.set_xlabel("radius r", labelpad=1)
axB.set_ylabel("ELLA intensity,\nsignificant genes", labelpad=2)
axB.set_ylim(0.15, 1.5)
axB.legend(
    frameon=False, fontsize=6.0, loc="lower left", handlelength=1.2, labelspacing=0.25
)
axB.spines[["top", "right"]].set_visible(False)
header("A", 0, 0.985, "Where transcripts peak")


# ---------------- B: bento nuclear share by region (classified calls)
prof = pd.read_csv(
    REPO / "results/bento/regions_summary/region_profiles.csv"
).set_index(["caste", "comp"])
ORDER = ["KC-IC", "KC-NC", "KC-OC", "lamina", "medulla", "lobula", "al"]
NAME = ["Small", "Large", "Class II", "Lamina", "Medulla", "Lobula", "AL"]
axC = fig.add_axes([0.42, TOP[0], 0.25, TOP[1]])
x = np.arange(len(ORDER))
for i, (caste, col) in enumerate((("queen", QUEEN_COLOR), ("worker", WORKER_COLOR))):
    axC.bar(
        x + (i - 0.5) * 0.38,
        [prof.loc[(caste, r), "frac_nuclear"] for r in ORDER],
        0.36,
        color=col,
        label=caste,
    )
axC.set_xticks(x)
axC.set_xticklabels(NAME, rotation=40, ha="right")
axC.set_ylabel("nuclear fraction of\nclassified calls")
axC.legend(frameon=False, fontsize=6.3, loc="upper left")
axC.spines[["top", "right"]].set_visible(False)
axC.axvline(2.5, color="#bbbbbb", lw=0.6)
axC.text(
    1.0,
    axC.get_ylim()[1] * 1.0,
    "Kenyon cells",
    ha="center",
    va="bottom",
    fontsize=6.2,
    color=KC_COL["Small-type"],
)
axC.text(
    4.5,
    axC.get_ylim()[1] * 1.0,
    "optic, antennal lobe",
    ha="center",
    va="bottom",
    fontsize=6.2,
    color=OPT_COL,
)
header("B", 0.35, 0.985, "Nuclear share by region")


# ---------------- C: controls
glm = pd.read_csv(REPO / "results/bento/regions_summary/size_control_glm.csv")
models = [
    "group only",
    "+ depth",
    "+ depth + cell area",
    "+ depth + cell area + nucleus area",
]
labs = ["none", "+ read depth", "+ cell area", "+ nucleus area"]
axD = fig.add_axes([0.82, TOP[0], 0.16, TOP[1]])
for i, (brain, col) in enumerate((("queen", QUEEN_COLOR), ("worker", WORKER_COLOR))):
    q = glm[glm.brain == brain].set_index("model").reindex(models)
    y = np.arange(len(models)) + (i - 0.5) * 0.24
    axD.errorbar(
        q.optic_vs_kc_OR,
        y,
        xerr=[q.optic_vs_kc_OR - q.lo, q.hi - q.optic_vs_kc_OR],
        fmt="o",
        ms=3.5,
        color=col,
        capsize=1.5,
        lw=1.1,
    )
axD.axvline(1, color="#888888", lw=0.6)
axD.set_yticks(range(len(models)))
axD.set_yticklabels(labs)
axD.invert_yaxis()
axD.set_xlim(0.8, 3.5)
axD.set_xlabel("odds ratio, nuclear call\n(optic, AL vs Kenyon)")
axD.tick_params(axis="y", labelsize=6.3)
axD.set_ylabel("adjusted for", labelpad=2)
axD.spines[["top", "right"]].set_visible(False)
header("C", 0.7, 0.985, "Not depth or cell size")


# ---------------- D-F: caste, per gene (run-confounded)
res = pd.read_csv(REPO / "results/bento/differential/pergene_caste_stats.csv")
rgm = pd.read_csv(REPO / "results/bento/differential/region_gene_means.csv")


def normalized_diffs(rgm):
    """Per gene: queen - worker mean of the region fractions after renormalizing
    each region x gene to the four called patterns (dropping "none" alone)."""
    det = 1.0 - rgm["none"]
    out = []
    for pat in ("nuclear", "cytoplasmic"):
        t = rgm[["gene_name", "brain"]].assign(v=rgm[pat] / det)
        # a region x gene whose classified calls are all "none" has no called
        # pattern to renormalize to (0/0)
        t = t[np.isfinite(t["v"])]
        m = t.groupby(["gene_name", "brain"])["v"].mean().unstack()
        d = (m["queen"] - m["worker"]).rename("diff").reset_index()
        out.append(d.rename(columns={"gene_name": "gene"}).assign(pattern=pat))
    return pd.concat(out, ignore_index=True)


def raw_diffs(rgm):
    """Per gene: queen - worker mean of the classified-call region fractions."""
    out = []
    for pat in ("nuclear", "cytoplasmic"):
        m = rgm.groupby(["gene_name", "brain"])[pat].mean().unstack()
        d = (m["queen"] - m["worker"]).rename("diff").reset_index()
        out.append(d.rename(columns={"gene_name": "gene"}).assign(pattern=pat))
    return pd.concat(out, ignore_index=True)


def sign_counts(diffs):
    return {
        pat: (int((g["diff"] > 0).sum()), int((g["diff"] < 0).sum()), len(g))
        for pat, g in diffs.groupby("pattern")
    }


norm = normalized_diffs(rgm)
# Leave one capture region out: the range of each sign count over the 28 refits.
loo = {}
for dropped in sorted(rgm["region"].unique()):
    sub = rgm[rgm["region"] != dropped]
    for stage, fn in (("raw", raw_diffs), ("normalized", normalized_diffs)):
        for pat, (q, w, _) in sign_counts(fn(sub)).items():
            loo.setdefault((pat, stage), []).append((q, w))
counts = []
for stage, diffs in (("raw", raw_diffs(rgm)), ("normalized", norm)):
    for pat, (q, w, n) in sign_counts(diffs).items():
        lq, lw = zip(*loo[(pat, stage)], strict=True)
        counts.append(
            {
                "pattern": pat,
                "stage": stage,
                "n_genes": n,
                "queen_higher": q,
                "worker_higher": w,
                "median_diff_pp": 100
                * float(diffs[diffs["pattern"] == pat]["diff"].median()),
                "loo_queen_min": min(lq),
                "loo_queen_max": max(lq),
                "loo_worker_min": min(lw),
                "loo_worker_max": max(lw),
            }
        )
counts = pd.DataFrame(counts)
counts.to_csv(DIFF_OUT / "normalization_sign_counts.csv", index=False)
# The run difference the normalization removes: the share of classified calls that
# are "none" only. The unclassified share (no flag at all) is reported alongside.
reg = rgm.groupby(["brain", "region"])[["none", "unclassified"]].mean().reset_index()
det = {}
for col, key in (("none", "none"), ("unclassified", "unclassified")):
    q, w = reg[reg.brain == "queen"][col], reg[reg.brain == "worker"][col]
    det |= {
        f"queen_mean_{key}_pct": 100 * q.mean(),
        f"worker_mean_{key}_pct": 100 * w.mean(),
        f"{key}_mwu_p": stats.mannwhitneyu(q, w, alternative="two-sided")[1],
    }
det |= {
    "n_queen_regions": int((reg.brain == "queen").sum()),
    "n_worker_regions": int((reg.brain == "worker").sum()),
}
pd.DataFrame([det]).to_csv(DIFF_OUT / "detection_by_run.csv", index=False)
print(counts.to_string(index=False), flush=True)
print(pd.DataFrame([det]).T.to_string(), flush=True)

named = ["tachykinins", "short neuropeptide F"]


def panel_effect(ax, res, pattern, highlight=()):
    """Per-gene queen - worker difference, ranked; colored by direction only."""
    d = res.sort_values(f"{pattern}_diff").reset_index(drop=True)
    diff = d[f"{pattern}_diff"].values
    colors = np.where(diff > 0, QUEEN_COLOR, WORKER_COLOR)
    ax.bar(np.arange(len(d)), diff * 100, color=colors, width=0.85, edgecolor="none")
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xlim(-1, len(d))
    lo, hi = diff.min() * 100, diff.max() * 100
    span = hi - lo
    ax.set_ylim(lo - 0.26 * span, hi + 0.22 * span)
    to_label = set(highlight) | {d.iloc[0]["gene"], d.iloc[-1]["gene"]}
    for i, row in d.iterrows():
        if row["gene"] not in to_label:
            continue
        y = row[f"{pattern}_diff"] * 100
        up = y > 0
        ax.annotate(
            pg._shorten(row["gene"], 44),
            xy=(i, y),
            xytext=(i + (-14 if up else 14), y + (0.1 if up else -0.1) * (hi - lo)),
            fontsize=6,
            ha="right" if up else "left",
            va="bottom" if up else "top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#666", shrinkA=0, shrinkB=1),
        )


axE = fig.add_axes([0.07, BOT[0], 0.33, BOT[1]])
axF = fig.add_axes([0.47, BOT[0], 0.33, BOT[1]])
axG = fig.add_axes([0.87, BOT[0], 0.12, BOT[1]])
for ax, pat in ((axE, "nuclear"), (axF, "cytoplasmic")):
    panel_effect(ax, res, pat, highlight=named)
    ax.set_xlabel(f"panel genes, ranked (n={len(res)})", fontsize=7)
    ax.set_ylabel("queen − worker\n(percentage points)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
axE.legend(
    [plt.Rectangle((0, 0), 1, 1, color=c) for c in (QUEEN_COLOR, WORKER_COLOR)],
    ["queen higher", "worker higher"],
    fontsize=6,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.86),
    handlelength=1.0,
)
order = [
    ("nuclear", "raw"),
    ("nuclear", "normalized"),
    ("cytoplasmic", "raw"),
    ("cytoplasmic", "normalized"),
]
cidx = counts.set_index(["pattern", "stage"])
qc = [int(cidx.loc[k, "queen_higher"]) for k in order]
wc = [int(cidx.loc[k, "worker_higher"]) for k in order]
xg = np.arange(len(order))
bw = 0.38
axG.bar(xg - bw / 2, qc, bw, color=QUEEN_COLOR)
axG.bar(xg + bw / 2, wc, bw, color=WORKER_COLOR)
for xi, (a, b) in enumerate(zip(qc, wc, strict=True)):
    axG.text(xi - bw / 2, a + 0.8, str(a), ha="center", fontsize=5.5)
    axG.text(xi + bw / 2, b + 0.8, str(b), ha="center", fontsize=5.5)
axG.set_ylim(0, max(qc + wc) * 1.25)
axG.set_xticks(range(4))
axG.set_xticklabels(
    ["nuc.\nall", "nuc.\npatt.", "cyto.\nall", "cyto.\npatt."], fontsize=6
)
axG.set_ylabel("genes", fontsize=7)
axG.tick_params(axis="y", labelsize=6.5)
axG.spines[["top", "right"]].set_visible(False)
header("D", 0, 0.47, "Nuclear fraction, queen − worker")
header("E", 0.4, 0.47, "Cytoplasmic fraction, queen − worker")
header("F", 0.81, 0.47, "Among patterns")

# The top row inherits set_style's larger defaults; match the bottom row.
for ax in (axB, axC, axD):
    ax.tick_params(labelsize=6.5)
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_fontsize(7)
# In E the tachykinin and sNPF callouts land on each other; lift tachykinins.
for t in axF.texts:
    if t.get_text().startswith("tachykinins"):
        x, y = t.xyann
        t.xyann = (x + 14, y + 1.2)

out = Path(SCRATCH) if SCRATCH else RES / "figures/si"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "figS_subcellular_regions.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "figS_subcellular_regions.pdf", bbox_inches="tight")
print("wrote figS_subcellular_regions", flush=True)
