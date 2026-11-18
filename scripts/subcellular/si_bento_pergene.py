#!/usr/bin/env python3
"""SI figure: per-gene caste-differential subcellular localization + confound controls.

Shows the per-gene evidence for caste differences in nuclear localization over
the tested panel genes; the per-region and per-cell-type figures are summaries
only.

Panels
    A  per-gene queen-worker difference in nuclear fraction (all 100 genes)
    B  same for cytoplasmic fraction
    C  volcano: effect size vs BH-adjusted significance, both patterns
    D  confound: region nuclear fraction vs nucleus:cell area ratio
    E  confound: region nuclear fraction vs transcript density
    F  confound: run-level shift in undetected ("no pattern") calls
    G  control: does the queen/worker split survive composition normalization?

Statistics
    Replicate unit = Xenium region (15 queen, 13 worker). Two-sided
    Mann-Whitney U on region-level mean localization fractions, Benjamini-
    Hochberg corrected across the 100 panel genes within each pattern.

    Note: the 28 regions are sections of one queen and one worker brain,
    collected on two different instrument runs (queen 20251204, worker
    20260115). They are technical replicates; n = 1 animal per caste. Panels
    D-G exist so the reader can judge how much of the effect is run-level.

Requires scripts/subcellular/bento_pergene_stats.py to have been run.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/subcellular/si_bento_pergene.py
"""

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import QUEEN_COLOR, WORKER_COLOR, add_panel_headers
from bimp_st.style import set_pub_style

REPO = Path(__file__).resolve().parents[2]  # resolved from this file, so a worktree
# writes to its own results/ rather than the main checkout's
DIFF = REPO / "results/bento/differential"
OUT = Path("results/figures/si/fig_s_bento_pergene.png")

NS_COLOR = "#c9c9c9"
ALPHA = 0.05
PATTERNS = ["cell_edge", "cytoplasmic", "nuclear", "nuclear_edge"]

PANEL_KW_BASE = dict(fontweight="bold", va="bottom", ha="right")


def _shorten(name, maxlen=30):
    s = str(name)
    for tag in (" isoform X1", " isoform X2", " isoform X3", " isoform X4"):
        s = s.replace(tag, "")
    s = s.replace("LOW QUALITY PROTEIN_ ", "").replace(" protein", "")
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def normalized_stats(rgm: pd.DataFrame) -> pd.DataFrame:
    """Re-test each pattern after renormalizing to the detected (non-none) total.

    With ``rgm`` on the classified denominator (bento_pergene_stats), ``none`` is
    the share of classified calls flagged only "none", so this re-tests each
    pattern among the four real patterns. It removes the run-level difference in
    that share; what remains is a shift in the composition of called patterns.
    """
    det = 1.0 - rgm["none"]
    rows = []
    for pat in PATTERNS:
        vals = rgm[pat] / det
        tmp = rgm[["gene_name", "brain"]].assign(v=vals)
        # A region x gene whose classified calls are all "none" has no
        # detected pattern to renormalize to (0/0); leave it out.
        tmp = tmp[np.isfinite(tmp["v"])]
        out = []
        for gene, d in tmp.groupby("gene_name"):
            q = d[d.brain == "queen"]["v"].values
            w = d[d.brain == "worker"]["v"].values
            _, p = stats.mannwhitneyu(q, w, alternative="two-sided")
            out.append({"gene": gene, "diff": q.mean() - w.mean(), "pval": p})
        o = pd.DataFrame(out)
        o["padj"] = multipletests(o["pval"], method="fdr_bh")[1]
        o["pattern"] = pat
        rows.append(o)
    return pd.concat(rows, ignore_index=True)


def loo_range(loo, pattern):
    """Min-max of each headline count across the 28 leave-one-section-out refits."""
    d = loo[loo["pattern"] == pattern]

    def rng(col):
        lo, hi = int(d[col].min()), int(d[col].max())
        return str(lo) if lo == hi else f"{lo}–{hi}"

    return (
        f"Leave-one-section-out (28 refits): {rng('n_sig')} significant, "
        f"{rng('n_queen')} queen-higher, {rng('n_worker')} worker-higher"
    )


def panel_pergene(ax, res, pattern, label, highlight=()):
    """Ordered per-gene caste difference, significant genes colored by direction."""
    d = res.sort_values(f"{pattern}_diff").reset_index(drop=True)
    diff = d[f"{pattern}_diff"].values
    sig = (d[f"{pattern}_padj"] < ALPHA).values

    colors = np.where(
        sig & (diff > 0),
        QUEEN_COLOR,
        np.where(sig & (diff < 0), WORKER_COLOR, NS_COLOR),
    )
    ax.bar(np.arange(len(d)), diff * 100, color=colors, width=0.85, edgecolor="none")
    ax.axhline(0, color="#333", lw=0.6)

    # No title: the caption gives this panel's n_q+n_w/len(d) breakdown
    # (A: 50/100, 31 queen-higher, 19 worker-higher; B: 72/100, 67/5).
    # The leave-one-out range is also in the caption (see main()).
    ax.set_xlabel(
        f"Panel genes, ranked by {label.lower()} difference (n=100)", fontsize=9
    )
    ax.set_ylabel("Queen − Worker\n(percentage points)", fontsize=9)
    ax.set_xlim(-1, len(d))
    ax.tick_params(labelsize=8)
    # Headroom so the callout labels stay inside the axes.
    lo, hi = diff.min() * 100, diff.max() * 100
    span = hi - lo
    ax.set_ylim(lo - 0.26 * span, hi + 0.22 * span)

    # Label the highlighted genes, plus the two extremes.
    to_label = set(highlight) | {d.iloc[0]["gene"], d.iloc[-1]["gene"]}
    for i, row in d.iterrows():
        if row["gene"] not in to_label:
            continue
        y = row[f"{pattern}_diff"] * 100
        up = y > 0
        ax.annotate(
            _shorten(row["gene"], 44),
            xy=(i, y),
            xytext=(i + (-14 if up else 14), y + (0.1 if up else -0.1) * (hi - lo)),
            fontsize=7,
            ha="right" if up else "left",
            va="bottom" if up else "top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#666", shrinkA=0, shrinkB=1),
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=QUEEN_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=WORKER_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=NS_COLOR),
    ]
    ax.legend(
        handles,
        ["Queen higher (FDR<0.05)", "Worker higher (FDR<0.05)", "n.s."],
        fontsize=7,
        frameon=False,
        loc="upper left",
    )


def panel_volcano(ax, res):
    """Effect size vs significance for both patterns — shows the full distribution."""
    for pat, marker, name in [
        ("nuclear", "o", "Nuclear"),
        ("cytoplasmic", "^", "Cytoplasmic"),
    ]:
        diff = res[f"{pat}_diff"].values * 100
        padj = res[f"{pat}_padj"].values
        sig = padj < ALPHA
        colors = np.where(
            sig & (diff > 0),
            QUEEN_COLOR,
            np.where(sig & (diff < 0), WORKER_COLOR, NS_COLOR),
        )
        ax.scatter(
            diff,
            -np.log10(padj),
            c=colors,
            marker=marker,
            s=26,
            alpha=0.85,
            linewidths=0.4,
            edgecolors="white",
            label=name,
        )

    ax.axhline(-np.log10(ALPHA), color="#666", ls="--", lw=0.7)
    ax.text(
        ax.get_xlim()[0],
        -np.log10(ALPHA),
        " FDR 0.05",
        fontsize=7,
        va="bottom",
        ha="left",
        color="#666",
    )
    ax.axvline(0, color="#333", lw=0.6)
    ax.set_xlabel("Queen − Worker (percentage points)", fontsize=9)
    ax.set_ylabel("−log₁₀ BH-adjusted p", fontsize=9)
    # No title: caption (C) already says "Effect size vs significance."
    ax.tick_params(labelsize=8)

    shape_handles = [
        plt.Line2D([], [], marker="o", ls="", color="#888", markersize=5),
        plt.Line2D([], [], marker="^", ls="", color="#888", markersize=5),
    ]
    ax.legend(
        shape_handles,
        ["Nuclear", "Cytoplasmic"],
        fontsize=7,
        frameon=False,
        loc="lower right",
    )


def panel_confound(ax, reg, xcol, xlabel, title):
    """Region-level nuclear fraction vs a segmentation/depth covariate."""
    for caste, color in [("queen", QUEEN_COLOR), ("worker", WORKER_COLOR)]:
        d = reg[reg["brain"] == caste]
        ax.scatter(
            d[xcol],
            d["nuclear"] * 100,
            color=color,
            s=34,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            label=f"{caste.capitalize()} (n={len(d)})",
        )

    rho, p = stats.spearmanr(reg[xcol], reg["nuclear"])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Region mean nuclear fraction (%)", fontsize=9)
    # No title: the x label names the covariate and the caption gives rho/p.
    ax.legend(fontsize=7, frameon=False, loc="best")
    ax.tick_params(labelsize=8)


def panel_detection(ax, reg):
    """Run-level shift in how often bento assigns no pattern at all."""
    rng = np.random.default_rng(0)
    for i, (caste, color) in enumerate(
        [("queen", QUEEN_COLOR), ("worker", WORKER_COLOR)]
    ):
        d = reg[reg["brain"] == caste]["none"].values * 100
        x = i + rng.uniform(-0.09, 0.09, len(d))
        ax.scatter(
            x,
            d,
            color=color,
            s=34,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.hlines(d.mean(), i - 0.24, i + 0.24, color="#333", lw=1.6, zorder=4)

    q = reg[reg["brain"] == "queen"]["none"]
    w = reg[reg["brain"] == "worker"]["none"]
    _, p = stats.mannwhitneyu(q, w, alternative="two-sided")
    print(
        f"  panel F Mann-Whitney p={p:.1e} (caption text, dropped from title)",
        flush=True,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Queen\n(run 20251204)\nn={len(q)}", f"Worker\n(run 20260115)\nn={len(w)}"],
        fontsize=8,
    )
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Region mean “no pattern” calls (%)", fontsize=9)
    # No title: caption (F) carries the description and the statistics.
    ax.tick_params(axis="y", labelsize=8)


def panel_normcontrol(ax, res, norm):
    """Raw vs composition-normalized queen/worker split of significant genes."""
    labels, q_counts, w_counts = [], [], []
    for pat, nice in [("nuclear", "Nuclear"), ("cytoplasmic", "Cytoplasmic")]:
        sig = res[f"{pat}_padj"] < ALPHA
        labels.append(f"{nice}\nraw")
        q_counts.append(int((sig & (res[f"{pat}_diff"] > 0)).sum()))
        w_counts.append(int((sig & (res[f"{pat}_diff"] < 0)).sum()))

        n = norm[norm["pattern"] == pat]
        nsig = n["padj"] < ALPHA
        labels.append(f"{nice}\nnormalized")
        q_counts.append(int((nsig & (n["diff"] > 0)).sum()))
        w_counts.append(int((nsig & (n["diff"] < 0)).sum()))

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, q_counts, w, color=QUEEN_COLOR, label="Queen higher")
    ax.bar(x + w / 2, w_counts, w, color=WORKER_COLOR, label="Worker higher")
    for xi, (a, b) in enumerate(zip(q_counts, w_counts, strict=False)):
        ax.text(xi - w / 2, a + 0.8, str(a), ha="center", fontsize=7.5)
        ax.text(xi + w / 2, b + 0.8, str(b), ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Significant genes (FDR<0.05)", fontsize=9)
    # No title: the x tick labels name the four conditions; caption (G).
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, max(q_counts + w_counts) * 1.25)


def confound_summary(reg: pd.DataFrame) -> str:
    """One-line summary of how the two runs differ mechanically."""
    q = reg[reg["brain"] == "queen"]
    w = reg[reg["brain"] == "worker"]

    def mwu(col):
        return stats.mannwhitneyu(q[col], w[col], alternative="two-sided")[1]

    parts = [
        f"cell area {q.cell_area.mean():.1f} vs {w.cell_area.mean():.1f} µm² "
        f"(p={mwu('cell_area'):.0e})",
        f"nucleus:cell ratio {q.nuc_frac_area.mean():.3f} vs "
        f"{w.nuc_frac_area.mean():.3f} (p={mwu('nuc_frac_area'):.3f})",
        f"transcript density {q.density.mean():.1f} vs {w.density.mean():.1f} "
        f"µm⁻² (p={mwu('density'):.0e})",
    ]
    return (
        "The two runs differ systematically in segmentation and detection: "
        + "; ".join(parts)
        + ".\nYet the global mean nuclear fraction does NOT differ by caste "
        f"({q.nuclear.mean() * 100:.1f}% vs {w.nuclear.mean() * 100:.1f}%, "
        f"p={mwu('nuclear'):.2f}) — the per-gene effects in A-C are "
        "redistribution among genes, not a global shift."
    )


def main():
    set_pub_style(savefig_dpi=300)

    res = pd.read_csv(DIFF / "pergene_caste_stats.csv")
    rgm = pd.read_csv(DIFF / "region_gene_means.csv")
    cov = pd.read_csv(DIFF / "region_covariates.csv")
    loo = pd.read_csv(DIFF / "loo_stability.csv")

    reg = (
        rgm.groupby(["brain", "region"])[["nuclear", "cytoplasmic", "none"]]
        .mean()
        .reset_index()
        .merge(cov, on=["brain", "region"], how="inner")
    )
    print(f"{len(res)} genes; {len(reg)} regions with covariates", flush=True)

    norm = normalized_stats(rgm)

    fig = plt.figure(figsize=(15.5, 12.0))
    gs = fig.add_gridspec(
        3,
        4,
        hspace=0.58,
        wspace=0.34,
        left=0.06,
        right=0.985,
        top=0.915,
        bottom=0.185,
    )

    axA = fig.add_subplot(gs[0, :2])
    axB = fig.add_subplot(gs[0, 2:])
    axC = fig.add_subplot(gs[1, :2])
    axD = fig.add_subplot(gs[1, 2])
    axE = fig.add_subplot(gs[1, 3])
    axF = fig.add_subplot(gs[2, :2])
    axG = fig.add_subplot(gs[2, 2:])

    named = ["tachykinins", "short neuropeptide F"]
    panel_pergene(axA, res, "nuclear", "Nuclear", highlight=named)
    panel_pergene(axB, res, "cytoplasmic", "Cytoplasmic", highlight=named)
    print(
        f"  leave-one-out (caption text): nuclear {loo_range(loo, 'nuclear')}; "
        f"cytoplasmic {loo_range(loo, 'cytoplasmic')}",
        flush=True,
    )
    panel_volcano(axC, res)
    panel_confound(
        axD,
        reg,
        "nuc_frac_area",
        "Region mean nucleus:cell area ratio",
        "Segmentation geometry",
    )
    panel_confound(
        axE, reg, "density", "Region mean transcripts per µm²", "Transcript density"
    )
    panel_detection(axF, reg)
    panel_normcontrol(axG, res, norm)

    add_panel_headers(
        fig,
        [
            (axA, "A", "Nuclear fraction"),
            (axB, "B", "Cytoplasmic fraction"),
            (axC, "C", "Volcano"),
            (axD, "D", "Segmentation"),
            (axE, "E", "Transcript density"),
            (axF, "F", "Detection by run"),
            (axG, "G", "Normalization control"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["fig_s_bento_pergene"],
    )

    # No figure or panel titles: the caption names the figure and carries the
    # per-panel numbers, statistics and leave-one-out range. confound_summary()
    # is printed to the run log so those caption numbers can be re-derived.
    print(
        "  confound summary (now caption text): "
        + confound_summary(reg).replace("\n", " "),
        flush=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
