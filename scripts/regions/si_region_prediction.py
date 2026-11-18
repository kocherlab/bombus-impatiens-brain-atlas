#!/usr/bin/env python3
"""SI figure for the region-prediction analysis (v2).

``region_prediction.py`` writes only CSVs
(``scheme_comparison.csv``, ``per_region_f1.csv``); this script renders them.

The figure carries three claims:

  A. Per-region F1 depends heavily on which vocabulary labels the cells, not
     just on adding spatial context. Cell-type-free geometry beats the
     Tangram-vocabulary composition model on some regions; the
     Leiden-expression vocabulary recovers the Kenyon boundary that Tangram
     composition cannot.
  B. Accuracy alone flatters the model (imbalanced classes; the majority
     baseline is read from scheme_comparison.csv at load time), so balanced
     accuracy and macro-F1 are shown with it, across CV schemes of increasing
     strictness including cross-brain transfer, with a permuted-label control
     showing "no signal" under the same pipeline.
  C. The k sweep, so the neighbourhood size is not a hidden free parameter.

Run:
    PYTHONPATH=src .venv/bin/python scripts/regions/si_region_prediction.py

Reads results/reframe/region_prediction_v3/{scheme_comparison,per_region_f1}.csv
Writes results/figures/si/figS_region_prediction.{png,pdf}

Every panel uses the canonical 17 (src/bimp_st/cluster_labels.py
CANONICAL_ORDER), read from the same grid in region_prediction_v3/.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bimp_st.nine_region_common import display_region
from bimp_st.paths import REPO  # noqa: E402
from bimp_st.si_common import (
    save as _save,
)
from bimp_st.si_common import set_style

# BIMP_REGION_PREDICTION_ROOT (as in region_prediction.py) reads a scratch rerun;
# BIMP_FIG_OUT renders elsewhere (a twice-render reproducibility check)
_ROOT = Path(
    os.environ.get("BIMP_REGION_PREDICTION_ROOT", REPO / "results" / "reframe")
)
DATA_DIR = _ROOT / "region_prediction_v3"
DATA_DIR_3D = _ROOT / "region_prediction_3d_v2"

# Drawn at the printed width (7.2 in canvas, full \linewidth), so these are
# close to the printed point sizes and the print-scale hook barely rescales.
TICK_SIZE = 6.5
LABEL_SIZE = 7.0
LEGEND_SIZE = 6.5

# Optic neuropils first, then the Kenyon cell subdivisions, then the antennal
# lobe (added to the statistical region set alongside the base six -- see
# nine_region_common.STATISTICAL_REGIONS; ellipsoid body and fan-shaped body
# stay out of every fit and so never appear here) -- matches the grouping
# used throughout the SI text (regional_volcanoes etc.).
REGION_ORDER = [
    "Lamina",
    "Medulla",
    "Lobula",
    "Small-type",
    "Large-type",
    "Class II",
    "Antennal lobe",
]
OPTIC = {"Lamina", "Medulla", "Lobula"}
KC = {"Small-type", "Large-type", "Class II"}

TANGRAM_COLOR = "#2A7F8E"  # teal -- Tangram canonical-17
LEIDEN_COLOR = "#C0642A"  # burnt orange -- Leiden expression clusters
GEOM_COLOR = "#8C8C8C"  # grey -- no cell-type information at all
BASELINE_COLOR = "#666666"

SCHEME_ORDER = [
    "random",
    "leave-section",
    "blocked",
    "cross-brain q2w",
    "cross-brain w2q",
    "permuted",
]
SCHEME_LABEL = {
    "random": "random",
    "leave-section": "leave-section",
    "blocked": "blocked",
    "cross-brain q2w": "q → w",
    "cross-brain w2q": "w → q",
    "permuted": "permuted",
}


# ── Panel A: per-region F1, vocabulary x feature-set ────────────────────────


def panel_region_f1(ax, per):
    # All four combos read the same canonical-17 grid as panels B and C.
    combos = [
        (
            "(none)",
            "geometry only",
            "leave-section",
            "Geometry only\n(no cell types)",
            GEOM_COLOR,
            None,
            per,
        ),
        (
            "Tangram canonical-17",
            "neighbourhood k=20",
            "leave-section",
            "Tangram atlas-17,\nneighborhood k=20",
            TANGRAM_COLOR,
            None,
            per,
        ),
        (
            "Leiden expression",
            "own cell type",
            "leave-section",
            "Leiden expression,\nown cell type",
            LEIDEN_COLOR,
            "///",
            per,
        ),
        (
            "Leiden expression",
            "neighbourhood k=20",
            "leave-section",
            "Leiden expression,\nneighborhood k=20",
            LEIDEN_COLOR,
            None,
            per,
        ),
    ]
    n_bars = len(combos)
    width = 0.8 / n_bars
    x = np.arange(len(REGION_ORDER))

    support = {}
    for i, (vocab, feat, scheme, label, color, hatch, data) in enumerate(combos):
        # geometry-only writes one per-region row per seed (its per-region F1 varies
        # by seed, most in the lobula), while Tangram/Leiden write seed 0 only.
        # groupby-mean averages the former and is a no-op for the latter.
        sub = (
            data[
                (data.vocabulary == vocab)
                & (data.features == feat)
                & (data.cv_scheme == scheme)
            ]
            .groupby("region", as_index=True)
            .mean(numeric_only=True)
        )
        vals = [
            sub.loc[r, "f1-score"] if r in sub.index else np.nan for r in REGION_ORDER
        ]
        if i == 1:  # support (test-set n) read off the headline combo
            support = sub["support"].to_dict()
        xpos = x + (i - (n_bars - 1) / 2) * width
        ax.bar(
            xpos,
            vals,
            width=width * 0.92,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.4,
            label=label,
            zorder=2,
        )
        # An F1 of exactly 0 draws no bar and would read as missing data (the
        # Leiden own-cell model never identifies the lobula), so print the zero.
        for xp, v in zip(xpos, vals, strict=False):
            if np.isfinite(v) and v < 0.005:
                # Horizontal and anchored above the baseline so it reads as a
                # value label over its own bar.
                ax.text(
                    xp,
                    0.045,
                    "0.00",
                    ha="center",
                    va="bottom",
                    fontsize=TICK_SIZE - 0.5,
                    color=color,
                    rotation=0,
                    zorder=3,
                )

    # Three groups: optic neuropils, Kenyon cell subdivisions, and the antennal
    # lobe on its own (neither optic nor Kenyon cell tissue). Computed from
    # REGION_ORDER rather than hardcoded dividers.
    groups = [
        ("optic neuropils", [r for r in REGION_ORDER if r in OPTIC]),
        ("Kenyon cell subdivisions", [r for r in REGION_ORDER if r in KC]),
        ("antennal lobe", [r for r in REGION_ORDER if r not in OPTIC and r not in KC]),
    ]
    pos = 0
    for gi, (label, members) in enumerate(groups):
        if not members:
            continue
        lo, hi = pos, pos + len(members)
        pos = hi
        if gi > 0:
            ax.axvline(lo - 0.5, color="#BBBBBB", lw=0.8, zorder=1)
        ax.text(
            (lo + hi - 1) / 2,
            1.015,
            label,
            ha="center",
            fontsize=TICK_SIZE,
            color="#555555",
            style="italic",
            transform=ax.get_xaxis_transform(),
        )

    # These are HELD-OUT test-set counts, not region sizes -- a test split
    # holds only part of each region's cells. Label them
    # as such; a bare "n=" here reads as the size of the region.
    xticklabels = [f"{r}\n(test n={int(support.get(r, 0)):,})" for r in REGION_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=TICK_SIZE)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Per-region F1 score", fontsize=LABEL_SIZE)
    # No title: caption (A) already reads "Per-region F1 under four models."
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        fontsize=LEGEND_SIZE,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.3,
    )


# ── Panel B: CV-scheme comparison, three metrics ─────────────────────────────


def _scheme_means(sch):
    """accuracy / balanced_accuracy / macro_f1, mean over seeds, per
    (vocabulary, scheme), for the k=20 neighbourhood feature set; the
    permuted-label control (leave-section + blocked averaged together,
    since that is all it was run under) is folded in as its own scheme."""
    main = sch[sch.features == "neighbourhood k=20"]
    g = (
        main.groupby(["vocabulary", "cv_scheme"])[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ]
        .mean()
        .reset_index()
    )

    perm = sch[sch.features.str.startswith("permuted")]
    gp = (
        perm.groupby("vocabulary")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )
    gp["cv_scheme"] = "permuted"
    return pd.concat([g, gp], ignore_index=True)


def chance_baselines(majority_baseline, n_classes=None):
    """Score of a classifier that always predicts the majority class, per
    metric. Accuracy is the majority-class rate p; balanced accuracy is
    1/K (recall 1 on one class, 0 on the rest); macro-F1 is that class's
    F1, 2p/(1+p), divided by K. The permuted-label control lands on these."""
    p = majority_baseline
    n_classes = n_classes or len(REGION_ORDER)
    return {
        "accuracy": p,
        "balanced_accuracy": 1.0 / n_classes,
        "macro_f1": (2 * p / (1 + p)) / n_classes,
    }


def panel_cv_schemes(axes, sch, majority_baseline):
    means = _scheme_means(sch)
    baselines = chance_baselines(majority_baseline)
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
    ]
    vocabs = [
        ("Tangram canonical-17", TANGRAM_COLOR),
        ("Leiden expression", LEIDEN_COLOR),
    ]
    x = np.arange(len(SCHEME_ORDER))
    width = 0.35

    for ax, (col, mlabel) in zip(axes, metrics, strict=False):
        for i, (vocab, color) in enumerate(vocabs):
            sub = means[means.vocabulary == vocab].set_index("cv_scheme")
            vals = [sub.loc[s, col] if s in sub.index else np.nan for s in SCHEME_ORDER]
            xpos = x + (i - 0.5) * width
            ax.bar(
                xpos,
                vals,
                width=width * 0.92,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
                label=vocab.replace("canonical", "atlas") if ax is axes[0] else None,
            )
        # Each metric gets its own baseline: the score of always predicting
        # the majority class. One 0.317 line on all three overstated the
        # floor for balanced accuracy and macro-F1, whose floor is far lower.
        base = baselines[col]
        ax.axhline(
            base,
            color=BASELINE_COLOR,
            lw=1.1,
            ls="--",
            zorder=1,
            label="majority-class predictor" if ax is axes[0] else None,
        )
        ax.text(
            len(SCHEME_ORDER) - 0.5,
            base + 0.012,
            f"{base:.3f}",
            ha="right",
            va="bottom",
            fontsize=TICK_SIZE - 0.5,
            color=BASELINE_COLOR,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SCHEME_LABEL[s] for s in SCHEME_ORDER],
            fontsize=TICK_SIZE,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylim(0, 1.0)
        ax.set_title(mlabel, fontsize=LABEL_SIZE, fontweight="bold")
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Score", fontsize=LABEL_SIZE)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    handles, labels = axes[0].get_legend_handles_labels()
    # Directly under the rotated scheme labels, centred on the three
    # sub-panels (anchored in figure coordinates by compose()).
    return handles, labels
    # No title and no footnote: caption (B) names the metrics and CV schemes and
    # defines q -> w / w -> q (train on one brain, test on the other) and permuted
    # (within-section label shuffle, averaged over leave-section and blocked), so
    # the artwork carries no definitions of its own.


# ── Panel C: k sweep ─────────────────────────────────────────────────────────


def panel_k_sweep(ax, sch):
    ks = sch[
        (sch.features.str.startswith("neighbourhood"))
        & (sch.cv_scheme == "leave-section")
    ]
    vocabs = [
        ("Tangram canonical-17", TANGRAM_COLOR),
        ("Leiden expression", LEIDEN_COLOR),
    ]
    metrics = [
        ("macro_f1", "macro-F1", "-", "o"),
        ("balanced_accuracy", "balanced accuracy", "--", "s"),
    ]

    for vocab, color in vocabs:
        sub = ks[ks.vocabulary == vocab]
        for col, _mname, ls, marker in metrics:
            g = sub.groupby("k")[col].agg(["mean", "std"]).reindex([10, 20, 40])
            ax.errorbar(
                g.index,
                g["mean"],
                yerr=g["std"],
                color=color,
                ls=ls,
                marker=marker,
                markersize=3,
                lw=1.1,
                capsize=1.8,
                zorder=2,
            )

    ax.set_xticks([10, 20, 40])
    ax.set_xlabel("Neighborhood size, k", fontsize=LABEL_SIZE)
    ax.set_ylabel("Score (leave-section CV)", fontsize=LABEL_SIZE)
    ax.set_ylim(0.5, 1.0)
    # No title: caption (C) already reads "Neighborhood-size sweep, k = 10,
    # 20, 40."
    ax.tick_params(labelsize=TICK_SIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # One legend, not two. Two adjacent frameless legends (colour and line
    # style) read as a single four-item list in which nothing indicates that
    # the two halves mean different things. Cross the two factors instead, so
    # each entry names exactly the line it points at.
    handles = [
        Line2D(
            [],
            [],
            color=color,
            ls=ls,
            marker=marker,
            markersize=3,
            lw=1.1,
            label=f"{vname}, {mname}",
        )
        for vocab, color, vname in [
            (vocabs[0][0], vocabs[0][1], "Tangram-17"),
            (vocabs[1][0], vocabs[1][1], "Leiden"),
        ]
        for col, mname, ls, marker in metrics
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=LEGEND_SIZE,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.3,
        borderaxespad=0.1,
    )


# ── Panel D: does a true 3D neighbourhood beat the in-section 2D one? ───────

D_2D_COLOR = TANGRAM_COLOR  # same teal as the 2D Tangram-17 series elsewhere
D_3D_COLOR = "#7A3F9E"  # purple -- distinct from every other series here


def panel_3d_vs_2d(ax, summary_3d):
    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("balanced_accuracy_mean", None, "Balanced\naccuracy"),
        ("macro_f1_mean", None, "Macro-F1"),
    ]
    sub = summary_3d[(summary_3d.cv_scheme == "leave-section")]
    row_2d = sub[sub.features == "neighbourhood 2D k=20"].iloc[0]
    row_3d = sub[sub.features == "neighbourhood 3D k=20"].iloc[0]

    x = np.arange(len(metrics))
    w = 0.32
    vals_2d = [row_2d[m] for m, _, _ in metrics]
    vals_3d = [row_3d[m] for m, _, _ in metrics]
    err_2d = [
        row_2d["accuracy_std"] if e == "accuracy_std" else 0 for _, e, _ in metrics
    ]
    err_3d = err_2d  # only accuracy_std is tracked per-metric in the CSV

    ax.bar(
        x - w / 2,
        vals_2d,
        w,
        yerr=err_2d,
        color=D_2D_COLOR,
        label="2D, in-section",
        capsize=1.8,
        error_kw=dict(lw=0.8),
        zorder=2,
    )
    ax.bar(
        x + w / 2,
        vals_3d,
        w,
        yerr=err_3d,
        color=D_3D_COLOR,
        label="3D, cross-section",
        capsize=1.8,
        error_kw=dict(lw=0.8),
        zorder=2,
    )

    # The same per-metric majority-class baselines as panel B, each drawn
    # only across its own metric's pair of bars.
    baselines = chance_baselines(float(row_2d["majority_baseline"]))
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    for xi, key in zip(x, keys, strict=True):
        ax.hlines(
            baselines[key],
            xi - w - 0.06,
            xi + w + 0.06,
            color=BASELINE_COLOR,
            ls="--",
            lw=0.9,
            zorder=3,
            label="majority-class predictor" if xi == x[0] else None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in metrics], fontsize=TICK_SIZE)
    # Bars start at zero: a truncated axis exaggerates the 2D-vs-3D gap.
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Score (leave-section CV)", fontsize=LABEL_SIZE)
    # No title: caption (D) already reads "3D neighbourhoods from registered
    # adjacent sections score below in-section 2D..."
    ax.tick_params(labelsize=TICK_SIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # In the empty band above the bars (they top out below 0.8).
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(x) for x in ("2D, in-section", "3D, cross-section")]
    order += [i for i in range(len(lab)) if i not in order]
    ax.legend(
        [h[i] for i in order],
        [lab[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.1,
    )


# ── Figure ───────────────────────────────────────────────────────────────────


def compose(per, sch, majority_baseline, summary_3d, output_dir):
    # 7.2 in wide: the printed width, so text is authored at printed size.
    fig = plt.figure(figsize=(7.2, 7.6))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 0.82],
        hspace=0.7,
        left=0.085,
        right=0.985,
        top=0.955,
        bottom=0.045,
    )

    ax_a = fig.add_subplot(outer[0])
    panel_region_f1(ax_a, per)

    # Row 2: B (three metrics) with its legend directly beneath.
    b_gs = outer[1].subgridspec(1, 3, wspace=0.10)
    axes_b = [fig.add_subplot(b_gs[i]) for i in range(3)]
    handles, labels = panel_cv_schemes(axes_b, sch, majority_baseline)

    # Row 3: C and D side by side, each a single plot of similar width.
    bottom = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
    ax_c = fig.add_subplot(bottom[0])
    panel_k_sweep(ax_c, sch)
    ax_d = fig.add_subplot(bottom[1])
    panel_3d_vs_2d(ax_d, summary_3d)

    # B's legend: one row, under the rotated scheme labels of the middle
    # sub-panel, so it reads as belonging to all three.
    axes_b[1].legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=3,
        fontsize=LEGEND_SIZE,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.6,
    )

    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [(ax_c, "C", "Neighborhood size"), (ax_d, "D", "3D neighborhoods")],
        width_frac=1.0,
    )
    # A carries its region-group labels, and B its metric titles, just above
    # the axes; lift those headers clear of them.
    add_panel_headers(fig, [(ax_a, "A", "Regional F1")], width_frac=1.0, dy_pts=13)
    add_panel_headers(
        fig, [(axes_b[0], "B", "Validation schemes")], width_frac=1.0, dy_pts=15
    )

    fig_dir = Path(os.environ.get("BIMP_FIG_OUT", "results/figures/si"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, fig_dir, "figS_region_prediction")
    # no CreationDate, so a re-render of the same artwork is byte-identical
    fig.savefig(
        fig_dir / "figS_region_prediction.pdf", metadata={"CreationDate": None}
    )
    print(f"    Saved {fig_dir / 'figS_region_prediction.pdf'}", flush=True)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        per = pd.read_csv(DATA_DIR / "per_region_f1.csv")
        per["region"] = per["region"].map(display_region)  # CSV names, e.g. 'KC-NC'
        sch = pd.read_csv(DATA_DIR / "scheme_comparison.csv")
        summary_3d = pd.read_csv(DATA_DIR_3D / "summary_2d_vs_3d.csv")
        majority_baseline = float(sch["majority_baseline"].iloc[0])
        print(f"majority-class baseline = {majority_baseline:.4f}", flush=True)
        print(
            f"per-metric baselines = {chance_baselines(majority_baseline)}",
            flush=True,
        )
        compose(per, sch, majority_baseline, summary_3d, DATA_DIR)
        print("Done.", flush=True)
