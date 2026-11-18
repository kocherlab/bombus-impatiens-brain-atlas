#!/usr/bin/env python3
"""Atlas quality figure: does Xenium recover the snRNA-seq reference structure?

Every panel here answers "is the atlas any good?" using evidence that is
independent of the Tangram cell-type labels being compared -- they never
touch a Tangram label on the snRNA side of the comparison. Hand-drawn
anatomy (A), de-novo unsupervised structure (B), within-modality
specificity (C). Panel D (per-gene held-out imputation against a random
gene-pair null) is not drawn in the figure: its bars repeat
Fig. tangram_qc A. ``plot_panel_d`` documents how the
random-pair null is computed; its median is read from
results/reframe/loocv_null_0af3056_20261020/summary.json.

Cross-modal concordance (circular by construction -- Tangram labels on both
sides) is not shown here; it lives in figS_panel_concordance. This figure
reports three non-circular tests.

Every panel plots its null/baseline alongside the observed value; a
similarity number with no null next to it is not evidence of anything.

Data sources (all pre-computed by the noncircular-battery / audit jobs):
  results/reframe/noncircular/testA1_markerscore.csv   (Panel A)
  results/reframe/noncircular/testA1_null_draws.csv    (Panel A)
  results/reframe/noncircular/testB_matching.csv       (Panel B)
  results/reframe/noncircular/testC_tau.csv            (Panel C)
  results/reframe/noncircular/testC_summary.csv        (Panel C)
  results/tangram/loocv_0af3056_20261020/combined_clustered_loocv_scores.csv  (Panel D)

Panel D's null (cross-gene cosine baseline: how similar do two unrelated
genes' observed Xenium spatial patterns look, from shared density/baseline
structure alone) has no precomputed source and is derived here directly from
the Xenium expression matrix (no Tangram involved) and cached to
``results/figures/quality/loocv_null_cache.csv``.

    PYTHONPATH=src:scripts .venv/bin/python scripts/atlas/si_atlas_quality.py
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import PANEL_CROSSWALK  # noqa: E402
from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import add_panel_headers, set_style

# ── Config ───────────────────────────────────────────────────────────────
NONCIRC = Path("results/reframe/noncircular")
# canonical-17, the working vocabulary (src/bimp_st/cluster_labels.py
# CANONICAL_ORDER). The canonical-15 run of the same analysis lives in
# spatial_coherence_c15/ and must not be read from here.
SPATIAL = Path("results/reframe/spatial_coherence_c17")
LOOCV_CSV = Path(
    "results/tangram/loocv_0af3056_20261020/combined_clustered_loocv_scores.csv"
)
COMBINED_H5AD = Path("pipelines/combined_atlas/data/combined_clustered.h5ad")
GENE_MAP_CSV = PANEL_CROSSWALK
OUTPUT_DIR = Path("results/figures/quality")  # LOO-CV cache
FIG_DIR = Path("results/figures/si")  # the registered figure
LOOCV_NULL_CACHE = OUTPUT_DIR / "loocv_null_cache.csv"

# ── Status palette (reserved; not reused for anything else) ────────────────
NOT_CIRCULAR = "#3E7C59"  # green  -- independent of the labels under test
FAILS_NULL = "#B5523B"  # brick  -- honest negative result, still not circular
# (not amber: gold is the queen colour in every figure)
OBSERVED = "#2F4E96"  # slate blue -- house PN-ad blue, reused for "the data"
OBSERVED_FILL = "#AEBBD8"  # pale OBSERVED: panel D bars, under the two lines
NULLC = "#ABABAB"  # grey   -- null / baseline
INK = "#2B2B2B"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out)
        print(f"    Saved {out}", flush=True)
    plt.close(fig)


# ── Panel D null: cross-gene cosine baseline (no Tangram involved) ─────────


def compute_loocv_null(genes_v43_lc):
    """How similar do two unrelated genes' observed Xenium patterns look,
    from shared cell-density/baseline structure alone? Computed directly
    from measured Xenium expression -- no Tangram, no imputation, no
    cell-type label. Cached because it needs a 2 GB backed h5ad read.
    """
    if LOOCV_NULL_CACHE.exists():
        cached = pd.read_csv(LOOCV_NULL_CACHE)
        if set(cached["gene"]) == set(genes_v43_lc):
            return cached
    import anndata as ad
    import scipy.sparse as sp

    mapping = read_panel_crosswalk()
    mapping["v43_lc"] = mapping["v43_id"].str.lower()
    v43_to_np31 = dict(zip(mapping["v43_lc"], mapping["np31_id"], strict=True))

    a = ad.read_h5ad(COMBINED_H5AD, backed="r")
    var_by_np31 = {v.split("|")[0]: v for v in a.var_names}

    matched_var, matched_lc = [], []
    for g in genes_v43_lc:
        np31 = v43_to_np31.get(g)
        if np31 in var_by_np31:
            matched_var.append(var_by_np31[np31])
            matched_lc.append(g)

    sub = a[:, matched_var].to_memory()
    X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
    X = X.astype(np.float64)
    # combined_clustered.h5ad .X is log1p; the LOO-CV cosines being compared
    # against were computed in linear space (tangram_mapping.cross_validate
    # explicitly expm1's this same file before scoring) -- match that so the
    # null is on the same scale as the observed statistic.
    X = np.expm1(X)
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    Xn = X / norms
    cos_mat = Xn.T @ Xn
    n = cos_mat.shape[0]
    per_gene_null = (cos_mat.sum(axis=1) - 1.0) / (n - 1)

    out = pd.DataFrame({"gene": matched_lc, "null_cross_gene_cosine": per_gene_null})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LOOCV_NULL_CACHE, index=False)
    return out


# ── Panel A: KC markers separate calyx from optic anatomy ──────────────────


def plot_panel_a(ax):
    ms = pd.read_csv(NONCIRC / "testA1_markerscore.csv")
    null = pd.read_csv(NONCIRC / "testA1_null_draws.csv")["null_auroc"].values

    obs_all = ms.loc[ms["split"] == "all", "auroc_set"].iloc[0]
    sections = ms[ms["split"].str.startswith("section:")]["auroc_set"].values

    ax.hist(
        null,
        bins=40,
        color=NULLC,
        edgecolor="none",
        alpha=0.85,
        label=f"null: 1000 matched\n10-gene sets\n(mean {null.mean():.3f})",
    )
    ax.axvline(
        obs_all,
        color=OBSERVED,
        lw=2.2,
        zorder=5,
        label=f"observed: {obs_all:.3f}",
    )

    ax.set_xlabel("set-level AUROC\n(KC markers, calyx vs. optic)")
    ax.set_ylabel("null draws")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)  # headroom for the legend
    # No title: caption (A) already reads "Anatomy recovery."
    # No stats block: the caption gives the effect and its P value, and the
    # legend already names the observed value and the null mean.
    print(
        f"  A: +{obs_all - null.mean():.3f} over null; sections median "
        f"{np.median(sections):.3f} range {sections.min():.3f}-"
        f"{sections.max():.3f}",
        flush=True,
    )
    # Upper left: the null histogram is empty above ~50 draws left of 0.88.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=8,
        handlelength=1.2,
    )


# ── Panel B: spatial coherence of the transferred labels ───────────────────


def plot_panel_b(ax):
    """Spatial coherence of the transferred labels.

    Uses spatial coherence because it appeals to something the reference and
    the mapping do not have: where the cells physically are. (A
    cluster-matching test -- de novo snRNA clusters vs Xenium clusters against
    a wrong-partition null -- is not used: cluster_correspondence_scaling.py shows
    its null swings from 1.00 to 18.6 with gene scaling alone while the
    observed count stays at 11-15, so it does not test correspondence.)
    """
    s = pd.read_csv(SPATIAL / "summary.csv")
    order = ["tangram argmax (canonical-17)", "leiden (panel expression)"]
    s = (
        s.set_index("label_set")
        .reindex([o for o in order if o in set(s["label_set"])])
        .reset_index()
    )
    nice = {
        # Short tick labels; the caption carries "per cell" / "upper reference".
        "tangram argmax (canonical-17)": "transferred\nlabels",
        "leiden (panel expression)": "panel's own\nclusters",
    }

    x = np.arange(len(s))
    w = 0.36
    ax.bar(
        x - w / 2, s["join_count_weighted"], width=w, color=OBSERVED, label="observed"
    )
    ax.bar(
        x + w / 2,
        s["null_weighted"],
        width=w,
        color=NULLC,
        label="null: labels permuted\nwithin section",
    )
    for i, row in s.iterrows():
        ax.text(
            i - w / 2,
            row["join_count_weighted"] + 0.012,
            f"{row['join_count_weighted']:.3f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=OBSERVED,
        )
        ax.text(
            i + w / 2,
            row["null_weighted"] + 0.012,
            f"{row['null_weighted']:.3f}",
            ha="center",
            fontsize=8,
            color="#555555",
        )
        # Ratio sits just under the bar top, inside the axes: above the taller
        # bar it collided with the legend.
        # centred on the observed bar (i - w/2), not on the group centre --
        # at the group centre it straddles the bar edge and gets half-clipped
        ax.text(
            i - w / 2,
            row["join_count_weighted"] * 0.5,
            f"{row['ratio']:.1f}×",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([nice.get(v, v) for v in s["label_set"]])
    ax.set_ylabel("fraction of spatial kNN edges\njoining same-label cells")
    # No title: caption (B) already reads "Spatial organization."
    ax.set_ylim(0, max(s["join_count_weighted"]) * 1.55)
    # upper left: the tall bar in this panel sits on the right
    ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.2)


# ── Panel C: tau specificity -- curated passes, de-novo-x-de-novo fails ────


def plot_panel_c(ax):
    s = pd.read_csv(NONCIRC / "testC_summary.csv")
    order = [
        "prior MAJOR (snRNA curated x Xenium leiden_original)",
        "prior FINE (snRNA curated fine x Xenium leiden)",
        "de novo x de novo (snRNA leiden res 0.3)",
        "de novo x de novo (snRNA leiden res 0.5)",
        "de novo x de novo (snRNA leiden res 1.0)",
    ]
    # "major"/"fine" as in the caption; the counts are the snRNA curated class
    # numbers read from the summary (k_snrna), not a Tangram vocabulary.
    kmaj, kfine = (int(s.set_index("pairing").loc[o, "k_snrna"]) for o in order[:2])
    print(f"  C: curated major k={kmaj}, fine k={kfine}", flush=True)
    labels = [
        "curated\nmajor",
        "curated\nfine",
        "de novo\nres 0.3",
        "de novo\nres 0.5",
        "de novo\nres 1.0",
    ]
    s = s.set_index("pairing").loc[order].reset_index()
    # Only the curated-label pairings pass cleanly. The de-novo-x-de-novo
    # variant does not replicate at any resolution (p = 0.05 / 0.054 / 0.29,
    # with res-0.3 sitting right at the boundary) -- shown as the negative
    # result it is, never as a marginal pass.
    passed = s["pairing"].str.startswith("prior")

    x = np.arange(len(s))
    w = 0.36
    colors = [NOT_CIRCULAR if p else FAILS_NULL for p in passed]
    ax.bar(x - w / 2, s["spearman_rho"], width=w, color=colors)
    # Pass/fail is carried by the bar colour (keyed here), so each bar's
    # label is just its p-value.

    ax.bar(
        x + w / 2,
        s["null_wrongpartition_mean"],
        width=w,
        color=NULLC,
    )
    for i, (_, row) in enumerate(s.iterrows()):
        ax.text(
            i,
            max(row["spearman_rho"], row["null_wrongpartition_mean"])
            + 0.02
            + (0.06 if i == 3 else 0),  # stagger: 3 and 4 sit level
            f"p={row['p_wrongpartition']:.3f}",
            ha="center",
            fontsize=8,
            color=(NOT_CIRCULAR if passed.iloc[i] else FAILS_NULL),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ (tau specificity,\nsnRNA vs. Xenium)")
    # No title: caption (C) already reads "Gene specificity."
    ax.set_ylim(
        0, max(s["spearman_rho"].max(), s["null_wrongpartition_mean"].max()) + 0.22
    )
    from matplotlib.patches import Patch

    handles = [
        Patch(color=NOT_CIRCULAR, label="observed ρ, clears null"),
        Patch(color=FAILS_NULL, label="observed ρ, fails null"),
        Patch(color=NULLC, label="null: 500 wrong\npartitions (mean)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
    )


# ── Panel D: LOO-CV held-out gene imputation ────────────────────────────────


def plot_panel_d(ax):
    df = pd.read_csv(LOOCV_CSV).sort_values("cosine_sim").reset_index(drop=True)
    # Use the median over all random gene pairs (loocv_null.py). The curve is
    # the per-gene mean, whose median is a different statistic.
    import json

    NULL_PAIR_MEDIAN = json.load(
        open("results/reframe/loocv_null_0af3056_20261020/summary.json")
    )["null_a_observed_vs_observed"]["median"]
    null = compute_loocv_null(df["gene"].tolist())
    null = null.set_index("gene").reindex(df["gene"]).reset_index()

    x = np.arange(len(df))
    # One fill: the bars are already sorted, so a colour ramp would only
    # re-encode rank.
    ax.bar(x, df["cosine_sim"], width=1.0, color=OBSERVED_FILL, edgecolor="none")

    null_sorted_for_display = null["null_cross_gene_cosine"].values
    ax.plot(
        x,
        np.sort(null_sorted_for_display),
        color=NULLC,
        lw=1.6,
        zorder=5,
        label=(
            "null: each gene vs. the other 99 genes' observed\n"
            f"patterns (mean); median over gene pairs {NULL_PAIR_MEDIAN:.3f}"
        ),
    )

    med, mean = df["cosine_sim"].median(), df["cosine_sim"].mean()
    ax.axhline(
        med,
        color=OBSERVED,
        ls="--",
        lw=1.3,
        label=f"observed median = {med:.2f}\n(mean {mean:.2f})",
    )

    ax.set_xlim(-1, len(df))
    ax.set_xlabel(f"gene, ranked by held-out cosine similarity (n = {len(df)})")
    ax.set_ylabel("cosine similarity\n(predicted vs. observed)")
    # No title: caption (D) already reads "Held-out gene imputation."
    ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.2)


# ── Compose ──────────────────────────────────────────────────────────────


def compose_figure():
    fig = plt.figure(figsize=(13.0, 3.9))
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[0.9, 0.95, 1.45],
        wspace=0.42,
        top=0.88,
        bottom=0.16,
        left=0.055,
        right=0.985,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b)
    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel_c(ax_c)

    add_panel_headers(
        fig,
        [
            (ax_a, "A", "Anatomy recovery"),
            (ax_b, "B", "Spatial organization"),
            (ax_c, "C", "Gene specificity"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["figS_atlas_quality"],
    )

    # The scope caveat is in the LaTeX caption only, not on the figure.

    save(fig, "figS_atlas_quality")


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        print("Composing atlas quality figure...", flush=True)
        compose_figure()
        print("Done.", flush=True)
