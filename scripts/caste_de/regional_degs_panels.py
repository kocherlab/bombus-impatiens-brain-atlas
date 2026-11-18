"""Panel drawing for si_regional_degs.py:
one function per panel or panel element.

See si_regional_degs.py (the composer) for the figure and how to render it.
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from caste_de.regional_degs_data import (  # noqa: E402
    CASTE_TEST_REGIONS,
    COMMON_REGIONS,
    _pooled_label_order,
    _pretty_region,
    assign_regions,
    gene_short_name,
)
from matplotlib.patches import Patch

from bimp_st.cluster_labels import (
    get_pooled_color,
    get_pooled_display,
)
from bimp_st.nine_region_common import (
    REGION_COLORS,
)
from bimp_st.si_common import (
    CASTE_DIVERGING_CMAP,
    LABEL_SIZE,
    LEGEND_SIZE,
    QUEEN_COLOR,
    TICK_SIZE,
    TITLE_SIZE,
    WORKER_COLOR,
)

BRAIN_COLORS = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}

# One palette across the SI: the shared nine_region_common colors (also used by
# the Hill-diversity figure), so a region keeps its color from figure to figure
# and Lamina / KC-NC get distinct colours.
REGION_PALETTE = {
    **{r: REGION_COLORS[r] for r in COMMON_REGIONS},
    "unassigned": "#BBBBBB",
}


def _plot_domain_heatmap(ax, ovr_results, pb_domain_df):
    """Panel A: expression heatmap of top domain markers."""
    # Collect top 5 marker genes per region (from queen)
    brain_key = "queen"
    if brain_key not in ovr_results:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    marker_genes = []
    marker_regions = []
    for region in sorted(ovr_results[brain_key].keys()):
        df = ovr_results[brain_key][region]
        top = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0)].head(5)
        if len(top) == 0:
            top = df.nlargest(3, "logfoldchanges")
        for _, row in top.iterrows():
            if row["names"] not in marker_genes:
                marker_genes.append(row["names"])
                marker_regions.append(region)

    if not marker_genes:
        ax.text(
            0.5, 0.5, "No markers", ha="center", va="center", transform=ax.transAxes
        )
        return

    # Build mean expression matrix from pseudobulk
    # Average queen + worker per region
    regions = sorted(pb_domain_df["region"].unique())
    mat = np.zeros((len(marker_genes), len(regions)))
    for j, region in enumerate(regions):
        rdf = pb_domain_df[pb_domain_df["region"] == region]
        for i, gene in enumerate(marker_genes):
            if gene in rdf.columns:
                mat[i, j] = rdf[gene].mean()

    # Z-score per gene (row)
    row_means = mat.mean(axis=1, keepdims=True)
    row_stds = mat.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1
    z_mat = (mat - row_means) / row_stds

    im = ax.imshow(z_mat, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(range(len(marker_genes)))
    ax.set_yticklabels(
        [gene_short_name(g) for g in marker_genes],
        fontsize=TICK_SIZE - 1,
    )

    # Color y-tick labels by source region
    for i, region in enumerate(marker_regions):
        color = REGION_PALETTE.get(region, "#666")
        ax.get_yticklabels()[i].set_color(color)

    plt.colorbar(im, ax=ax, shrink=0.7, label="z-score")
    ax.set_title("Domain-specific markers\n(z-scored expression)")


def _plot_domain_pca(ax, pca_df, pca_model):
    """Panel B: PCA of slice-level pseudobulk by domain."""
    if pca_df is None or len(pca_df) == 0 or pca_model is None:
        ax.text(
            0.5, 0.5, "No PCA data", ha="center", va="center", transform=ax.transAxes
        )
        return

    brain_markers = {"queen": "o", "worker": "s"}
    for region in sorted(pca_df["region"].unique()):
        color = REGION_PALETTE.get(region, "#CCCCCC")
        for brain, marker in brain_markers.items():
            mask = (pca_df["region"] == region) & (pca_df["brain"] == brain)
            subset = pca_df[mask]
            if len(subset) == 0:
                continue
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                c=color,
                marker=marker,
                s=50,
                alpha=0.8,
                edgecolors="k",
                linewidths=0.3,
                label=f"{region} ({brain[0].upper()})",
            )

    var = pca_model.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} var)")
    ax.set_title("Pseudobulk PCA by domain")
    ax.legend(
        fontsize=LEGEND_SIZE - 1, ncol=2, loc="best", framealpha=0.9, edgecolor="none"
    )


def _plot_ovr_volcanos(axes, brain_results, regions):
    """Panel C: volcano plots for 1-vs-rest DEGs per region.

    Styled after the panel-2i volcanoes of Yao et al. Nat Neurosci 2025:
    three-tier coloring (NS gray, FDR<0.01 pink, |FC|>1 red),
    black italic gene labels with thin leader lines.
    """
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    FDR_THRESH = 0.01
    FC_THRESH = 1.0  # log2FC threshold for "strong" hits

    COL_NS = "#BBBBBB"
    COL_SIG = "#FF9999"  # light red / pink
    COL_STRONG = "#CC0000"  # dark red

    for i, region in enumerate(regions):
        ax = axes[i]
        if region not in brain_results:
            ax.set_title(_pretty_region(region), fontsize=TITLE_SIZE, fontweight="bold")
            continue

        df = brain_results[region].copy()
        pvals = np.clip(df["pvals_adj"].values, 1e-300, None)
        neg_log_p = -np.log10(pvals)
        fc = df["logfoldchanges"].values

        sig = df["pvals_adj"] < FDR_THRESH
        strong = sig & (np.abs(fc) > FC_THRESH)
        sig_only = sig & ~strong
        ns = ~sig

        # Layer 1: NS — small gray
        ax.scatter(
            fc[ns],
            neg_log_p[ns],
            s=6,
            alpha=0.4,
            color=COL_NS,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        # Layer 2: significant but |FC| ≤ 1 — pink/salmon
        ax.scatter(
            fc[sig_only],
            neg_log_p[sig_only],
            s=8,
            alpha=0.6,
            color=COL_SIG,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        # Layer 3: significant AND |FC| > 1 — dark red
        ax.scatter(
            fc[strong],
            neg_log_p[strong],
            s=12,
            alpha=0.8,
            color=COL_STRONG,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )

        # Dashed FDR threshold line
        ax.axhline(
            -np.log10(FDR_THRESH),
            color="k",
            ls="--",
            lw=0.6,
            alpha=0.4,
        )

        # Label top genes — black italic with leader lines
        n_label = 5
        top_pos = df[strong & (fc > 0)].nlargest(
            n_label,
            "logfoldchanges",
        )
        top_neg = df[strong & (fc < 0)].nsmallest(
            min(3, (strong & (fc < 0)).sum()),
            "logfoldchanges",
        )
        to_label = pd.concat([top_pos, top_neg])

        texts = []
        for _, row in to_label.iterrows():
            x = row["logfoldchanges"]
            y = -np.log10(max(row["pvals_adj"], 1e-300))
            txt = ax.text(
                x,
                y,
                row["gene_short"],
                fontsize=6,
                fontstyle="italic",
                color="k",
                ha="left" if x > 0 else "right",
                va="bottom",
                zorder=10,
            )
            texts.append(txt)

        if texts:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(
                    arrowstyle="-",
                    color="k",
                    lw=0.3,
                    alpha=0.5,
                ),
                expand=(1.4, 1.6),
                force_text=(0.6, 0.9),
                only_move={"text": "xy"},
            )

        # Axis labels
        ax.set_xlabel("log₂(FC)", fontsize=LABEL_SIZE)
        if i == 0:
            ax.set_ylabel("-log₁₀(FDR)", fontsize=LABEL_SIZE)

        ax.set_title(_pretty_region(region), fontsize=TITLE_SIZE, fontweight="bold")

        # Legend in first panel
        if i == 0:
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markersize=3.5,
                    markerfacecolor=COL_NS,
                    label="NS",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markersize=4,
                    markerfacecolor=COL_SIG,
                    label="FDR < 0.01",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markersize=5,
                    markerfacecolor=COL_STRONG,
                    label="log₂FC > 1",
                ),
            ]
            ax.legend(
                handles=legend_elements,
                fontsize=LEGEND_SIZE - 1,
                loc="upper left",
                framealpha=0.9,
                edgecolor="none",
                handletextpad=0.3,
                borderpad=0.3,
            )


def _plot_domain_violins(ax_left, ax_right, ovr_results, adatas):
    """Panel D: violin plots of top domain-specific genes across regions."""
    brain_key = "queen"
    if brain_key not in ovr_results:
        return

    # Collect top 1 marker per region
    top_genes = {}
    for region in sorted(ovr_results[brain_key].keys()):
        df = ovr_results[brain_key][region]
        up = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0)]
        if len(up) > 0:
            top_genes[region] = up.iloc[0]
        elif len(df) > 0:
            top_genes[region] = df.nlargest(1, "logfoldchanges").iloc[0]

    if not top_genes:
        return

    # Build combined adata with region labels
    combined_obs = []
    combined_X = []
    for brain, adata in adatas.items():
        regions = assign_regions(adata)
        mask = regions != "unassigned"
        sub = adata[mask.values]
        obs = sub.obs.copy()
        obs["region"] = regions[mask].values
        obs["brain"] = brain
        combined_obs.append(obs)
        combined_X.append(
            sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        )

    all_obs = pd.concat(combined_obs, ignore_index=True)
    all_X = np.vstack(combined_X)
    gene_names = list(adatas["queen"].var_names)
    all_regions = sorted(all_obs["region"].unique())

    items = list(top_genes.items())
    # Split across two axes
    half = (len(items) + 1) // 2
    for ax, gene_items in [(ax_left, items[:half]), (ax_right, items[half:])]:
        if not gene_items:
            ax.set_visible(False)
            continue

        positions = []
        pos = 0
        for region, row in gene_items:
            gene_name = row["names"]
            if gene_name not in gene_names:
                continue
            gi = gene_names.index(gene_name)
            expr_by_region = []
            region_labels = []
            for r in all_regions:
                r_mask = all_obs["region"] == r
                vals = all_X[r_mask.values, gi]
                expr_by_region.append(vals)
                region_labels.append(r)

            bp = ax.violinplot(
                expr_by_region,
                positions=list(range(pos, pos + len(all_regions))),
                showmedians=True,
                widths=0.7,
            )
            for j, pc in enumerate(bp["bodies"]):
                r = region_labels[j]
                c = REGION_PALETTE.get(r, "#CCC")
                pc.set_facecolor(c)
                pc.set_alpha(0.6 if r != region else 0.9)
                pc.set_edgecolor(c)
            for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
                if key in bp:
                    bp[key].set_color("k")
                    bp[key].set_linewidth(0.5)

            # Gene label
            mid = pos + len(all_regions) / 2 - 0.5
            ax.text(
                mid,
                ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                gene_short_name(gene_name),
                ha="center",
                va="bottom",
                fontsize=TICK_SIZE,
                fontweight="bold",
                color=REGION_PALETTE.get(region, "#333"),
            )
            positions.extend(range(pos, pos + len(all_regions)))
            pos += len(all_regions) + 1

        ax.set_xticks(positions)
        tick_labels = region_labels * len(gene_items)
        ax.set_xticklabels(
            tick_labels[: len(positions)],
            rotation=45,
            ha="right",
            fontsize=TICK_SIZE - 2,
        )
        ax.set_ylabel("Expression", fontsize=LABEL_SIZE)
        ax.set_title("Top domain markers", fontsize=TITLE_SIZE)


# Neutral ink for the lollipop figure, whose only colour channel is the
# queen/worker direction of effect (see _plot_region_lollipop).
STEM_COLOR = "#9a9a9a"
LABEL_COLOR = "#333333"


def _plot_regional_shift(ax, comp_df, cap=2.0, near_zero=0.005):
    """Panel C: the caste contrast of panel B, as log2FC worker/queen.

    It sits beside panel B because it is the difference between the two bars
    B draws side by side, and the expression-only-vote circularity disclosure
    then lives once, in this figure's caption.

    Computed from ``comp_df`` -- the same table panel B plots -- so the two
    panels cannot disagree. It is deliberately not read back from
    results/figures/fig2/regional_abundance.csv, which would make this figure
    depend on the `fig:caste_comparison` analysis step having been rerun.

    ``near_zero``: below this share of a region's cells in both castes, a
    log2FC is noise with a loud colour -- a tiny residual per-cluster leakage
    (e.g. lamina cells voting Kenyon) yields a near-saturated cell for a
    population indistinguishable from zero. Such cells are greyed.
    """
    df = comp_df[comp_df["region"] != "unassigned"].copy()
    # Single source of truth for the canonical order.
    region_order = COMMON_REGIONS
    regions = [r for r in region_order if r in df["region"].unique()]
    cats = _pooled_label_order(df["cluster"].unique().tolist())

    prop = (
        df.set_index(["region", "brain", "cluster"])["proportion"]
        .groupby(level=[0, 1, 2])
        .first()
    )
    M = np.full((len(cats), len(regions)), np.nan)
    absent = np.zeros_like(M, dtype=bool)
    for j, region in enumerate(regions):
        for i, cat in enumerate(cats):
            q = float(prop.get((region, "queen", cat), 0.0))
            w = float(prop.get((region, "worker", cat), 0.0))
            if q < near_zero and w < near_zero:
                absent[i, j] = True
                continue
            # Same epsilon on both sides, so a category present in one caste
            # and truly absent in the other saturates rather than dividing by
            # zero; the near_zero gate above has already removed the cases
            # where that would be meaningless.
            M[i, j] = np.clip(np.log2((w + 1e-9) / (q + 1e-9)), -cap, cap)

    cmap = CASTE_DIVERGING_CMAP.copy()
    cmap.set_bad("#dedede")
    im = ax.imshow(M, cmap=cmap, vmin=-cap, vmax=cap, aspect="auto")
    for i in range(len(cats)):
        for j in range(len(regions)):
            if absent[i, j]:
                ax.text(
                    j,
                    i,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#8a8a8a",
                    style="italic",
                )

    # Rows and columns are two different label vocabularies (cell type vs.
    # anatomical region) that happen to share a name -- "KC" (row short code,
    # cluster_labels._POOLED_SHORT) sits right above "KC-IC"/"KC-NC"/"KC-OC"
    # (columns). Say so explicitly rather than relying on the reader to infer
    # it from position alone.
    # As axis titles, not corner notes: a note above the axes collided with
    # the panel header.
    ax.set_xlabel("Region (columns)", fontsize=LABEL_SIZE - 1)
    ax.set_ylabel("Cell type (rows)", fontsize=LABEL_SIZE - 1)

    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(
        [_pretty_region(r) for r in regions],
        fontsize=TICK_SIZE,
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(len(cats)))
    # Full display names, wrapped, not short codes: here the cell-type and
    # anatomical-region vocabularies share one plot, and "KC"/"OL" are also
    # region codes. The "(all subtypes)" marker says which categories are
    # pools. Panel B's legend carries the same marker and needs the wide
    # gutter (wspace 1.0); if they collide, widen the gutter or shrink the
    # legend rather than abbreviating.
    ax.set_yticklabels(
        [
            get_pooled_display(c).replace(" (all subtypes)", "\n(all subtypes)")
            for c in cats
        ],
        fontsize=TICK_SIZE,
    )
    for t, c in zip(ax.get_yticklabels(), cats, strict=False):
        t.set_color(get_pooled_color(c) if c != "Mixed" else "#9a9a9a")
        t.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, len(regions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03, aspect=14)
    cb.set_label(
        f"log$_2$FC, capped $\\pm${cap:.0f}\n(gold=queen, blue=worker)",
        fontsize=LEGEND_SIZE,
    )
    cb.ax.tick_params(labelsize=TICK_SIZE)
    cb.outline.set_visible(False)
    # The n/a key sits directly under the colour bar it extends.
    cb.ax.legend(
        handles=[
            Patch(
                facecolor="#dedede",
                edgecolor="none",
                label=f"n/a: <{near_zero * 100:.1f}% of\ncells, both castes",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(-0.6, -0.04),
        frameon=False,
        fontsize=TICK_SIZE,
        handlelength=1.1,
        handleheight=1.1,
        handletextpad=0.4,
        borderaxespad=0.0,
    )


def _plot_density(ax, density_df):
    """Panel A: paired bar chart of cell counts by region."""
    # Single source of truth for the canonical order.
    region_order = COMMON_REGIONS
    all_regions = [
        r
        for r in region_order
        if r in density_df["region"].unique() and r != "unassigned"
    ]

    x = np.arange(len(all_regions))
    width = 0.38

    for i, brain in enumerate(["queen", "worker"]):
        bd = density_df[density_df["brain"] == brain]
        counts = []
        for region in all_regions:
            row = bd[bd["region"] == region]
            counts.append(int(row["n_cells"].iloc[0]) if len(row) else 0)
        ax.bar(
            x + (i - 0.5) * width,
            counts,
            width,
            label=brain.title(),
            color=BRAIN_COLORS[brain],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_pretty_region(r) for r in all_regions],
        rotation=20,
        ha="right",
        fontsize=TICK_SIZE,
    )
    ax.set_ylabel("Number of cells", fontsize=LABEL_SIZE)
    # No title: caption (A) already reads "Cells per region."
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    # Thousands as "k" tick labels: a scientific "1e4" offset text sits at
    # the top-left of the axes, exactly where the panel letter goes.
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plot_composition(ax, comp_df, top_n=8):
    """Panel B: stacked bars of pooled cell type proportions per region.

    Cell types are the expression-only per-cluster vote over the combined
    Tangram labels (``_composition_labels``; deliberately not the anatomy-guided
    vote — that would be circular in a region-conditioned panel) — already
    pooled to the coarse display groups, so the fine ``OL_*``/glia/KC subtypes
    the 101-gene panel cannot resolve never appear here. Groups beyond *top_n*
    in total abundance are aggregated into 'Other' for legibility (a no-op at
    the current group count); bars are ordered by the canonical POOLED_ORDER.
    """
    df = comp_df[comp_df["region"] != "unassigned"].copy()

    # Single source of truth for the canonical order.
    region_order = COMMON_REGIONS
    regions = [r for r in region_order if r in df["region"].unique()]
    brains = ["queen", "worker"]

    totals = df.groupby("cluster")["proportion"].sum().sort_values(ascending=False)
    keep = _pooled_label_order(totals.head(top_n).index.tolist())
    df["plot_label"] = df["cluster"].where(df["cluster"].isin(keep), "Other")
    label_order = keep + (["Other"] if (df["plot_label"] == "Other").any() else [])

    n_bars = len(regions) * len(brains)
    x = np.arange(n_bars, dtype=float)
    # Group queen+worker pairs together with a small intra-pair gap
    for k, _region in enumerate(regions):
        for j, _ in enumerate(brains):
            x[k * 2 + j] = k * 2.4 + (j - 0.5) * 0.85
    bar_labels = []
    bar_centers = []
    for k, region in enumerate(regions):
        bar_centers.append(k * 2.4)
        bar_labels.append(_pretty_region(region))

    bottoms = np.zeros(n_bars)
    for label in label_order:
        heights = []
        for region in regions:
            for brain in brains:
                sub = df[
                    (df["region"] == region)
                    & (df["brain"] == brain)
                    & (df["plot_label"] == label)
                ]
                heights.append(float(sub["proportion"].sum()))
        color = get_pooled_color(label) if label != "Other" else "#888888"
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.78,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += np.array(heights)

    # Per-bar Q/W minor ticks under the bars
    for k, _region in enumerate(regions):
        for j, brain in enumerate(brains):
            ax.text(
                k * 2.4 + (j - 0.5) * 0.85,
                -0.025,
                brain[0].upper(),
                ha="center",
                va="top",
                fontsize=TICK_SIZE,
                color="0.35",
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xticks(bar_centers)
    ax.set_xticklabels(
        bar_labels,
        fontsize=TICK_SIZE,
        rotation=20,
        ha="right",
    )
    ax.tick_params(axis="x", length=0, pad=14)
    ax.set_ylabel("Cell type proportion", fontsize=LABEL_SIZE)
    # No title: caption (B) already reads "Cell-type composition, queen (Q)
    # and worker (W)..."
    ax.set_ylim(0, 1.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    # Wrapped the same way panel C wraps its y-tick labels (below): the
    # unwrapped "(all subtypes)" suffix is long enough to run into panel C's
    # labels across the gutter (see the wspace note on make_overview_figure).
    labels = [
        get_pooled_display(label).replace(" (all subtypes)", "\n(all subtypes)")
        for label in labels
    ]
    ax.legend(
        handles,
        labels,
        fontsize=LEGEND_SIZE,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        handlelength=1.0,
        handleheight=1.0,
    )


# print_scale rescales every text artist at save time, so the layout reserves
# that much room around each label. The 8.4 in canvas is authored near print
# size, so the growth is small.
PRINT_GROWTH = 1.15
# Pixel clearance between a volcano label's box and any plotted point.
LABEL_CLEAR_PX = 12
VOLC_HEADROOM = 0.35  # y units above the data (max 1): the gene-label band
# Figure-only short forms for long gene names on the volcano panels (the CSVs
# keep the full names). Shorter labels fit beside their points instead of
# being sent across the panel on a long leader.
VOLCANO_LABEL_ABBREV = {
    "5-hydroxytryptamine receptor 1": "5-HT receptor 1",
    "5-hydroxytryptamine receptor 2A": "5-HT receptor 2A",
    "dopamine D2-like receptor": "D2-like receptor",
    "G-protein coupled receptor moody": "GPCR moody",
}


def _place_volcano_labels(ax, specs, pts_xy, guides_x=(), guide_top=0.0):
    """Place gene labels on one volcano panel without collisions.

    ``specs`` is [(x, y, text)] in data coordinates, ``pts_xy`` every plotted
    point. A label may sit anywhere inside the axes (candidate centres on a
    pixel grid) provided its box covers no plotted point, overlaps no other
    label, and its straight leader to its own point crosses no other label.
    Among the admissible slots the one with the shortest leader wins, and every
    placement order is tried until one lays out all labels (shortest total
    leader). Pure arithmetic on pixel boxes: identical on every run (unlike
    adjustText), and nothing can sit on a point or run over a
    spine.
    """
    import itertools

    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(rend)
    inv = ax.transData.inverted()
    pts_px = ax.transData.transform(pts_xy)
    guide_px = [ax.transData.transform((g, guide_top)) for g in guides_x]
    pad = 3.0
    items = []
    for x, y, txt in specs:
        t = ax.text(
            0,
            0,
            txt,
            fontsize=6.5,
            zorder=10,
            ha="center",
            va="center",
            linespacing=1.05,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
        )
        bb = t.get_window_extent(rend)
        items.append(
            (
                x,
                y,
                t,
                bb.width * PRINT_GROWTH + 2 * pad,
                bb.height * PRINT_GROWTH + 2 * pad,
                ax.transData.transform((x, y)),
            )
        )

    def hit(box, seg):
        for f in np.linspace(0, 1, 30):
            px = seg[0][0] + f * (seg[1][0] - seg[0][0])
            py = seg[0][1] + f * (seg[1][1] - seg[0][1])
            if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                return True
        return False

    def layout(order):
        boxes, leaders, out, total = [], [], [], 0.0
        for i in order:
            _, _, _, w, h, pt = items[i]
            best = None
            xs = np.arange(axbox.x0 + w / 2 + 2, axbox.x1 - w / 2 - 2, 8)
            ys = np.arange(axbox.y0 + h / 2 + 2, axbox.y1 - h / 2 - 2, 8)
            for yc in ys:
                for xc in xs:
                    box = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
                    # nearest point of the box to this label's own point
                    sx = min(max(pt[0], box[0]), box[2])
                    sy = min(max(pt[1], box[1]), box[3])
                    cost = float(np.hypot(sx - pt[0], sy - pt[1]))
                    if best is not None and cost >= best[0]:
                        continue
                    # clearance (LABEL_CLEAR_PX): a label hugging a point reads as
                    # belonging to its neighbour, and a visible leader says
                    # which point it names.
                    if (
                        (pts_px[:, 0] >= box[0] - LABEL_CLEAR_PX)
                        & (pts_px[:, 0] <= box[2] + LABEL_CLEAR_PX)
                        & (pts_px[:, 1] >= box[1] - LABEL_CLEAR_PX)
                        & (pts_px[:, 1] <= box[3] + LABEL_CLEAR_PX)
                    ).any():
                        continue
                    # keep clear of the guide lines (they end at guide_top)
                    if any(
                        box[0] - 2 < g[0] < box[2] + 2 and box[1] < g[1]
                        for g in guide_px
                    ):
                        continue
                    if any(
                        not (
                            box[2] < q[0]
                            or box[0] > q[2]
                            or box[3] < q[1]
                            or box[1] > q[3]
                        )
                        for q in boxes
                    ):
                        continue
                    seg = ((sx, sy), (pt[0], pt[1]))
                    if any(hit(q, seg) for q in boxes):
                        continue
                    if any(hit(box, ldr) for ldr in leaders):
                        continue
                    # a leader running over other dots is ambiguous: penalize
                    d = np.array(seg[1]) - np.array(seg[0])
                    L2 = float(d @ d) or 1.0
                    tt = np.clip(((pts_px - np.array(seg[0])) @ d) / L2, 0, 1)
                    near = (
                        np.hypot(*(pts_px - (np.array(seg[0]) + tt[:, None] * d)).T) < 6
                    )
                    cost += 40.0 * max(int(near.sum()) - 1, 0)
                    if best is not None and cost >= best[0]:
                        continue
                    best = (cost, box, seg)
            if best is None:
                return None
            total += best[0]
            boxes.append(best[1])
            leaders.append(best[2])
            out.append((i, best[1], best[2]))
        return total, out

    # Every placement order (at most 4! = 24), keeping the shortest total
    # leader; ties go to the earlier order, so the result is deterministic.
    # (Taking the first order that fits sent one Lobula leader across the
    # whole panel when a later order fitted it beside its point.)
    best = None
    base = sorted(range(len(items)), key=lambda i: -abs(items[i][0]))
    for order in [base] + [
        o for o in itertools.permutations(range(len(items))) if list(o) != base
    ]:
        r = layout(order)
        if r is not None and (best is None or r[0] < best[0]):
            best = r
    if best is None:
        raise RuntimeError("no collision-free volcano label layout")
    for i, box, seg in best[1]:
        x, y, t, _, _, _ = items[i]
        cx, cy = inv.transform(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        t.set_position((cx, cy))
        lx0, ly0 = inv.transform(seg[0])
        ax.plot(
            [lx0, x], [ly0, y], color="0.4", lw=0.5, zorder=6, solid_capstyle="round"
        )


def _plot_volcanos(axes, deg_dfs):
    """Panel C: per-region caste effect sizes (descriptive; queen vs worker).

    The spatial atlas has one queen and one worker brain, so slices are not
    independent caste replicates. These panels therefore present descriptive
    effect sizes only — no caste-level significance is claimed or tested.

    Conventions:
      - x = log2FC (worker / queen). Negative = queen-enriched, positive = worker.
      - y = slice consistency, |rank-biserial r| of the two brains' slice
        means (1 = every worker section on one side of every queen section).
        This reflects how reproducibly a gene's direction holds across slices;
        no P value is computed.
      - Points coloured by effect: queen-enriched (log2FC < -FC_THRESH) gold,
        worker-enriched (log2FC > +FC_THRESH) blue, near-zero grey.
      - Vertical guides at ±FC_THRESH mark the near-zero effect band; no
        significance threshold line is drawn.
    """
    FC_THRESH = 0.5  # |log2FC| band separating near-zero (grey) from coloured

    # Compute global x and y limits. Both are shared across the six panels so
    # effect sizes stay visually comparable. y (|slice consistency|) runs to 1,
    # plus headroom for the gene labels.
    all_fc, all_nlp = [], []
    for region in CASTE_TEST_REGIONS:
        if region in deg_dfs and len(deg_dfs[region]):
            all_fc.append(deg_dfs[region]["log2fc"].values)
            all_nlp.append(deg_dfs[region]["slice_consistency"].abs().values)
    if all_fc:
        all_fc = np.concatenate(all_fc)
        # The largest |log2FC| must lie inside the axes (a 99.5th-percentile
        # limit would clip the lamina's tachykinins at -3.09 while labelling
        # them).
        x_lim = max(2.5, float(np.nanmax(np.abs(all_fc))) * 1.08)
    else:
        x_lim = 3.0
    y_max = 1.0  # |rank-biserial r| <= 1
    # Headroom above the data is a reserved band for the gene labels (see
    # _place_volcano_labels), so the axis runs past the last point.
    y_lim = y_max + VOLC_HEADROOM
    y_top_tick = y_max

    for i, region in enumerate(CASTE_TEST_REGIONS):
        ax = axes[i]
        if region not in deg_dfs or len(deg_dfs[region]) == 0:
            ax.set_title(
                f"{_pretty_region(region)}\n(no data)",
                fontsize=TITLE_SIZE,
            )
            continue

        df = deg_dfs[region]
        fc = df["log2fc"].values
        nlp = df["slice_consistency"].abs().values

        # Colour by effect (direction + magnitude of log2FC), not by any
        # significance call: queen-enriched gold, worker-enriched blue,
        # near-zero grey.
        queen_eff = fc < -FC_THRESH
        worker_eff = fc > FC_THRESH
        near_zero = ~(queen_eff | worker_eff)

        # Background near-zero-effect points
        ax.scatter(
            fc[near_zero],
            nlp[near_zero],
            s=8,
            alpha=0.35,
            color="#bbbbbb",
            edgecolor="none",
            rasterized=True,
        )
        # Queen-enriched (left)
        ax.scatter(
            fc[queen_eff],
            nlp[queen_eff],
            s=18,
            alpha=0.85,
            color=QUEEN_COLOR,
            edgecolor="white",
            linewidth=0.3,
            rasterized=True,
        )
        # Worker-enriched (right)
        ax.scatter(
            fc[worker_eff],
            nlp[worker_eff],
            s=18,
            alpha=0.85,
            color=WORKER_COLOR,
            edgecolor="white",
            linewidth=0.3,
            rasterized=True,
        )

        # Effect-band guides: vertical bars mark the near-zero |log2FC| band.
        # No horizontal significance line (this is not a caste test).
        ax.plot(
            [-FC_THRESH] * 2, [0, y_top_tick], color="k", ls="--", lw=0.6, alpha=0.4
        )
        ax.plot([FC_THRESH] * 2, [0, y_top_tick], color="k", ls="--", lw=0.6, alpha=0.4)
        ax.plot([0, 0], [0, y_top_tick], color="k", lw=0.4, alpha=0.25)

        # Label the strongest-effect genes: the 2 largest |log2FC| per side
        # (at most 4 per panel), so the labels have room.
        labelable = queen_eff | worker_eff
        if labelable.any():
            df_hit = df[labelable].copy()
            df_hit["nlp"] = nlp[labelable]
            df_hit["abs_fc"] = df_hit["log2fc"].abs()
            up_top = (
                df_hit[df_hit["log2fc"] > 0]
                .sort_values(["abs_fc", "nlp"], ascending=False)
                .head(2)
            )
            down_top = (
                df_hit[df_hit["log2fc"] < 0]
                .sort_values(["abs_fc", "nlp"], ascending=False)
                .head(2)
            )
            top_labels = pd.concat([up_top, down_top])
            # Deterministic placement (adjustText varies run to run and can drop
            # labels on points and over the spines).
            # Limits must be fixed first: the layout is measured in pixels.
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(0, y_lim)
            specs = [
                (
                    float(r["log2fc"]),
                    float(abs(r["slice_consistency"])),
                    textwrap.fill(
                        VOLCANO_LABEL_ABBREV.get(
                            str(r["gene_short"]), str(r["gene_short"])
                        ),
                        width=16,
                        break_long_words=False,
                        break_on_hyphens=False,
                    ),
                )
                for _, r in top_labels.iterrows()
            ]
            # ring the named points: at the top of the axes many genes tie in
            # slice consistency, and the ring says which of the row a leader names
            ax.scatter(
                [sp[0] for sp in specs],
                [sp[1] for sp in specs],
                s=30,
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,
                zorder=7,
            )
            _place_volcano_labels(
                ax,
                specs,
                np.column_stack([fc, nlp]),
                guides_x=(-FC_THRESH, 0.0, FC_THRESH),
                guide_top=y_top_tick,
            )

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(0, y_lim)
        ax.set_yticks([0, 0.5, 1])
        ax.spines["left"].set_bounds(0, y_top_tick)
        ax.set_xlabel("log$_2$ FC (worker / queen)", fontsize=TICK_SIZE)
        if i % 3 == 0:
            ax.set_ylabel(
                "slice consistency\n(|rank-biserial $r$|)",
                fontsize=TICK_SIZE,
            )

        # Region name only. The per-panel counts of queen- and worker-enriched
        # genes live in the LaTeX caption, not the artwork (titles are identity
        # labels, not statistics).
        ax.set_title(
            _pretty_region(region), fontsize=TICK_SIZE + 1, fontweight="bold", pad=2
        )
        ax.tick_params(labelsize=TICK_SIZE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def _plot_interaction_heatmap(ax, interaction_df, adatas, top_n=15):
    """Heatmap of top region × brain interaction genes (z-scored)."""
    if interaction_df is None or len(interaction_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No interaction\ngenes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=LABEL_SIZE,
        )
        ax.set_title(
            "Region × brain interaction",
            fontsize=TITLE_SIZE,
        )
        return

    top = interaction_df.head(top_n)
    if len(top) == 0:
        ax.set_visible(False)
        return

    mean_cols = [c for c in top.columns if c.startswith("mean_")]
    # Order columns: queen first then worker, regions in canonical order
    # Single source of truth for the canonical order.
    region_order = COMMON_REGIONS
    ordered_cols = []
    for brain in ("queen", "worker"):
        for region in region_order:
            tag = f"mean_{brain}_{region}"
            if tag in mean_cols:
                ordered_cols.append(tag)
    if not ordered_cols:
        ordered_cols = mean_cols
    mat = top[ordered_cols].values.astype(float)

    row_means = mat.mean(axis=1, keepdims=True)
    row_stds = mat.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1
    z_mat = (mat - row_means) / row_stds

    col_labels = []
    for c in ordered_cols:
        tag = c.replace("mean_", "")
        brain, region = tag.split("_", 1)
        col_labels.append(f"{brain.title()}\n{_pretty_region(region)}")

    im = ax.imshow(
        z_mat,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-2,
        vmax=2,
    )
    # Vertical line between queen and worker block
    n_per_brain = len(ordered_cols) // 2
    if n_per_brain * 2 == len(ordered_cols):
        ax.axvline(n_per_brain - 0.5, color="black", lw=1.2)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(
        col_labels,
        rotation=45,
        ha="right",
        fontsize=TICK_SIZE - 1,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene_short"].values, fontsize=TICK_SIZE)
    # No in-figure title: the LaTeX caption already says "Region × caste
    # interaction effects ... Top 15 genes".
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label("Row z-score", fontsize=LABEL_SIZE - 1)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 1)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_spatial_regions(axes, adatas):
    """Spatial maps with cells coloured by anatomical region."""
    spans = []
    for i, (brain, adata) in enumerate(adatas.items()):
        ax = axes[i]
        regions = assign_regions(adata)
        coords = adata.obsm["spatial_aligned"]

        n = len(coords)
        if n > 120_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, 120_000, replace=False)
        else:
            idx = np.arange(n)

        # Background (unassigned) cells first as a faint grey
        bg_mask = regions.values[idx] == "unassigned"
        if bg_mask.any():
            ax.scatter(
                coords[idx[bg_mask], 0],
                coords[idx[bg_mask], 1],
                s=0.4,
                alpha=0.18,
                c="#dddddd",
                edgecolor="none",
                rasterized=True,
            )

        # Then each labelled region
        labelled = [
            r
            for r in REGION_PALETTE.keys()
            if r != "unassigned" and r in regions.unique()
        ]
        for region in labelled:
            mask = regions.values[idx] == region
            if not mask.any():
                continue
            ax.scatter(
                coords[idx[mask], 0],
                coords[idx[mask], 1],
                s=1.4,
                alpha=0.85,
                c=REGION_PALETTE[region],
                edgecolor="none",
                rasterized=True,
            )

        ax.set_aspect("equal")
        spans.append(
            coords[idx].min(axis=0).tolist() + coords[idx].max(axis=0).tolist()
        )
        ax.set_title(brain.title(), fontsize=LABEL_SIZE, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if i == 1:
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=REGION_PALETTE[r],
                    markersize=6,
                    label=_pretty_region(r),
                )
                for r in labelled
            ]
            # under both maps, centred on the gap between them
            ax.legend(
                handles=handles,
                fontsize=LEGEND_SIZE,
                loc="upper center",
                bbox_to_anchor=(-0.01, 0.0),
                ncol=3,
                frameon=False,
                handlelength=1.0,
                handletextpad=0.3,
                columnspacing=1.0,
            )
    # One shared window size, centred on each brain: equal-aspect axes with
    # different extents drew the two brains at different scales and put the
    # Queen / Worker titles on different baselines.
    if spans:
        sp = np.array(spans)
        half_w = float((sp[:, 2] - sp[:, 0]).max()) / 2 * 1.02
        half_h = float((sp[:, 3] - sp[:, 1]).max()) / 2 * 1.02
        for ax, (x0, y0, x1, y1) in zip(axes, spans, strict=False):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy - half_h, cy + half_h)


def _plot_region_lollipop(ax, df, region, n_top=5):
    """Top n_top up- and down-genes for one region.

    The most positive and most negative log2FC among every gene.
    """
    sig = df.copy()
    up = sig.nlargest(n_top, "log2fc")
    down = sig.nsmallest(n_top, "log2fc")
    top = pd.concat([up, down]).drop_duplicates(subset="gene")
    # Descending, then invert: worker-enriched at the top of each panel,
    # queen-enriched at the bottom.
    top = top.sort_values("log2fc", ascending=False)

    labels = []
    for y, (_, row) in enumerate(top.iterrows()):
        fc = float(row["log2fc"])
        # Stems and labels stay neutral: this figure's one colour channel is
        # the queen/worker direction of effect; region is the panel title.
        ax.plot([0, fc], [y, y], color=STEM_COLOR, lw=1.4, alpha=0.55)
        ax.plot(
            fc,
            y,
            "o",
            color=WORKER_COLOR if fc > 0 else QUEEN_COLOR,
            markersize=6,
            alpha=0.95,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        # Resolve the display name here rather than trusting the cached
        # gene_short column, which was written when the DEG tables were
        # computed and would not carry a later correction (the Vg-family
        # override).
        labels.append(
            gene_short_name(row["gene"]) if "gene" in row.index else row["gene_short"]
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=TICK_SIZE - 1.5, color=LABEL_COLOR)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    # Pad the x range so the extreme dots clear the right edge.
    xl = ax.get_xlim()
    ax.set_xlim(xl[0], xl[1] + 0.06 * (xl[1] - xl[0]))
    from matplotlib.ticker import MultipleLocator

    ax.xaxis.set_major_locator(MultipleLocator(1.0))  # 0.5 steps collided
    ax.set_xlabel("log$_2$ FC (worker / queen)", fontsize=LABEL_SIZE - 1)
    ax.tick_params(axis="x", labelsize=TICK_SIZE - 1.5)
    # Region names the panel.
    # No title: the region name rides on the panel letter (make_lollipop_figure).
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
