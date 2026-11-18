#!/usr/bin/env python3
"""Per-region cell density, DEGs, and domain-specific marker genes.

Loads per-brain aligned_clustered.h5ad files (which have in_* region
boolean columns), computes region-level density, composition, pseudobulk
DEGs (queen vs worker), cross-region interaction effects, and
1-vs-rest domain marker genes (inspired by Yao et al. Nat Neurosci 2025).

    PYTHONPATH=src .venv/bin/python scripts/caste_de/si_regional_degs.py
"""


import anndata as ad
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from caste_de.regional_degs_data import (  # noqa: E402
    BRAINS,  # noqa: F401 (re-exported)
    CASTE_TEST_REGIONS,  # noqa: F401 (re-exported)
    COMMON_REGIONS,  # noqa: F401 (re-exported)
    EFFECTS_DIR,  # noqa: F401 (re-exported)
    FIG_DIR,
    MIN_CELLS_PER_SLICE,  # noqa: F401 (re-exported)
    OUTPUT_DIR,
    _pretty_region,
    composition_by_region,
    compute_density,
    cross_region_compare,
    get_region_cols,
    one_vs_rest_degs,
    pseudobulk_by_domain,
    pseudobulk_pca,
    regional_degs,  # noqa: F401 (re-exported)
)
from caste_de.regional_degs_panels import (  # noqa: E402
    _plot_composition,
    _plot_density,
    _plot_domain_heatmap,
    _plot_domain_pca,
    _plot_domain_violins,
    _plot_interaction_heatmap,
    _plot_ovr_volcanos,
    _plot_region_lollipop,
    _plot_regional_shift,
    _plot_spatial_regions,
    _plot_volcanos,
)

from bimp_st.nine_region_common import (
    BASE_REGIONS,
    attach_region_flags,
)
from bimp_st.paths import explain_missing_inputs
from bimp_st.si_common import (
    LEGEND_SIZE,
    PANEL_LABEL_SIZE,
    QUEEN_COLOR,
    WORKER_COLOR,
    add_panel_headers,
    save,
    set_style,
)

# ── Domain marker figure ─────────────────────────────────────────────────


def make_domain_marker_figure(ovr_results, pb_domain_df, pca_df, pca_model, adatas):
    """Generate domain-specific marker figure (inspired by Yao et al.)."""
    print("\nGenerating domain marker figure...", flush=True)
    set_style()

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.4,
        wspace=0.35,
        height_ratios=[1, 1.2, 1],
    )

    # ── Panel A: Pseudobulk heatmap (genes × regions) ─────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_domain_heatmap(ax_a, ovr_results, pb_domain_df)
    ax_a.text(
        -0.08,
        1.08,
        "A",
        transform=ax_a.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
    )

    # ── Panel B: PCA of pseudobulk by domain ──────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_domain_pca(ax_b, pca_df, pca_model)
    ax_b.text(
        -0.08,
        1.08,
        "B",
        transform=ax_b.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
    )

    # ── Panel C: Volcano plots per domain (1-vs-rest) ─────────────────
    # Use queen results (or combined if available)
    brain_key = "queen"
    regions_to_plot = sorted(ovr_results.get(brain_key, {}).keys())
    n_regions = len(regions_to_plot)
    if n_regions > 0:
        gs_c = gs[1, :].subgridspec(1, n_regions, wspace=0.3)
        axes_c = [fig.add_subplot(gs_c[0, i]) for i in range(n_regions)]
        _plot_ovr_volcanos(axes_c, ovr_results[brain_key], regions_to_plot)
        axes_c[0].text(
            -0.25,
            1.12,
            "C",
            transform=axes_c[0].transAxes,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="top",
        )

    # ── Panel D: Violin plots of top domain-specific genes ────────────
    ax_d_gs = gs[2, :].subgridspec(1, 2, wspace=0.3)
    ax_d0 = fig.add_subplot(ax_d_gs[0, 0])
    ax_d1 = fig.add_subplot(ax_d_gs[0, 1])
    _plot_domain_violins(ax_d0, ax_d1, ovr_results, adatas)
    ax_d0.text(
        -0.12,
        1.08,
        "D",
        transform=ax_d0.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
    )

    return fig


def make_overview_figure(density_df, comp_df, adatas):
    """Anatomical region overview figure.

    Four panels:
      A — cell counts per region (queen vs worker, paired bars)
      B — pooled cell type composition stacked bars
      C — caste contrast of B (log2FC worker/queen, heatmap)
      D — spatial maps with region overlays (queen, worker)

    The caption's circularity exemption names panel B by letter, so keep A
    and B lettered as they are.
    """
    print("\nBuilding overview figure...", flush=True)
    set_style()

    # No suptitle: the LaTeX caption already carries the figure title.
    # 8.4 in wide: at 0.95\linewidth the authored tick size (TICK_SIZE) then
    # prints at about the house 6 pt without rescaling. Two rows: A and B on
    # top; C beside the two brain maps underneath. The maps are ~0.7 as wide as
    # tall, so a full-width row of two maps would be half the figure's height
    # and mostly blank; sharing the row with C uses that width.
    fig = plt.figure(figsize=(8.4, 6.6))
    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        hspace=0.42,
        height_ratios=[1.0, 1.32],
        left=0.075,
        right=0.99,
        top=0.955,
        bottom=0.075,
    )
    gs_top = gs[0].subgridspec(1, 4, width_ratios=[1.0, 0.22, 1.25, 0.62], wspace=0)
    ax_a = fig.add_subplot(gs_top[0, 0])
    _plot_density(ax_a, density_df)

    ax_b = fig.add_subplot(gs_top[0, 2])
    _plot_composition(ax_b, comp_df)

    # C: B's caste contrast; the spacer column holds its two-line row labels.
    gs_bot = gs[1].subgridspec(1, 4, width_ratios=[0.42, 1.0, 0.62, 1.62], wspace=0)
    ax_c = fig.add_subplot(gs_bot[0, 1])
    _plot_regional_shift(ax_c, comp_df)

    gs_d = gs_bot[0, 3].subgridspec(1, 2, wspace=0.02)
    axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(2)]
    _plot_spatial_regions(axes_d, adatas)
    add_panel_headers(
        fig,
        [
            (ax_a, "A", "Cells per region"),
            (ax_b, "B", "Cell type composition"),
            (ax_c, "C", "Caste contrast"),
            # no tag: the Queen / Worker map titles sit on that line
            (axes_d[0], "D", None),
        ],
        width_frac=0.95,
    )

    return fig


def make_volcano_figure(deg_dfs):
    """Per-region caste effect sizes (queen vs worker): `fig:regional_volcanoes`.

    Six panels in a 2x3 grid: lamina, medulla, lobula on row 1; KC ic, nc, oc
    on row 2. Descriptive effect-size volcanoes (x = log2FC) with the
    near-zero effect band marked; points coloured by effect direction --
    queen-enriched (gold) on the left, worker-enriched (blue) on the right --
    with deterministic label placement. The atlas is n=1 brain per caste, so no
    caste-level significance is claimed.

    CASTE_TEST_REGIONS, not COMMON_REGIONS: this is a caste-contrast figure,
    and the antennal lobe has no caste contrast to show. It is traced on all
    sixteen queen sections and six of the worker's sixteen, so it carries no
    per-slice consistency score and would be an empty panel. It is still
    fitted everywhere a caste
    contrast is not involved -- region prediction, diversity, composition.
    ellipsoid_body and fan_shaped_body never reach this figure at all; see the
    COMMON_REGIONS comment.
    """
    print("\nBuilding volcano figure...", flush=True)
    set_style()

    # Sized off the region list rather than hardcoded, so the grid follows
    # whatever is actually plotted.
    n_regions = len(CASTE_TEST_REGIONS)
    ncols = 3
    nrows = -(-n_regions // ncols)  # ceil

    # No suptitle: the LaTeX caption already carries the figure title and the
    # n=1-per-caste / descriptive caveat.
    # Narrower than the old 11.5 in canvas so the authored sizes print near the
    # house sizes at full width.
    fig = plt.figure(figsize=(VOLC_W, VOLC_ROW_H * nrows))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        hspace=0.42,
        wspace=0.28,
        left=0.085,
        right=0.985,
        top=0.925,
        bottom=0.075,
    )
    axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(n_regions)]
    _plot_volcanos(axes, deg_dfs)
    row_tags = ["Optic lobe regions", "Kenyon cell compartments", None]
    add_panel_headers(
        fig,
        [(axes[r * ncols], "ABC"[r], row_tags[r]) for r in range(nrows)],
        width_frac=1.0,
        dy_pts=17,
    )  # lifted clear of the region sub-headers

    # No suptitle. The LaTeX caption already opens "Per-region caste effect sizes
    # (queen vs. worker), descriptive." and carries the n=1-per-caste caveat, so a
    # banner here just repeats it and steals vertical space from the panels.
    return fig


def make_lollipop_figure(deg_dfs):
    """SI: top DEG lollipop summary, one panel per region.

    One panel per region gives a normal landscape aspect (a single stacked
    axes is far taller than any other SI figure) and gives each gene label the
    full panel width.
    """
    print("\nBuilding lollipop figure...", flush=True)
    set_style()

    drawn = [r for r in COMMON_REGIONS if r in deg_dfs and len(deg_dfs[r]) > 0]
    ncol = 2
    nrow = int(np.ceil(len(drawn) / ncol)) if drawn else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 4.1 * nrow), squeeze=False)
    for k, region in enumerate(drawn):
        _plot_region_lollipop(axes[k // ncol][k % ncol], deg_dfs[region], region)
    # The region name is the panel's tag (one panel = one region).
    add_panel_headers(
        fig,
        [
            (axes[k // ncol][k % ncol], "ABCDEF"[k], _pretty_region(region))
            for k, region in enumerate(drawn)
        ],
        width_frac=1.0,
    )
    # Any unused cell is removed rather than left as an empty frame.
    for k in range(len(drawn), nrow * ncol):
        axes[k // ncol][k % ncol].remove()

    from matplotlib.lines import Line2D

    # Invisible anchor at the figure's left edge: under bbox='tight' the longest
    # y label otherwise defines the canvas and starts flush against it.
    fig.add_artist(
        Line2D(
            [0],
            [0.5],
            transform=fig.transFigure,
            marker=".",
            markersize=1,
            color="white",
            clip_on=False,
        )
    )
    # One encoding, one legend: marker colour = direction of the caste effect.
    # It sits inside the Lobula panel, in its empty lower-right corner (the
    # queen-enriched stems all run left), so it is next to what it explains and
    # clear of every x-axis label.
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=QUEEN_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            label="queen-enriched",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=WORKER_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
            label="worker-enriched",
        ),
    ]
    xl = axes[1][0].get_xlim()  # widen the right side to give the legend room
    axes[1][0].set_xlim(xl[0], xl[1] + 1.1)
    axes[1][0].legend(
        handles=handles,
        fontsize=LEGEND_SIZE,
        loc="lower right",
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    # Room for the longest gene label to the left of every column (the second
    # column's labels hang into wspace), including the long 5-HT receptor
    # name.
    fig.subplots_adjust(
        left=0.25, right=0.985, top=0.965, bottom=0.07, wspace=0.95, hspace=0.62
    )
    return fig


def make_interaction_figure(interaction_df, adatas):
    """Region × brain interaction heatmap.

    Top 15 genes with strongest region × caste interaction effects,
    z-scored expression by region × brain.
    """
    print("\nBuilding interaction figure...", flush=True)
    set_style()

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    _plot_interaction_heatmap(ax, interaction_df, adatas, top_n=15)
    fig.subplots_adjust(left=0.235, right=0.90, top=0.985, bottom=0.215)
    return fig
VOLC_W, VOLC_ROW_H = 9.6, 3.5  # volcano figure canvas (in)


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with explain_missing_inputs():
        import sys

        # --only degs: load the brains, compute the per-region caste effects and
        # draw the volcano figure only (skips density, composition, interaction and
        # domain markers)
        ONLY_DEGS = sys.argv[1:] == ["--only", "degs"]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Load per-brain data
        adatas = {}
        for brain, path in BRAINS.items():
            print(f"Loading {brain} from {path}...", flush=True)
            adata = ad.read_h5ad(path)
            print(
                f"  {adata.shape[0]:,} cells, {adata.shape[1]} genes",
                flush=True,
            )
            # Refresh the six in_<region> flags from the crop sidecars. The copy
            # baked into aligned_clustered.h5ad predates polygon repair: four
            # polygons on worker 0068401 R1 move 119 cells from the lobula to the
            # antennal lobe
            # (stored worker lobula 2,435 vs current 2,316), which would otherwise
            # contaminate the lobula volcano.
            #
            # BASE_REGIONS deliberately, not STATISTICAL_REGIONS: the antennal
            # lobe stays out of this script's fits, because it is drawn on all
            # sixteen queen sections and seven of the worker's sixteen, so a
            # between-brain comparison inside it would measure how much was
            # annotated.
            attach_region_flags(adata, regions=BASE_REGIONS)
            bool_cols = get_region_cols(adata)
            print(f"  Region columns: {bool_cols}", flush=True)
            adatas[brain] = adata

        if ONLY_DEGS:
            deg_dfs = regional_degs(adatas)
            EFFECTS_DIR.mkdir(parents=True, exist_ok=True)
            for region, df in deg_dfs.items():
                if len(df):
                    df.to_csv(EFFECTS_DIR / f"degs_{region}.csv", index=False)
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            save(make_volcano_figure(deg_dfs), FIG_DIR, "figS_regional_volcanoes")
            raise SystemExit(0)

        # 1. Cell density
        print("\n" + "=" * 60, flush=True)
        density_df = compute_density(adatas)
        density_df.to_csv(
            OUTPUT_DIR / "cell_density_by_region.csv",
            index=False,
        )
        print("  Saved cell_density_by_region.csv", flush=True)

        # 2. Cell type composition
        print("\n" + "=" * 60, flush=True)
        comp_df, comp_test_df = composition_by_region(adatas)
        comp_df.to_csv(
            OUTPUT_DIR / "composition_by_region.csv",
            index=False,
        )
        if len(comp_test_df):
            comp_test_df.to_csv(
                OUTPUT_DIR / "composition_tests.csv",
                index=False,
            )
        print("  Saved composition CSVs", flush=True)

        # 3. Per-region DEGs
        print("\n" + "=" * 60, flush=True)
        deg_dfs = regional_degs(adatas)
        EFFECTS_DIR.mkdir(parents=True, exist_ok=True)
        for region, df in deg_dfs.items():
            if len(df):
                df.to_csv(
                    EFFECTS_DIR / f"degs_{region}.csv",
                    index=False,
                )
                print(f"  Saved caste_effects/degs_{region}.csv", flush=True)

        # 4. Cross-region interaction
        print("\n" + "=" * 60, flush=True)
        interaction_df = cross_region_compare(adatas)
        if len(interaction_df):
            interaction_df.to_csv(
                OUTPUT_DIR / "region_brain_interaction.csv",
                index=False,
            )
            print(
                "  Saved region_brain_interaction.csv",
                flush=True,
            )

        # 5. Three SI figures
        print("\n" + "=" * 60, flush=True)
        fig_overview = make_overview_figure(density_df, comp_df, adatas)
        save(fig_overview, OUTPUT_DIR, "si_regional_overview")

        fig_volcano = make_volcano_figure(deg_dfs)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save(fig_volcano, FIG_DIR, "figS_regional_volcanoes")

        fig_lollipop = make_lollipop_figure(deg_dfs)
        save(fig_lollipop, OUTPUT_DIR, "si_regional_lollipop")

        fig_interaction = make_interaction_figure(interaction_df, adatas)
        save(fig_interaction, OUTPUT_DIR, "si_regional_interaction")

        # 6. One-vs-rest domain markers
        print("\n" + "=" * 60, flush=True)
        ovr_results = one_vs_rest_degs(adatas)
        for brain, brain_res in ovr_results.items():
            for region, df in brain_res.items():
                if len(df):
                    df.to_csv(
                        OUTPUT_DIR / f"domain_markers_{brain}_{region}.csv",
                        index=False,
                    )
        print("  Saved domain marker CSVs", flush=True)

        # 7. Pseudobulk by domain
        print("\n" + "=" * 60, flush=True)
        pb_domain_df = pseudobulk_by_domain(adatas)
        pb_domain_df.to_csv(
            OUTPUT_DIR / "pseudobulk_by_domain.csv",
            index=False,
        )

        # 8. Pseudobulk PCA
        print("\n" + "=" * 60, flush=True)
        pca_df, pca_model = pseudobulk_pca(adatas)
        if len(pca_df):
            pca_df.to_csv(
                OUTPUT_DIR / "pseudobulk_pca.csv",
                index=False,
            )

        # 9. Domain marker figure
        print("\n" + "=" * 60, flush=True)
        fig2 = make_domain_marker_figure(
            ovr_results,
            pb_domain_df,
            pca_df,
            pca_model,
            adatas,
        )
        save(fig2, OUTPUT_DIR, "domain_markers")

        print(f"\nDone. Results in {OUTPUT_DIR}/", flush=True)
