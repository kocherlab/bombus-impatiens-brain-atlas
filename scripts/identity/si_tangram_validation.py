#!/usr/bin/env python3
"""SI figure: Tangram cell-type mapping validation.

Three panels that support the Tangram cell-type transfer:

  A: LOO-CV ranked cosine-similarity waterfall (per-gene imputation accuracy
     across the 101-gene Xenium panel).
  B: Cluster correspondence heatmap (spatial Leiden clusters x pooled snRNA
     cell types, row-normalized proportions).
  C: LOO-CV cosine similarity vs reference sparsity and vs mean expression
     (dependence of imputation accuracy on gene detectability).

The panel-drawing functions (``plot_loocv_ranked``, ``plot_cluster_heatmap``,
``plot_loocv_vs_sparsity``) and their helpers/loaders are copied from a frozen
snapshot of an earlier composition script so this SI figure stays independent of
``scripts/regions/caste_comparison_sccoda.py``, which does not export them.

    PYTHONPATH=src:scripts .venv/bin/python scripts/identity/si_tangram_validation.py
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from bimp_st.cluster_labels import (
    POOLED_ORDER,
    get_pooled,
    get_pooled_display,
    get_spatial_short,
)
from bimp_st.paths import CANONICAL_TANGRAM
from bimp_st.si_common import (
    OUTPUT_DIR,
    TICK_SIZE,
    add_panel_headers,
    maybe_subsample_adata,
    save,
    set_style,
)

# ── Configuration ────────────────────────────────────────────────────────────
LOOCV_CSV = "results/tangram/loocv_0af3056_20261020/combined_clustered_loocv_scores.csv"
# Canonical object (cell-type columns keyed by name, not position): panel B's
# cluster-correspondence heatmap reads tangram_cell_type directly, so
# it must come from the canonical object, matching the purity numbers cited for
# fig:tangram_qc.
TANGRAM_H5AD = str(CANONICAL_TANGRAM)


# ── Panel A: LOO-CV ranked cosine similarity ────────────────────────────────
# (lifted verbatim from the frozen caste_comparison_sccoda.py: plot_loocv_ranked)


def plot_loocv_ranked(loocv_df, output_dir=None, gene_lookup=None, ax=None):
    """LOO-CV ranked cosine-similarity waterfall.

    Standalone (``ax=None``): builds its own figure and saves ``loocv_ranked``.
    Embedded (``ax`` provided, e.g. from the SI validation figure): draws on
    that axis and does not save. Drawing logic is identical in both paths.
    """
    print("  Panel A: LOO-CV ranked barplot...", flush=True)
    df = loocv_df.sort_values("cosine_sim", ascending=True).reset_index(drop=True)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9.5, 5.1))

    # Colour encodes reference sparsity rather than repeating bar length: it
    # shows the panel-C correlation on the ranked axis, so the
    # failing tail reads as the sparse tail.
    spars = df["sparsity_sc"].values
    cmap = plt.colormaps.get_cmap("viridis_r")
    norm = plt.Normalize(vmin=float(np.nanmin(spars)), vmax=float(np.nanmax(spars)))
    ax.barh(
        np.arange(len(df)),
        df["cosine_sim"].values,
        color=cmap(norm(spars)),
        height=0.85,
        edgecolor="none",
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.020, pad=0.012, aspect=32)
    # Colorbar label/ticks at the shared tick size, so the print-scale hook
    # brings them to the same printed size as every other tick in the figure.
    cb.set_label(
        "Reference sparsity\n(fraction zero)", fontsize=TICK_SIZE, linespacing=1.15
    )
    cb.ax.tick_params(labelsize=TICK_SIZE)
    cb.outline.set_visible(False)

    # Drop per-gene labels — too many genes to be readable; the message is
    # the score distribution, not individual gene identities
    ax.set_yticks([])
    ax.set_ylabel(f"Genes (n={len(df)}, ranked)")
    ax.set_xlabel("Cosine similarity (LOO-CV)")
    # Reserve headroom above the tallest bars so the median/mean labels
    # (anchored at axes-fraction y=1.0 below) sit in clear space instead of
    # overlapping the top few bars, which reach nearly to the top of the
    # unpadded default y-range.
    ax.set_ylim(-0.5, len(df) - 0.5 + max(2.0, len(df) * 0.05))
    if standalone:
        # Only needed when this panel isn't embedded under a lettered SI
        # panel with its own LaTeX caption sentence.
        ax.set_title("Leave-one-out cross-validation")

    median_val = df["cosine_sim"].median()
    mean_val = df["cosine_sim"].mean()
    ax.axvline(median_val, color="#222222", lw=1.0, ls="--", alpha=0.85)
    ax.axvline(mean_val, color="#222222", lw=1.0, ls=":", alpha=0.85)
    # Labelled at the lines rather than with a keyed legend, so the two
    # similar grey dashes are easy to tell apart.
    # Both labels sit above the axes, on opposite sides of their lines (mean
    # to the left of its line, median to the right), so neither can land on a
    # bar -- inside the axes the top bars reach 0.8-0.9 and run under the text.
    for val, lab, side in (
        (median_val, f"median {median_val:.2f}", 1),
        (mean_val, f"mean {mean_val:.2f}", -1),
    ):
        ax.annotate(
            lab,
            xy=(val, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3 * side, 2),
            textcoords="offset points",
            ha="left" if side > 0 else "right",
            va="bottom",
            fontsize=TICK_SIZE,
            color="#222222",
            annotation_clip=False,
        )

    ax.set_xlim(0, 1.0)
    if standalone:
        fig.tight_layout()
        save(fig, output_dir, "loocv_ranked")


# ── Panel B: Cluster correspondence heatmap ─────────────────────────────────
# (lifted verbatim from the frozen caste_comparison_sccoda.py: _pool_ct_table +
#  plot_cluster_heatmap)


def _pool_ct_table(ct_tab):
    """Aggregate fine snRNA cell type columns into the coarse pooled groups.

    Groups columns by ``get_pooled()`` (OL_1..OL_13 → Optic lobe, the glia
    subtypes → Glia, projection neurons pooled) and orders them by
    ``POOLED_ORDER`` so the correspondence heatmap shows one clean column per
    coarse group.
    """
    pooled = {}
    for col in ct_tab.columns:
        label = get_pooled(col)
        pooled.setdefault(label, []).append(col)
    ordered = [g for g in POOLED_ORDER if g in pooled]
    ordered += [g for g in pooled if g not in ordered]  # any unmapped, keep
    out = pd.DataFrame(
        {label: ct_tab[pooled[label]].sum(axis=1) for label in ordered},
        index=ct_tab.index,
    )
    return out


def plot_cluster_heatmap(tangram, snrna=None, output_dir=None, ax=None):
    """Heatmap: spatial Leiden clusters (rows) vs pooled snRNA cell types (cols).

    Cell counts come from the tangram h5ad's ``tangram_cell_type`` column
    and are aggregated by pooled snRNA short label (KC-l.* → KC-l,
    Glia-a/b → Glia) so each cell type appears once. ``snrna`` is accepted for
    call-site compatibility but is not used.

    Standalone (``ax=None``): builds its own figure and saves ``cluster_heatmap``.
    Embedded (``ax`` provided): draws on that axis and does not save.
    """
    print("  Panel B: Cluster correspondence heatmap...", flush=True)
    # Use tangram_cell_type directly for the cross-tab
    ct_tab = pd.crosstab(
        tangram.obs["leiden"],
        tangram.obs["tangram_cell_type"],
    )

    # Rows = spatial Leiden (sorted numerically), cols = cell type strings
    # Drop degenerate micro-clusters (<100 cells): their row-normalized
    # proportions are small-count noise (e.g. a "0.75" from 3 cells) but would
    # render with the same visual weight as 90k-cell clusters.
    # Subsampling: main() runs the atlas through maybe_subsample_adata()
    # (default BIMP_FIG_SUBSAMPLE_FRACTION=0.1), and at 10% a cluster needs
    # 1,000 real cells to clear a nominal 100-cell bar, so whole Leiden
    # clusters (e.g. S19-S21) would drop out. The bar is scaled by the loaded
    # fraction; main() also refuses to save a registered copy below full data.
    from bimp_st.si_common import FIG_SUBSAMPLE_FRACTION as _FRAC

    _MIN_CELLS = max(10, int(round(100 * min(1.0, _FRAC))))
    row_order = [
        r
        for r in sorted(ct_tab.index, key=lambda x: int(x))
        if ct_tab.loc[r].sum() >= _MIN_CELLS
    ]
    col_order = sorted(ct_tab.columns)
    ct_tab = ct_tab.reindex(index=row_order, columns=col_order, fill_value=0)

    # Aggregate columns by pooled short label
    ct_tab = _pool_ct_table(ct_tab)

    # Normalize: proportion within each spatial cluster (row-norm)
    ct_normed = ct_tab.div(ct_tab.sum(axis=1), axis=0).fillna(0)

    # Labels. Columns use the full display names, not the pooled short codes:
    # "KC" and "OL" are also anatomical region codes in this paper, so the same
    # two letters meant a cell type here and a piece of tissue elsewhere. The
    # "(all subtypes)" marker also tells the reader which columns are pools.
    row_labels = [get_spatial_short(r) for r in row_order]
    col_labels = [
        get_pooled_display(c).replace(" (all subtypes)", "\n(all subtypes)")
        for c in ct_tab.columns
    ]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
    sns.heatmap(
        ct_normed,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Proportion within spatial cluster"},
    )
    ax.set_xlabel("snRNA cell type")
    ax.set_ylabel("Spatial Leiden cluster")
    if standalone:
        ax.set_title("Cluster correspondence", fontsize=11)
    # Row labels at the shared tick size (they were 2 pt under it and printed
    # at ~3 pt); the canvas is sized so 26 rows fit at that size.
    ax.tick_params(axis="y", labelsize=TICK_SIZE, length=0, pad=1.5)
    ax.tick_params(axis="x", length=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_label("Proportion within spatial cluster", fontsize=TICK_SIZE + 1)
    if standalone:
        fig.tight_layout()
        save(fig, output_dir, "cluster_heatmap")


# ── Panel C: LOO-CV score vs reference sparsity / expression ────────────────
# (lifted verbatim from the frozen caste_comparison_sccoda.py: _add_mean_expression +
#  plot_loocv_vs_sparsity)


def _add_mean_expression(loocv_df, tangram):
    """Add mean normalized expression (log10 CPM) to LOO-CV dataframe."""
    from scipy.sparse import issparse

    if "mean_expr" in loocv_df.columns:
        return loocv_df

    loocv_df = loocv_df.copy()
    if tangram is not None:
        # Gene names in loocv CSV are v4.3 lowercase; tangram var_names may differ
        gene_lookup = {g.lower(): g for g in tangram.var_names}
        mean_exprs = []
        for gene in loocv_df["gene"]:
            key = gene.lower()
            if key in gene_lookup:
                col = tangram[:, gene_lookup[key]].X
                if issparse(col):
                    col = col.toarray()
                # Spatial .X is raw counts — compute mean directly
                linear = np.asarray(col).flatten()
                linear = np.clip(linear, 0, None)  # guard negative
                mean_cpm = linear.mean()
                mean_exprs.append(np.log10(mean_cpm + 1))
            else:
                mean_exprs.append(np.nan)
        loocv_df["mean_expr"] = mean_exprs
    else:
        # Fallback: use 1 - sparsity as proxy (detection rate)
        loocv_df["mean_expr"] = 1 - loocv_df["sparsity_sc"]
    return loocv_df


def _annotate_rho(ax, rho):
    """Spearman rho inside the axes, upper-right."""
    ax.annotate(
        f"$\\rho$ = {rho:.2f}",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=TICK_SIZE + 0.5,
        fontweight="bold",
        color="#222222",
    )


def _plot_clipped_fit(ax, x, y):
    """Least-squares trend, drawn only where it is meaningful.

    Both relationships are visibly curved, so the line is a trend indicator,
    not a model: it is clipped to [0, 1] on y (the range cosine similarity can
    take) and dropped where it leaves that range.
    """
    z = np.polyfit(x, y, 1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ys = np.polyval(z, xs)
    keep = (ys >= 0.0) & (ys <= 1.0)
    if keep.sum() > 1:
        ax.plot(xs[keep], ys[keep], "--", color="#E15759", lw=1.4, alpha=0.75, zorder=1)


def plot_loocv_vs_sparsity(loocv_df, output_dir=None, tangram=None, axes=None):
    """LOO-CV cosine similarity vs reference sparsity and mean expression.

    Standalone (``axes=None``): builds its own 2-panel figure and saves
    ``loocv_vs_sparsity``. Embedded (a length-2 array of axes provided, e.g.
    from the SI validation figure): draws on those axes and does not save /
    add a figure suptitle. Drawing logic is identical in both paths.
    """
    print("  Panel C: LOO-CV score vs reference sparsity / expression...", flush=True)
    from scipy.stats import spearmanr

    loocv_df = _add_mean_expression(loocv_df, tangram)

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    y = loocv_df["cosine_sim"].values

    # Left: vs reference sparsity (kept for standalone diagnostics)
    ax = axes[0]
    x = loocv_df["sparsity_sc"].values
    ax.scatter(
        x, y, c="#4E79A7", s=14, alpha=0.7, edgecolors="white", lw=0.3, rasterized=True
    )
    rho, pval = spearmanr(x, y)
    ax.set_xlabel("Reference sparsity (fraction zero)")
    ax.set_ylabel("Cosine similarity (LOO-CV)")
    # rho in-axes, not as a title, which would collide with the "C" letter.
    _annotate_rho(ax, rho)
    _plot_clipped_fit(ax, x, y)

    # Right: vs mean expression
    ax = axes[1]
    x2 = loocv_df["mean_expr"].values
    ax.scatter(
        x2, y, c="#59A14F", s=14, alpha=0.7, edgecolors="white", lw=0.3, rasterized=True
    )
    valid = np.isfinite(x2) & np.isfinite(y)
    rho2, _ = spearmanr(x2[valid], y[valid]) if valid.sum() > 2 else (np.nan, np.nan)
    ax.set_xlabel("Mean expression (log$_{10}$ CPM)")
    ax.set_ylabel("Cosine similarity (LOO-CV)")
    _annotate_rho(ax, rho2)
    if valid.sum() > 2 and np.ptp(x2[valid]) > 0:
        _plot_clipped_fit(ax, x2[valid], y[valid])

    for a in axes:
        a.set_ylim(-0.1, 1.05)
    if standalone:
        fig.suptitle(
            f"LOO-CV gene imputation: mean={y.mean():.3f}, median={np.median(y):.3f}",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        save(fig, output_dir, "loocv_vs_sparsity")


# ── Composite SI figure ─────────────────────────────────────────────────────


def compose_figure(loocv_df, tangram, output_dir):
    """Assemble A (LOO-CV waterfall), B (cluster heatmap), C (scatters).

    Layout (two rows, wide rather than a tall single-column stack):
      Row 1 — A: LOO-CV ranked cosine-similarity waterfall (full width)
      Row 2 — B: cluster-correspondence heatmap | C: LOO-CV vs sparsity |
               LOO-CV vs mean expression (three axes side by side)
    B and C side by side keep the figure short enough to need no LaTeX
    height cap.
    """
    print("  Composing SI Tangram validation figure...", flush=True)

    # ~7.4 in wide (prints at 6.7 in): close to print size, so the hook's
    # single text factor stays near 1 and nothing has to be inflated.
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.85, 1.0],
        hspace=0.38,
        top=0.965,
        bottom=0.075,
        left=0.075,
        right=0.985,
    )

    # ── A: LOO-CV ranked cosine-similarity waterfall (full width) ──
    ax_a = fig.add_subplot(gs[0])
    plot_loocv_ranked(loocv_df, ax=ax_a)

    # ── Row 2: B (heatmap) + C (two scatters), side by side ──
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[1],
        width_ratios=[1.25, 1.0, 1.0],
        wspace=0.45,
    )
    ax_b = fig.add_subplot(gs_row2[0, 0])
    plot_cluster_heatmap(tangram, ax=ax_b)

    ax_c0 = fig.add_subplot(gs_row2[0, 1])
    ax_c1 = fig.add_subplot(gs_row2[0, 2])
    plot_loocv_vs_sparsity(loocv_df, tangram=tangram, axes=[ax_c0, ax_c1])

    # ── Panel letters + tags ──
    add_panel_headers(
        fig,
        [
            (ax_a, "A", "Imputation accuracy"),
            (ax_b, "B", "Cluster correspondence"),
            (ax_c0, "C", "Accuracy drivers"),
        ],
        width_frac=1.0,
    )

    # No figure-level title: the LaTeX caption already carries it.
    save(fig, output_dir, "figS_tangram_validation")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Loading {LOOCV_CSV}...", flush=True)
        loocv_df = pd.read_csv(LOOCV_CSV)
        print(f"  {len(loocv_df)} genes", flush=True)

        from bimp_st.si_common import FIG_SUBSAMPLE_FRACTION

        if FIG_SUBSAMPLE_FRACTION < 1.0:
            print(
                f"\n  !! PREVIEW MODE (BIMP_FIG_SUBSAMPLE_FRACTION="
                f"{FIG_SUBSAMPLE_FRACTION}). Panel B's cluster rows depend on "
                f"cell counts, so this render is NOT publication-ready -- "
                f"re-run with BIMP_FIG_SUBSAMPLE_FRACTION=1 before shipping.\n",
                flush=True,
            )

        print(f"Loading {TANGRAM_H5AD}...", flush=True)
        tangram = sc.read_h5ad(TANGRAM_H5AD)
        print(f"  {tangram.n_obs:,} cells", flush=True)
        tangram = maybe_subsample_adata(tangram)

        compose_figure(loocv_df, tangram, OUTPUT_DIR)

        print(f"\nDone. Output in {OUTPUT_DIR}/", flush=True)
