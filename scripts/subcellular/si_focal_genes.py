#!/usr/bin/env python3
"""Supplementary figure: Focal gene spatial expression.

Merged 5-row figure (``fig_s_focal_genes``): one row per focal gene
(Vg, yl, Kr-h1, GAD1, apolpp) x three columns (UMAP, queen spatial,
worker spatial), with a shared colour scale and colorbar per row. Five
single-gene figures (``fig_s_focal_<gene>``) are also written for the
LaTeX sources that reference them.

    PYTHONPATH=src .venv/bin/python scripts/subcellular/si_focal_genes.py

Rendering notes:

1. ``adata.X`` is log1p of a *large*-target normalisation, so its
   smallest non-zero value is ~4.5, not ~0.7. Colour-mapping it with
   ``vmin=0`` would put a single detected transcript ~80% of the way up
   the colormap, making the scale effectively binary. We re-normalise
   from the raw counts in ``adata.raw`` to counts-per-median log1p,
   so 0 -> 1 -> 2 -> 20 transcripts are visually distinct.
2. The spatial panels project ~16 z-slices onto one plane, so every
   screen pixel covers tens of cells. Drawing cells sorted by ascending
   expression (the usual scanpy idiom) then renders the *maximum* of
   each stack, which paints any broadly-detected gene as uniformly hot.
   We draw cells in random order instead: each pixel shows a uniformly
   sampled cell, so the rendered colour distribution is an unbiased
   image of the underlying population.
"""

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import (
    FOCAL_GENES,
    OUTPUT_DIR,
    SPATIAL_H5AD,
    TICK_SIZE,
    add_panel_headers,
    gene_display,
    get_expr,
    resolve_gene,
    save,
    set_style,
    strip_spines,
)

# Row order for the merged figure: Vg is paired with its receptor (yl).
ROW_ORDER = ["Vg", "yl", "Kr-h1", "GAD1", "apolpp"]

# Non-expressing cells. Deliberately a neutral grey that is distinct from
# the low end of FOCAL_CMAP, so "detected at 1 transcript" and "not
# detected" are not the same colour.
ZERO_COLOR = "#E4E4E4"

# Same warm ramp as EXPR_CMAP but starting at a pale cream instead of a
# near-white grey, so faint detections stay readable against ZERO_COLOR.
FOCAL_CMAP = LinearSegmentedColormap.from_list(
    "focal_expr",
    [
        "#FFF9C4",
        "#FEE08B",
        "#F46D43",
        "#D53E4F",
        "#9E0142",
    ],
    N=256,
)

# Xenium pixel size (µm/px) for the scale bars.
PIXEL_SCALE = 0.2125
SCALEBAR_UM = 1000.0

VMAX_PCTL = 99  # clip at the 99th pctile of *expressing* cells
RNG_SEED = 0

# Below this expressing fraction, detections are drawn on top of the grey
# background at an enlarged marker size (they are otherwise invisible among
# ~1.3M cells, and at this prevalence they do not overlap each other).
SPARSE_FRAC = 0.05
SPARSE_SIZE_BOOST = 12.0


# ── Expression ───────────────────────────────────────────────────────────────


def get_display_expr(adata, var_name, total_counts, median_total):
    """Counts-per-median log1p expression for one gene.

    ``adata.X`` is already normalised, but with a target sum large enough
    that its minimum non-zero value is ~4.5 -- useless as a colour scale
    anchored at 0. Recompute from the raw counts when they are available
    so the scale starts at a true zero.
    """
    counts = None
    if adata.raw is not None and var_name in adata.raw.var_names:
        X = adata.raw[:, var_name].X
        vals = (
            np.asarray(X.todense()).flatten()
            if hasattr(X, "todense")
            else np.asarray(X).flatten()
        )
        if np.allclose(vals, np.round(vals)):  # integer -> genuine counts
            counts = vals.astype(np.float32)

    if counts is None:
        # No raw counts: fall back to the stored values and let the caller
        # anchor vmin at the detection floor instead of at 0.
        vals = get_expr(adata, var_name)
        return vals, False

    scaled = counts / np.maximum(total_counts, 1.0) * median_total
    return np.log1p(scaled).astype(np.float32), True


# ── Rendering primitives ─────────────────────────────────────────────────────


def _scatter_cells(ax, xy, values, idx, vmin, vmax, size, rng, frac):
    """Scatter cells; zeros grey, detections colour-mapped.

    Two regimes, both chosen to avoid the max-bias described in the module
    docstring:

    * ``frac >= SPARSE_FRAC`` -- draw every cell in one pass in random
      order, so each screen pixel shows a uniformly sampled member of the
      cell stack underneath it.
    * ``frac < SPARSE_FRAC`` -- detections are rare enough that they
      essentially never overlap each other, so they can be drawn on top of
      the grey background at an enlarged marker size without biasing
      anything, and are otherwise invisible.
    """
    norm = Normalize(vmin, vmax)
    if frac < SPARSE_FRAC:
        pos = values[idx] > vmin
        zero_idx, pos_idx = idx[~pos], idx[pos]
        ax.scatter(
            xy[zero_idx, 0],
            xy[zero_idx, 1],
            c=ZERO_COLOR,
            s=size,
            edgecolors="none",
            rasterized=True,
        )
        pos_idx = pos_idx[rng.permutation(pos_idx.size)]
        ax.scatter(
            xy[pos_idx, 0],
            xy[pos_idx, 1],
            c=values[pos_idx],
            cmap=FOCAL_CMAP,
            norm=norm,
            s=size * SPARSE_SIZE_BOOST,
            edgecolors="none",
            rasterized=True,
        )
    else:
        order = idx[rng.permutation(idx.size)]
        vals = values[order]
        colors = FOCAL_CMAP(norm(vals))
        colors[vals <= vmin] = to_rgba(ZERO_COLOR)
        ax.scatter(
            xy[order, 0],
            xy[order, 1],
            c=colors,
            s=size,
            edgecolors="none",
            rasterized=True,
        )
    # Stand-alone mappable so colorbars carry the true scale.
    return plt.cm.ScalarMappable(norm=norm, cmap=FOCAL_CMAP)


def _frame_spatial(ax, sp, idx, span):
    """Centre the panel on the brain bbox using a span shared queen/worker."""
    xlo, xhi = np.percentile(sp[idx, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(sp[idx, 1], [0.5, 99.5])
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect("equal")


def _inset_scalebar(ax, bar_um=SCALEBAR_UM, label="1 mm"):
    """1 mm scale bar drawn inside the axes (coords are Xenium pixels)."""
    bar_px = bar_um / PIXEL_SCALE
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    span_x, span_y = xhi - xlo, yhi - ylo
    x0 = xlo + 0.05 * span_x
    y0 = ylo + 0.06 * span_y
    ax.plot(
        [x0, x0 + bar_px],
        [y0, y0],
        lw=2.2,
        color="black",
        solid_capstyle="butt",
        zorder=10,
    )
    ax.text(
        x0 + bar_px / 2,
        y0 + 0.015 * span_y,
        label,
        ha="center",
        va="bottom",
        fontsize=TICK_SIZE,
        zorder=10,
    )


def _pct_label(ax, values, idx):
    """Report the expressing fraction on the panel (it is the key number)."""
    frac = float((values[idx] > 0).mean()) * 100
    txt = f"{frac:.1f}% expressing" if frac < 10 else f"{frac:.0f}% expressing"
    ax.text(
        0.98,
        0.02,
        txt,
        transform=ax.transAxes,
        fontsize=TICK_SIZE,
        ha="right",
        va="bottom",
        color="0.35",
    )


def _gene_scales(values, has_zero_anchor):
    """(vmin, vmax) for one gene, shared by all panels in its row."""
    detected = values[values > 0]
    if detected.size == 0:
        return 0.0, 1.0
    vmin = 0.0 if has_zero_anchor else float(detected.min())
    vmax = float(np.percentile(detected, VMAX_PCTL))
    if vmax <= vmin:
        vmax = float(detected.max()) if detected.max() > vmin else vmin + 1.0
    return vmin, vmax


# ── Figures ──────────────────────────────────────────────────────────────────


def _prepare(adata):
    """Shared arrays + per-gene display expression and colour scales."""
    total = np.asarray(adata.obs["total_counts"].values, dtype=np.float64)
    median_total = float(np.median(total))

    q_mask = adata.obs["brain"].values == "queen"
    q_idx = np.where(q_mask)[0]
    w_idx = np.where(~q_mask)[0]
    sp = np.asarray(adata.obsm["spatial_aligned"])
    span = (
        max(
            np.ptp(sp[q_idx, 0]),
            np.ptp(sp[q_idx, 1]),
            np.ptp(sp[w_idx, 0]),
            np.ptp(sp[w_idx, 1]),
        )
        * 1.06
    )

    genes = {}
    for name in FOCAL_GENES:
        var_name = resolve_gene(name, adata.var_names)
        if var_name is None:
            print(f"    {name}: not found", flush=True)
            continue
        values, has_zero = get_display_expr(adata, var_name, total, median_total)
        vmin, vmax = _gene_scales(values, has_zero)
        frac = float((values > 0).mean())
        genes[name] = dict(values=values, vmin=vmin, vmax=vmax, frac=frac)
        print(
            f"    {name}: {frac:.2%} expressing "
            f"(queen {(values[q_idx] > 0).mean():.2%}, "
            f"worker {(values[w_idx] > 0).mean():.2%}); "
            f"scale 0-{vmax:.2f}",
            flush=True,
        )

    return dict(
        umap=np.asarray(adata.obsm["X_umap"]),
        sp=sp,
        q_idx=q_idx,
        w_idx=w_idx,
        span=span,
        genes=genes,
        n_obs=adata.n_obs,
    )


def _draw_row(
    ctx, name, axes, rng, *, umap_size=0.3, sp_size=0.3, show_col_titles=False
):
    """Draw one gene's (UMAP, queen, worker) triple. Returns the mappable."""
    g = ctx["genes"][name]
    values, vmin, vmax, frac = g["values"], g["vmin"], g["vmax"], g["frac"]
    all_idx = np.arange(values.size)

    ax = axes[0]
    sm = _scatter_cells(
        ax, ctx["umap"], values, all_idx, vmin, vmax, umap_size, rng, frac
    )
    ax.set_aspect("equal")
    strip_spines(ax)
    if show_col_titles:
        # Short column label, matching "Queen"/"Worker" alongside it; the
        # caption describes the rows, and a longer title collides with the
        # row's panel letter at (-0.04, 1.0).
        ax.set_title("UMAP")

    for ax, idx, label in (
        (axes[1], ctx["q_idx"], "Queen"),
        (axes[2], ctx["w_idx"], "Worker"),
    ):
        _scatter_cells(ax, ctx["sp"], values, idx, vmin, vmax, sp_size, rng, frac)
        _frame_spatial(ax, ctx["sp"], idx, ctx["span"])
        strip_spines(ax)
        _inset_scalebar(ax)
        _pct_label(ax, values, idx)
        if show_col_titles:
            ax.set_title(label)

    return sm


def plot_focal_genes_merged(ctx, output_dir, names=ROW_ORDER, tag="fig_s_focal_genes"):
    """Merged N-row x 3-column focal-gene figure for an arbitrary gene subset."""
    print(f"  Merged focal-gene figure ({tag})...", flush=True)
    rows = [g for g in names if g in ctx["genes"]]
    rng = np.random.default_rng(RNG_SEED)

    fig = plt.figure(figsize=(12.4, 3.85 * len(rows) + 0.25))
    gs = fig.add_gridspec(
        len(rows),
        4,
        width_ratios=[1, 1, 1, 0.045],
        wspace=0.03,
        hspace=0.04,
        left=0.055,
        right=0.935,
        top=1 - 0.5 / fig.get_figheight(),  # 0.5 in for the column titles
        bottom=0.008,
    )

    header_items = []
    for r, name in enumerate(rows):
        axes = [fig.add_subplot(gs[r, c]) for c in range(3)]
        sm = _draw_row(ctx, name, axes, rng, show_col_titles=False)
        if r == 0:
            # Column titles on one baseline, from the grid cells rather than
            # the equal-aspect axes (whose heights differ, so set_title put
            # "UMAP" lower than "Queen"/"Worker").
            for c, col in enumerate(("UMAP", "Queen", "Worker")):
                cell = gs[0, c].get_position(fig)
                fig.text(
                    (cell.x0 + cell.x1) / 2,
                    cell.y1 + 0.004,
                    col,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

        info = FOCAL_GENES[name]
        # Row label: italic gene symbol over its full name (two rotated
        # lines, symbol outermost). x=-0.11 keeps it in a separate column
        # from the panel letter at -0.04; otherwise its vertical extent hits
        # the letter and print_scale's collision check reverts all letters to
        # raw size.
        axes[0].text(
            -0.11,
            0.5,
            info["full"],
            transform=axes[0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
        )
        axes[0].annotate(
            gene_display(name),
            xy=(-0.11, 0.5),
            xycoords=axes[0].transAxes,
            xytext=(-1.25, 0),
            textcoords="offset fontsize",
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            fontstyle="italic",
        )
        # Letter only: the row label already names the gene.
        header_items.append((axes[0], "ABCDE"[r], None))

        cax = fig.add_subplot(gs[r, 3])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Norm. expression (log1p)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_visible(False)

    add_panel_headers(fig, header_items, width_frac=PRINTED_WIDTH_FRAC.get(tag, 1.0))
    save(fig, output_dir, tag)


def plot_focal_genes_single(ctx, output_dir):
    """Per-gene 3-panel figures (referenced by the LaTeX sources)."""
    print("  Per-gene focal figures...", flush=True)
    rng = np.random.default_rng(RNG_SEED)

    for name in ROW_ORDER:
        if name not in ctx["genes"]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
        sm = _draw_row(ctx, name, axes, rng, show_col_titles=True)

        info = FOCAL_GENES[name]
        fig.suptitle(
            f"{gene_display(name)} ({info['full']})",
            fontsize=12,
            fontweight="bold",
            y=0.99,
        )
        add_panel_headers(
            fig, [(axes[j], "ABC"[j], None) for j in range(3)], width_frac=1.0
        )
        fig.tight_layout(rect=(0, 0.02, 0.93, 0.96))
        cax = fig.add_axes((0.945, 0.15, 0.012, 0.65))
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Norm. expression (log1p)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(False)
        save(fig, output_dir, f"fig_s_focal_{name}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Loading {SPATIAL_H5AD}...", flush=True)
        adata = sc.read_h5ad(SPATIAL_H5AD)
        print(f"  {adata.n_obs:,} cells, {adata.n_vars} genes", flush=True)

        ctx = _prepare(adata)
        print(
            f"  Cells per panel: UMAP {ctx['n_obs']:,}; "
            f"queen {ctx['q_idx'].size:,}; worker {ctx['w_idx'].size:,}",
            flush=True,
        )

        plot_focal_genes_merged(ctx, OUTPUT_DIR)
        plot_focal_genes_merged(
            ctx, OUTPUT_DIR, names=["Vg", "yl", "Kr-h1"], tag="figS_focal_vg_krh1"
        )
        plot_focal_genes_single(ctx, OUTPUT_DIR)
        print(f"\nDone. Output in {OUTPUT_DIR}/", flush=True)
