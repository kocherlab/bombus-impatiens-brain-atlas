"""Panel drawing for fig_neuropeptide_spatial.py:
one function per panel or panel element.

See fig_neuropeptide_spatial.py (the composer) for the figure and how to render it.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap as _ListedCmap  # noqa: E402
from neuropeptides.neuropeptide_spatial_data import (  # noqa: E402
    _image_window,
    _normalize_z,
    _panel_k_cb,
)

from bimp_st.si_common import (
    PIXEL_SCALE,
)
from bimp_st.si_common import (
    strip_spines as _strip_spines,
)
from bimp_st.style import (
    QUEEN_COLOR,
    WORKER_COLOR,
)

# MALDI ion-image colormap: a light-background ramp (pale gold -> deep red) that
# suits the white page and stays distinct from the viridis Xenium panels. The
# palest end is truncated so faint tissue still shows; off-tissue stays white.
MALDI_CMAP = _ListedCmap(plt.get_cmap("YlOrRd")(np.linspace(0.06, 1.0, 256)))
MALDI_CMAP.set_bad("white")

# Solid viridis for the spatial panels (low end stays visible as dark blue, not
# faded to white) + a sqrt power-norm so a wide dynamic range shows both the
# broad/low signal and the enriched high end without saturating.
from matplotlib.colors import PowerNorm as _PowerNorm  # noqa: E402

SOLID_VIRIDIS = plt.get_cmap("viridis").copy()
SOLID_VIRIDIS.set_bad("white")
SPATIAL_GAMMA = 0.5  # color-norm gamma (0.5 = sqrt); <1 lifts the low end


def _spatial_scatter(
    ax,
    sp,
    values,
    mask,
    cmap,
    vmax,
    title,
    shared_span,
    s=0.3,
    alpha=0.25,
    show_mean=True,
    vmin=0.0,
):
    """Full brain projection scatter. Axes are in microns (sp is in Xenium
    pixels -> x PIXEL_SCALE) so the window + 1 mm scale bar match the MALDI
    panels exactly. shared_span is passed in pixels. Cells below *vmin* (the
    per-peptide floor) are not drawn, so low-expression noise is suppressed."""
    idx = np.where(mask)[0]
    spum = sp * PIXEL_SCALE
    span_um = shared_span * PIXEL_SCALE
    # Plot only cells above the floor (drops low-expression haze); framing still
    # uses the full brain so the window/outline stays stable across panels.
    keep = idx[values[idx] >= vmin]
    order = keep[np.argsort(values[keep])]
    sc_obj = ax.scatter(
        spum[order, 0],
        spum[order, 1],
        c=values[order],
        cmap=cmap,
        s=s,
        alpha=alpha,
        edgecolors="none",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=7.5)
    # Frame tight to the brain (1-99.5 pctile drops scattered outliers/halo),
    # centered on the bbox; shared span keeps queen/worker bars comparable.
    xlo, xhi = np.percentile(spum[idx, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(spum[idx, 1], [0.5, 99.5])
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    ax.set_xlim(cx - span_um / 2, cx + span_um / 2)
    ax.set_ylim(cy - span_um / 2, cy + span_um / 2)
    ax.set_aspect("equal")
    _strip_spines(ax)
    if show_mean:
        mean_val = values[idx].mean()
        pct_pos = (values[idx] > 0).mean() * 100
        ax.text(
            0.98,
            0.02,
            f"mean={mean_val:.2f}\n{pct_pos:.0f}% expr.",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="0.3",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )
    return sc_obj


def _plot_density(ax, img, extent, vmin, vmax, title, cmap=None):
    """Render a smoothed density heatmap on a sqrt (power) color scale so a wide
    dynamic range shows both the broad/low signal and the enriched high end.
    Solid viridis (off-tissue white); low end stays visible (dark blue)."""
    ax.set_facecolor("white")
    norm = _PowerNorm(gamma=SPATIAL_GAMMA, vmin=0.0, vmax=vmax)
    im = ax.imshow(
        img,
        origin="lower",
        extent=extent,
        cmap=SOLID_VIRIDIS,
        norm=norm,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_title(title, fontsize=7.5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _strip_spines(ax)
    return im


def _apply_window(ax, window):
    """Apply a (cx, cy, span) square window with the y-axis inverted to match
    imshow(origin='upper')."""
    cx, cy, span = window
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy + span / 2, cy - span / 2)


def _plot_maldi_panel(ax, mip, extent, vmin, vmax, title, cmap="viridis", window=None):
    """Render a MALDI map with imshow. *window*=(cx, cy, span) frames the panel
    tight to the MALDI brain (shared across queen/worker so bars stay
    comparable); falls back to this map's own brain bbox."""
    ax.set_facecolor("white")
    # Mask empty (off-tissue) bins so the panel background is white.
    mip_m = np.ma.masked_less_equal(mip, 0.0)
    _cmap = MALDI_CMAP.copy()
    _cmap.set_bad("white")
    ax.imshow(
        mip_m,
        cmap=_cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(extent[0], extent[1], extent[3], extent[2]),
        origin="upper",
        interpolation="bilinear",
        aspect="equal",
    )

    _apply_window(ax, window if window is not None else _image_window(mip, extent))
    ax.set_title(title, fontsize=7.5, fontweight="bold")
    _strip_spines(ax)
    return vmax


def _panel_scalebar(ax, um=1000.0, label=None, color="black", right=True):
    """Scale bar at the lower right (or left) in axes coordinates, robust to
    view clipping and inverted y; its length is um divided by the visible data
    width, so it is a true 1 mm whenever the axes limits are in microns. All
    image panels share one physical span, so the figure needs one bar, drawn
    on panel A and labelled there."""
    w = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    frac = um / w if w else 0.2
    x_start = (0.94 - frac) if right else 0.06
    y = 0.05
    ax.plot(
        [x_start, x_start + frac],
        [y, y],
        transform=ax.transAxes,
        color=color,
        lw=2.5,
        solid_capstyle="butt",
        clip_on=False,
    )
    if label:
        ax.text(
            x_start + frac / 2,
            y - 0.02,
            label,
            transform=ax.transAxes,
            color=color,
            fontsize=_FS_TICK,
            ha="center",
            va="top",
        )


def _draw_ap_xenium(
    ax, expr_tk, expr_snpf, queen_mask, z_levels, fs=9, headroom=1.28, legend=True
):
    """Draw the Xenium A-P gradient lines (thick, left-axis data) onto ax.

    Xenium and MALDI are drawn in separate panels, each with its own y-axis:
    overlaying them on twin axes makes traces in different units look as if
    they start from the same value.
    """
    worker_mask = ~queen_mask
    z_norm = _normalize_z(z_levels, queen_mask)
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    styles = [("Tk", expr_tk, "-"), ("sNPF", expr_snpf, "--")]
    for gene_name, gene_expr, ls in styles:
        for brain_label, mask, color in [
            ("Queen", queen_mask, QUEEN_COLOR),
            ("Worker", worker_mask, WORKER_COLOR),
        ]:
            z = z_norm[mask]
            gx = gene_expr[mask]
            bin_idx = np.clip(np.digitize(z, bin_edges) - 1, 0, n_bins - 1)
            means = np.array(
                [
                    gx[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                    for i in range(n_bins)
                ]
            )
            sems = np.array(
                [
                    gx[bin_idx == i].std() / np.sqrt((bin_idx == i).sum())
                    if (bin_idx == i).sum() > 1
                    else 0
                    for i in range(n_bins)
                ]
            )
            valid = ~np.isnan(means)
            ax.plot(
                bin_centers[valid],
                means[valid],
                color=color,
                ls=ls,
                lw=2,
                label=f"{gene_name} {brain_label}",
            )
            ax.fill_between(
                bin_centers[valid],
                means[valid] - sems[valid],
                means[valid] + sems[valid],
                color=color,
                alpha=0.1,
            )
    ax.set_xlabel("Anterior → Posterior", fontsize=fs, labelpad=1)
    ax.set_ylabel("Xenium expr. (log1p)", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["A", "mid", "P"], fontsize=fs)
    ax.set_title("Xenium mRNA", fontsize=fs + 1, fontweight="bold", pad=8)

    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * headroom)
    if legend:
        from matplotlib.lines import Line2D

        leg = [
            Line2D([0], [0], color=QUEEN_COLOR, lw=2, label="Queen"),
            Line2D([0], [0], color=WORKER_COLOR, lw=2, label="Worker"),
            Line2D([0], [0], color="0.35", lw=2, ls="-", label="Tk"),
            Line2D([0], [0], color="0.35", lw=2, ls="--", label="sNPF"),
        ]
        # One row in the empty headroom band above the traces, below the title.
        ax.legend(
            handles=leg,
            frameon=False,
            fontsize=fs - 1.5,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            columnspacing=1.0,
            handlelength=2.0,
            handletextpad=0.4,
            borderaxespad=0.2,
        )


_PANEL_K_CB: dict = {}
# Optic-lobe prevalence row (n=1 dissection/caste): worker%, queen%, real_signal?.
# Source: data/sn/BIMP_atlas_qw_combined.h5ad, major_cell_type=='Optic lobe neurons'.
# Descriptive only: with one dissection per caste, a cell-level test would be
# pseudoreplicated, so no p-value. Replicated significance is the n=3 forest.
_PANEL_K_OL_PREV = {"sNPF": (0.2, 0.6, False), "Tk": (2.0, 12.6, True)}


def _draw_forest(ax, gene, show_ylabels):
    """C: central-brain caste log2FC forest for one peptide (DESeq2 n=3);
    filled + star at FDR<0.05."""
    if not _PANEL_K_CB:
        _PANEL_K_CB.update(_panel_k_cb())
    rows = _PANEL_K_CB[gene]
    y = np.arange(len(rows))[::-1]
    for yi, (_lab, lfc, se, padj) in zip(y, rows, strict=False):
        sig = np.isfinite(padj) and padj < 0.05
        ax.errorbar(
            lfc,
            yi,
            xerr=1.96 * se,
            fmt="o",
            ms=5.5,
            color=QUEEN_COLOR,
            ecolor="#B0842A",
            elinewidth=1.1,
            capsize=2.2,
            mfc=QUEEN_COLOR if sig else "white",
            mec=QUEEN_COLOR,
            mew=1.4,
            zorder=3,
        )
        if sig:
            ax.text(
                lfc,
                yi + 0.2,
                "✱",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#8A5A00",
                fontweight="bold",
            )
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([r[0] for r in rows], fontsize=_FS_TICK)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.55, len(rows) - 0.2)
    ax.set_xlim(-0.65, 2.3)
    ax.set_xlabel(
        "central brain caste log₂FC (queen-up →)", fontsize=_FS_LAB, labelpad=2
    )
    ax.set_title(gene, fontsize=_FS_TITLE, fontweight="bold", loc="left", pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=_FS_TICK, length=2, pad=1.5)


def _draw_optic_prev(axO):
    """D: optic-lobe recruitment dumbbells, worker -> queen % of optic-lobe
    neurons expressing; descriptive, n=1 dissection/caste."""
    genes = ["sNPF", "Tk"]
    yy = np.arange(len(genes))[::-1]
    for gy, gene in zip(yy, genes, strict=False):
        w, q, real = _PANEL_K_OL_PREV[gene]
        col = "#8A5A00" if real else "#9A9A9A"
        axO.plot(
            [w, q],
            [gy, gy],
            color=col if real else "#CFCFCF",
            lw=2.0,
            zorder=1,
            solid_capstyle="round",
        )
        axO.plot(w, gy, "o", ms=6.5, color=WORKER_COLOR, mec="white", mew=1.0, zorder=3)
        axO.plot(q, gy, "o", ms=7, color=QUEEN_COLOR, mec="white", mew=1.0, zorder=3)
        axO.text(
            w - 0.3,
            gy,
            f"{w:.1f}%",
            ha="right",
            va="center",
            fontsize=_FS_LAB,
            color=WORKER_COLOR,
            fontweight="bold",
        )
        axO.text(
            q + 0.3,
            gy,
            f"{q:.1f}%",
            ha="left",
            va="center",
            fontsize=_FS_LAB,
            color=QUEEN_COLOR,
            fontweight="bold",
        )
    axO.set_yticks(yy)
    axO.set_yticklabels(genes, fontsize=_FS_TICK, fontweight="bold")
    axO.set_ylim(-0.6, len(genes) - 0.4)
    axO.set_xlim(-1.4, 14.3)
    axO.set_xlabel(
        "% of optic lobe neurons expressing (worker ● → queen ●; n=1 dissection/caste)",
        fontsize=_FS_LAB,
        labelpad=2,
    )
    for s in ("top", "right"):
        axO.spines[s].set_visible(False)
    axO.tick_params(axis="y", length=0)
    axO.tick_params(axis="x", labelsize=_FS_TICK, length=2, pad=1.5)
_FS_TITLE = 7
_FS_LAB = 6.5
_FS_TICK = 6
