"""Panel drawing for fig_kc_subtypes.py:
one function per panel or panel element.

See fig_kc_subtypes.py (the composer) for the figure and how to render it.
"""
from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports


import matplotlib

matplotlib.use("Agg")
import kenyon.snrna_kc_subclusters as snuc_kc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import shapely.ops as sops
from kenyon.kc_subtypes_data import (  # noqa: E402
    COMP_COLORS,
    COMP_ORDER,
    DATA,
    GENE_ID,
    N_MARKERS_PER_TYPE,
    SEED,
    SM_GRID,
    SUB_CLUSTERS,
    SUB_COLORS,
    SUB_LABELS,
    SUB_SHORT,
    SUBZONE_STATS,
    TYPE_COLORS,
    TYPE_ORDER,
    TYPE_SHORT,
    ZOOM_Z,
    _chosen_lobes,
    _hill1,
    _kc_marker_expr,
    _kc_type_markers,
    _lighten,
    _lobe_view,
    _marker_label,
    _morph_outline,
    _record_sn_caste_stats,
    _sankey_ribbon,
    _simplex_rgb,
    _smooth_sections,
    _sn_table,
    _subzone_data,
    load_kc5pop,
    load_ltkc,
    section_cells,
    zoom_window,
)
from matplotlib import patheffects as pe
from matplotlib import transforms as mtransforms
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle
from scipy.spatial import cKDTree

from bimp_st.si_common import PIXEL_SCALE
from bimp_st.style import (  # noqa: E402
    CASTE_COLORS,
)

rng = np.random.default_rng(SEED)
N_BOOT = 2000  # bootstrap resamples for _diversity_series()
# Drawn names of the compartments: each is named for the population that
# dominates it.  Code identifiers and data keys keep KC-xx.
COMP_DISPLAY = {"KC-IC": "Small-type", "KC-NC": "Large-type", "KC-OC": "Class II"}


# One title size shared by both rows.  The narrow spatial crops (C, F, G) are
# the tightest constraint; main() prints every title's width as a multiple of
# its panel width, so an overhang shows up in the log.
PANEL_TITLE_SM = 7.1


def strip(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def scale_bar(ax, bar_um, label, lw=3.0):
    """Horizontal scale bar for a spatial crop, drawn in the margin strip
    below the axes rather than on the section.

    Data coordinates are Xenium image pixels, not micrometres (same convention
    as _si_common.add_scalebar), hence the PIXEL_SCALE conversion.  The bar
    length is converted to a fraction of the x-window, so the three crops
    (C, F, G; one magnification via zoom_window()) print bars of identical
    length only if their axes are also the same physical width.  C and F/G
    sit in different sub-gridspecs, so main() prints all three widths every
    run; if they drift, retune the rows, not the type or the bar.

    The bar sits at the right end of the strip; the crop's legend or counts
    sit at the left end.  Offsets are in points below the axes, not axes
    fractions, so they do not drift on re-layout.
    """
    length = bar_um / PIXEL_SCALE
    x0, x1 = ax.get_xlim()
    frac = length / abs(x1 - x0)
    fig = ax.figure
    tr_bar = mtransforms.offset_copy(ax.transAxes, fig=fig, x=0, y=-9, units="points")
    ax.plot(
        [1.0 - frac, 1.0],
        [0.0, 0.0],
        transform=tr_bar,
        color="black",
        lw=lw,
        solid_capstyle="butt",
        clip_on=False,
        zorder=20,
    )
    tr_txt = mtransforms.offset_copy(ax.transAxes, fig=fig, x=0, y=-12, units="points")
    ax.text(
        1.0 - frac / 2,
        0.0,
        label,
        transform=tr_txt,
        ha="center",
        va="top",
        fontsize=8,
        clip_on=False,
        zorder=20,
    )
    # The bar occupies the right `frac` of the strip and the legend / counts
    # the left.  If frac approaches ~0.5 the two meet; shorten the bar, not
    # the type.
    print(f"    scale bar: {bar_um:.0f} um = {frac:.2f} of the crop width", flush=True)


def frame_crop(ax, brain):
    """Set the shared crop window on a spatial axes and draw the grey tissue
    context beneath.  Returns (window, in-window mask, section dict)."""
    centres, hx, hy = zoom_window()
    xc, yc = centres[brain]
    xlo, xhi, ylo, yhi = xc - hx, xc + hx, yc - hy, yc + hy
    sec = section_cells(brain, ZOOM_Z)
    ctx = sec["xy"]
    inwin = (
        (ctx[:, 0] >= xlo)
        & (ctx[:, 0] <= xhi)
        & (ctx[:, 1] >= ylo)
        & (ctx[:, 1] <= yhi)
    )
    ax.scatter(
        ctx[inwin, 0],
        ctx[inwin, 1],
        c="#DCDCDC",
        s=1.6,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.set_aspect("equal")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(yhi, ylo)  # inverted: image y runs downwards
    strip(ax)
    return (xlo, xhi, ylo, yhi), inwin, sec


# ── Panel A: all Kenyon-cell nuclei, snRNA reference ───────────────────────
def panel_a(ax):
    df = pd.read_parquet(DATA / "kc_umap.parquet")
    o = rng.permutation(len(df))
    d = df.iloc[o]
    # Large-type is outlined so its link to panel E (one arrow, drawn in
    # main()) is obvious.  The cloud is not convex and has two dense cores
    # joined by a sparse bridge, so a convex hull balloons over neighbouring
    # populations and an alpha shape splits it into two rings with a notch
    # between them; _morph_outline() encloses every point in one piece.
    ax.scatter(
        d["umap1"],
        d["umap2"],
        c=[TYPE_COLORS.get(t, "#CCCCCC") for t in d["cell_type"]],
        s=1.4,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
    lt = "Large-type Kenyon cells"
    pts = df.loc[(df["cell_type"] == lt).values, ["umap1", "umap2"]].values
    rings = []
    if len(pts) > 50:
        # close_r swept 0.10-0.30 UMAP units: 0.15 is the smallest radius
        # that merges the two dense cores into one piece through the sparse
        # bridge (the piece count is data-driven, not forced) while keeping
        # few nuclei of the other four KC populations inside; larger radii
        # take in more of them for no further gain.  The nuclei left outside
        # are isolated strays inside other populations' clouds.
        rings = _morph_outline(pts, close_r=0.15)
        for ring in rings:
            ax.add_patch(
                Polygon(
                    ring,
                    closed=True,
                    fill=False,
                    edgecolor=TYPE_COLORS[lt],
                    lw=2.8,
                    alpha=1.0,
                    zorder=5,
                    path_effects=[pe.withStroke(linewidth=5.2, foreground="white")],
                )
            )
        if rings:
            union = sops.unary_union([sgeom.Polygon(r) for r in rings])
            n_out = int(sum(not union.contains(sgeom.Point(p)) for p in pts))
        else:
            n_out = len(pts)
        print(
            f"  panel A: Large-type outline drew {len(rings)} piece(s) "
            f"from all {len(pts):,} nuclei ({n_out} outside the drawn "
            f"boundary, {n_out / len(pts):.2%} -- isolated strays inside "
            f"other populations' clouds, not a second bridging clump)",
            flush=True,
        )
    # Direct labels rather than a legend box: they cannot collide with the
    # cloud they name, and they save the reader a colour-matching step.  Sited
    # on the median so a long tail cannot drag a label off its own cloud.
    # Small-/Middle-type and Class II/Other KC sit close enough that their
    # white haloes touch at this panel width, so each pair is nudged apart by
    # a few per cent of the x-range in opposite directions.
    nudge = {
        "Small-type Kenyon cells": -0.16,
        "Middle-type Kenyon cells": 0.13,
        "Class II Kenyon cells": -0.10,
        "Other Kenyon cells": 0.11,
    }
    xspan = float(df["umap1"].max() - df["umap1"].min())
    label_texts = []
    for t in TYPE_ORDER:
        m = (df["cell_type"] == t).values
        if m.sum() < 30:
            continue
        cx, cy = df.loc[m, "umap1"].median(), df.loc[m, "umap2"].median()
        cx += nudge.get(t, 0.0) * xspan
        # 7.0 pt keeps neighbouring labels apart at this panel width.
        txt = ax.text(
            cx,
            cy,
            f"{TYPE_SHORT[t]}\n{int(m.sum()):,}",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=TYPE_COLORS[t],
            zorder=6,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )
        label_texts.append((t, txt))

    # Measured overlap check on the render, since the nudges cannot be
    # predicted reliably.  Boxes are inflated by the 3.0 pt white halo, the
    # part that visibly touches.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    halo_in = 3.0 / 72.0
    boxes = {}
    for t, txt in label_texts:
        bb = txt.get_window_extent(renderer)
        pad = halo_in * ax.figure.dpi
        boxes[t] = (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
    names = list(boxes)
    for i, t1 in enumerate(names):
        for t2 in names[i + 1 :]:
            b1, b2 = boxes[t1], boxes[t2]
            overlap = not (
                b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1]
            )
            if overlap:
                print(f"  panel A DIAG: label overlap {t1} <-> {t2}", flush=True)
    strip(ax)
    ax.set_xlabel("UMAP 1", labelpad=2)
    ax.set_ylabel("UMAP 2", labelpad=2)
    # "Kenyon cell nuclei", not "reference": these nuclei are the five
    # central-brain populations.  A sixth, generic Kenyon entry lives only in
    # the optic-lobe atlas, so this is not the whole reference.
    ax.set_title("Kenyon cell nuclei", fontsize=PANEL_TITLE_SM, pad=5)
    # No text label for the A -> E link: the arrow and panel E's title carry
    # it, and a label here would sit in the arrow's path.
    return len(df), rings


# ── Panel E: subclustering the Large-type population ──────────────────────
def panel_b(ax_umap, ax_markers, n_markers=5):
    """Reuses compute_kc_subcluster_data() from the standalone SI figure so
    the subclustering has one definition, but draws it locally: the SI
    version's third sub-panel (what fraction of the markers the Xenium panel
    carries) has its own SI figure, so it is not repeated here."""
    data = snuc_kc.compute_kc_subcluster_data(seed=SEED)
    ref, counts = data["ref"], data["counts"]
    col, lab_of = snuc_kc.COLORS, snuc_kc.LABELS
    clusters = snuc_kc.CLUSTERS

    xy = ref.obsm["X_umap"]
    lab = ref.obs[snuc_kc.CLUSTER_COL].values
    o = np.random.default_rng(SEED).permutation(len(xy))
    ax_umap.scatter(
        xy[o, 0],
        xy[o, 1],
        c=[col[lab[i]] for i in o],
        s=4.5,
        linewidths=0,
        alpha=0.75,
        rasterized=True,
    )
    xr, yr = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    nudge = {"Subcluster 2": (0.0, -0.12), "Subcluster 1": (0.07, -0.05)}
    for c in clusters:
        m = lab == c
        if not m.any():
            continue
        dx, dy = nudge.get(lab_of[c], (0.0, 0.0))
        ax_umap.text(
            xy[m, 0].mean() + dx * xr,
            xy[m, 1].mean() + dy * yr,
            f"{lab_of[c]}\n{counts.get(c, 0):,}",
            ha="center",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=col[c],
            zorder=6,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )
    strip(ax_umap)
    ax_umap.set_xlabel("UMAP 1", labelpad=2)
    ax_umap.set_ylabel("UMAP 2", labelpad=7)
    # The title and the arrow from A (see main()) tie this panel to A's
    # Large-type population.  It stays short: title width against this
    # column is the binding constraint, and main() prints every title's
    # width against its panel.
    ax_umap.set_title("Large-type subclusters", fontsize=PANEL_TITLE_SM, pad=5)

    panel_receptors(ax_markers)
    return ref.n_obs


# ── Panel C: where the three compartments are ──────────────────────────────
def panel_c(ax):
    """The anatomical frame for panels F and G, on the same section and the
    same magnification, so the reader can carry a location across.

    Compartment membership comes from the atlas (every cell of the section):
    this panel is about where the compartments are, and the counts have to
    mean that.
    """
    _, _, sec = frame_crop(ax, "queen")
    counts = {}
    for comp in COMP_ORDER:
        m = sec["comp"] == comp
        counts[comp] = int(m.sum())
        # s=4.0 matches the equivalent spatial crops of the other figures
        # (e.g. the spatial panels of `fig:atlas_overview`); only F/G use smaller
        # dots.
        ax.scatter(
            sec["xy"][m, 0],
            sec["xy"][m, 1],
            c=COMP_COLORS[comp],
            s=4.0,
            alpha=0.95,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )
    ax.set_title("Kenyon compartments", fontsize=PANEL_TITLE_SM, pad=5)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markersize=6.5,
            markerfacecolor=COMP_COLORS[c],
            label=f"{COMP_DISPLAY[c]} ({counts[c]:,})",
        )
        for c in COMP_ORDER
    ]
    # Legend below the axes, not on the tissue: the crop's upper left carries
    # grey context cells.  The axes are equal-aspect and anchored "N" in a
    # cell taller than the section, so this strip is empty page.  One column:
    # three across would not fit at this width without shrinking the type.
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.02),
        fontsize=6.6,
        frameon=False,
        handletextpad=0.2,
        borderpad=0.0,
        labelspacing=0.22,
    )
    scale_bar(ax, 200.0, "200 µm")
    print(
        "  panel C: queen z=%d — all atlas cells: " % ZOOM_Z
        + ", ".join(f"{c} {counts[c]:,}" for c in COMP_ORDER),
        flush=True,
    )
    return counts


def _sankey_minimize_crossings(tab, left_order, right_order, n_iter=12):
    """Order left/right nodes to minimize ribbon crossings (barycenter
    heuristic, alternating passes) instead of a fixed declared order, which
    just moves the "which end looks arbitrary" problem from one side to the
    other on a 5x3 table with several genuinely double-digit-percent cross
    links. Same method already used for the standalone
    celltype_region_sankey.py exploratory figure. `node_positions`
    below stacks bottom-up, so this returns each side ascending (first
    name lowest / last name highest) for a "biggest near the top" read."""
    left = list(left_order)
    right = list(right_order)

    def barycenter(order_self, order_other, axis):
        rank_other = {name: i for i, name in enumerate(order_other)}
        scores = {}
        for name in order_self:
            row = tab.loc[name] if axis == 0 else tab[name]
            w = row[row > 0]
            scores[name] = (
                sum(rank_other[k] * v for k, v in w.items()) / w.sum()
                if w.sum()
                else -1
            )
        return sorted(order_self, key=lambda n: scores[n])

    for _ in range(n_iter):
        new_left = barycenter(left, right, axis=0)
        new_right = barycenter(right, new_left, axis=1)
        if new_left == left and new_right == right:
            break
        left, right = new_left, new_right
    return left, right


def panel_sankey(ax):
    """Named KC population -> Kenyon compartment, as a sankey (panel B).

    Data are load_kc5pop()'s reclassification, restricted to
    Kenyon-compartment cells and A's 5 named populations with no
    escape-hatch categories.  Both castes are pooled: the compartment
    correspondence is caste-independent (see load_kc5pop()'s docstring).
    Ribbon width shows directly which population dominates which
    compartment, so no separate diversity summary is drawn.
    """
    d = load_kc5pop()
    argmax_type = np.array(TYPE_ORDER)[d["argmax"]]
    tab = pd.crosstab(argmax_type, d["comp"]).reindex(
        index=TYPE_ORDER, columns=COMP_ORDER, fill_value=0
    )

    # Reference (snRNA) nuclei counts per population, same source and same
    # counts as panel A's on-cloud labels -- printed beside each left node
    # here so a reader can tie this panel's population back to A's without
    # flipping between two different count systems (this panel's own node
    # heights are Xenium/spatial cell counts, not reference nuclei counts).
    ref_df = pd.read_parquet(DATA / "kc_umap.parquet")
    ref_n = {t: int((ref_df["cell_type"] == t).sum()) for t in TYPE_ORDER}

    left_totals = tab.sum(axis=1)
    right_totals = tab.sum(axis=0)
    total = int(left_totals.sum())
    gap = total * 0.028

    def node_positions(totals, order):
        pos, y = {}, 0.0
        for name in order:
            h = totals[name]
            pos[name] = (y, y + h)
            y += h + gap
        return pos, y - gap

    # Crossing-minimized order (barycenter heuristic), not a fixed list --
    # see _sankey_minimize_crossings(). node_positions stacks bottom-up
    # (first name lowest), so pass it reversed for a "biggest/most-connected
    # near the top" read.
    left_order, right_order = _sankey_minimize_crossings(tab, TYPE_ORDER, COMP_ORDER)
    left_pos, left_h = node_positions(left_totals, left_order[::-1])
    right_pos, right_h = node_positions(right_totals, right_order[::-1])
    y_shift = (left_h - right_h) / 2
    right_pos = {k: (v0 + y_shift, v1 + y_shift) for k, (v0, v1) in right_pos.items()}

    x_left, x_right, node_w = 0.0, 1.0, 0.03
    left_cursor = {k: v0 for k, (v0, v1) in left_pos.items()}
    right_cursor = {k: v0 for k, (v0, v1) in right_pos.items()}
    links = sorted(
        (
            (t, c, int(tab.loc[t, c]))
            for t in TYPE_ORDER
            for c in COMP_ORDER
            if tab.loc[t, c] > 0
        ),
        key=lambda r: r[2],
        reverse=True,
    )
    for t, c, n in links:
        ly0 = left_cursor[t]
        ly1 = ly0 + n
        left_cursor[t] = ly1
        ry0 = right_cursor[c]
        ry1 = ry0 + n
        right_cursor[c] = ry1
        path = _sankey_ribbon(x_left + node_w, ly0, ly1, x_right - node_w, ry0, ry1)
        ax.add_patch(
            PathPatch(
                path,
                facecolor=TYPE_COLORS[t],
                edgecolor="none",
                alpha=0.68,
                lw=0,
                zorder=2,
            )
        )

    for t in TYPE_ORDER:
        y0, y1 = left_pos[t]
        ax.add_patch(
            Rectangle(
                (x_left, y0),
                node_w,
                y1 - y0,
                facecolor=TYPE_COLORS[t],
                edgecolor="white",
                lw=0.5,
                zorder=3,
            )
        )
        ax.text(
            x_left - 0.03,
            (y0 + y1) / 2,
            f"{TYPE_SHORT[t]}\n(sn {ref_n[t]:,})",
            ha="right",
            va="center",
            fontsize=7.4,
            color=TYPE_COLORS[t],
            fontweight="bold",
            linespacing=1.1,
        )
    for c in COMP_ORDER:
        y0, y1 = right_pos[c]
        ax.add_patch(
            Rectangle(
                (x_right - node_w, y0),
                node_w,
                y1 - y0,
                facecolor=COMP_COLORS[c],
                edgecolor="white",
                lw=0.5,
                zorder=3,
            )
        )
        ax.text(
            x_right + 0.03,
            (y0 + y1) / 2,
            f"{COMP_DISPLAY[c]}\n({int(right_totals[c]):,})",
            ha="left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=COMP_COLORS[c],
            linespacing=1.1,
        )

    # Fixed small xlim margin, not one fitted to measured label extents: a
    # fitted xlim is tied to the axes' physical width at measurement time and
    # silently stops matching when the column's gridspec ratio is retuned.
    # Re-check visually if the column width or label lengths change a lot.
    ax.set_ylim(-total * 0.02, max(left_h, right_h) + total * 0.02)
    ax.set_xlim(-0.30, 1.30)
    ax.axis("off")
    ax.set_title("Population to compartment", fontsize=PANEL_TITLE_SM, pad=6)
    print(
        f"  panel B (sankey): n={total:,} Kenyon-compartment cells, "
        + ", ".join(
            f"{t}->{tab.loc[t].idxmax()} "
            f"({100 * tab.loc[t].max() / left_totals[t]:.0f}%)"
            for t in TYPE_ORDER
        ),
        flush=True,
    )
    return tab


# ── Panels F/G: the subcluster projection in tissue ────────────────────────
def panel_zoom(ax, brain, letter):
    """Per-cell ltKC subcluster probability vector, colour-blended (not
    argmaxed) over every KC-NC cell, on the same section and magnification as
    panel C.  `letter` is this call's own panel letter (F for queen, G for
    worker), passed in for the printed diagnostics.

    Colour comes from _simplex_rgb(): white where the three probabilities are
    equal, the pure subcluster colour only where one dominates.  An argmax
    colour would make a 40/35/25 cell look as confident as a 90/5/5 one and
    turn every subcluster boundary into an abrupt switch, which misstates a
    projection that does not form discrete territories.

    The subclusters are derived from Large-type Kenyon nuclei, so the only
    compartment where that vocabulary means anything is the non-compact one,
    KC-NC.  KC-IC and KC-OC are drawn as grey anatomy: painting a
    Large-type-derived identity across them would read as a gradient over the
    whole calyx.

    The projection covers every Kenyon-compartment cell of the current atlas
    with no cell-type gate (see load_ltkc()), so every KC-NC cell has a vector
    and is coloured; the printed coverage check flags any regression.
    """
    d = load_ltkc()
    win, _, sec = frame_crop(ax, brain)
    # Grey anatomy: every Kenyon-compartment cell of the section, so panel C's
    # geometry is still legible under the colour.
    kc = sec["comp"] != ""
    # Small dots (1.1 pt) so individual cells stay separable and the local
    # mixture reads through; panel C keeps the shared dot size.
    ax.scatter(
        sec["xy"][kc, 0],
        sec["xy"][kc, 1],
        c="#BDBDBD",
        s=1.1,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    n_nc_all = int((sec["comp"] == "KC-NC").sum())

    m = (d["brain"] == brain) & (d["z"] == ZOOM_Z) & (d["comp"] == "KC-NC")
    xy, am, pr = d["xy"][m], d["argmax"][m], d["probs"][m]
    # Smooth field: each pixel takes the mean probability vector of its 6
    # nearest KC-NC cells (adaptive, so no length scale is imposed), coloured
    # by _simplex_rgb() and drawn only within ~1.6 median nearest-neighbour
    # spacings of a cell.
    (xlo, xhi, ylo, yhi) = win
    tree = cKDTree(xy)
    nn = tree.query(xy, k=2)[0][:, 1]
    gx = np.linspace(xlo, xhi, 330)
    gy = np.linspace(ylo, yhi, max(int(330 * (yhi - ylo) / (xhi - xlo)), 20))
    GX, GY = np.meshgrid(gx, gy)
    dd, ii = tree.query(np.column_stack([GX.ravel(), GY.ravel()]), k=6)
    rgb = _simplex_rgb(pr[ii].mean(axis=1)).reshape(GX.shape + (3,))
    alpha = (dd[:, 0] < 1.6 * np.median(nn)).reshape(GX.shape).astype(float)
    ax.imshow(
        np.dstack([rgb, alpha]),
        origin="lower",
        extent=[xlo, xhi, ylo, yhi],
        zorder=5,
        interpolation="bilinear",
    )
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(yhi, ylo)
    n = int(m.sum())
    ax.set_title(f"{brain.capitalize()} subclusters", fontsize=PANEL_TITLE_SM, pad=5)
    # The n and the colour key are data and stay in the figure, but in the
    # margin strip below the section, not on the tissue.  Offsets are in
    # points from the axes' lower-left corner, so they do not drift on
    # re-layout.  No white halo: there is nothing behind this text.  The
    # argmax-vs-blend distinction is explained in the caption.
    ax.annotate(
        f"{n:,} cells\n(Large-type, all)",
        xy=(0.0, 0.0),
        xycoords="axes fraction",
        xytext=(0, -7),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6.6,
        color="#333333",
        fontweight="bold",
        linespacing=1.25,
        annotation_clip=False,
    )
    shares = {}
    counts = {}
    for i, c in enumerate(SUB_CLUSTERS):
        k = int((am == i).sum())
        shares[SUB_LABELS[c]] = 100 * k / n
        counts[SUB_LABELS[c]] = k
    # Colour key only; the shares are printed to stdout.
    keys = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markersize=4.5,
            markerfacecolor=SUB_COLORS[c],
            label=SUB_SHORT[c],
        )
        for c in SUB_CLUSTERS
    ]
    ax.legend(
        handles=keys,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.0),
        bbox_transform=ax.transAxes
        + mtransforms.ScaledTranslation(0, -27.5 / 72, ax.figure.dpi_scale_trans),
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handletextpad=0.05,
        columnspacing=0.6,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    # No caveat text on the panel; it belongs in the caption.
    # 200 um keeps the bar's fraction of the window (~0.18) well clear of the
    # counts at the left of the strip (see scale_bar()).
    scale_bar(ax, 200.0, "200 µm")
    # Coverage check: n (cells of this section with a subcluster vector) must
    # equal n_nc_all (KC-NC cells the atlas has there).  If they diverge, the
    # projection has been re-gated and F/G are drawing a subset.
    print(
        f"  panel {letter}: {brain} z={int(ZOOM_Z)}"
        f" coloured n={n:,} of {n_nc_all:,} KC-NC cells in the section"
        f" ({'COMPLETE' if n == n_nc_all else 'INCOMPLETE — re-gated!'}) — "
        + ", ".join(f"{k} {v:.1f}% ({counts[k]:,})" for k, v in shares.items()),
        flush=True,
    )
    return shares
# Wes Anderson hues for the genes: Grand Budapest pink (darkened for contrast on
# white) for Oamb, Moonrise charcoal for sNPF.
GENE_COLOR = {"Oamb": "#D1749F", "sNPF": "#29211F", "AChE": "#5E7681"}
GENE_RAMP = {
    "Oamb": ["#FFFFFF", "#E6A0C4", "#B34D82"],
    "sNPF": ["#FFFFFF", "#29211F"],
    "AChE": ["#FFFFFF", "#A9BAC0", "#4F6772"],
}
GENE_TAG = {
    "Oamb": ("S1 marker", "0"),
    "sNPF": ("S2 marker", "1"),
    "AChE": ("S3 marker", "2"),
}
ZONE_UM = 120.0  # near-core zone used for the pooled caste fold


def panel_markers(axH):
    """S1 and S2 markers measured in tissue, and where the caste effect sits.
    Top: Oamb (S1 marker) and sNPF (S2 marker) detection as smooth maps over
    the Large-type cells of one cup-shaped lobe per caste, with the cell
    outlines of the surrounding tissue.  Bottom: detection against distance to
    the Small-type core from ALL Large-type cells in ALL sections (bands:
    bootstrap over sections), queen and worker."""
    pick, _ = _chosen_lobes()
    views = {b: _lobe_view(b, pick[b]) for b in ("queen", "worker")}
    vmax = {
        g: max(np.nanpercentile(views[b]["field"][g], 98) for b in views)
        for g in GENE_ID
    }
    ramp = {
        g: LinearSegmentedColormap.from_list("r" + g, GENE_RAMP[g]) for g in GENE_ID
    }
    axH.axis("off")
    axH.set_title("S1, S2 and S3 markers in tissue", fontsize=PANEL_TITLE_SM, pad=5)
    ctx_other = "#D8D8D8"
    ctx_core = _lighten(COMP_COLORS["KC-IC"], 0.45)
    rowy = {"Oamb": 0.680, "sNPF": 0.340, "AChE": 0.0}
    mh, mw = 0.298, 0.37
    for g in GENE_ID:
        for ci, b in enumerate(("queen", "worker")):
            ax = axH.inset_axes([0.17 + ci * (mw + 0.02), rowy[g], mw, mh])
            axH.__dict__.setdefault("map_axes", []).append(ax)
            V = views[b]
            ax.add_collection(
                PolyCollection(
                    [V["polys"][q] for q in V["other"]],
                    facecolors=[ctx_other] * len(V["other"]),
                    edgecolors="white",
                    linewidths=0.08,
                    zorder=1,
                )
            )
            ax.add_collection(
                PolyCollection(
                    [V["polys"][q] for q in V["core"]],
                    facecolors=ctx_core,
                    edgecolors="white",
                    linewidths=0.08,
                    zorder=2,
                )
            )
            ax.add_collection(
                PolyCollection(
                    [V["polys"][q] for q in V["large"]],
                    facecolors="#E6E6E6",
                    edgecolors="white",
                    linewidths=0.08,
                    zorder=2.5,
                )
            )
            ax.imshow(
                V["field"][g],
                extent=[V["gx"][0], V["gx"][-1], V["gy"][-1], V["gy"][0]],
                cmap=ramp[g],
                vmin=0,
                vmax=vmax[g],
                interpolation="bilinear",
                zorder=3,
            )
            cu = (
                sops.unary_union(
                    [
                        sgeom.Polygon(V["polys"][q]).buffer(0.3)
                        for q in V["core"]
                        if len(V["polys"][q]) > 2
                    ]
                )
                .buffer(2)
                .buffer(-2)
            )
            for gp in cu.geoms if hasattr(cu, "geoms") else [cu]:
                xs, ys = gp.exterior.xy
                ax.plot(xs, ys, color="#4a3a52", lw=0.4, zorder=4)
            c, W, WY = V["c"], V["W"], V["WY"]
            ax.set_xlim(c[0] - W, c[0] + W)
            ax.set_ylim(c[1] + WY, c[1] - WY)
            ax.set_aspect("equal")
            strip(ax)
            if g == "Oamb":
                ax.set_title(
                    b.capitalize(),
                    fontsize=7.6,
                    fontweight="bold",
                    color=CASTE_COLORS[b],
                    pad=2,
                )
            if ci == 0:
                ax.text(
                    -0.05,
                    0.5,
                    g,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=7.6,
                    fontweight="bold",
                    style="italic",
                    color=GENE_COLOR[g],
                )
                tag, k = GENE_TAG[g]
                ax.text(
                    -0.27,
                    0.5,
                    tag,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=6.4,
                    fontweight="bold",
                    color=SUB_COLORS[k],
                )
            if g == "AChE" and ci == 0:
                # scale bar below the last map row, clear of the tissue
                ax.plot(
                    [c[0] - W, c[0] - W + 100],
                    [-0.06] * 2,
                    color="k",
                    lw=1.4,
                    clip_on=False,
                    transform=ax.get_xaxis_transform(),
                )
                ax.text(
                    c[0] - W + 50,
                    -0.11,
                    "100 µm",
                    fontsize=5.2,
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                )
        cax = axH.inset_axes([0.945, rowy[g] + 0.06, 0.018, mh * 0.62])
        cb = plt.colorbar(
            plt.cm.ScalarMappable(cmap=ramp[g], norm=plt.Normalize(0, vmax[g])), cax=cax
        )
        cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
        cax.set_title("% cells", fontsize=5.4, pad=2)
    axH.legend(
        handles=[
            Patch(facecolor=ctx_core, edgecolor="#4a3a52", label="Small-type core"),
            Patch(facecolor=ctx_other, label="other tissue"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=5.6,
        handlelength=1.1,
        columnspacing=0.9,
    )


def panel_distance(axp):
    """Detection against distance to the Small-type core from ALL Large-type
    cells in ALL sections (bands: bootstrap over sections), queen and worker;
    sNPF solid, Oamb dashed."""
    d = _subzone_data()
    axp.set_title(
        "Caste effect by distance from the core", fontsize=PANEL_TITLE_SM, pad=5
    )
    x = 10**SM_GRID
    styles = (
        ("sNPF", "-", 1.5),
        ("Oamb", (0, (3, 2)), 1.25),
        ("AChE", (0, (1, 1.2)), 1.4),
    )
    for b in ("queen", "worker"):
        u = np.isfinite(d["dist"]) & (d["brain"] == b)
        for g, ls, lw in styles:
            e, lo, hi = _smooth_sections(
                d["dist"][u], d["det"][g][u] * 100.0, d["sl"][u]
            )
            axp.plot(x, e, color=CASTE_COLORS[b], lw=lw, ls=ls)
            if g == "sNPF":
                axp.fill_between(x, lo, hi, color=CASTE_COLORS[b], alpha=0.18, lw=0)
    # vertical key to the right of the plot: gene line styles, then the castes
    gene_h = [Line2D([], [], color="#444444", lw=lw, ls=ls) for _, ls, lw in styles]
    caste_h = [
        Line2D([], [], color=CASTE_COLORS[c_], lw=1.6) for c_ in ("queen", "worker")
    ]
    gap_h = Line2D([], [], lw=0)
    keys = ("sNPF", "Oamb", "AChE", " ", "queen", "worker")
    lg = axp.legend(
        handles=gene_h + [gap_h] + caste_h,
        labels=list(keys),
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.09, 0.5),
        ncol=1,
        fontsize=5.4,
        handlelength=1.6,
        labelspacing=0.45,
        borderaxespad=0.0,
        handletextpad=0.4,
    )
    for t_, k_ in zip(lg.get_texts(), keys, strict=False):
        t_.set_fontweight("bold")
        if k_ in GENE_COLOR:
            t_.set_color(GENE_COLOR[k_])
            t_.set_style("italic")
        elif k_ in CASTE_COLORS:
            t_.set_color(CASTE_COLORS[k_])
    axp.set_xscale("log")
    axp.set_xlim(6, 250)
    axp.set_ylim(0, 105)
    axp.set_yticks([0, 25, 50, 75, 100])
    axp.set_xticks([10, 30, 100, 250], ["10", "30", "100", "250"])
    axp.set_xlabel("Distance from Small-type core (µm)", fontsize=6.4, labelpad=1)
    axp.set_ylabel("% cells", fontsize=6.4, labelpad=1)
    for s_ in ("top", "right"):
        axp.spines[s_].set_visible(False)
    axp.tick_params(length=2.0, width=0.5, labelsize=5.8, pad=1.5)
    # summary numbers
    for b in ("queen", "worker"):
        u = np.isfinite(d["dist"]) & (d["brain"] == b)
        SUBZONE_STATS[f"n_cells_{b}"] = int(u.sum())
        SUBZONE_STATS[f"n_sections_{b}"] = int(len(np.unique(d["sl"][u])))
        near = []
        for s_ in np.unique(d["sl"][u]):
            mm = u & (d["sl"] == s_) & (d["dist"] < 30)
            if mm.sum() > 100:
                near.append(d["det"]["sNPF"][mm].mean() * 100)
        SUBZONE_STATS[f"snpf_pct_near30_{b}"] = float(np.median(near))
        near = []
        for s_ in np.unique(d["sl"][u]):
            mm = u & (d["sl"] == s_) & (d["dist"] < 30)
            if mm.sum() > 100:
                near.append(d["det"]["Oamb"][mm].mean() * 100)
        SUBZONE_STATS[f"oamb_pct_near30_{b}"] = float(np.median(near))
    use = np.isfinite(d["dist"])

    def _fold(lo, hi):
        a = d["det"]["sNPF"]
        q = a[use & (d["brain"] == "queen") & (d["dist"] >= lo) & (d["dist"] < hi)]
        w_ = a[use & (d["brain"] == "worker") & (d["dist"] >= lo) & (d["dist"] < hi)]
        return float(q.mean() / w_.mean())

    SUBZONE_STATS["snpf_fold_within120"] = _fold(0, ZONE_UM)
    SUBZONE_STATS["snpf_fold_beyond120"] = _fold(ZONE_UM, 1e9)
    SUBZONE_STATS["snpf_fold_all"] = _fold(0, 1e9)
    print(
        "  panel G: "
        + ", ".join(
            f"{k} {v:.2f}"
            for k, v in SUBZONE_STATS.items()
            if k.startswith(("snpf_fold", "snpf_pct"))
        ),
        flush=True,
    )


def panel_receptors(ax):
    """The three panel genes most enriched in each subcluster, so the markers
    mapped in panel H are visibly the top of a ranking.  Enrichment = a
    subcluster's detection over the highest other subcluster's (counts + 1),
    among named genes detected in >= 10% of its nuclei; each gene appears once,
    under the subcluster where it ranks.  Panel H maps the highest-ranked
    receptor, neuropeptide or transmitter enzyme gene of each subcluster
    (bold, in its panel H colour).  Dot area = % nuclei."""
    T, pers = _sn_table()
    for gname in pers:
        _record_sn_caste_stats(gname, pers[gname])
    T = T[~T.name.str.fullmatch(r"\d+")]
    rows, seen = [], set()
    for k in ("S1", "S2", "S3"):
        o = [c for c in ("S1", "S2", "S3") if c != k]
        c = (
            T[T[k] >= 10]
            .assign(fold=(T[k] + 1) / (T[o].max(axis=1) + 1))
            .sort_values("fold", ascending=False)
        )
        got = 0
        for r in c.itertuples():
            if r.name not in seen and got < 3:
                rows.append((k, r))
                seen.add(r.name)
                got += 1
    n = len(rows)
    ys, y = [], 0.0
    for i, (k, _) in enumerate(rows):
        if i and rows[i - 1][0] != k:
            y -= 0.55
        ys.append(y)
        y -= 1.0
    names = [
        r.name.replace("neuronal acetylcholine receptor", "nAChR").replace(
            "LIM domain only protein 3", "LIM protein 3"
        )
        for _, r in rows
    ]
    for (k, r), nm, yv in zip(rows, names, ys, strict=False):
        for j, c in enumerate(SUB_CLUSTERS):
            v = (r.S1, r.S2, r.S3)[j]
            ax.scatter(
                j, yv, s=v * 1.9 + 4, color=SUB_COLORS[c], alpha=0.92, linewidths=0
            )
        mapped = nm in GENE_COLOR
        ax.text(
            -0.55,
            yv,
            nm,
            ha="right",
            va="center",
            fontsize=6.0,
            fontweight="bold" if mapped else "normal",
            color=GENE_COLOR.get(nm, "black"),
        )
        ax.text(
            2.95,
            yv,
            f"{r.fold:.1f}x",
            ha="left",
            va="center",
            fontsize=5.8,
            color=SUB_COLORS[SUB_CLUSTERS[int(k[1]) - 1]],
            fontweight="bold" if mapped else "normal",
        )
    ax.text(
        2.95, 0.95, "enrichment", ha="left", va="bottom", fontsize=5.4, color="#555555"
    )
    for i in range(1, n):
        if rows[i - 1][0] != rows[i][0]:
            ax.axhline(
                (ys[i] + ys[i - 1]) / 2, xmin=0.02, xmax=1.0, color="#CCCCCC", lw=0.5
            )
    ax.set_yticks([])
    ax.set_xticks(range(3), [SUB_SHORT[c] for c in SUB_CLUSTERS], fontsize=6.6)
    for t, c in zip(ax.get_xticklabels(), SUB_CLUSTERS, strict=False):
        t.set_color(SUB_COLORS[c])
        t.set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.set_xlim(-2.4, 3.7)
    ax.set_ylim(min(ys) - 3.6, 1.6)
    ax.tick_params(length=0, pad=1.5)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("Top three enriched panel genes\nper subcluster", fontsize=6.4, pad=14)
    hs = [
        ax.scatter([], [], s=p * 1.9 + 4, color="#9A9A9A", linewidths=0)
        for p in (10, 30, 60)
    ]
    ax.legend(
        hs,
        ["10%", "30%", "60%"],
        title="dot area = % nuclei",
        title_fontsize=5.6,
        fontsize=5.6,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.45, -0.01),
        handletextpad=0.3,
        columnspacing=1.4,
        borderpad=0.2,
        labelspacing=0.6,
    )
    SUBZONE_STATS["dot_genes"] = ";".join(names)


def panel_caste_dots(ax, gene):
    """Nuclei detecting `gene` by subcluster, queen vs worker, per sample
    (3 vs 3); black bars are means."""
    _, pers = _sn_table()
    per = pers[gene]
    for j, k in enumerate(SUB_CLUSTERS):
        for ci, b in enumerate(("queen", "worker")):
            v = per[per.caste == b].sort_index(key=lambda i: i.str[-2:])[k].values
            xx = j + (ci - 0.5) * 0.36
            ax.scatter(
                xx + np.array([-0.05, 0, 0.05]),
                v,
                s=9,
                color=CASTE_COLORS[b],
                zorder=3,
                linewidths=0,
            )
            ax.hlines(v.mean(), xx - 0.15, xx + 0.15, color="k", lw=1.1, zorder=4)
    ax.set_xticks(range(3), [SUB_SHORT[c] for c in SUB_CLUSTERS], fontsize=6.6)
    for t, c in zip(ax.get_xticklabels(), SUB_CLUSTERS, strict=False):
        t.set_color(SUB_COLORS[c])
        t.set_fontweight("bold")
    ax.set_xlim(-0.55, 2.55)
    ax.set_title(
        f"{gene}\n({GENE_TAG[gene][0]})",
        fontsize=6.4,
        fontweight="bold",
        style="italic",
        color=GENE_COLOR[gene],
        pad=2,
    )
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=2.0, width=0.5, labelsize=6.2, pad=1.5)


def _diversity_series(ax, groups, depth, marker, filled, dodge=0.17):
    """One estimate per compartment x caste with a 95% bootstrap interval,
    reported as the Hill number of order 1 (effective number of
    equally-common categories, exp(Shannon)) rather than raw Shannon nats.

    Raw Shannon needs its ceiling (ln K) before a value can be judged; Hill-1
    reads directly ("behaves like 5.6 of 25 equally common categories") and
    its ceiling is the bare count K.  This is the project's convention
    elsewhere (`fig:caste_comparison` D, si_hill_diversity.py, both built on
    region_diversity.py's Hill-number machinery).

    `groups` maps (compartment, caste) -> category counts, resampled N_BOOT
    times with SEED at a rarefaction depth `depth` common to all groups.
    `marker` and `filled` set the two-tone marker convention shared with
    other diversity panels in this figure set.
    """
    boot = np.random.default_rng(SEED)
    xs = np.arange(len(COMP_ORDER))
    out = {}
    for caste, sgn in [("queen", -1), ("worker", 1)]:
        col = CASTE_COLORS[caste]
        for i, comp in enumerate(COMP_ORDER):
            n = np.asarray(groups[(comp, caste)], dtype=float)
            est = _hill1(n)
            draws = boot.multinomial(depth, n / n.sum(), size=N_BOOT)
            s = np.array([_hill1(row) for row in draws])
            lo, hi = np.percentile(s, [2.5, 97.5])
            xc = xs[i] + sgn * dodge
            ax.errorbar(
                xc,
                est,
                yerr=[[est - lo], [hi - est]],
                fmt="none",
                ecolor=col,
                elinewidth=1.5,
                capsize=4.0,
                capthick=1.5,
                zorder=3,
            )
            ax.scatter(
                [xc],
                [est],
                marker=marker,
                s=58,
                c=col if filled else "none",
                edgecolors="white" if filled else col,
                linewidths=1.1 if filled else 1.6,
                zorder=5,
            )
            out[(comp, caste)] = (est, lo, hi, int(n.sum()))
    ax.set_xlim(-0.55, len(COMP_ORDER) - 0.45)
    return out


def _fit_ylim(ax, out, ceiling=None):
    lo_all = min(v[1] for v in out.values())
    hi_all = max(v[2] for v in out.values())
    if ceiling is not None:
        hi_all = max(hi_all, ceiling)
    pad = 0.10 * (hi_all - lo_all)
    ax.set_ylim(lo_all - pad, hi_all + pad)


# ── Panel D: KC-type marker gene expression, by compartment and caste ──────
def panel_h(ax, n_per_type=N_MARKERS_PER_TYPE):
    markers = _kc_type_markers(n_per_type)
    expr = _kc_marker_expr()
    piv = expr.pivot_table(index="gene", columns=["comp", "brain"], values="mean_expr")
    # Row order: grouped by TYPE_ORDER (not by expression magnitude), so a
    # reader can read the panel as "here is Large-type's pair, here is
    # Small-type's pair, ..." matching panel A's population order, rather
    # than an order that scrambles genes across types by how bright their
    # colour happens to be.
    type_rank = {t: i for i, t in enumerate(TYPE_ORDER)}
    markers_sorted = markers.assign(
        _torder=markers["cell_type"].map(type_rank)
    ).sort_values(["_torder", "rank"])
    piv = piv.loc[markers_sorted["gene"].values]
    piv = piv[[(c, b) for c in COMP_ORDER for b in ("queen", "worker")]]

    # Z-score each gene across its own six compartment x caste means: red =
    # higher than this gene's own average across the six groups, not an
    # absolute-expression comparison between genes on different scales.
    z = piv.sub(piv.mean(axis=1), axis=0).div(piv.std(axis=1) + 1e-9, axis=0)
    vmax = float(np.abs(z.values).max())
    im = ax.imshow(
        z.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(-0.5, piv.shape[1] - 0.5, len(piv) - 0.5, -0.5),
    )

    gene_to_type = markers.set_index("gene")["cell_type"]
    gene_to_name = markers.set_index("gene")["gene_name"]
    row_labels = [_marker_label(g, gene_to_name) for g in piv.index]

    # Left colour strip: which KC population this gene marks, in panel A's
    # own TYPE_COLORS, tying the panel back to A instead of a new key.
    ax.set_xlim(-1.55, piv.shape[1] - 0.5)
    for i, g in enumerate(piv.index):
        ax.add_patch(
            plt.Rectangle(
                (-1.42, i - 0.42),
                0.62,
                0.84,
                facecolor=TYPE_COLORS[gene_to_type[g]],
                edgecolor="none",
                clip_on=False,
                zorder=4,
            )
        )
    for i in range(n_per_type, len(piv), n_per_type):
        ax.axhline(i - 0.5, color="white", lw=2.5, zorder=5)

    ax.set_xticks(range(piv.shape[1]), ["Q", "W"] * len(COMP_ORDER), fontsize=8.5)
    # 6.6 pt: these labels are what reaches left toward C (see gCD's wspace).
    ax.set_yticks(range(len(piv)), row_labels, fontsize=6.6)
    for lbl, g in zip(ax.get_yticklabels(), piv.index, strict=True):
        lbl.set_color(TYPE_COLORS[gene_to_type[g]])
        lbl.set_fontweight("bold")
    # No ylabel: the tick labels are gene names, self-explanatory given the
    # title, and a ylabel placed outside these long labels would push into
    # panel C at this column width.
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, comp in enumerate(COMP_ORDER):
        if i:
            ax.axvline(2 * i - 0.5, color="white", lw=2.5, zorder=5)
        ax.text(
            2 * i + 0.5,
            -0.085,
            COMP_DISPLAY[comp],
            ha="center",
            va="top",
            fontsize=6.2,
            fontweight="bold",
            color=COMP_COLORS[comp],
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Expression (z-scored per gene)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("Marker genes", fontsize=PANEL_TITLE_SM, pad=6)

    print(
        f"  panel D: {len(piv)} marker genes ({n_per_type}/type x "
        f"{len(TYPE_ORDER)} types), z-scored across the 6 compartment x "
        f"caste columns",
        flush=True,
    )
    for g in piv.index:
        row = piv.loc[g]
        print(
            f"    {gene_to_name[g]:10s} ({TYPE_SHORT[gene_to_type[g]]}): "
            + ", ".join(
                f"{c[0][3:]}-{c[1][0].upper()}={v:.2f}" for c, v in row.items()
            ),
            flush=True,
        )
    return piv
