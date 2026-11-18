"""Kenyon cells: finer gradation than the classical scheme.

Draws figure ``fig:kc_subtypes``,
results/figures/main/kc_subtypes/fig_kc_subtypes.*.

Layout, two rows, read left to right:

  Row 1  A  all-KC snRNA UMAP, five named populations, Large-type outlined
         B  sankey: named KC population -> Kenyon compartment
         C  queen section (z = 260): the three Kenyon compartments
         D  KC-type marker-gene expression, compartment x caste
  Row 2  E  Large-type subclusters: UMAP + top enriched panel genes
         F, G  queen / worker subcluster projection on KC-NC cells
         H  S1/S2/S3 marker detection in one calyx lobe per caste
         I  marker detection against distance from the Small-type core

One arrow in Large-type's colour runs from panel A's Large-type outline down
into E, the panel that subclusters that population.

The message:
  1. The classical morphological scheme names three to four Kenyon-cell (KC)
     classes.  The snRNA-seq reference resolves five populations in the
     central brain, and the largest (Large-type, "ltKC" -- Large-type Kenyon
     cells, never "long-tail") splits again into three de-novo
     transcriptomic subclusters (A, E).
  2. Each anatomical compartment (KC-IC, KC-NC, KC-OC) is dominated by a
     different named population (B, restricted-Tangram reclassification,
     load_kc5pop()), and each population's marker genes track that across
     compartment and caste (D).
  3. The three subclusters can be projected onto the KC-NC compartment as
     per-cell Tangram probability vectors (F, G); their markers are mapped in
     tissue and against distance from the core (H, I).
  4. Caste is descriptive throughout: the spatial cohort is one brain per
     caste, so no spatial panel carries a caste test, a P value or a
     significance mark.

The subcluster projection (F, G).  Only 1 of the top 50 marker genes per
subcluster is on the 101-gene Xenium panel (3 of 150 in total), so the genes
that separate the subclusters are nearly invisible to the spatial assay and
the projection is soft.  It is a projection, not a measured spatial gradient;
do not caption it as "Kenyon cells form a continuum".  The blended colouring
keeps unresolved cells white.  The subclusters come from Large-type nuclei,
and Large-type is expected in the non-compact compartment, so only KC-NC
cells are coloured and KC-IC / KC-OC are grey anatomy.  Inside KC-NC the
S1-vs-S2 call is largely position (distance to the nearest KC-IC cell
predicts it, within every cell type sharing the compartment),
so F and G must be captioned as position, never as subtype territories.

Text on the artwork.  Panels carry data only (axis and tick labels, titles,
legends, n's, scale bars); caveats belong in the caption.  No text sits on
the tissue: the crops' legends, counts and scale bars are drawn in the empty
strip below each equal-aspect axes.  That strip lies outside the axes bbox,
so a blank-pixel scan inside the axes does not check its clearance; check
the render.

Which Tangram run.  F and G read ltKC_CB_spatial_subclusters_dsource.h5ad
(scripts/kenyon/project_large_type_subclusters.py): the query is every cell in
a Kenyon compartment of the current atlas, with no cell-type gate, and the
reference is the CB atlas's own 17,553 Large-type nuclei (see load_ltkc()).
Other Tangram runs are not substitutes for these.

Panel D ranks marker genes per KC population in the whole-transcriptome
snRNA reference, keeps those on the Xenium panel, and reads each gene's
measured spatial expression per compartment x caste.  No per-cell type call,
Tangram argmax or unsupervised clustering is involved.  Ranking genes within
spatial Leiden clusters or compartments instead gives no usable structure on
the 100-gene panel, and the raw per-cell Tangram KC-type calls are unreliable
("Small-type" acts as a brain-wide attractor).  See _kc_type_markers() and
_kc_marker_expr().

Naming.  Anatomical regions are KC-IC / KC-NC / KC-OC in code, never "calyx"
for one compartment and never bare IC/NC/OC; on the artwork each is drawn
under the name of its dominant population (COMP_DISPLAY).  Populations use
the reference's display names (Large-type, Middle-type, Small-type, Class II,
Other KC; "Class II" is class two).  The ltKC subclusters are
"Subcluster 1/2/3", taken from snuc_kc.LABELS so the artwork cannot drift
from the SI figure.

Statistics.  Spatial caste contrasts are one brain per caste: descriptive,
no P values, and the caption is the place that must say so.

Inputs:
  results/fig5_kc/          kc_umap.parquet (panel A; built from the snRNA
                            reference only, so the tangram column correction
                            does not affect it),
                            kc_type_marker_genes.csv, kc_type_marker_spatial_expr.csv
                            (panel D; this script's own cache, regenerated
                            automatically if missing or stale).
                            Not kc_compartment_diversity_per_slice.csv, which
                            is built from other labels and is not read here.
  data/sn/BIMP_atlas_qw_CentralBrain.h5ad  panel D's marker-gene ranking (snRNA
                            reference, restricted to the 5 KC populations)
  results/tangram_ltKC_cb/  ltKC_CB_spatial_subclusters_dsource.h5ad -- the ltKC
                            subcluster Tangram projection onto every cell of the
                            three KC compartments (331,816), written by
                            project_large_type_subclusters.py against the
                            CB reference's own Large-type nuclei.  The
                            ltKC-reference run in results/tangram_ltKC/ is not
                            read.  *_meanS.h5ad is a hybrid alternative, not
                            for use.
  results/tangram_ltKC/     kc5pop_spatial_allKC_dsource.h5ad (panel B; see
                            load_kc5pop())
  results/fig5_kc/          CB_LargeType_subclustered.h5ad -- the snRNA-side
                            subcluster labels feeding the Tangram run above and
                            panel E's own marker-gene ranking (via
                            snrna_kc_subclusters.py), written by
                            large_type_kc_subclusters.py.
  results/tangram/          paths.CANONICAL_TANGRAM, i.e.
                            combined_clustered_g2_R2_s42_tangram.h5ad
                            (grey tissue context; also panel D's
                            spatial expression source)
Panel E calls snrna_kc_subclusters.compute_kc_subcluster_data() so the ltKC
subclustering has one definition shared with its standalone SI figure.

Run at full resolution (this script reads precomputed tables and never calls
_si_common.subsample(), so the variable is only a safeguard):
  BIMP_FIG_SUBSAMPLE_FRACTION=1 PYTHONPATH=src:scripts \
      .venv/bin/python scripts/kenyon/fig_kc_subtypes.py
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kenyon.kc_subtypes_data import (  # noqa: E402
    _FIG_OUT,
    COMP_COLORS,  # noqa: F401 (re-exported)
    COMP_ORDER,  # noqa: F401 (re-exported)
    DATA,
    FIG_DIR,
    KC5POP_H5AD,  # noqa: F401 (re-exported)
    LTKC_H5AD,  # noqa: F401 (re-exported)
    SUB_CLUSTERS,  # noqa: F401 (re-exported)
    SUB_COLORS,  # noqa: F401 (re-exported)
    SUB_LABELS,  # noqa: F401 (re-exported)
    SUB_SHORT,  # noqa: F401 (re-exported)
    SUBZONE_STATS,
    TYPE_COLORS,  # noqa: F401 (re-exported)
    TYPE_ORDER,  # noqa: F401 (re-exported)
    _simplex_rgb,  # noqa: F401 (re-exported)
    _sn_table,  # noqa: F401 (re-exported)
    _subzone_data,  # noqa: F401 (re-exported)
)
from kenyon.kc_subtypes_panels import (  # noqa: E402
    COMP_DISPLAY,  # noqa: F401 (re-exported)
    PANEL_TITLE_SM,
    ZONE_UM,  # noqa: F401 (re-exported)
    panel_a,
    panel_b,
    panel_c,
    panel_distance,
    panel_h,
    panel_markers,
    panel_sankey,
    panel_zoom,
)
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

from bimp_st.style import (  # noqa: E402
    panel_header,
)

SUBZONE_STATS_CSV = (FIG_DIR if _FIG_OUT else DATA) / "subzone_stats.csv"


# ── Compose ────────────────────────────────────────────────────────────────
def main():
    # Two-row layout (see the module docstring for panel content):
    #   Row 1  A | B | C + D (C and D in their own sub-grid)
    #   Row 2  E (UMAP over gene dots) | F | G | H over I
    #
    # Sizing constraints.  The spatial crops C, F and G are equal-aspect and
    # height-bound, so their physical width follows their row height, not
    # their width_ratio; they must still print at the same width in inches,
    # because scale_bar() sets bar length as a fraction of the axes width.
    # C must also line up horizontally with G (row 1's and row 2's third
    # columns), which is set explicitly below rather than by ratio tuning.
    # Labels, legends and point-offset annotations extend past each row's
    # nominal cell, so hspace is tuned from the measured ink gap (the DIAG
    # row-gap print), not from gridspec arithmetic.  Re-check the DIAG lines
    # and check_manuscript_layout.py after any retune.
    fig = plt.figure(figsize=(12.2, 9.45))
    # Row 1's height ratio is what sets C's width: C is an equal-aspect,
    # height-bound crop like F/G, so it is matched to F/G's height here.
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.976, 1.46],
        hspace=0.26,
        left=0.044,
        right=0.976,
        top=0.955,
        bottom=0.075,
    )

    # ── Row 1: A | B | C+D (C and D in their own 1x2 sub-grid) ──────────────
    # C is height-bound, so A's and B's ratios only set where C starts, not
    # its width.
    g0 = gs[0].subgridspec(1, 3, width_ratios=[0.95, 0.90, 2.89], wspace=0.30)
    axA = fig.add_subplot(g0[0, 0])
    axB = fig.add_subplot(g0[0, 1])
    # C and D get their own sub-grid with a wider internal gap: D's y-tick
    # labels are gene names (e.g. "acetylcholine transporter-like") and are
    # drawn outside D's left edge, toward C's crop.  Re-verify on the render.
    gCD = g0[0, 2].subgridspec(1, 2, width_ratios=[1.22, 1.16], wspace=0.35)
    axC = fig.add_subplot(gCD[0, 0])
    axD = fig.add_subplot(gCD[0, 1])

    # ── Row 2: E (UMAP over gene dots) | F | G | H over I ──────────────────
    # A nested 2-row sub-grid: E, F and G span both sub-rows, H takes the top
    # one and I the bottom (I is repositioned under H's maps below), so H/I
    # get extra height without changing F/G's size.  F/G's column ratios are
    # tight because their height-bound boxes leave the rest of the cell
    # empty.  hspace=0: nothing else occupies sub-row 1.
    g1r = gs[1].subgridspec(
        2,
        4,
        height_ratios=[3.25, 1.5],
        width_ratios=[0.70, 0.88, 0.88, 1.30],
        wspace=0.28,
        hspace=0.0,
    )
    gE = g1r[:, 0].subgridspec(2, 1, height_ratios=[1.0, 1.9], hspace=0.40)
    axE1 = fig.add_subplot(gE[0, 0])
    axE2 = fig.add_subplot(gE[1, 0])
    axF = fig.add_subplot(g1r[:, 1])
    axG = fig.add_subplot(g1r[:, 2])
    axH = fig.add_subplot(g1r[0, 3])
    axI = fig.add_subplot(g1r[1, 3])
    # Equal-aspect axes shrink inside their gridspec cell; anchoring them to
    # the top of the cell rather than its centre keeps every title in a row on
    # one line and stops the leftover height opening as a gap above the row.
    for ax in (axF, axG, axH):
        ax.set_anchor("N")
    # C is anchored NW, not N: its cell is wider than its height-bound box,
    # and pinning the box to the cell's left edge makes its x-position exact.
    axC.set_anchor("NW")
    # Set C's x0 directly to G's (C is row 1's third column, G row 2's): the
    # row-1 widths are coupled, so ratio tuning cannot hit this exactly.  C
    # keeps its own y0, width and height.
    axG_pos = axG.get_position()
    axC_pos = axC.get_position()
    axC.set_position([axG_pos.x0, axC_pos.y0, axC_pos.width, axC_pos.height])

    print("panel A ...", flush=True)
    n_kc, lt_rings = panel_a(axA)
    print("panel B ...", flush=True)
    panel_sankey(axB)
    print("panel C ...", flush=True)
    panel_c(axC)
    print("panel D ...", flush=True)
    panel_h(axD)
    print("panel E ...", flush=True)
    n_lt = panel_b(axE1, axE2)
    print("panel F ...", flush=True)
    panel_zoom(axF, "queen", letter="F")
    print("panel G ...", flush=True)
    panel_zoom(axG, "worker", letter="G")
    print("panel H ...", flush=True)
    panel_markers(axH)
    panel_distance(axI)
    # Place H: its leftmost ink (row labels) one F-G gutter right of G's
    # rightmost ink, its rightmost ink (x label) at the right edge of the top
    # row (D's heatmap + colourbar).  Narrows the plot, not the label column.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _gut = axG.get_tightbbox(_r).x0 - axF.get_tightbbox(_r).x1
    _left = axG.get_tightbbox(_r).x1 + _gut
    _right = max(
        a_.get_tightbbox(_r).x1 for a_ in fig.axes if a_.get_position().y0 > 0.5
    )
    _hh = axH.get_window_extent(_r).height
    for _ in range(3):
        _bb = axH.get_window_extent(_r)
        _tb = axH.get_tightbbox(_r)
        _x0 = _left + (_bb.x0 - _tb.x0)
        _x1 = _right - (_tb.x1 - _bb.x1)
        _pos = axH.get_position()
        axH.set_position(
            [_x0 / fig.bbox.width, _pos.y0, (_x1 - _x0) / fig.bbox.width, _pos.height]
        )
        axH.set_box_aspect(_hh / (_x1 - _x0))
        fig.canvas.draw()
    _need = 0.0
    print(
        f"  panel H: narrowed by {max(_need, 0) / fig.dpi:.3f} in "
        f"(F-G gutter {_gut / fig.dpi:.3f} in)",
        flush=True,
    )

    # Panel I sits under H, spanning the width of its two map columns.
    fig.canvas.draw()
    _r2 = fig.canvas.get_renderer()
    _hp = axH.get_position()
    _y_top = axH.get_tightbbox(_r2).y0 / fig.bbox.height - 0.045
    _cell = axI.get_subplotspec().get_position(fig)
    _y_bot = _cell.y0 + 0.032
    _mx0 = min(a_.get_position().x0 for a_ in axH.map_axes)
    _mx1 = max(a_.get_position().x1 for a_ in axH.map_axes)
    axI.set_position([_mx0, _y_bot, _mx1 - _mx0, _y_top - _y_bot])

    # E's dot plot ends level with the lowest ink of the other panels, so the
    # figure's bottom edge is straight.
    fig.canvas.draw()
    _r3 = fig.canvas.get_renderer()
    # (the S1/S2/S3 key under F and G sets the line; I rises to meet it)
    _target = min(axF.get_tightbbox(_r3).y0, axG.get_tightbbox(_r3).y0)
    _d = (axE2.get_tightbbox(_r3).y0 - _target) / fig.bbox.height
    _pe = axE2.get_position()
    axE2.set_position([_pe.x0, _pe.y0 - _d, _pe.width, _pe.height + _d])
    _di = (_target - axI.get_tightbbox(_r3).y0) / fig.bbox.height
    _pi = axI.get_position()
    axI.set_position([_pi.x0, _pi.y0 + _di, _pi.width, _pi.height - _di])

    # Panel letters: draw once so equal-aspect axes settle into their final
    # (shrunk) boxes, then hang each letter at that box's left edge at a
    # shared per-row y so the row reads level.  If a centred title extends
    # further left than the letter would sit, the letter is pushed left of
    # the title by a fixed clearance; other panels are unaffected.
    fig.canvas.draw()
    # The clearance is measured before the print-scale save hook
    # (bimp_st.print_scale) enlarges every text artist.  A centred
    # title grows outward from its centre, so 0.015 is sized against the saved
    # PNG, not the pre-scale gap.
    _letter_title_gap = 0.015

    def _header(anchor, letter, title):
        panel_header(
            fig,
            anchor,
            letter,
            title,
            letter_size=15,
            tag_size=PANEL_TITLE_SM,
            dy_pts=0,
            letter_ha="right",
        )

    for row in (
        [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")],
        [(axE1, "E"), (axF, "F"), (axG, "G"), (axH, "H")],
    ):
        ry = max(ax.get_position().y1 for ax, _ in row)
        for ax, lab in row:
            x0 = ax.get_position().x0 - 0.010
            renderer = fig.canvas.get_renderer()
            title_bb = ax.title.get_window_extent(renderer)
            if title_bb.width > 0:
                title_x0_fig = fig.transFigure.inverted().transform(
                    (title_bb.x0, title_bb.y0)
                )[0]
                x0 = min(x0, title_x0_fig - _letter_title_gap)
            # House panel header (bimp_st.style): letter hung left of x0,
            # bold title on its baseline to the right; the axes' own
            # (invisible) title is kept only so downstream geometry that reads
            # its extent still works.
            _header((x0, ry + 0.012), lab, ax.title.get_text())
            if lab == "H":
                _letter_h = x0
            ax.title.set_alpha(0)  # invisible but keeps its extent

    _pI = axI.get_position()
    _header((_letter_h, _pI.y1 + 0.012), "I", axI.title.get_text())
    axI.title.set_alpha(0)

    # The row-to-row connector: one arrow, A -> E, in Large-type's colour.
    # The tail sits on the bottom of panel A's Large-type outline (taken from
    # the drawn rings each run, not a hand-picked point).  If the outline
    # ever splits into two pieces, two unheaded prongs from a branch point
    # touch each ring and the single headed arrow still makes one claim
    # (len(ring_anchors) == 2 below); with one piece only the arrow is drawn.
    pa, pb = axA.get_position(), axB.get_position()
    lt_color = TYPE_COLORS["Large-type Kenyon cells"]

    def _ring_bottom_xy(ring):
        disp = axA.transData.transform(ring)
        fxy = fig.transFigure.inverted().transform(disp)
        return fxy[np.argmin(fxy[:, 1])]

    ring_anchors = sorted(
        (_ring_bottom_xy(r) for r in lt_rings), key=lambda p: p[0]
    )  # left ring first
    print(
        "  connector ring anchors (fig fraction): "
        + ", ".join(f"({x:.4f}, {y:.4f})" for x, y in ring_anchors),
        flush=True,
    )

    x_branch = pa.x1 + 0.35 * (pb.x0 - pa.x1)
    y_branch = pa.y0 - 0.006
    # The head is placed from E1's measured title and subcluster-label boxes,
    # not fixed offsets, so it stays clear if they move or wrap.
    title_bbox = axE1.title.get_window_extent(fig.canvas.get_renderer())
    title_bottom_fig = fig.transFigure.inverted().transform(
        (title_bbox.x0, title_bbox.y0)
    )[1]
    _renderer = fig.canvas.get_renderer()
    _inv = fig.transFigure.inverted()
    _label_boxes = []
    for t in axE1.texts:
        if "Subcluster" not in t.get_text():
            continue
        bb = t.get_window_extent(_renderer)
        (bx0, by0), (bx1, by1) = (
            _inv.transform((bb.x0, bb.y0)),
            _inv.transform((bb.x1, bb.y1)),
        )
        _label_boxes.append((t.get_text().splitlines()[0], bx0, by0, bx1, by1))
    for name, bx0, by0, bx1, by1 in _label_boxes:
        print(
            f"  DIAG label box: {name!r} x=[{bx0:.4f},{bx1:.4f}] "
            f"y=[{by0:.4f},{by1:.4f}]",
            flush=True,
        )
    _topmost_label_top = min(
        (by1 for _, _, _, _, by1 in _label_boxes), default=title_bottom_fig - 0.06
    )
    print(
        f"  DIAG connector headroom: title_bottom={title_bottom_fig:.4f}  "
        f"topmost subcluster label top={_topmost_label_top:.4f}  "
        f"available slack={title_bottom_fig - _topmost_label_top:.4f}",
        flush=True,
    )
    # The safe channel into E1 is the gap between the top of the Subcluster 3
    # label and the bottom of the title, clear of all four boxes at any x;
    # the head targets its midpoint.
    _subcluster3_top = max(
        by1 for name, _, _, _, by1 in _label_boxes if name.startswith("Subcluster 3")
    )
    _tx1 = fig.transFigure.inverted().transform((title_bbox.x1, title_bbox.y1))[0]
    # The title is a short, hidden tag, so its right edge does not mark the
    # text column; use the axes' right edge less a margin instead.
    _tx1 = max(_tx1, axE1.get_position().x1 - 0.02)
    # The arrow is one smooth arc: from the bottom of the Large-type outline,
    # bowing out through the gutter right of the text column, into E just
    # right of its title.  The rendered path is checked against every
    # measured text box below.
    _ring_x, _ring_y = ring_anchors[-1]  # bottom of the LT outline
    _title_top_fig = fig.transFigure.inverted().transform(
        (title_bbox.x1, title_bbox.y1)
    )[1]

    # Panel A's x-label ("UMAP 1") sits in the band the arc crosses, so it is
    # measured too.
    _xlab_bb = axA.xaxis.label.get_window_extent(_renderer)
    (_xl0, _yl0), (_xl1, _yl1) = _inv.transform(
        [(_xlab_bb.x0, _xlab_bb.y0), (_xlab_bb.x1, _xlab_bb.y1)]
    )
    print(
        f"  DIAG connector: ring=({_ring_x:.4f},{_ring_y:.4f}) "
        f"title_top={_title_top_fig:.4f} "
        f"A-xlabel x=[{_xl0:.4f},{_xl1:.4f}] y=[{_yl0:.4f},{_yl1:.4f})",
        flush=True,
    )

    # Both control points are pushed out by the same bulge, so the path curves
    # along its whole length rather than turning at a joint.  The head ends
    # right of the title's x1: any endpoint inside the title's x-span
    # (0.073-0.194 on the current render) would cross it at title height.
    _tail = (_ring_x, _ring_y - 0.004)
    _title_x1 = _tx1
    _head = (_title_x1 + 0.002, (title_bottom_fig + _subcluster3_top) / 2)
    _bulge = 0.055
    _dy = _tail[1] - _head[1]
    elbow_path = MplPath(
        [
            _tail,
            (_tail[0] + _bulge, _tail[1] - 0.28 * _dy),
            (_head[0] + _bulge, _head[1] + 0.28 * _dy),
            _head,
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )

    # The bulge must not wander into panel B. Control points may sit outside
    # the free corridor (they are not on the curve), so test the rendered
    # path's actual rightmost point, not the hull.
    _max_x = elbow_path.interpolated(400).vertices[:, 0].max()
    assert _max_x < pb.x0 - 0.004, (
        f"A->E arc bulges to x={_max_x:.4f}, into panel B (x0={pb.x0:.4f}) "
        f"-- reduce _bulge"
    )

    # Verify the drawn path by sampling the interpolated curve against every
    # box measured above -- the title, all three subcluster labels, and panel
    # A's x-label -- rather than reasoning about control points.
    _obstacles = [
        (
            "axE1 title",
            *_inv.transform(
                [(title_bbox.x0, title_bbox.y0), (title_bbox.x1, title_bbox.y1)]
            ).ravel(),
        ),
        ("A xlabel", _xl0, _yl0, _xl1, _yl1),
    ]
    _obstacles += [
        (name, bx0, by0, bx1, by1) for name, bx0, by0, bx1, by1 in _label_boxes
    ]
    _pts = elbow_path.interpolated(400).vertices
    for _name, _ox0, _oy0, _ox1, _oy1 in _obstacles:
        _hit = (
            (_pts[:, 0] >= _ox0)
            & (_pts[:, 0] <= _ox1)
            & (_pts[:, 1] >= _oy0)
            & (_pts[:, 1] <= _oy1)
        )
        assert not _hit.any(), (
            f"A->E connector crosses {_name!r} "
            f"(x=[{_ox0:.4f},{_ox1:.4f}] y=[{_oy0:.4f},{_oy1:.4f}]) at "
            f"{int(_hit.sum())} of {len(_pts)} sampled points -- retune the "
            f"connector; do not just widen the tolerance"
        )
    print(
        f"  DIAG connector path clears all {len(_obstacles)} measured "
        f"boxes (400 samples)",
        flush=True,
    )

    if len(ring_anchors) == 2:
        for (xr, yr), rad in zip(ring_anchors, (0.18, -0.12), strict=True):
            prong = FancyArrowPatch(
                (x_branch, y_branch),
                (xr, yr - 0.004),
                transform=fig.transFigure,
                color=lt_color,
                arrowstyle="-",
                lw=1.6,
                alpha=0.85,
                shrinkA=0,
                shrinkB=0,
                zorder=49,
                connectionstyle=f"arc3,rad={rad}",
                capstyle="round",
            )
            fig.add_artist(prong)
        fig.add_artist(
            plt.Circle(
                (x_branch, y_branch),
                0.0028,
                transform=fig.transFigure,
                color=lt_color,
                zorder=51,
            )
        )
    # else: the outline is not exactly two rings -- skip the prongs and
    # branch dot and keep the plain arrow.
    # lw 2.0 / mutation_scale 15 keep the arrow lighter than the data strokes.
    arrow = FancyArrowPatch(
        path=elbow_path,
        transform=fig.transFigure,
        color=lt_color,
        arrowstyle="-|>",
        mutation_scale=15,
        lw=2.0,
        shrinkA=0,
        shrinkB=0,
        zorder=50,
        joinstyle="round",
        capstyle="round",
    )
    fig.add_artist(arrow)

    # Panel titles are the binding width constraint on this figure, so
    # measure them rather than trusting a character count.  >1.0 means the
    # title is wider than its own panel and is leaning into a neighbour.
    for ax, lab in (
        (axA, "A"),
        (axB, "B"),
        (axC, "C"),
        (axD, "D"),
        (axE1, "E"),
        (axF, "F"),
        (axG, "G"),
        (axH, "H"),
    ):
        t = ax.title.get_window_extent(fig.canvas.get_renderer())
        w = ax.get_window_extent(fig.canvas.get_renderer()).width
        print(f"  title {lab}: {t.width / w:.2f} x panel width", flush=True)

    # Scale-bar consistency check (see scale_bar()'s docstring): C sits in
    # row 1's sub-gridspec, F and G in row 2's, so a shared magnification
    # does not guarantee equal physical width.  All three must agree for the
    # scale bars to match.
    renderer = fig.canvas.get_renderer()
    w_c = axC.get_window_extent(renderer).width / fig.dpi
    w_f = axF.get_window_extent(renderer).width / fig.dpi
    w_g = axG.get_window_extent(renderer).width / fig.dpi
    print(
        f"  crop widths (in): C={w_c:.3f}  F={w_f:.3f}  G={w_g:.3f}"
        f"  (must match within ~1% or the scale bars disagree)",
        flush=True,
    )
    # DIAG: is H's rendered box width- or height-bound inside its gridspec
    # cell?  Compared against the cell's nominal box (before equal-aspect
    # shrink-to-fit) so it's obvious which dimension has slack.
    hbox = axH.get_window_extent(renderer)
    hcell = axH.get_subplotspec().get_position(fig)
    print(
        f"  DIAG panel H: rendered (shrunk) box {hbox.width / fig.dpi:.3f} x "
        f"{hbox.height / fig.dpi:.3f} in; nominal gridspec cell "
        f"{hcell.width * fig.get_figwidth():.3f} x "
        f"{hcell.height * fig.get_figheight():.3f} in",
        flush=True,
    )
    # DIAG: the same width- vs height-bound check for the spatial crops C, F
    # and G, to tell horizontal slack from vertical.
    for lab, ax in (("C", axC), ("F", axF), ("G", axG)):
        box = ax.get_window_extent(renderer)
        cell = ax.get_subplotspec().get_position(fig)
        print(
            f"  DIAG panel {lab}: rendered (shrunk) box {box.width / fig.dpi:.3f} x "
            f"{box.height / fig.dpi:.3f} in; nominal gridspec cell "
            f"{cell.width * fig.get_figwidth():.3f} x "
            f"{cell.height * fig.get_figheight():.3f} in",
            flush=True,
        )
    # C/G horizontal-position check: C's x0 is set from G's rendered x0
    # above, so this should match to float precision.
    pC, pG = axC.get_position(), axG.get_position()
    x0_gap = pG.x0 - pC.x0
    print(
        f"  crop x-position (fig fraction): C x0={pC.x0:.4f}  G x0={pG.x0:.4f}"
        f"  gap={x0_gap:+.4f}  ({'ALIGNED' if abs(x0_gap) < 0.001 else 'MISALIGNED'})",
        flush=True,
    )
    # DIAG: the actual ink gap between row 1 and row 2, from get_tightbbox()
    # (which includes legends and annotations outside the axes boxes, e.g.
    # the crops' margin strips).  Bottom of row 1 = min tightbbox.y0 over A-D;
    # top of row 2 = max tightbbox.y1 over E-H.  Retune hspace from this
    # number, not from gridspec geometry.
    row1_bottom = min(ax.get_tightbbox(renderer).y0 for ax in (axA, axB, axC, axD))
    row2_top = max(ax.get_tightbbox(renderer).y1 for ax in (axE1, axE2, axF, axG, axH))
    print(
        f"  DIAG row gap: row-1 bottom ink y={row1_bottom / fig.dpi:.3f} in, "
        f"row-2 top ink y={row2_top / fig.dpi:.3f} in, "
        f"actual gap={(row1_bottom - row2_top) / fig.dpi:.3f} in",
        flush=True,
    )

    out = FIG_DIR / "fig_kc_subtypes.png"
    # 600 dpi, as in fig_atlas_overview.py: the small, semi-transparent rasterized
    # markers in F/G (1.1-1.8 pt) blur at 300 dpi.
    fig.savefig(out, dpi=600, bbox_inches="tight")
    # dpi= applies here too: without it, the PDF's rasterized scatter layers
    # embed at matplotlib's figure.dpi default (100), not 600.
    fig.savefig(FIG_DIR / "fig_kc_subtypes.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.Series(SUBZONE_STATS).rename("value").to_csv(
        SUBZONE_STATS_CSV, index_label="stat"
    )
    print(f"Saved {out}  ({n_kc:,} KC nuclei in A, {n_lt:,} in E)")


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
