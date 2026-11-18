#!/usr/bin/env python3
"""Whole-brain spatial transcriptomic atlas panels.

Generator of figure ``fig:atlas_overview``,
figures/main/fig_atlas_overview.png, from
results/figures/main/atlas_overview/fig_atlas_overview.png.

Balanced 2-row composite (6 data panels, no external-render placeholders):

  Row 1  A: 30-section transcript-density grid (queen)
         B: 3D anatomical-region render (FOCAL_BRAIN)
         C: 3D cell-type render (FOCAL_BRAIN) + heavy black KC-IC selection box
  Row 2  D: Representative Queen | Worker spatial slices, majority-vote-labeled
         E: snRNA reference UMAPs (Central Brain | Optic Lobe, 29 fine types)
         F: whole UMAP (grey) with the KC-IC box cells highlighted on top

Panel-F callout style is set by ``F_COMPOSITE_OPTION``; standalone variants
``fig_F_option{1,2,3}.png`` are always rendered so the style can be compared.

    PYTHONPATH=src:scripts .venv/bin/python scripts/atlas/fig_atlas_overview.py \
        --compose-only
"""

import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import (
    Axes3D,  # noqa: F401
    proj3d,
)

from bimp_st.cluster_labels import (
    CELL_TYPE_ORDER,
    POOLED_ORDER,
    get_color,
    get_pooled,
    get_pooled_color,
    get_pooled_display,
    get_short,
    get_spatial_color,
    hierarchical_vote,
)
from bimp_st.nine_region_common import (
    DESCRIPTIVE_REGIONS,
    KC_POPULATION_COLORS,
    REGION_COLORS,
    attach_region_flags,
)
from bimp_st.nine_region_common import (
    display as region_display,
)
from bimp_st.paths import (
    ALIGNED_QUEEN,
    ALIGNED_WORKER,
    CANONICAL_TANGRAM,
    SN_CB,
    SN_OL,
    SPATIAL_COMBINED,
)
from bimp_st.si_common import (
    PIXEL_SCALE,
    TICK_SIZE,
    maybe_subsample_adata,
)
from bimp_st.si_common import (
    add_scalebar as _add_scalebar,
)
from bimp_st.si_common import (
    strip_spines as _strip_spines,
)
from bimp_st.style import panel_header, set_pub_style

# ── Configuration ────────────────────────────────────────────────────────────
QUEEN_INPUT = str(ALIGNED_QUEEN)
WORKER_INPUT = str(ALIGNED_WORKER)
COMBINED_INPUT = str(SPATIAL_COMBINED)
# That object's own tangram_cell_type is not the canonical labelling, so
# the vote panels take labels from the canonical object.
TANGRAM_LABELS = str(CANONICAL_TANGRAM)
# Panel E shows the two separate annotated snRNA atlases (union =
# 29 fine cell types), not the merged/pooled reference.
CB_REF = str(SN_CB)  # 15 fine types
OL_REF = str(SN_OL)  # 20 fine types
OUTPUT_DIR = Path("results/figures/main/atlas_overview")
CLUSTER_COL = "leiden"
# True-scale depth for the 3D views. get_3d_coords() puts section i at z = 10 * i
# (the 10 um section spacing, as a plain number); x and y are registration-frame
# pixels of PIXEL_SCALE um each. Multiplying z by 1 / PIXEL_SCALE converts the
# 10 um spacing into frame pixels (47.06 px per section), so x, y and z share
# one scale and the brain is drawn as the flat slab it is. No exaggeration.
Z_TRUE = 1.0 / PIXEL_SCALE

# Panel F callout style used in the composite fig_atlas_overview.png. Standalone
# variants (fig_F_option{1,2,3}.png) are always rendered for comparison; set this to the
# one you want baked into the composite.
#   1 = box cells in pooled cell-type colors (mostly KC-orange)
#   2 = box cells in one flat KC-accent color
#   3 = pooled colors + a small unobtrusive KC-centroid marker
F_COMPOSITE_OPTION = 1

# The optic-lobe reference resolves 13 numbered subclusters (OL_1..OL_13)
# that are unannotated placeholders (cluster_labels.py) -- real, distinct
# expression clusters, but not yet given biological names. Labeling all 13
# individually in Panel E would hand the reader a 13-entry vocabulary that
# used nowhere else (they collapse to one "Optic lobe"
# pool everywhere else, per CANONICAL_MAP). Panel E keeps every one of them
# drawn in its own color -- the point of this panel is to show the reference
# really does resolve this many clusters -- but tags the group with one
# bracket label instead of thirteen.
_OL_PLACEHOLDER = [f"OL_{i}" for i in range(1, 14)]

# Region palette for anatomical annotations -- shared across figures via
# nine_region_common.REGION_COLORS so a region is never a different colour in
# two figures (see also si_exploded_regions.py, fig_atlas_structure.py).
REGION_PALETTE = dict(REGION_COLORS)


def set_style():
    set_pub_style(savefig_dpi=600)  # house style; higher DPI for the atlas figure


# ── Helpers ──────────────────────────────────────────────────────────────────


def section_order(adata):
    """Slice names in section order (the order Panel A and the box use)."""
    return sorted(adata.obs["slice_name"].unique())


def get_3d_coords(adata, z_spacing=10.0):
    """Build (N, 3) coordinates from obsm['spatial'] + slice_name."""
    # Same section order as Panel A and as _kc_ic_region_mask (plain sorted slice
    # names), so the selection box's three consecutive sections are three
    # consecutive planes here. A suffix-keyed sort would interleave the two
    # slides' sections and scatter the box's cells over ~30 planes.
    slice_names = section_order(adata)
    slice_to_z = {s: i * z_spacing for i, s in enumerate(slice_names)}
    coords_2d = adata.obsm["spatial"]
    z_coords = adata.obs["slice_name"].map(slice_to_z).values
    return np.column_stack([coords_2d, z_coords])


def get_region_labels(adata):
    """Extract region labels from in_* boolean columns.

    al/ellipsoid_body/fan_shaped_body are nullable booleans -- pd.NA means
    "this section has no tracing for that region yet", not "outside the
    region". ``.values.astype(bool)`` would raise on the NA entries; a plain
    ``.fillna(False)`` would raise a caste difference out of missing data. For
    this single-brain coverage render, NA cells simply get no region tag (they
    fall into "unlabeled" alongside genuinely out-of-region cells) --
    dropping them from the scatter instead would leave visible holes in the
    3D brain, so this is a display simplification, not a quantitative claim.
    """
    in_cols = [c for c in adata.obs.columns if c.startswith("in_")]
    labels = np.full(adata.n_obs, "unlabeled", dtype=object)
    for col in in_cols:
        region = col.replace("in_", "")
        mask = adata.obs[col].to_numpy(dtype=bool, na_value=False)
        labels[mask] = region
    return labels


def _label3d(ax, x, y, z, text, dx_pts, dy_pts, **kw):
    """Place text a fixed pixel offset from a projected 3D point.

    mplot3d's screen-space mapping can be heavily compressed along one axis
    (here, Z: the true-scale section depth is short once projected), so a
    text offset specified in 3D data units can land right on top of a
    neighbouring tick label. Projecting the anchor once with
    ``proj3d`` and then nudging in points (via annotate's offset-points
    textcoords) guarantees clearance regardless of that compression.
    """
    x2d, y2d, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    ax.annotate(
        text,
        xy=(x2d, y2d),
        xycoords=ax.transData,
        xytext=(dx_pts, dy_pts),
        textcoords="offset points",
        **kw,
    )


def style_3d_ax(ax, elev=25, azim=-65, z_scale=Z_TRUE, label_lxmult=7.0):
    """Style a 3D axis: transparent panes, visible axes, correct Z um.

    ``label_lxmult`` scales how far left of the origin the Y/Z axis labels sit
    (in x-tick units). Panel B uses a smaller value so its Y/Z labels do not
    collide with Panel A's colorbar sitting immediately to B's left.
    """
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.patch.set_visible(False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    o = [xlim[0], ylim[0], zlim[0]]
    kw = dict(color="k", lw=0.8)

    # Axis lines
    ax.plot([o[0], xlim[1]], [o[1], o[1]], [o[2], o[2]], **kw)
    ax.plot([o[0], o[0]], [o[1], ylim[1]], [o[2], o[2]], **kw)
    ax.plot([o[0], o[0]], [o[1], o[1]], [o[2], zlim[1]], **kw)

    # Ticks with nice round values. X/Y are relabeled in physical mm using the
    # same convention as Panel D's scale bar (PIXEL_SCALE µm/px → mm), so B/C and
    # D read at one consistent physical scale. Z is drawn at the same true scale
    # (10 µm per section), so its ticks are physical mm too.
    from matplotlib.ticker import MaxNLocator

    tkw = dict(color="k", lw=0.6)
    tick_len_y = (ylim[1] - ylim[0]) * 0.015
    tick_len_x = (xlim[1] - xlim[0]) * 0.015

    mm_per_coord = PIXEL_SCALE / 1000.0  # coordinate px → mm
    coord_per_mm = 1.0 / mm_per_coord

    def _fmt_mm(v_mm):
        """Physical mm tick label (no trailing '.0' for whole mm)."""
        if abs(v_mm) < 1e-9:
            return "0"
        if abs(v_mm - round(v_mm)) < 1e-6:
            return f"{v_mm:.0f}"
        return f"{v_mm:.1f}"

    def _mm_ticks(lo, hi, n):
        """Round mm tick positions (returned as (coord, mm)) within [lo, hi] px."""
        mm_vals = MaxNLocator(n).tick_values(lo * mm_per_coord, hi * mm_per_coord)
        out = []
        for m in mm_vals:
            c = m * coord_per_mm
            if lo <= c <= hi:
                out.append((c, m))
        return out

    # X ticks (mm)
    for t, m in _mm_ticks(xlim[0], xlim[1], 3):
        ax.plot([t, t], [o[1], o[1] - tick_len_y], [o[2], o[2]], **tkw)
        ax.text(
            t,
            o[1] - tick_len_y * 4,
            o[2],
            _fmt_mm(m),
            fontsize=6,
            ha="center",
            va="top",
        )

    # Y ticks (mm) — use 2 ticks to avoid crowding
    for t, m in _mm_ticks(ylim[0], ylim[1], 2):
        ax.plot([o[0] - tick_len_x, o[0]], [t, t], [o[2], o[2]], **tkw)
        ax.text(
            o[0] - tick_len_x * 4,
            t,
            o[2],
            _fmt_mm(m),
            fontsize=6,
            ha="right",
            va="center",
        )

    # Z axis in physical mm. Each section is 10 µm apart; displayed z is
    # slice_index * 10 * Z_TRUE, so physical mm = displayed_z / (z_scale * 1000).
    z_mm_lo = zlim[0] / (z_scale * 1000.0)
    z_mm_hi = zlim[1] / (z_scale * 1000.0)
    for m in MaxNLocator(2).tick_values(z_mm_lo, z_mm_hi):
        t = m * z_scale * 1000.0
        if not (zlim[0] + 0.10 * (zlim[1] - zlim[0]) < t <= zlim[1]):
            continue
        ax.plot([o[0] - tick_len_x, o[0]], [o[1], o[1]], [t, t], **tkw)
        ax.text(
            o[0] - tick_len_x * 5,
            o[1],
            t,
            _fmt_mm(m),
            fontsize=6,
            ha="right",
            va="center",
        )

    # Axis labels — placed beyond the outermost tick with extra clearance. The
    # Y/Z labels use ``label_lxmult`` so panels with a neighbour to their left
    # (Panel B ↔ Panel A colorbar) can pull them in and avoid a collision.
    ax.text(
        xlim[1],
        o[1] - tick_len_y * 7,
        o[2],
        "X (mm)",
        fontsize=7,
        ha="center",
        va="top",
    )
    ax.text(
        o[0] - tick_len_x * label_lxmult,
        ylim[1] + (ylim[1] - ylim[0]) * 0.10,
        o[2],
        "Y (mm)",
        fontsize=7,
        ha="right",
        va="center",
    )
    # Z (mm) label is NOT placed here -- see _place_z_label below. It must be
    # placed after _set_box_aspect_from_ranges (called later, in
    # compose_figure), because box_aspect changes ax.get_proj() and a
    # pixel-anchored label computed against the pre-box_aspect projection
    # drifts away from the axis it is supposed to label.

    ax.dist = 3.5
    # Minimize 3D subplot margins
    ax.set_position(ax.get_position().expanded(1.4, 1.4))


def _place_z_label(ax, label_lxmult=7.0, side="left"):
    """Place the "Z (mm)" axis title, anchored a fixed pixel offset above the
    topmost Z tick (see _label3d for why a pixel offset, not a data offset).

    ``side="left"`` (default, Panel C): anchor left of the origin by
    ``label_lxmult`` x-tick-units, growing further left -- matches the Y-label
    convention and the open whitespace Panel C has to spare on that side.

    ``side="right"``: anchor just right of the origin, growing right, into the
    gap between the Z axis and the data cloud. Use this whenever something
    else already occupies the left side at this height -- Panel B sits right
    next to Panel A's colorbar, and *no* left-side anchor clears both the Z
    tick AND the colorbar's tall rotated "Transcripts per cell" label: pulling
    ``label_lxmult`` in stops the tick collision but lands right in the
    colorbar text instead (the two problems trade off against each other on
    that side). The right side of B's origin is empty
    all the way to the data, so anchoring there sidesteps the trade-off
    entirely instead of tuning further within it.

    Must be called after the axes' projection is fully finalized (view_init,
    dist and box_aspect all set): box_aspect changes ax.get_proj(), so calling
    this any earlier freezes the anchor at a stale projection and the label
    drifts elsewhere in the panel.
    """
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    o = [xlim[0], ylim[0], zlim[0]]
    tick_len_x = (xlim[1] - xlim[0]) * 0.015
    if side == "right":
        _label3d(
            ax,
            o[0] + tick_len_x * 1.5,
            o[1],
            zlim[1],
            "Z (mm)",
            4,
            15,
            fontsize=7,
            ha="left",
            va="bottom",
        )
    else:
        _label3d(
            ax,
            o[0] - tick_len_x * label_lxmult,
            o[1],
            zlim[1],
            "Z (mm)",
            2,
            15,
            fontsize=7,
            ha="right",
            va="bottom",
        )


def subsample_idx(n_obs, max_n=80000, seed=42):
    """Return a single subsample index (or arange) for dense 3D rendering.

    One index is used for the coords and every per-cell mask/label, so the
    plotted geometry and the in-region highlight always agree.
    """
    if n_obs <= max_n:
        return np.arange(n_obs)
    return np.random.default_rng(seed).choice(n_obs, max_n, replace=False)


def _square_umap_limits(ax, coords, pad_frac=0.06):
    """Set symmetric square data limits (equal data units per axis)."""
    x, y = coords[:, 0], coords[:, 1]
    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())
    half = 0.5 * max(x.max() - x.min(), y.max() - y.min())
    half *= 1.0 + pad_frac
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def _set_box_aspect_from_ranges(ax, coords):
    """Set the 3D box aspect proportional to the axis limits (true scale).

    The limits, not the data ranges, set what is drawn, so x, y and z each get
    a box edge proportional to the span they cover in the same units.
    """
    del coords  # kept for the call sites; the limits carry the spans
    spans = np.array(
        [
            abs(np.subtract(*ax.get_xlim3d()[::-1])),
            abs(np.subtract(*ax.get_ylim3d()[::-1])),
            abs(np.subtract(*ax.get_zlim3d()[::-1])),
        ]
    )
    ax.set_box_aspect(spans / spans.max())


def load_spatial_celltypes(combined):
    """Build cell type labels from spatial Leiden clusters (combined-order)."""
    print("  Using spatial Leiden clusters for cell types...")
    ct_labels = combined.obs[CLUSTER_COL].astype(str).values
    ct_cats = sorted(np.unique(ct_labels), key=lambda s: int(s))
    ct_color_map = {c: get_spatial_color(c) for c in ct_cats}
    print(f"  {len(ct_cats)} spatial clusters, {len(ct_labels):,} cells")
    return ct_labels, ct_cats, ct_color_map


def _ct_color(name):
    """Colour for a display category, fine Kenyon subtypes included.

    get_pooled_color folds every Kenyon subtype back onto the single pooled
    orange, which would make the refined split invisible. The five named
    Kenyon populations take KC_POPULATION_COLORS (each named population in its
    namesake compartment's colour from panel B), the palette fig:kc_subtypes
    and the SI
    KC-population figure use, so the figures stay consistent.
    """
    # Display overrides first, and applied here rather than at the legend, so
    # the swatch and the points it describes can never disagree: pooled "Glia"
    # is the same hex as Perineurial glia, and naming glial subtypes beside the
    # residual drew two categories in one colour.
    from bimp_st.cluster_labels import DISPLAY_COLOR_OVERRIDES

    if name in DISPLAY_COLOR_OVERRIDES:
        return DISPLAY_COLOR_OVERRIDES[name]
    if name in KC_POPULATION_COLORS:
        return KC_POPULATION_COLORS[name]
    return get_pooled_color(name)


KC5POP_CLASSGATED = Path("results/tangram_ltKC/kc5pop_spatial_classgated.h5ad")


def _refine_kenyon(combined):
    """Split the pooled 'Kenyon cells' block into its five named subtypes.

    The whole-atlas 29-way run cannot do this: it resolves few Kenyon
    cells to a named type, because the unspecified 'Kenyon cells' reference
    entry and 'Other neurons' act as escape hatches it falls back on. A
    restricted re-projection can -- narrow the reference to the five named
    populations and the escape hatches are gone.

    The run used here (scripts/kenyon/project_kc_populations_class_gated.py) gates its
    query on the pooled call, never on anatomy, unlike the fig:kc_subtypes run which
    gates on the
    Kenyon compartments. That distinction matters here specifically:
    this figure plots cell types against those compartments, so an
    anatomy-gated label would be circular. Gating on the call keeps the
    anatomy free, and the subtype breakdown per compartment (KC-IC, KC-NC,
    KC-OC), in queen and worker separately, then reads as a test. The
    queried cells that lie outside every compartment are the control on the
    forced choice: their top-subtype share is compared with that inside a
    compartment. "Other Kenyon cells" is a real reference population, not
    the escape hatch this run removes.

    Falls back to the pooled label if the object is absent, so the figure
    still renders on a checkout without it.
    """
    ct = combined.obs["cluster_vote"].astype(str).values
    p = KC5POP_CLASSGATED
    if not p.exists():
        print(f"  [KC refine] {p} absent; keeping the pooled Kenyon block", flush=True)
        return ct
    import anndata as _ad

    sub = _ad.read_h5ad(p, backed="r")
    call = sub.obs["kc5pop_call"].astype(str).values
    sub.file.close()
    # Align positionally. combined.obs_names carries 562,028 duplicates, so
    # a name join would silently expand; the saved object is a boolean subset
    # of this same file in this same order, so recomputing the mask and
    # writing into it in order is exact. The assert is what makes that safe.
    mask = ct == "Kenyon cells"
    if mask.sum() != len(call):
        print(
            f"  [KC refine] mask {mask.sum():,} != saved {len(call):,}; "
            f"keeping the pooled block",
            flush=True,
        )
        return ct
    out = ct.copy()
    out[mask] = call
    from collections import Counter

    n = Counter(call).most_common()
    print(
        f"  [KC refine] {mask.sum():,} Kenyon cells split into "
        f"{len(n)} named subtypes: " + ", ".join(f"{k.split()[0]} {v:,}" for k, v in n),
        flush=True,
    )
    return out


def _ensure_cluster_vote(combined):
    """Ensure combined.obs['cluster_vote'] (expression-only hierarchical labels).

    Recomputes every call (cheap groupby) so a vote persisted in the object,
    which may follow a different pooling scheme (e.g. an "Optic lobe
    (lamina/medulla)" label), is never used.

    Anatomy-free: ``hierarchical_vote`` reads the per-cluster Tangram calls
    and nothing else, so the labels on this figure come from
    expression alone and the anatomy is independent evidence rather than an
    input.  The anatomy-guided variant (``anatomy_guided_vote``) forced any
    Leiden cluster more than half inside calyx anatomy to "Kenyon cells" and
    more than half inside optic anatomy to "Optic lobe", reading the same
    ``in_*`` flags the figure plots against.  The two votes are compared in
    ``results/reframe/celltype_anatomy/``.
    """
    if "leiden" not in combined.obs.columns:
        return
    if not combined.uns.get("_bimp_labels_corrected"):
        from bimp_st.paths import keyed_obs_column

        # joined on (slice_name, cell_id), never by position; raises on any
        # unmatched or repeated cell rather than falling back to the in-file column
        combined.obs["tangram_cell_type"] = keyed_obs_column(
            combined, "tangram_cell_type", TANGRAM_LABELS
        ).astype(str)
        combined.uns["_bimp_labels_corrected"] = True
    # hierarchical_vote with its defaults (min_purity=0.5, lead_ratio=1.5,
    # Kenyon capped to "Kenyon cells"), exactly as fig_caste_comparison.py calls
    # it, so the two
    # main figures agree on the Mixed share. Not cluster_vote -- see
    # caste_comparison_sccoda for why.
    hierarchical_vote(
        combined,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        dst="cluster_vote",
    )


# ── KC-IC selection geometry (shared by 3D cube + UMAP callout) ─────────────

# Which brain carries panels B, C and F. Panel D deliberately keeps both castes
# -- it is the queen-vs-worker comparison -- but everywhere else the figure shows
# a single brain so the reader is not switching caste between panels.
FOCAL_BRAIN = "queen"

# Spatial box over 3 mid-brain sections, per brain. The two brains are not in a
# common coordinate frame, so a box cannot be carried across; the queen box was
# found by scripts/atlas/atlas_overview_callout_box.py (400 px boxes swept over
# 3-section
# windows inside the KC-IC annotation).
#
# Both counts below use the metric panel C prints -- every cell inside the box,
# and the fraction of those carrying in_kc_ic.  Kenyon purity on cluster_vote,
# % in_kc_ic and raw Tangram purity are different quantities and not
# comparable; quote this one.
_KC_BOXES = {
    # Sizes are in the registration frame's pixel units, 0.2125 um each (PIXEL_SCALE),
    # not um: the 1,200 box is 255 um across, over 3 sections x 10 um = 30 um deep.
    # At 400 px (85 um) the drawn cube prints at ~4 pt and is unreadable.
    # Swept 800/1000/1200 px over 3- and 5-section windows (stride 25 px around
    # the optimum); the inner compact compartment is narrow in this frame,
    # so its purity falls as the box grows; the 1,200 px box trades purity
    # for a cube that is readable in print.
    "queen": dict(x=(-3025, -1825), y=(675, 1875), slices=[14, 15, 16]),
    "worker": dict(x=(-1854, -1254), y=(2363, 2963), slices=[19, 20, 21]),
}
_KC_BOX_STATS = {  # verified against both h5ads
    "queen": "2205 cells, 72% in_kc_ic",
    "worker": "859 cells, 85% in_kc_ic",
}
_KC_X0, _KC_X1 = _KC_BOXES[FOCAL_BRAIN]["x"]
_KC_Y0, _KC_Y1 = _KC_BOXES[FOCAL_BRAIN]["y"]
_KC_SLICE_IDXS = _KC_BOXES[FOCAL_BRAIN]["slices"]

# Panel C selection box: a crisp thin black wireframe cube (no halo; a halo
# filled the tiny box into a white blob) inside a zoom ring. All sizes are in
# printed points, converted to figure points from the page scale at draw time.
CUBE_PRINT_PT = 1.0
CUBE_HALO_PT = 0.0  # >0 adds a white halo under the edges; 0 disables it
DRAW_CUBE = True  # the true selection box (1,200 px = 255 um) can draw itself
LEADER_PRINT_PT = 0.35  # hairline leaders from the box's extreme corners to Panel F


def _kc_ic_region_mask(focal):
    """Boolean mask (focal-brain order) of the KC-IC cube cells."""
    f_slices = section_order(focal)
    f_sp = focal.obsm["spatial"]
    xy = (
        (f_sp[:, 0] >= _KC_X0)
        & (f_sp[:, 0] <= _KC_X1)
        & (f_sp[:, 1] >= _KC_Y0)
        & (f_sp[:, 1] <= _KC_Y1)
    )
    in_region = np.zeros(focal.n_obs, dtype=bool)
    for fsi in _KC_SLICE_IDXS:
        if fsi >= len(f_slices):
            continue
        sl_mask = (focal.obs["slice_name"] == f_slices[fsi]).values
        in_region |= sl_mask & xy
    return in_region


# ── Standalone: snRNA reference UMAP helpers ──────────────────────────────


def _norm_ct_arr(adata):
    """Whitespace-normalized fine ``cell_type`` labels as a str array."""
    return np.array([" ".join(str(x).split()) for x in adata.obs["cell_type"].values])


def _add_umap_frame(ax):
    """Dashed rounded-rectangle border matching the composite panel frames."""
    m = 0.03
    ax.add_patch(
        FancyBboxPatch(
            (m, m),
            1 - 2 * m,
            1 - 2 * m,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="k",
            facecolor="none",
            linestyle="--",
            zorder=9,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def _density_peak(xx, yy):
    """Robust on-plot label anchor: the densest bin of a 2D histogram. Beats the
    median for types whose cells have scattered outliers (e.g. the few, spread-
    out central-brain hemocytes), which would otherwise drag a median label off
    into empty space."""
    if len(xx) < 30:
        return float(np.median(xx)), float(np.median(yy))
    h, xe, ye = np.histogram2d(xx, yy, bins=24)
    i, j = np.unravel_index(int(np.argmax(h)), h.shape)
    return 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])


def _draw_ref_umap(
    ax,
    adata,
    rng,
    point_size=0.8,
    alpha=0.65,
    label=False,
    label_fs=6.5,
    min_frac_label=0.001,
    collapse_types=None,
    collapse_label=None,
):
    """Scatter one annotated snRNA atlas UMAP colored by fine cell_type.

    Draws types in ``CELL_TYPE_ORDER`` so large KC blobs go down first and the
    small distinct OL_* / glia types land on top. With ``label=True`` each
    present type gets a white-haloed short-name tag at its density peak (used by
    the enlarged Panel E). Returns the list of present fine types (canonical
    order); the caller's "(N types)" title counts every entry in this list
    (labeled individually or folded into ``collapse_label``), so the title and
    the visible tags never silently disagree.

    ``collapse_types`` (optional): fine types that stay drawn in their own
    color but get ONE shared text tag (``collapse_label``) at the density peak
    of their union, instead of one tag each -- for a set of real but
    unannotated/placeholder subclusters (e.g. OL_1..OL_13) that would
    otherwise hand the reader a vocabulary of numbered labels used nowhere
    else.

    ``min_frac_label`` = 0.001 sits below the smallest fine type of either
    reference, so every type counted in the title gets a
    label, while the absolute floor below (<20 cells) keeps a negligible
    fragment from getting a label at a meaningless density peak.
    """
    coords = adata.obsm["X_umap"]
    fine = _norm_ct_arr(adata)
    present_set = set(np.unique(fine))
    present = [c for c in CELL_TYPE_ORDER if c in present_set]
    present += sorted(present_set - set(present))  # any stragglers, gray
    n_tot = len(fine)
    for ct in present:
        mask = fine == ct
        if not mask.any():
            continue
        order = rng.permutation(int(mask.sum()))
        ax.scatter(
            coords[mask][order, 0],
            coords[mask][order, 1],
            c=KC_POPULATION_COLORS.get(ct, get_color(ct)),
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )
    if label:
        collapse_set = set(collapse_types or [])
        for ct in present:
            if ct in collapse_set:
                continue
            mask = fine == ct
            if mask.sum() < max(20, min_frac_label * n_tot):
                continue
            cx, cy = _density_peak(coords[mask, 0], coords[mask, 1])
            ax.text(
                cx,
                cy,
                get_short(ct),
                fontsize=label_fs,
                fontweight="bold",
                ha="center",
                va="center",
                color="#171717",
                zorder=10,
                path_effects=[
                    pe.Stroke(linewidth=1.9, foreground="white"),
                    pe.Normal(),
                ],
            )
        if collapse_set and collapse_label:
            cmask = np.isin(fine, list(collapse_set))
            if cmask.any():
                cx, cy = _density_peak(coords[cmask, 0], coords[cmask, 1])
                ax.text(
                    cx,
                    cy,
                    collapse_label,
                    fontsize=label_fs - 0.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="#171717",
                    zorder=10,
                    path_effects=[
                        pe.Stroke(linewidth=1.9, foreground="white"),
                        pe.Normal(),
                    ],
                )
    _square_umap_limits(ax, coords, pad_frac=0.06)
    ax.set_box_aspect(1)
    _strip_spines(ax)
    return present


# ── Panel D: majority-vote-labeled spatial slices ───────────────────────────


def _select_two_slices(adata):
    """Pick [queen_slice, worker_slice] — mid-brain slice with most cells."""
    slices_arr = adata.obs["slice_name"].values.astype(str)
    brains_arr = adata.obs["brain"].values.astype(str)
    aligned = adata.obsm["spatial_aligned"]
    unique_slices = np.unique(slices_arr)
    slice_z, slice_brain, slice_n = {}, {}, {}
    for sl in unique_slices:
        pos = np.where(slices_arr == sl)[0]
        slice_z[sl] = (
            aligned[pos, 2].mean()
            if aligned.shape[1] > 2
            else float(np.where(unique_slices == sl)[0][0])
        )
        slice_brain[sl] = brains_arr[pos[0]]
        slice_n[sl] = len(pos)
    sorted_slices = sorted(unique_slices, key=lambda s: slice_z[s])
    picks = []
    for brain in ("queen", "worker"):
        bs = [s for s in sorted_slices if slice_brain[s] == brain]
        if bs:
            mid = bs[len(bs) // 3 : 2 * len(bs) // 3] or bs
            picks.append(max(mid, key=lambda s: slice_n[s]))
    return picks


def _spatial_rect_bounds(sp_xy, *, pct=(0.0, 100.0), pad=0.03):
    """Axis-aligned (x_lo, x_hi, y_lo, y_hi) tight on the point cloud."""
    x = sp_xy[:, 0].astype(float)
    y = sp_xy[:, 1].astype(float)
    if len(x) < 10:
        x_lo, x_hi = float(x.min()), float(x.max())
        y_lo, y_hi = float(y.min()), float(y.max())
    else:
        x_lo, x_hi = np.percentile(x, pct)
        y_lo, y_hi = np.percentile(y, pct)
    px = (x_hi - x_lo) * pad if x_hi > x_lo else max(float(np.ptp(x)) * pad, 1e-6)
    py = (y_hi - y_lo) * pad if y_hi > y_lo else max(float(np.ptp(y)) * pad, 1e-6)
    return x_lo - px, x_hi + px, y_lo - py, y_hi + py


def draw_majority_vote_slices(ax, combined, rng):
    """Draw representative Queen | Worker slices, majority-vote labeled.

    Per-cluster vote over the combined Tangram labels (expression-only; see
    _ensure_cluster_vote); display-only, colored by the pooled cell-type
    palette. Returns the present pooled cell types (canonical order).
    """
    _ensure_cluster_vote(combined)
    # Pooled vocabulary by default, matching panels C and F and every
    # downstream analysis. Regional diversity and composition
    # run on the same hierarchical_vote; the cell-type 3D adjacency runs on
    # canonical-17 raw per-cell labels, where the named Kenyon subtypes are
    # near-empty against the generic entry. Nothing downstream consumes the
    # five subtypes, and fig:kc_subtypes
    # derives
    # its own from a separately gated projection, so splitting them here would
    # add a third Kenyon vocabulary. Set BIMP_FIG1_KC_SPLIT=1 to use the
    # class-gated split (scripts/kenyon/project_kc_populations_class_gated.py,
    # _refine_kenyon).
    if os.environ.get("BIMP_FIG1_KC_SPLIT") == "1":
        ct = _refine_kenyon(combined)
    else:
        ct = combined.obs["cluster_vote"].astype(str).values
    pick = _select_two_slices(combined)
    slices_arr = combined.obs["slice_name"].values.astype(str)
    sp_coords = combined.obsm["spatial"]

    present = {c for c in np.unique(ct) if c not in ("-1", "")}
    _order = []
    for g in POOLED_ORDER:
        if g == "Kenyon cells":
            # Expand to the named subtypes when they are present (the
            # BIMP_FIG1_KC_SPLIT path). The substring test also matches the
            # generic "Kenyon cells" entry itself, so on the pooled path this
            # branch and the append below both add it; the dedup below keeps
            # the legend from listing it twice.
            _order += [
                c for c in CELL_TYPE_ORDER if "Kenyon cells" in c and c in present
            ]
        if g in present:
            _order.append(g)
    _seen = set()
    ct_unique = [c for c in _order if c in present and not (c in _seen or _seen.add(c))]
    ct_unique += [c for c in sorted(present) if c not in ct_unique]
    ct_colors = {c: _ct_color(c) for c in ct_unique}

    slice_data = []
    for sl in pick[:2]:
        pos = np.where(slices_arr == sl)[0]
        slice_data.append((sp_coords[pos], ct[pos]))
    (sp_q, ct_q), (sp_w, ct_w) = slice_data[0], slice_data[1]

    xq_lo, xq_hi, yq_lo, yq_hi = _spatial_rect_bounds(sp_q)
    xw_lo, xw_hi, yw_lo, yw_hi = _spatial_rect_bounds(sp_w)
    gap = (xq_hi - xq_lo) * 0.08  # small gutter between the two brains
    dx = xq_hi - xw_lo + gap  # place worker just right of queen

    # One scatter per slice, points in random order, rather than a loop over
    # categories drawn largest-first.  A per-category loop paints each type as
    # an opaque layer, so whichever is drawn last covers the rest where they
    # interleave.  With Kenyon subtypes of comparable size sharing the same
    # compartments, painted area then tracks draw order, not abundance (a
    # small category drawn last can cover more pixels than a large one).
    # Interleaving the points removes the systematic priority.
    for sl_sp, sl_ct, xoff in ((sp_q, ct_q, 0.0), (sp_w, ct_w, dx)):
        keep = np.isin(sl_ct, list(ct_colors))
        pts, lab = sl_sp[keep], sl_ct[keep]
        if len(pts) == 0:
            continue
        order = rng.permutation(len(pts))
        ax.scatter(
            pts[order, 0] + xoff,
            pts[order, 1],
            c=[ct_colors[c] for c in lab[order]],
            s=4.0,
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
        )

    x_left, x_right = xq_lo, xw_hi + dx
    y_lo, y_hi = min(yq_lo, yw_lo), max(yq_hi, yw_hi)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")

    cx_q = (xq_lo + xq_hi) / 2
    cx_w = (xw_lo + xw_hi) / 2 + dx
    y_title = y_hi + (y_hi - y_lo) * 0.02
    for cx_t, brain_key in ((cx_q, "Queen"), (cx_w, "Worker")):
        ax.text(
            cx_t,
            y_title,
            brain_key,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="normal",
            clip_on=False,
        )
    _strip_spines(ax)
    # Scale bar under the worker (right) brain so it can never collide with the
    # centered legend row that sits under the panel.
    _add_scalebar(ax, x_right - x_left, center_x=cx_w)
    return ct_unique


# ── Panel F: whole-UMAP context + highlighted box cells ─────────────────────


def prepare_kc_selection(combined, focal):
    """Boolean mask (combined-order) of the KC-IC cube cells + KC accent color.

    Joins the worker-order cube selection onto the combined object by cell id so
    the exact same cells highlighted in Panel C are highlighted in Panel F.
    """
    _ensure_cluster_vote(combined)
    cvote = combined.obs["cluster_vote"].astype(str).values
    focal_mask = combined.obs["brain"].values == FOCAL_BRAIN
    combined_idx = combined.obs.index.astype(str)
    in_region = _kc_ic_region_mask(focal)
    focal_ids = focal.obs.index.astype(str).values
    focal_in_region_ids = set(focal_ids[in_region])
    in_box_combined = np.zeros(combined.n_obs, dtype=bool)
    for i in np.where(focal_mask)[0]:
        if combined_idx[i] in focal_in_region_ids:
            in_box_combined[i] = True
    kc_hex = get_pooled_color("Kenyon cells")
    return cvote, in_box_combined, kc_hex


def draw_panel_f(
    ax_f, combined, cvote, in_box_combined, kc_hex, rng, *, option=1, ctx_grey="#E4E4E4"
):
    """Whole UMAP (faint grey) + the box cells highlighted on top.

    A plain context+highlight -- no ring, no heavy KC-IC label. The three options
    differ only in how the box cells are marked:
      1  pooled cell-type colors (mostly KC-orange, honest ~93% Kenyon)
      2  one flat KC-accent color (unambiguous "these are the box cells")
      3  pooled colors + a small unobtrusive ring at the KC centroid
    Returns the highlighted (box) UMAP coordinates.
    """
    coords_umap = combined.obsm["X_umap"]
    n_sub_f = min(200_000, combined.n_obs)
    sub_idx_f = rng.choice(combined.n_obs, n_sub_f, replace=False)
    f_ctx = coords_umap[sub_idx_f]
    order_ctx = rng.permutation(len(f_ctx))
    ax_f.scatter(
        f_ctx[order_ctx, 0],
        f_ctx[order_ctx, 1],
        c=ctx_grey,
        s=1.5,
        alpha=0.30,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    box_coords = coords_umap[in_box_combined]
    box_types = cvote[in_box_combined]
    if box_coords.shape[0] > 0:
        ob = rng.permutation(len(box_coords))
        if option == 2:
            colors = kc_hex
        else:
            colors = np.array([get_pooled_color(t) for t in box_types])[ob]
        ax_f.scatter(
            box_coords[ob, 0],
            box_coords[ob, 1],
            c=colors,
            s=14,
            alpha=0.95,
            edgecolors="none",
            rasterized=True,
            zorder=10,
        )
        if option == 3:
            is_kc = np.array([get_pooled(t) == "Kenyon cells" for t in box_types])
            kc_pts = box_coords[is_kc] if is_kc.any() else box_coords
            cx, cy = float(kc_pts[:, 0].mean()), float(kc_pts[:, 1].mean())
            ax_f.scatter(
                [cx],
                [cy],
                s=340,
                facecolors="none",
                edgecolors=kc_hex,
                linewidths=1.5,
                zorder=11,
            )
    _square_umap_limits(ax_f, f_ctx, pad_frac=0.06)
    ax_f.set_box_aspect(1)
    _strip_spines(ax_f)
    return box_coords


def render_f_variants(combined, focal, output_dir):
    """Save standalone Panel-F variants for comparing callout styles."""
    cvote, in_box_combined, kc_hex = prepare_kc_selection(combined, focal)
    titles = {
        1: "Option 1 — box cells in cell-type colors (mostly KC-orange)",
        2: "Option 2 — box cells in one flat KC accent color",
        3: "Option 3 — cell-type colors + subtle KC-centroid marker",
    }
    for opt in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        rng = np.random.default_rng(42)  # identical context across variants
        draw_panel_f(ax, combined, cvote, in_box_combined, kc_hex, rng, option=opt)
        ax.set_anchor("N")
        _add_umap_frame(ax)
        ax.set_xlabel("UMAP1", fontsize=8)
        ax.set_ylabel("UMAP2", fontsize=8)
        ax.set_title(titles[opt], fontsize=9, pad=6)
        out = output_dir / f"fig_F_option{opt}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out}", flush=True)


# ── Composite ───────────────────────────────────────────────────────────────


def compose_figure(
    combined,
    queen,
    worker,
    output_dir,
    ct_labels=None,
    ct_cats=None,
    ct_color_map=None,
    cb=None,
    ol=None,
):
    """Assemble the 6 data panels into the composite (2 rows x 3 cols)."""
    print("  Composing final figure...", flush=True)

    fig = plt.figure(figsize=(16.5, 11.5))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.18, 1.0, 1.0],
        # Row 1 gets more height than row 0: the 3D panels (B, C) do not
        # fill their cell (an ink-bbox scan of the render showed ~20% of the
        # row-0 cell as dead space, with A's grid ending at the same row), so
        # this ratio pulls row 1 up into that space.
        height_ratios=[0.85, 1.15],
        hspace=0.15,
        wspace=0.12,
    )
    # Row 1 gets its OWN column widths (decoupled from row 0's A/B/C): the
    # majority-vote slices (D) on the left, then a packed strip of three equal-
    # size square UMAPs — CB | OL (Panel E) and the KC-IC zoom (Panel F) — all the
    # same size with minimal whitespace between them.
    gs_row1 = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs[1, :],
        width_ratios=[1.0, 2.45],
        wspace=0.10,
    )
    gs_umaps = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs_row1[1],
        wspace=0.05,
    )

    rng = np.random.default_rng(42)

    # Panels B, C and F all show FOCAL_BRAIN; panel D keeps both castes.
    focal = queen if FOCAL_BRAIN == "queen" else worker
    focal_mask = combined.obs["brain"].values == FOCAL_BRAIN
    combined_idx = combined.obs.index.astype(str)

    # ── A: Serial-section grid (transcript density) ──
    slices = sorted(queen.obs["slice_name"].unique())
    if hasattr(queen.X, "toarray"):
        total_counts = np.asarray(queen.X.sum(axis=1)).ravel()
    else:
        total_counts = queen.X.sum(axis=1).ravel()
    vmin_c = np.percentile(total_counts, 2)
    vmax_c = np.percentile(total_counts, 98)

    n_slices = len(slices)
    ncols_g = int(np.ceil(np.sqrt(n_slices)))
    nrows_g = int(np.ceil(n_slices / ncols_g))
    gs_a = GridSpecFromSubplotSpec(
        nrows_g,
        ncols_g,
        subplot_spec=gs[0, 0],
        hspace=0.10,
        wspace=0.08,
    )
    all_sp = queen.obsm["spatial"]
    sp_xmin, sp_xmax = all_sp[:, 0].min(), all_sp[:, 0].max()
    sp_ymin, sp_ymax = all_sp[:, 1].min(), all_sp[:, 1].max()
    sp_pad = max(sp_xmax - sp_xmin, sp_ymax - sp_ymin) * 0.02
    slice_axes = []
    for si, sl in enumerate(slices):
        r, c = divmod(si, ncols_g)
        ax_sl = fig.add_subplot(gs_a[r, c])
        mask_sl = queen.obs["slice_name"] == sl
        sl_idx = np.where(mask_sl.values)[0]
        s_sp = queen[mask_sl].obsm["spatial"]
        s_counts = total_counts[sl_idx]
        if len(s_sp) > 0:
            ax_sl.scatter(
                s_sp[:, 0],
                s_sp[:, 1],
                c=s_counts,
                cmap="viridis",
                s=0.10,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
                vmin=vmin_c,
                vmax=vmax_c,
            )
        ax_sl.set_xlim(sp_xmin - sp_pad, sp_xmax + sp_pad)
        ax_sl.set_ylim(sp_ymin - sp_pad, sp_ymax + sp_pad)
        ax_sl.set_aspect("equal")
        _strip_spines(ax_sl)
        ax_sl.set_title(f"z={si}", fontsize=6, pad=1)
        slice_axes.append(ax_sl)
    for si in range(n_slices, nrows_g * ncols_g):
        r, c = divmod(si, ncols_g)
        ax_empty = fig.add_subplot(gs_a[r, c])
        ax_empty.set_visible(False)
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=vmin_c, vmax=vmax_c),
    )
    cbar = fig.colorbar(sm, ax=slice_axes, shrink=0.6, aspect=22, pad=0.02)
    cbar.set_label("Transcripts per cell", fontsize=8)
    cbar.ax.tick_params(labelsize=6.5)
    # Sit the colorbar just right of Panel A's slice grid -- near the brains but
    # not overlapping them -- at near-full width.
    fig.canvas.draw()
    _cbp = cbar.ax.get_position()
    _a_right = max(a.get_position().x1 for a in slice_axes)
    cbar.ax.set_position([_a_right + 0.004, _cbp.y0, _cbp.width * 0.9, _cbp.height])

    # ── B: 3D anatomical-region render (focal brain) ──
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    b_coords = get_3d_coords(focal)
    b_coords[:, 2] *= Z_TRUE
    b_regions = get_region_labels(focal)
    b_idx = subsample_idx(focal.n_obs)
    b_coords_s = b_coords[b_idx]
    b_regions_s = b_regions[b_idx]

    # The unlabelled cells are about two thirds of the brain.  Drawn too faint
    # they vanish in print, and B reads as a hollow shell of regions floating
    # in space rather than as a whole brain with regions picked out.  Dark
    # enough to read as tissue, light enough that the region colours carry.
    unlab = b_regions_s == "unlabeled"
    ax_b.scatter(
        b_coords_s[unlab, 0],
        b_coords_s[unlab, 1],
        b_coords_s[unlab, 2],
        c="#C8C8C8",
        s=0.45,
        alpha=0.16,
        edgecolors="none",
        rasterized=True,
        depthshade=False,
    )
    handles_r = []
    # All nine drawn regions (six original + al/ellipsoid_body/fan_shaped_body),
    # in the shared preferred order from nine_region_common.
    region_order = DESCRIPTIVE_REGIONS
    present = set(b_regions_s) - {"unlabeled"}
    for region in region_order:
        if region not in present:
            continue
        rm = b_regions_s == region
        color = REGION_PALETTE.get(region, "#999999")
        ax_b.scatter(
            b_coords_s[rm, 0],
            b_coords_s[rm, 1],
            b_coords_s[rm, 2],
            c=color,
            s=1.2,
            alpha=0.7,
            edgecolors="none",
            rasterized=True,
            depthshade=False,
        )
        # Small-type / Large-type / Class II are the display names for the
        # three Kenyon compartments in every figure (matches
        # fig_s_regional_diversity, fig_s_regional_composition,
        # figS_tangram_confidence, si_exploded_regions). Sourced from
        # nine_region_common.display() rather than re-derived here, so a
        # newly drawn region gets the same label everywhere.
        label = region_display(region)
        handles_r.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=7,
                label=label,
            )
        )
    # Pull B's Y/Z labels IN (small left offset) so they hug B's own axis and
    # stay well clear of Panel A's "Transcripts per cell" colorbar to B's left.
    style_3d_ax(ax_b, label_lxmult=1.2)
    _set_box_aspect_from_ranges(ax_b, b_coords_s)
    # NB "Z (mm)" for B is placed later, post fig.canvas.draw() -- see there.
    # Two columns: the eight regions in one column made a tall thin strip down
    # the side of B's 3D axes, crowding the cloud it labels.
    ax_b.legend(
        handles=handles_r,
        fontsize=TICK_SIZE,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.3,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.88),
    )

    # ── C: 3D cell-type render (focal brain) + KC-IC zoom cube ──
    ax_c = fig.add_subplot(gs[0, 2], projection="3d")
    c_coords = get_3d_coords(focal)
    c_coords[:, 2] *= Z_TRUE
    # Cell-type colors in focal-brain order, get_pooled_color(cluster_vote):
    # the pooled scheme, not raw Leiden cluster colors, matching panels D and
    # F.  With BIMP_FIG1_KC_SPLIT=1, D splits the Kenyon block into the five
    # populations (KC_POPULATION_COLORS), while pooled 'Kenyon cells' here
    # stays #C8781E.
    _ensure_cluster_vote(combined)
    cvote = combined.obs["cluster_vote"].astype(str).values
    cvote_by_id = {combined_idx[i]: cvote[i] for i in np.where(focal_mask)[0]}
    focal_ids = focal.obs.index.astype(str).values
    c_colors = np.array([_ct_color(cvote_by_id.get(fid, "Mixed")) for fid in focal_ids])

    in_region = _kc_ic_region_mask(focal)  # focal-brain order, FULL
    n_box = int(in_region.sum())
    if "in_kc_ic" in focal.obs.columns:
        n_kc = int((in_region & focal.obs["in_kc_ic"].astype(bool).values).sum())
        pct_kc = 100 * n_kc / n_box if n_box else 0
    else:
        pct_kc = float("nan")
    print(
        f"    Small-type compartment cube: {n_box:,} {FOCAL_BRAIN} cells "
        f"({pct_kc:.0f}% in_kc_ic; expected {_KC_BOX_STATS[FOCAL_BRAIN]})",
        flush=True,
    )

    # ONE subsample index shared by coords, colors AND the in-region mask.
    c_idx = subsample_idx(focal.n_obs)
    c_coords_s = c_coords[c_idx]
    c_colors_s = c_colors[c_idx]
    in_region_s = in_region[c_idx]

    # All cells colored by pooled cell type, drawn at one uniform size.  The
    # box cells are not highlighted here (they read as floating dots above
    # the cloud); Panel F highlights them, and the black box alone marks the
    # KC-IC region, so C reads as a coherent cell-type brain with a box.
    ax_c.scatter(
        c_coords_s[:, 0],
        c_coords_s[:, 1],
        c_coords_s[:, 2],
        c=c_colors_s,
        s=1.4,
        alpha=0.75,
        edgecolors="none",
        rasterized=True,
        depthshade=False,
    )

    # Wireframe cube = actual 3D bounding box of the selected cells (true z scale),
    # so it encloses the highlighted cells by construction.
    sel = c_coords_s[in_region_s]
    bx0, bx1 = sel[:, 0].min(), sel[:, 0].max()
    by0, by1 = sel[:, 1].min(), sel[:, 1].max()
    bz0, bz1 = sel[:, 2].min(), sel[:, 2].max()
    # Pad half a section each side: the box is three sections (30 um) deep,
    # measured plane to plane plus the slab each outer plane stands for.
    padz = 0.5 * 10.0 * Z_TRUE
    # x/y are the box's own bounds (the selection is exactly this rectangle);
    # only z, which is quantised to section planes, takes the padded extent.
    bx0, bx1 = float(_KC_X0), float(_KC_X1)
    by0, by1 = float(_KC_Y0), float(_KC_Y1)
    bz0, bz1 = bz0 - padz, bz1 + padz
    # The box itself is drawn as 2D edges after the first draw(), where the
    # projection is final (3D zorder is unreliable here, so it would vanish in
    # the cloud). No text tag on the box.
    _kc_hex = get_pooled_color("Kenyon cells")

    style_3d_ax(ax_c)
    _set_box_aspect_from_ranges(ax_c, c_coords_s)
    # NB "Z (mm)" for C is placed later, post fig.canvas.draw() -- see there.

    # ── D: Queen | Worker majority-vote-labeled spatial slices ──
    ax_d = fig.add_subplot(gs_row1[0])
    d_cats = draw_majority_vote_slices(ax_d, combined, rng)
    ax_d.set_anchor("N")  # top-align the slices with the E/F UMAP row
    from bimp_st.cluster_labels import DISPLAY_NAME_OVERRIDES, display_legend_sort

    def _d_label(c):
        n = DISPLAY_NAME_OVERRIDES.get(str(c), get_pooled_display(c))
        return n.replace(" (all subtypes)", "\n(all subtypes)").replace(
            " (unresolved)", "\n(unresolved)"
        )

    d_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_ct_color(str(c)),
            markersize=6,
            label=_d_label(c),
        )
        for c in display_legend_sort(d_cats)
    ]
    if d_handles:
        # Full display names, wrapped -- not the pooled short codes. "KC" and
        # "OL" are also anatomical region codes in this paper (panel B labels
        # the regions nearby), so the same two letters would mean a cell type
        # here and a piece of tissue there. The "(all subtypes)" marker says
        # which categories are pools.
        #
        # Wrapped at up to five columns, anchored below the scalebar so the
        # "1 mm" label never collides with the legend.  A single legend row
        # would set the figure's width under bbox_inches="tight" when the
        # Kenyon split is on, squashing every panel when LaTeX scales the
        # figure to \linewidth.
        ax_d.legend(
            handles=d_handles,
            fontsize=6.5,
            frameon=False,
            # Largest ncol <= 5 that divides the handle count evenly, so the
            # last row is never a single orphan entry. At six categories that
            # is 3 (two rows of three); min(5, n) would give five plus one.
            ncol=next(
                (
                    k
                    for k in range(min(5, len(d_handles)), 1, -1)
                    if len(d_handles) % k == 0
                ),
                min(5, len(d_handles)),
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            handletextpad=0.25,
            columnspacing=0.7,
            labelspacing=0.3,
        )

    # ── E: snRNA reference — CB + OL as TWO separate annotated atlases ──
    # Side-by-side in the wide row-1 cell, so each square UMAP is large enough to
    # carry on-plot cluster labels directly.
    ax_e1 = fig.add_subplot(gs_umaps[0])
    ax_e2 = fig.add_subplot(gs_umaps[1])
    ax_e1.set_anchor("N")  # top-align the three square UMAPs (E CB, E OL, F)
    ax_e2.set_anchor("N")

    # CB and OL are two separate annotated atlases that share 6 fine types
    # (5 glia + hemocytes). Titles are the plain region names; the type counts
    # belong in the caption.
    if cb is not None and "X_umap" in cb.obsm:
        _draw_ref_umap(
            ax_e1, cb, rng, point_size=1.4, alpha=0.6, label=True, label_fs=6.0
        )
        _add_umap_frame(ax_e1)
        ax_e1.set_title("Central brain", fontsize=8, fontweight="normal", pad=3)
    else:
        ax_e1.text(
            0.5,
            0.5,
            "(CB ref n/a)",
            ha="center",
            va="center",
            fontsize=8,
            color="#AAAAAA",
            style="italic",
            transform=ax_e1.transAxes,
        )
        _strip_spines(ax_e1)

    if ol is not None and "X_umap" in ol.obsm:
        _draw_ref_umap(
            ax_e2,
            ol,
            rng,
            point_size=1.4,
            alpha=0.6,
            label=True,
            label_fs=6.0,
            collapse_types=_OL_PLACEHOLDER,
            collapse_label="OL subclusters\n(numbered, n=13)",
        )
        _add_umap_frame(ax_e2)
        ax_e2.set_title("Optic lobe", fontsize=8, fontweight="normal", pad=3)
    else:
        ax_e2.text(
            0.5,
            0.5,
            "(OL ref n/a)",
            ha="center",
            va="center",
            fontsize=8,
            color="#AAAAAA",
            style="italic",
            transform=ax_e2.transAxes,
        )
        _strip_spines(ax_e2)

    # ── F: whole UMAP + KC-IC cube cells highlighted (context + highlight) ──
    # Same size as the two Panel-E UMAPs (third square in the packed strip).
    ax_f = fig.add_subplot(gs_umaps[2])

    # Join the (full) selected focal-brain cells into combined by cell index —
    # the same cells the box highlights in Panel C.
    focal_in_region_ids = set(focal_ids[in_region])
    in_box_combined = np.zeros(combined.n_obs, dtype=bool)
    for i in np.where(focal_mask)[0]:
        if combined_idx[i] in focal_in_region_ids:
            in_box_combined[i] = True

    # Plain whole-manifold context + highlighted box cells (no ring, no label).
    draw_panel_f(
        ax_f, combined, cvote, in_box_combined, _kc_hex, rng, option=F_COMPOSITE_OPTION
    )
    ax_f.set_anchor("N")  # top-align with Panels D/E
    _add_umap_frame(ax_f)
    # No panel title: the caption names F. (F is the spatial Xenium embedding,
    # not the snRNA reference, so it must not read as continuous with E1/E2.)
    # No UMAP1/UMAP2 axis labels (matches the label-free Panel-E reference UMAPs).

    # Zoom callout: the cube sits inside a zoom ring; the two solid leader lines
    # start at the ring's left and right tangent points and run down to Panel
    # F's two TOP frame corners. The ring→frame trapezoid then reads as an
    # accurate "zoom into this region".
    fig.canvas.draw()
    # B/C's "Z (mm)" titles are placed here, not right after their own
    # box_aspect calls above: mplot3d only applies box_aspect's effect on
    # ax.get_proj() during an actual draw pass, so a pixel-anchored label
    # placed before any draw() resolves against a stale (pre-box_aspect)
    # projection and drifts -- exactly like this block's own corner
    # projections below, which is why this is the first draw() in the
    # function that touches ax_b/ax_c's final 3D state.
    _place_z_label(ax_b, side="right")
    # Panel C takes the right-side anchor too. With the left anchor its label
    # lands ~7 x-tick-units left of C's origin, in the empty gap between B's
    # data cloud and C's axis, where it reads as a stray label belonging to B.
    # C has no colorbar to its left, so the right anchor costs nothing and
    # keeps the label beside its own axis, matching B.
    _place_z_label(ax_c, side="right")
    _corners3d = [(x, y, z) for x in (bx0, bx1) for y in (by0, by1) for z in (bz0, bz1)]
    _proj2d = [
        proj3d.proj_transform(x, y, z, ax_c.get_proj())[:2] for (x, y, z) in _corners3d
    ]
    # Depth check: print the drawn 3D box aspect and the projected edge lengths
    # of the selection cube, so a depth distortion is visible in the log.
    _dp = [ax_c.transData.transform(p) for p in _proj2d]
    _ei = {
        k: float(np.hypot(*(np.array(_dp[a]) - np.array(_dp[b]))))
        for k, (a, b) in dict(x=(0, 4), y=(0, 2), z=(0, 1)).items()
    }
    _asp = [round(float(v), 3) for v in ax_c.get_box_aspect()]
    print(
        f"    3D depth check: box_aspect C={_asp}; cube edges "
        f"{bx1 - bx0:.0f} x {by1 - by0:.0f} x {bz1 - bz0:.0f} data units; "
        f"projected px x={_ei['x']:.1f} y={_ei['y']:.1f} z={_ei['z']:.1f}",
        flush=True,
    )
    # Draw the selection box as its 12 cube edges in 2D (the same transData
    # frame as the leader lines below), at a fixed printed-point weight. The
    # cube is a thin black wireframe inside a zoom ring: at this box size a
    # heavy stroke or white halo fills it into a blob. Edges join corners that
    # differ in exactly one of x, y, z.
    from bimp_st.print_scale import page_scale

    _s = page_scale(fig, 1.0)
    if not _s:
        _s = 6.67 / 13.83
    _cube_lw = CUBE_PRINT_PT / _s
    _cube_halo = _cube_lw + 2 * CUBE_HALO_PT / _s
    # Halo only when requested; with CUBE_HALO_PT = 0 the edges are plain strokes.
    _cube_effects = (
        [pe.Stroke(linewidth=_cube_halo, foreground="white"), pe.Normal()]
        if CUBE_HALO_PT > 0
        else []
    )
    for _i in range(8 if DRAW_CUBE else 0):
        for _j in range(_i + 1, 8):
            _ci, _cj = _corners3d[_i], _corners3d[_j]
            if sum(a != b for a, b in zip(_ci, _cj, strict=True)) != 1:
                continue
            (_x0, _y0), (_x1, _y1) = _proj2d[_i], _proj2d[_j]
            fig.add_artist(
                Line2D(
                    [_x0, _x1],
                    [_y0, _y1],
                    transform=ax_c.transData,
                    color="black",
                    lw=_cube_lw,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=6,
                    clip_on=False,
                    path_effects=_cube_effects,
                )
            )
    # No separate zoom marker: the cube is the selection. The two leaders start
    # at its leftmost and rightmost projected corners.
    _left = min(_proj2d, key=lambda p: p[0])
    _right = max(_proj2d, key=lambda p: p[0])
    # Panel F frame (FancyBboxPatch: margin _m, rounded pad ~0.02) — land the two
    # connectors on the top-left / top-right corners of its straight top edge.
    # boxstyle "round,pad=_frame_pad" inflates the (_m, _m, 1-2_m, 1-2_m) box
    # outward by _frame_pad on every side and rounds the new corners with
    # radius _frame_pad.  The two cancel: the corner arc runs from the
    # post-pad edge (_m - _frame_pad) out to radius _frame_pad, back to
    # x = _m, so the flat part of the top edge starts at x = _m with no extra
    # offset.  A small fixed epsilon keeps the line off the exact tangent point.
    _m = 0.03
    _frame_pad = 0.02  # must match _add_umap_frame's boxstyle pad
    _top_y = 1 - _m + _frame_pad
    _corner_eps = 0.004
    tgt_top = [(_m + _corner_eps, _top_y), (1 - _m - _corner_eps, _top_y)]
    # Left corner to F's top-left, right corner to F's top-right (no crossing).
    for (cx, cy), (tx, ty) in zip((_left, _right), tgt_top, strict=True):
        fig.add_artist(
            ConnectionPatch(
                xyA=(cx, cy),
                coordsA=ax_c.transData,
                xyB=(tx, ty),
                coordsB=ax_f.transAxes,
                color=_kc_hex,
                lw=LEADER_PRINT_PT / _s,
                ls="-",
                alpha=0.7,
                arrowstyle="-",
                zorder=5,
                shrinkA=0,
                shrinkB=0,
            )
        )

    # ── Aligned panel labels: bold letter at each panel's top-left corner,
    # followed on the same baseline by a short bold tag (one size, one offset
    # for every panel). Counts and caveats live in the caption. ──
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    _gap_px = 13.0 * fig.dpi / 72.0  # clear header band: > one tag height
    def _letter_tag(letter, tag, x_fig, y_fig):
        # house panel header (bimp_st.style); baseline at y_fig
        panel_header(
            fig, (x_fig, y_fig), letter, tag, letter_size=16, tag_size=10.5, dy_pts=0
        )

    # Row 0 (A, B, C): top-left of the gridspec cell, on one shared baseline
    # lifted above the z-index titles of A's top row.
    _y0_fig = max(
        fig.transFigure.inverted().transform(
            (0, a.title.get_window_extent(renderer).y1 + _gap_px)
        )[1]
        for a in slice_axes
    )
    for letter, tag, cell in zip(
        "ABC",
        ["Serial sections", "Annotated regions", "Cell types"],
        [gs[0, 0], gs[0, 1], gs[0, 2]],
        strict=True,
    ):
        pos = cell.get_position(fig)
        _letter_tag(letter, tag, pos.x0, _y0_fig)

    # Row 1 (D, E, F): left edge of the visible frame (axes box for D), on one
    # shared baseline lifted above D's Queen/Worker headers and E's sub-titles.
    _ref_x = []
    _tops = []
    for ax in (ax_d, ax_e1, ax_f):
        ref = ax.patches[-1] if ax.patches else ax
        bb = ref.get_window_extent(renderer)
        _ref_x.append(bb.x0)
        _tops.append(bb.y1)
    for t in list(ax_d.texts) + [ax_e1.title, ax_e2.title]:
        if t.get_text():
            _tops.append(t.get_window_extent(renderer).y1)
    _y_fig = fig.transFigure.inverted().transform((0, max(_tops) + _gap_px))[1]
    for letter, tag, xpx in zip(
        "DEF",
        ["Representative sections", "snRNA-seq reference", "Xenium UMAP"],
        _ref_x,
        strict=True,
    ):
        # F's zoom leader runs up-left from the frame's top-left corner through
        # the letter's position; start F's letter right of it (10 pt).
        if letter == "F":
            xpx += 17.0 * fig.dpi / 72.0
        x_fig = fig.transFigure.inverted().transform((xpx, 0))[0]
        _letter_tag(letter, tag, x_fig, _y_fig)

    out = output_dir / "fig_atlas_overview.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        import sys

        compose_only = "--compose-only" in sys.argv

        set_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Combined dataset for UMAP (both brains, 1.3M cells)
        print(f"Loading {COMBINED_INPUT}...", flush=True)
        combined = sc.read_h5ad(COMBINED_INPUT)
        print(f"  {combined.n_obs:,} cells, {combined.n_vars} genes", flush=True)
        combined = maybe_subsample_adata(combined)

        # Queen brain for the density grid (A)
        print(f"\nLoading {QUEEN_INPUT}...", flush=True)
        queen = sc.read_h5ad(QUEEN_INPUT)
        print(f"  {queen.n_obs:,} cells, {queen.n_vars} genes", flush=True)
        queen = maybe_subsample_adata(queen)
        # aligned_clustered.h5ad only ships the original six in_* columns; the
        # three new regions are attached from the nine-region sidecar table, keyed
        # on (slice_name, cell_id) rather than position -- see nine_region_common.
        # Refresh all nine flags: the six baked into the atlas object are frozen
        # at whatever labeling.py last wrote and are out of date (four polygons on
        # worker 0068401 R1 move 119 cells from the lobula into the antennal
        # lobe).  Attaching only the three new regions would leave those cells
        # flagged as both.
        attach_region_flags(queen, regions=DESCRIPTIVE_REGIONS)

        # Worker brain for the 3D panels (B, C) — has KC subtypes
        print(f"\nLoading {WORKER_INPUT}...", flush=True)
        worker = sc.read_h5ad(WORKER_INPUT)
        print(f"  {worker.n_obs:,} cells, {worker.n_vars} genes", flush=True)
        worker = maybe_subsample_adata(worker)
        attach_region_flags(worker, regions=DESCRIPTIVE_REGIONS)

        # Spatial Leiden cell types (combined-order) for reference/consistency.
        ct_labels, ct_cats, ct_color_map = load_spatial_celltypes(combined)

        # snRNA references for panel E: the two separate annotated atlases.
        print(f"\nLoading {CB_REF}...", flush=True)
        cb = sc.read_h5ad(CB_REF)
        print(
            f"  {cb.n_obs:,} cells, {cb.obs['cell_type'].nunique()} types", flush=True
        )
        print(f"\nLoading {OL_REF}...", flush=True)
        ol = sc.read_h5ad(OL_REF)
        print(
            f"  {ol.n_obs:,} cells, {ol.obs['cell_type'].nunique()} types", flush=True
        )

        # Composite (all panels are drawn inline in compose_figure)
        compose_figure(
            combined,
            queen,
            worker,
            OUTPUT_DIR,
            ct_labels=ct_labels,
            ct_cats=ct_cats,
            ct_color_map=ct_color_map,
            cb=cb,
            ol=ol,
        )

        # Standalone Panel-F callout variants, for comparing callout styles.
        print("\nRendering Panel F variants...", flush=True)
        render_f_variants(
            combined, queen if FOCAL_BRAIN == "queen" else worker, OUTPUT_DIR
        )

        print(f"\nDone. Panels + composite in {OUTPUT_DIR}/", flush=True)
