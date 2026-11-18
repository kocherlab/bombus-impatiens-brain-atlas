"""Data for fig_kc_subtypes.py:
loading, computation and the constants they use.

See fig_kc_subtypes.py (the composer) for the figure and how to render it.
"""
from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import functools
import glob as _glob
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import kenyon.snrna_kc_subclusters as snuc_kc
import numpy as np
import pandas as pd
import scanpy as sc
import shapely.geometry as sgeom
from anndata.io import read_elem
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull, cKDTree
from skimage import measure

from bimp_st.nine_region_common import KC_POPULATION_COLORS, REGION_COLORS
from bimp_st.paths import (
    CANONICAL_TANGRAM,
    REPO,  # noqa: E402
)
from bimp_st.style import (  # noqa: E402
    set_pub_style,
)

DATA = REPO / "results" / "fig5_kc"
# BIMP_FIG_KC_OUT renders the figure and its subzone_stats.csv to a scratch dir
# (render-identity checks) instead of the shared results/.
_FIG_OUT = os.environ.get("BIMP_FIG_KC_OUT")
FIG_DIR = Path(_FIG_OUT or REPO / "results" / "figures" / "main" / "kc_subtypes")
FIG_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0

# ltKC subcluster projection onto the three Kenyon compartments, run on the CB
# reference's own Large-type nuclei (see load_ltkc() and
# scripts/kenyon/project_large_type_subclusters.py).  The ltKC-reference run at
# results/tangram_ltKC/ltKC_spatial_subclusters_allKC_dsource.h5ad is not read.
LTKC_H5AD = (
    REPO / "results" / "tangram_ltKC_cb" / "ltKC_CB_spatial_subclusters_dsource.h5ad"
)
# Restricted 5-KC-population reclassification for panel B -- see
# project_kc_populations.py and load_kc5pop() below.
KC5POP_H5AD = REPO / "results" / "tangram_ltKC" / "kc5pop_spatial_allKC_dsource.h5ad"
CTX_H5AD = CANONICAL_TANGRAM
# This script never reads tangram_cell_type -- only compartment/leiden/brain/z
# flags and raw gene expression -- so neither the Tangram column correction nor
# the choice of whole-atlas mapping changes any value here.

# Matched queen/worker section for every spatial crop (panels C, F, G).  Chosen
# from a six-candidate diagnostic over z with all three compartments present in
# both brains: z = 260 has the highest cell count of the candidates in both
# brains (queen 1,690 / worker 1,244 Kenyon-compartment cells) and shows three
# complete, well-separated calyx cups with their KC-OC rim intact in each.
ZOOM_Z = 260.0
ZOOM_ASP = 0.58  # crop width / height floor; the Kenyon band is tall + narrow

# ── Palette ────────────────────────────────────────────────────────────────
# Separate semantic axes use visually different kinds of palette, so a reader
# cannot carry a colour meaning across them:
#   KC population / compartment  the atlas region colours       (A, B, C, D)
#   ltKC subcluster              the reference's own triad      (E-H)
#   caste                        the house gold/blue            (H, I)
#
# Compartment colours are `fig:atlas_overview`'s
# (bimp_st.nine_region_common.REGION_COLORS,
# also what fig_atlas_overview.py draws), read from the source rather than copied, so
# a compartment is the same colour in both figures.  scripts/
# region_diversity.py's REGION_GROUP_COLOR carries the same three
# hexes as literals -- change both or neither.
COMP_COLORS = {
    "KC-IC": REGION_COLORS["kc_ic"],
    "KC-NC": REGION_COLORS["kc_nc"],
    "KC-OC": REGION_COLORS["kc_oc"],
}
COMP_ORDER = ["KC-IC", "KC-NC", "KC-OC"]
COMP_KEY = {"kc_ic": "KC-IC", "kc_nc": "KC-NC", "kc_oc": "KC-OC"}

# A named population takes the colour of its namesake compartment, so a colour
# means one thing across the figure: Large-type orange, Small-type mauve,
# Class II yellow, whether the panel shows nuclei (A, B left, D strip) or
# tissue (B right, C, D columns), so yellow always means Class II.  Middle-type
# and Other KC have no compartment and get
# neutral tones that collide with nothing on the canvas (compartments, the
# S1-S3 triad, the H gene colours, caste gold/blue, the #DCDCDC context grey)
# and stay dark enough to read as label text.
# The palette itself lives in bimp_st.nine_region_common.KC_POPULATION_COLORS
# (shared with `fig:atlas_overview` E, the SI KC-population figure and the
# fine-type palette
# in bimp_st.cluster_labels itself); TYPE_COLORS keeps its
# name because fig_xenium_celltype_overlay.py reads F5.TYPE_COLORS.
TYPE_COLORS = dict(KC_POPULATION_COLORS)
assert all(
    TYPE_COLORS[t] == COMP_COLORS[c]
    for t, c in {
        "Large-type Kenyon cells": "KC-NC",
        "Small-type Kenyon cells": "KC-IC",
        "Class II Kenyon cells": "KC-OC",
    }.items()
)
TYPE_SHORT = {
    "Large-type Kenyon cells": "Large-type",
    "Small-type Kenyon cells": "Small-type",
    "Middle-type Kenyon cells": "Middle-type",
    "Class II Kenyon cells": "Class II",
    "Other Kenyon cells": "Other KC",
}
TYPE_ORDER = [
    "Large-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Middle-type Kenyon cells",
    "Other Kenyon cells",
]

# ltKC subcluster identity is imported, never re-typed, so panels E-H and the
# standalone SI figure cannot drift apart.
SUB_CLUSTERS = list(snuc_kc.CLUSTERS)  # ["0", "1", "2"]
SUB_COLORS = dict(snuc_kc.COLORS)
SUB_LABELS = dict(snuc_kc.LABELS)  # "Subcluster 1/2/3"
SUB_SHORT = {c: SUB_LABELS[c].replace("Subcluster ", "S") for c in SUB_CLUSTERS}
SUB_RGB = np.array([to_rgb(SUB_COLORS[c]) for c in SUB_CLUSTERS])

# Display symbol per panel gene, keyed by lower-cased v4.3 id (panel D reads
# BIMP_v4.3_*, panel E the spatial lower-case var_names).  Follows
# Supplementary Table S1's symbol column, abbreviated where S1 gives a full
# protein name.  Keyed by id, not description: 009611 and 009612 share the
# lookup description "octopamine receptor beta-1R" and only S1 separates them
# (009611 is the "-like" paralogue).  Without this, D showed trailing-two-word
# fragments ("receptor beta-2R", "protein 3") and E a front-truncated
# "nAChR...".
PANEL_SYMBOL = {
    "bimp_v4.3_009307": "oct-β2R",
    "bimp_v4.3_009612": "oct-β1R",
    "bimp_v4.3_009611": "oct-β1R-like",
    "bimp_v4.3_001355": "LIM protein 3",
    "bimp_v4.3_006026": "furin-like protease 1",
    "bimp_v4.3_000756": "VAChT",
    "bimp_v4.3_009119": "nAChRα10-like",
    "bimp_v4.3_006191": "GlyRα1",
    "bimp_v4.3_004045": "AChE",
    "bimp_v4.3_005730": "sNPF",
}


def _wrap_label(label, width=20):
    """Split a long multi-word label onto two lines at its first space."""
    if len(label) > width and " " in label:
        return label.replace(" ", "\n", 1)
    return label

# The shared house style rather than a local rcParams block, so type sizes and
# bold subpanel titles match the neighbouring figures.
set_pub_style()


# ── ltKC subcluster projection: one load, shared by the spatial panels ────
@functools.lru_cache(maxsize=1)
def load_ltkc():
    """Per-cell ltKC subcluster probability vectors for every cell of the three
    Kenyon compartments.

    Reads the run on the CB reference (project_large_type_subclusters.py).
    Its reference (results/fig5_kc/CB_LargeType_subclustered.h5ad, built by
    large_type_kc_subclusters.py) subclusters the CB atlas's own
    Large-type nuclei -- the reference panel A uses -- with both castes in
    every cluster and spread across all 3 queen + 3 worker libraries.

    Leiden resolution 0.05 is the only point in a 0.05-0.4 sweep giving
    exactly 3 clusters; every higher resolution splits the largest one into
    near-equal halves.  The cluster densities are compared with the ltKC
    reference's own 3-way split, as a check that the structure is not an
    ltKC-dataset artefact.  Query
    set, marker recipe, clusters mode, epoch count and the classic-tangram
    `d_source` prior are identical to the ltKC-reference run; only the reference and
    the cluster densities differ.

    The S1/S2 axis is largely position: the AUROC of distance to the nearest
    KC-IC cell for the call, and the S2 share across the eight most numerous
    cell types sharing KC-NC, are computed in
    (scripts/kenyon/large_type_projection_anatomy_test.py,
    results/ltkc_diag/p3_cb_summary.txt). Caption F and G as position, never as subtype
    territories.

    The zoom window and panels F and G all come from this one object, so they
    never mix two mappings.
    """
    ad = sc.read_h5ad(LTKC_H5AD)
    ad.obs_names_make_unique()
    p = ad.obsm["tangram_ct_pred"]
    if not isinstance(p, pd.DataFrame):
        p = pd.DataFrame(np.asarray(p), columns=SUB_CLUSTERS)
    p.columns = [str(c) for c in p.columns]
    p = p[SUB_CLUSTERS]
    probs = p.div(p.sum(axis=1), axis=0).fillna(1.0 / 3).values.astype(float)
    o = ad.obs
    ic = o["in_kc_ic"].values.astype(bool)
    nc = o["in_kc_nc"].values.astype(bool)
    # A handful of cells carry two compartment flags; resolve inner -> outer so
    # every cell lands in exactly one compartment and the three sum to n.
    comp = np.where(ic, "KC-IC", np.where(nc, "KC-NC", "KC-OC"))
    return dict(
        probs=probs,
        argmax=probs.argmax(axis=1),
        comp=comp,
        brain=o["brain"].astype(str).values,
        z=o["z_level"].values.astype(float),
        xy=np.asarray(ad.obsm["spatial"], dtype=float),
    )


@functools.lru_cache(maxsize=1)
def load_kc5pop():
    """Per-cell probability vectors over panel A's 5 named KC populations, for
    every cell of the three Kenyon compartments -- feeds panel B only (the
    other spatial panels use load_ltkc()'s 3 ltKC subclusters; the two answer
    independent questions).

    Built by project_kc_populations.py as a restricted
    reclassification: the query is Kenyon-compartment cells only
    (in_kc_ic | in_kc_nc | in_kc_oc, no cell-type gate, as in load_ltkc())
    and the reference is only the 5 KC populations (data/sn/BIMP_atlas_qw_
    CentralBrain.h5ad filtered to panel A's 5 cell_type values; the file has
    no generic "Kenyon cells" cluster, so this recovers exactly panel A's
    nuclei), removing every escape-hatch category.  Same
    classic-tangram d_source-restored config as load_ltkc() (500 epochs,
    lr=0.1, seed=42, cluster SUM expression, reference-density prior).

    The whole-atlas Tangram calls do not answer this question: their argmax
    sends most KC-compartment cells to the generic "Kenyon cells"/"Other
    neurons" buckets rather than one of the 5, and renormalizing their soft
    probabilities onto the 5 columns does not preserve the reference
    population sizes (project_kc_populations.py checks both).

    The run log (results/tangram_ltKC/slurm_kc5pop_12790171.out) records the
    renormalized-confidence summary and the compartment x call table, per
    caste.

    Built against a different corrected whole-atlas mapping. Its inputs do
    not depend on that mapping: against paths.CANONICAL_TANGRAM the query is
    the same cells in the same order, with identical in_kc_* flags and
    identical expression of all 100 panel genes; reference and Tangram config
    are the snRNA side only. Its obs also carries that mapping's
    tangram_cell_type, which nothing here reads.
    """
    ad = sc.read_h5ad(KC5POP_H5AD)
    ad.obs_names_make_unique()
    p = ad.obsm["tangram_ct_pred"]
    if not isinstance(p, pd.DataFrame):
        p = pd.DataFrame(np.asarray(p), columns=TYPE_ORDER)
    p = p[TYPE_ORDER]
    probs = p.div(p.sum(axis=1), axis=0).values.astype(float)
    o = ad.obs
    ic = o["in_kc_ic"].values.astype(bool)
    nc = o["in_kc_nc"].values.astype(bool)
    comp = np.where(ic, "KC-IC", np.where(nc, "KC-NC", "KC-OC"))
    return dict(
        probs=probs,
        argmax=probs.argmax(axis=1),
        comp=comp,
        brain=o["brain"].astype(str).values,
    )


@functools.lru_cache(maxsize=1)
def atlas_cells():
    """One read of the combined atlas, shared by every consumer below.

    The file is 2.3 GB on a network filesystem and opening it dominates this
    script's runtime, so read it once and slice.

    Compartment membership uses the same inner -> outer precedence as
    load_ltkc(), so a cell carrying two flags lands in the same compartment in
    every panel of the figure.
    """
    ad = sc.read_h5ad(CTX_H5AD, backed="r")
    o = ad.obs
    ic = o["in_kc_ic"].values.astype(bool)
    nc = o["in_kc_nc"].values.astype(bool)
    oc = o["in_kc_oc"].values.astype(bool)
    comp = np.where(ic, "KC-IC", np.where(nc, "KC-NC", np.where(oc, "KC-OC", "")))
    return dict(
        xy=np.asarray(ad.obsm["spatial"], dtype=float),
        comp=comp,
        brain=o["brain"].astype(str).values,
        z=o["z_level"].values.astype(float),
    )


def section_cells(brain, z):
    """Every cell of one section from the combined atlas, with its true
    compartment membership.

    Two jobs: (1) light-grey anatomical context under the spatial crops, and
    (2) the compartments themselves, for panel C and for the grey anatomy in
    panels F/G.  Both live in the same aligned frame as the ltKC object.
    Taking compartments from the atlas means panel C's counts describe every
    cell of each compartment.
    """
    a = atlas_cells()
    m = (a["brain"] == brain) & (a["z"] == z)
    return dict(xy=a["xy"][m], comp=a["comp"][m])


# There is deliberately no per-compartment "cells called Large-type" table:
# per-cell subtype argmax is noise here (see the module docstring).


@functools.lru_cache(maxsize=1)
def zoom_window():
    """One magnification shared by every spatial crop, with a per-brain centre.

    The window size is the max over both brains and applied to both; only the
    centre is per-brain.  Sizing each crop from its own extent would magnify
    one caste relative to the other and print two different scale bars.
    ZOOM_ASP is a floor, not a target: it widens the window only if the data
    would otherwise be narrower than that portrait ratio.  The padding and
    ZOOM_ASP match the data's own padded
    width:height ratio across both brains, keeping background to a minimum.
    """
    d = load_ltkc()
    centres, half_x, half_y = {}, [], []
    for b in ("queen", "worker"):
        c = d["xy"][(d["brain"] == b) & (d["z"] == ZOOM_Z)]
        x0, x1 = c[:, 0].min(), c[:, 0].max()
        y0, y1 = c[:, 1].min(), c[:, 1].max()
        centres[b] = ((x0 + x1) / 2, (y0 + y1) / 2)
        pad = 0.04 * (y1 - y0)
        hy = (y1 - y0) / 2 + pad
        half_y.append(hy)
        half_x.append(max((x1 - x0) / 2 + pad, ZOOM_ASP * hy))
    hy = max(half_y)
    return centres, max(max(half_x), ZOOM_ASP * hy), hy


def _morph_outline(points, close_r, cell_frac=0.006, min_area_frac=0.02):
    """Smooth boundary around a 2D point cloud that encloses every point in
    the returned piece(s) by construction, not by a density/circumradius
    threshold that can drop points.

    An alpha shape (a Delaunay circumradius cutoff) is a density threshold:
    it drops points in sparse triangles, and widening the cutoff never reached
    zero exclusion on the Large-type cloud -- past ~8x the median
    nearest-neighbour distance it got worse, as other triangles started to
    drop.  The result was a visible notch where a sparse bridge of Large-type
    nuclei joins the cloud's two dense cores.

    This function is not density-thresholded: it rasterizes `points` onto a
    fine grid (cell size `cell_frac` of the cloud's span), so every point
    fills at least one grid cell before any smoothing, even at close_r=0;
    then morphologically closes the mask by `close_r` (bridging gaps of
    roughly that radius, which merges the two cores through the bridge) and
    lightly Gaussian-blurs it to soften the pixel staircase before
    contouring.  `min_area_frac` then drops fragments under that fraction of
    the largest piece's area -- the only source of exclusion, deliberate and
    inspectable: it keeps a few isolated stray nuclei deep inside another
    population's cloud (e.g. one in Class II's territory) from each getting
    a tiny ring.
    """
    points = np.asarray(points)
    if len(points) < 4:
        return []
    span = max(
        points[:, 0].max() - points[:, 0].min(), points[:, 1].max() - points[:, 1].min()
    )
    cell = max(span * cell_frac, 1e-6)
    pad = 5 * close_r + 2 * cell
    xlim = (points[:, 0].min() - pad, points[:, 0].max() + pad)
    ylim = (points[:, 1].min() - pad, points[:, 1].max() + pad)
    nx = int(np.ceil((xlim[1] - xlim[0]) / cell)) + 1
    ny = int(np.ceil((ylim[1] - ylim[0]) / cell)) + 1
    ix = np.clip(((points[:, 0] - xlim[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((points[:, 1] - ylim[0]) / cell).astype(int), 0, ny - 1)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[iy, ix] = True
    r = max(int(round(close_r / cell)), 1) if close_r > 0 else 0
    struct = ndi.generate_binary_structure(2, 2)
    if r:
        mask = ndi.binary_dilation(mask, structure=struct, iterations=r)
        mask = ndi.binary_closing(mask, structure=struct, iterations=r)
    sm = ndi.gaussian_filter(mask.astype(float), sigma=max(r / 3, 0.6))
    pieces = []
    for c in measure.find_contours(sm, level=0.5):
        pieces.append(
            np.column_stack([xlim[0] + c[:, 1] * cell, ylim[0] + c[:, 0] * cell])
        )
    if not pieces:
        return pieces
    areas = [sgeom.Polygon(p).area if len(p) >= 3 else 0.0 for p in pieces]
    max_area = max(areas)
    return [
        p for p, a in zip(pieces, areas, strict=False) if a >= min_area_frac * max_area
    ]


def _sankey_ribbon(x0, y0a, y0b, x1, y1a, y1b):
    """Smooth cubic-bezier sankey ribbon between a left and a right segment."""
    xm = (x0 + x1) / 2
    verts = [
        (x0, y0a),
        (xm, y0a),
        (xm, y1a),
        (x1, y1a),
        (x1, y1b),
        (xm, y1b),
        (xm, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


# ── Panels H/I: subcluster markers in tissue and by distance ──────────────
KC5_LOG = REPO / "results" / "tangram_ltKC" / "slurm_kc5pop_12790171.out"
H_POPS = [
    ("Large", "Large-type Kenyon cells"),
    ("Middle", "Middle-type Kenyon cells"),
    ("Small", "Small-type Kenyon cells"),
    ("ClassII", "Class II Kenyon cells"),
    ("Other", "Other Kenyon cells"),
]
H_COMPS = ["KC-NC", "KC-IC", "KC-OC"]  # Large-type, Small-type, Class II


RUN_DIRS = {
    "queen": REPO / "data/20251204__194847___20251204",
    "worker": REPO / "data/20260115__193720__20260115",
}
CROP_DIRS = {
    "queen": REPO / "pipelines/wholebrain/queen/data",
    "worker": REPO / "pipelines/wholebrain/worker/data",
}
# Two genes that mark the two spatially ordered subclusters on the Xenium
# panel: Oamb is the strongest S1-specific panel marker, sNPF the S2 marker the
# caste story rests on.  Colours: one meaning each (S1/S2/S3 triad = subclusters,
# gold/blue = caste, orange/lilac/yellow = compartments), so genes get their own.
GENE_ID = {
    "Oamb": "bimp_v4.3_009024",
    "sNPF": "bimp_v4.3_005730",
    "AChE": "bimp_v4.3_004045",
}
# Calyx lobe cross-sections (_chosen_lobes), all distances in section microns:
LOBE_MIN_CORE_CELLS = 150  # a section needs this many Small-type cells to be scanned
LOBE_SUBSAMPLE = 3  # every 3rd Small-type cell enters the linkage (speed)
LOBE_LINK_UM = 120  # single-linkage distance that splits Small-type cores
LOBE_MIN_CLUSTER = 60  # subsampled cells for a core to count as a lobe
LOBE_RADIUS_UM = 250  # Large-type cells within this radius of the core centre
LOBE_MIN_LARGE = 150  # ... and at least this many of them
RING_UM = (40, 200)  # annulus around the core scored for angular coverage
RING_BINS = 16  # angular bins of the annulus
RING_MIN_CELLS = 5  # Large-type cells for a bin to count as covered
EXAMPLE_MIN_COV = 0.6  # example-lobe candidates: coverage at least this ...
EXAMPLE_MIN_LARGE = 300  # ... and this many Large-type cells
CUP_MIN_COV = 0.75  # coverage of a "cup-shaped" lobe (lobes_cup_* counts)
SM_GRID = np.linspace(np.log10(6), np.log10(250), 36)
SUBZONE_STATS = {}  # summary numbers; written to CSV


def _lighten(hex_colour, f):
    r, g, b = to_rgb(hex_colour)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


@functools.lru_cache(maxsize=1)
def _subzone_data():
    """Large-type compartment cells with real (section-um) coordinates, the
    distance to the nearest Small-type compartment cell in the same section,
    and Oamb / sNPF detection (>= 1 transcript).  Raw per-crop coordinates are
    joined on (slice_name, cell_id); the atlas frame is not in micrometres."""
    ad = sc.read_h5ad(LTKC_H5AD)
    o = ad.obs
    n = ad.n_obs
    xy = np.full((n, 2), np.nan)
    for (b, sl), idx in o.groupby(
        ["brain", "slice_name"], observed=True
    ).indices.items():
        with h5py.File(CROP_DIRS[b] / f"{sl}.h5ad", "r") as h:
            cid = np.asarray(read_elem(h["obs"]["cell_id"])).astype(str)
            raw = h["obsm/spatial"][:].astype(float)
        pos = pd.Series(np.arange(len(cid)), index=cid)
        ids = o["cell_id"].values[idx].astype(str)
        ok = pd.Index(ids).isin(pos.index)
        xy[idx[ok]] = raw[pos[ids[ok]].values]
    ic = o["in_kc_ic"].values.astype(bool)
    oc = o["in_kc_oc"].values.astype(bool)
    nc = o["in_kc_nc"].values.astype(bool) & ~ic
    have = np.isfinite(xy[:, 0])
    brain = o["brain"].astype(str).values
    sl_all = o["slice_name"].astype(str).values
    dist = np.full(n, np.nan)
    for idx in o.groupby(["brain", "slice_name"], observed=True).indices.values():
        m = np.zeros(n, bool)
        m[idx] = True
        a_, n_ = m & ic & have, m & nc & have
        if a_.any() and n_.any():
            dist[n_] = cKDTree(xy[a_]).query(xy[n_])[0]
    X = np.asarray(ad.X)
    var = {v.lower(): i for i, v in enumerate(ad.var_names)}
    det = {g: X[:, var[GENE_ID[g]]] > 0 for g in GENE_ID}
    z_lvl = o["z_level"].values.astype(float)
    p_ = ad.obsm["tangram_ct_pred"]
    p_ = np.asarray(p_.values if isinstance(p_, pd.DataFrame) else p_, float)
    sub = p_.argmax(axis=1)
    return dict(
        brain=brain,
        sl=sl_all,
        z=z_lvl,
        sub=sub,
        xy=xy,
        ic=ic,
        oc=oc,
        nc=nc,
        dist=dist,
        det=det,
        cell_id=o["cell_id"].astype(str).values,
    )


@functools.lru_cache(maxsize=1)
def _chosen_lobes():
    """The example calyx lobe (one per caste) for panel H: the lobe of the section
    shown in panels F and G with the most even S1/S2/S3 populations (largest
    share of the rarest subcluster) among cup-shaped lobes (Small-type core
    wrapped by Large-type cells over >= 60% of 16 angular bins), a rule that
    never looks at the gradient.  Also returns the lobe table for the
    all-lobe consistency numbers."""
    from scipy.stats import spearmanr

    d = _subzone_data()
    xy, nc, ic = d["xy"], d["nc"], d["ic"]
    rows = []
    for b in ("queen", "worker"):
        for sl in np.unique(d["sl"][d["brain"] == b]):
            m = (d["sl"] == sl) & np.isfinite(xy[:, 0])
            P = xy[m & ic]
            if len(P) < LOBE_MIN_CORE_CELLS:
                continue
            Ps = P[::LOBE_SUBSAMPLE]
            lab = fcluster(linkage(Ps, "single"), LOBE_LINK_UM, criterion="distance")
            for lb in np.unique(lab):
                Q = Ps[lab == lb]
                if len(Q) < LOBE_MIN_CLUSTER:
                    continue
                c = Q.mean(0)
                area = ConvexHull(Q).volume
                sel = m & nc & (np.hypot(*(xy - c).T) < LOBE_RADIUS_UM)
                if sel.sum() < LOBE_MIN_LARGE:
                    continue
                dd = np.hypot(*(xy[sel] - c).T)
                ang = np.arctan2(*(xy[sel] - c).T[::-1])
                ring = (dd > RING_UM[0]) & (dd < RING_UM[1])
                edges = np.linspace(-np.pi, np.pi, RING_BINS + 1)
                bins = np.digitize(ang[ring], edges) - 1
                per_bin = np.bincount(bins, minlength=RING_BINS)[:RING_BINS]
                cov = float(np.mean(per_bin >= RING_MIN_CELLS))
                fr = np.bincount(d["sub"][sel], minlength=3) / sel.sum()
                r = dict(
                    brain=b,
                    sec=sl,
                    cx=c[0],
                    cy=c[1],
                    area=area,
                    n=int(sel.sum()),
                    cov=cov,
                    minshare=float(fr.min()),
                )
                for g in GENE_ID:
                    v = d["det"][g][sel].astype(float)
                    r["rho_" + g] = (
                        spearmanr(d["dist"][sel], v).statistic
                        if len(np.unique(v)) > 1
                        else np.nan
                    )
                rows.append(r)
    LB = pd.DataFrame(rows)
    # the example lobes come from the same section as panels F and G (z = ZOOM_Z
    # in each brain): the lobe with the most angular coverage there
    pick = {}
    for b in ("queen", "worker"):
        secs = np.unique(d["sl"][(d["brain"] == b) & (d["z"] == ZOOM_Z)])
        assert len(secs) == 1, (b, secs)
        c = LB[(LB.brain == b) & (LB.sec == secs[0])]
        cc = c[(c["cov"] >= EXAMPLE_MIN_COV) & (c.n >= EXAMPLE_MIN_LARGE)]
        c = cc if len(cc) else c
        pick[b] = c.sort_values(["minshare", "cov"], ascending=False).iloc[0]
        print(
            f"  panel H lobe ({b}): section z={int(ZOOM_Z)}, coverage "
            f"{pick[b]['cov']:.2f}, {int(pick[b].n)} Large-type cells, "
            f"rarest subcluster {pick[b]['minshare']:.2f}",
            flush=True,
        )
    for b in ("queen", "worker"):
        s = LB[LB.brain == b]
        SUBZONE_STATS[f"lobes_{b}"] = len(s)
        SUBZONE_STATS[f"lobes_cup_{b}"] = int((s["cov"] >= CUP_MIN_COV).sum())
        for g, sgn in (("Oamb", 1), ("sNPF", -1)):
            x = s["rho_" + g].dropna()
            SUBZONE_STATS[f"lobe_sign_{g}_{b}"] = float(
                100 * (np.sign(x) == sgn).mean()
            )
    SUBZONE_STATS["lobes_total"] = len(LB)
    return pick, LB


def _smooth_sections(dist, y, secs, bw=0.10, boot=300):
    """Kernel smooth over ALL cells; 95% band from a bootstrap over sections
    (the unit of replication), not over cells."""
    ld = np.log10(np.clip(dist, 3, None))
    us = np.unique(secs)
    num = np.zeros((len(us), len(SM_GRID)))
    den = np.zeros_like(num)
    for i, s_ in enumerate(us):
        m = secs == s_
        w = np.exp(-0.5 * ((ld[m, None] - SM_GRID[None]) / bw) ** 2)
        num[i], den[i] = (w * y[m, None]).sum(0), w.sum(0)
    est = num.sum(0) / den.sum(0)
    r = np.random.default_rng(SEED)
    bs = np.array(
        [
            num[ix].sum(0) / den[ix].sum(0)
            for ix in (r.integers(0, len(us), len(us)) for _ in range(boot))
        ]
    )
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def _lobe_view(brain, lobe, W=270.0, WY=270.0):
    """Cell outlines + smoothed Oamb / sNPF detection on the Large-type cells of
    one lobe window (section um)."""
    d = _subzone_data()
    c = np.array([lobe.cx, lobe.cy])
    sname = lobe.sec
    parts = sname.split("__")
    with h5py.File(CROP_DIRS[brain] / f"{sname}.h5ad", "r") as h:
        cid = np.asarray(read_elem(h["obs"]["cell_id"])).astype(str)
        raw = h["obsm/spatial"][:].astype(float)
    inw = (np.abs(raw[:, 0] - c[0]) < W) & (np.abs(raw[:, 1] - c[1]) < WY)
    cap = _glob.glob(
        str(RUN_DIRS[brain] / f"output-XETG00151__{parts[1]}__{parts[2]}__*")
    )[0]
    bd = pd.read_parquet(
        cap + "/cell_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"]
    )
    bd["cell_id"] = bd["cell_id"].astype(str)
    bd = bd[bd.cell_id.isin(set(cid[inw]))]
    polys = {
        k: g[["vertex_x", "vertex_y"]].values
        for k, g in bd.groupby("cell_id", sort=False)
    }
    m = d["sl"] == sname
    ids_kc = d["cell_id"][m]
    role_kc = np.where(d["ic"][m], "core", np.where(d["nc"][m], "large", "classII"))
    ser = pd.Series(role_kc, index=ids_kc)
    ser = ser[~ser.index.duplicated()]
    role = pd.Series("other", index=cid[inw])
    common = role.index.intersection(ser.index)
    role.loc[common] = ser.loc[common]
    # keep this lobe only: a neighbouring lobe's core / Large-type cells in the
    # taller window are drawn as plain tissue
    d_c = pd.Series(np.hypot(raw[inw, 0] - c[0], raw[inw, 1] - c[1]), index=cid[inw])
    role[(role == "core") & (d_c >= 260)] = "other"
    role[(role == "large") & (d_c >= 340)] = "other"
    sel = m & (np.abs(d["xy"][:, 0] - c[0]) < W) & (np.abs(d["xy"][:, 1] - c[1]) < WY)
    Pn = d["xy"][sel & d["nc"]]
    gx = np.arange(c[0] - W, c[0] + W, 3.0)
    gy = np.arange(c[1] - WY, c[1] + WY, 3.0)
    GX, GY = np.meshgrid(gx, gy)
    G_ = np.column_stack([GX.ravel(), GY.ravel()])
    dmin = cKDTree(Pn).query(G_)[0].reshape(GX.shape)
    d2 = (G_[:, None, 0] - Pn[None, :, 0]) ** 2 + (G_[:, None, 1] - Pn[None, :, 1]) ** 2
    w = np.exp(-d2 / (2 * 16.0**2))
    w = w / (w.sum(1, keepdims=True) + 1e-9)
    field = {}
    for g in GENE_ID:
        v = (w @ d["det"][g][sel & d["nc"]].astype(float) * 100).reshape(GX.shape)
        v[(dmin > 12) | (np.hypot(GX - c[0], GY - c[1]) > 340)] = np.nan
        field[g] = v
    core = [k for k in role.index if role[k] == "core" and k in polys]
    other = [k for k in role.index if role[k] in ("other", "classII") and k in polys]
    large = [k for k in role.index if role[k] == "large" and k in polys]
    return dict(
        c=c,
        W=W,
        WY=WY,
        gx=gx,
        gy=gy,
        field=field,
        polys=polys,
        role=role,
        core=core,
        other=other,
        large=large,
    )


# ── Panel E, lower half: what the subclusters are ─────────────────────────
SHORT_GENE = {
    "octopamine receptor beta-1R": "Octβ1R",
    "octopamine receptor beta-2R": "Octβ2R",
    "furin-like protease 1": "Furin-like 1",
    "acetylcholinesterase": "AChE",
    "short neuropeptide F": "sNPF",
}


@functools.lru_cache(maxsize=1)
def _sn_table():
    """% of Large-type nuclei detecting each Xenium-panel gene, by subcluster,
    and the per-sample sNPF detection by subcluster (3 queen, 3 worker)."""
    sn = sc.read_h5ad(DATA / "CB_LargeType_subclustered.h5ad")
    panel = sc.read_h5ad(LTKC_H5AD, backed="r").var_names
    names = snuc_kc._load_gene_names()
    low = {g.lower(): g for g in sn.var_names}
    ids = [low[g.lower()] for g in panel if g.lower() in low]
    C = sn.layers["counts"]
    C = C.tocsc() if hasattr(C, "tocsc") else C
    col_of = {g: i for i, g in enumerate(sn.var_names)}
    sub = sn.obs["leiden_res0_05"].astype(str).values
    rows = []
    for g in ids:
        v = np.asarray(C[:, col_of[g]].todense()).ravel() > 0
        p = [100 * v[sub == k].mean() for k in SUB_CLUSTERS]
        nm = names.get(g, "") or g[-6:]
        nm = PANEL_SYMBOL.get(g.lower(), SHORT_GENE.get(nm, nm))
        rows.append((g, nm, 100 * v.mean(), *p))
    T = pd.DataFrame(rows, columns=["gene", "name", "pct_all", "S1", "S2", "S3"])
    T["spread"] = T[["S1", "S2", "S3"]].max(1) - T[["S1", "S2", "S3"]].min(1)
    pers = {}
    for gname in ("sNPF", "AChE"):
        gid = low[GENE_ID[gname]]
        v = np.asarray(C[:, col_of[gid]].todense()).ravel() > 0
        df = pd.DataFrame({"det": v, "sub": sub, "sample": sn.obs["sample"].values})
        per = df.groupby(["sample", "sub"]).det.mean().unstack() * 100
        per["all"] = df.groupby("sample").det.mean() * 100
        per["caste"] = ["queen" if "Queen" in i else "worker" for i in per.index]
        pers[gname] = per
    return T, pers


def _record_sn_caste_stats(gene, per):
    """Per-subcluster caste means, Welch / paired t-tests (3 queen vs 3 worker
    samples, paired by trial) and trial counts for `gene` into SUBZONE_STATS
    (-> subzone_stats.csv).  Called from
    panel_receptors, which draws on every run, so these rows always reach the
    CSV (panel_caste_dots is not called)."""
    from scipy.stats import ttest_ind, ttest_rel

    key = gene.lower()

    def by_trial(caste, col):
        return per[per.caste == caste].sort_index(key=lambda i: i.str[-2:])[col].values

    for k in SUB_CLUSTERS:
        q, w = by_trial("queen", k), by_trial("worker", k)
        SUBZONE_STATS[f"sn_{key}_{SUB_SHORT[k]}_queen"] = float(q.mean())
        SUBZONE_STATS[f"sn_{key}_{SUB_SHORT[k]}_worker"] = float(w.mean())
        SUBZONE_STATS[f"sn_{key}_{SUB_SHORT[k]}_welch_p"] = float(
            ttest_ind(q, w, equal_var=False).pvalue
        )
        SUBZONE_STATS[f"sn_{key}_{SUB_SHORT[k]}_paired_p"] = float(
            ttest_rel(q, w).pvalue
        )
        SUBZONE_STATS[f"sn_{key}_{SUB_SHORT[k]}_trials_queen_higher"] = int(
            (q > w).sum()
        )
    SUBZONE_STATS[f"sn_{key}_all_queen"] = float(by_trial("queen", "all").mean())
    SUBZONE_STATS[f"sn_{key}_all_worker"] = float(by_trial("worker", "all").mean())


def snuc_kc_gene(name):
    return GENE_ID[name].replace("bimp", "BIMP")


# _simplex_rgb: the soft probability-blend colouring panels F/G use.
def _simplex_rgb(p):
    """Per-cell colour: white where the three subcluster probabilities are
    equal, the pure subcluster colour at each vertex, blended in between.

    This is the one mapping that renders an unresolved projection as
    unresolved.  A plain probability-weighted blend would paint the
    equal-probability centre a dark mixture, which reads as a confident third
    identity; here a cell the mapping cannot place stays white.  Saturation is
    the total probability excess over uniform (max 2/3, hence the 1.5 scale),
    so the mapping is exact at the vertices and at the centre.
    """
    e = np.clip(p - 1.0 / 3, 0.0, None)
    s = np.clip(1.5 * e.sum(axis=1), 0.0, 1.0)[:, None]
    w = e / e.sum(axis=1, keepdims=True).clip(min=1e-12)
    return np.clip((1 - s) * 1.0 + s * (w @ SUB_RGB), 0, 1)


# ── Compartment diversity helpers (not drawn in the current layout) ───────
def _shannon(counts):
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _hill1(counts):
    """Hill number of order q=1: exp(Shannon entropy), the "effective number
    of equally-common categories."  A strictly increasing (monotonic)
    transform of _shannon(), so percentile bootstrap intervals commute with
    it -- exp() of the Shannon CI bounds IS the Hill-1 CI, to the same
    coverage, without redrawing anything.  See _diversity_series()'s
    docstring for why diversity is reported this way."""
    return float(np.exp(_shannon(counts)))


SN_REF_H5AD = REPO / "data" / "sn" / "BIMP_atlas_qw_CentralBrain.h5ad"
KC_MARKER_CSV = DATA / "kc_type_marker_genes.csv"
KC_MARKER_EXPR_CSV = DATA / "kc_type_marker_spatial_expr.csv"
N_MARKERS_PER_TYPE = 2


@functools.cache
def _kc_type_markers(n_per_type=N_MARKERS_PER_TYPE):
    """Panel D's marker genes: `n_per_type` per snRNA-defined KC population
    (TYPE_ORDER, panel A's five populations), restricted to genes the Xenium
    101-gene panel carries, so every row is a gene a reader can look up, not
    an unsupervised cluster id or a raw per-cell Tangram call.

    Independent of panel B, which rests on a per-cell Tangram type call
    (restricted, but still an argmax over a probability vector).  This panel
    ranks markers in the whole-transcriptome snRNA reference and then reads
    each gene's measured spatial expression directly, with no per-cell type
    call anywhere -- two lines of evidence for the same claim (each KC
    population's molecular signature is spatially recoverable).

    Ranking: Wilcoxon rank_genes_groups on the snRNA reference, restricted
    to the 5 real KC populations (the nuclei of panel A). A gene
    that ranks well for more than one type (this panel is only 100 genes
    wide, so a handful of strong markers double up) is assigned to its
    single best-ranked type only, round-robin across types by rank, so no
    gene appears twice under two different type colours.
    """
    if KC_MARKER_CSV.exists():
        print(f"  loading cached {KC_MARKER_CSV.name} ...", flush=True)
        mk = pd.read_csv(KC_MARKER_CSV)
    else:
        print("  computing KC-type marker genes (snRNA reference) ...", flush=True)
        ref = sc.read_h5ad(SN_REF_H5AD)
        ref = ref[ref.obs["cell_type"].astype(str).isin(TYPE_ORDER)].copy()
        sc.tl.rank_genes_groups(
            ref, "cell_type", groups=TYPE_ORDER, method="wilcoxon", n_genes=200
        )
        rg = ref.uns["rank_genes_groups"]

        # Spatial panel coverage: same v4.3 gene ids as the snRNA reference,
        # only the case differs (spatial var_names are lower-cased --
        # "bimp_v4.3_..." vs the reference's "BIMP_v4.3_...").
        sp = sc.read_h5ad(CTX_H5AD, backed="r")
        panel_genes = {g.lower() for g in sp.var_names}

        rows = []
        for t in TYPE_ORDER:
            for rank, (g, lfc, padj) in enumerate(
                zip(
                    rg["names"][t],
                    rg["logfoldchanges"][t],
                    rg["pvals_adj"][t],
                    strict=False,
                )
            ):
                rows.append(
                    dict(
                        cell_type=t,
                        rank=rank + 1,
                        gene=g,
                        log2fc=float(lfc),
                        padj=float(padj),
                        on_panel=g.lower() in panel_genes,
                    )
                )
        mk = pd.DataFrame(rows)
        mk.to_csv(KC_MARKER_CSV, index=False)

    gene_names = snuc_kc._load_gene_names()
    taken, chosen = set(), []
    for _ in range(n_per_type):
        for t in TYPE_ORDER:
            cand = mk[
                (mk.cell_type == t)
                & mk.on_panel
                & (mk.padj < 0.05)
                & (~mk.gene.isin(taken))
            ].sort_values("rank")
            if len(cand):
                g = cand.iloc[0]
                taken.add(g["gene"])
                chosen.append(
                    dict(
                        cell_type=t,
                        gene=g["gene"],
                        gene_name=gene_names.get(g["gene"], g["gene"]),
                        rank=int(g["rank"]),
                        log2fc=g["log2fc"],
                        padj=g["padj"],
                    )
                )
    print(
        f"  panel D marker selection ({n_per_type}/type, best-rank deduplicated):",
        flush=True,
    )
    for c in chosen:
        print(
            f"    {TYPE_SHORT[c['cell_type']]:12s} {c['gene_name']:10s} "
            f"({c['gene']}) rank={c['rank']:>3d} log2fc={c['log2fc']:+.2f} "
            f"padj={c['padj']:.1e}",
            flush=True,
        )
    return pd.DataFrame(chosen)


@functools.cache
def _kc_marker_expr():
    """Mean spatial expression of `_kc_type_markers()`'s genes, per
    compartment x caste -- the six columns every other panel in this figure
    uses.  Reads raw expression from CTX_H5AD; the tangram_cell_type column
    correction affected only the fine cell-type call columns, not the
    expression matrix.
    """
    markers = _kc_type_markers()
    if KC_MARKER_EXPR_CSV.exists():
        cached = pd.read_csv(KC_MARKER_EXPR_CSV)
        if set(cached["gene"]) == set(markers["gene"]):
            print(f"  loading cached {KC_MARKER_EXPR_CSV.name} ...", flush=True)
            return cached
        print(
            f"  {KC_MARKER_EXPR_CSV.name} is stale (gene set changed) -- recomputing",
            flush=True,
        )
    print("  loading spatial expression for marker genes ...", flush=True)
    sp = sc.read_h5ad(CTX_H5AD)
    var_lower = {v.lower(): v for v in sp.var_names}
    comp = np.where(
        sp.obs["in_kc_ic"].values.astype(bool),
        "KC-IC",
        np.where(
            sp.obs["in_kc_nc"].values.astype(bool),
            "KC-NC",
            np.where(sp.obs["in_kc_oc"].values.astype(bool), "KC-OC", ""),
        ),
    )
    brain = sp.obs["brain"].astype(str).values
    rows = []
    for _, m in markers.iterrows():
        col = var_lower.get(m["gene"].lower())
        x = np.asarray(sp[:, col].X).ravel() if col else None
        for c in COMP_ORDER:
            for b in ("queen", "worker"):
                sel = (comp == c) & (brain == b)
                rows.append(
                    dict(
                        gene=m["gene"],
                        cell_type=m["cell_type"],
                        comp=c,
                        brain=b,
                        mean_expr=float(x[sel].mean()) if x is not None else np.nan,
                        n=int(sel.sum()),
                    )
                )
    ex = pd.DataFrame(rows)
    ex.to_csv(KC_MARKER_EXPR_CSV, index=False)
    return ex


@functools.cache
def _raw_gene_descriptions():
    df = pd.read_csv(snuc_kc.GENE_LOOKUP_TSV, sep="\t")
    return dict(zip(df["gene_id"], df["description"], strict=True))


def _marker_label(gene_id, gene_names):
    """Row label for one marker gene: the real symbol if there is one (no
    spaces -- e.g. "CaMKII", "Nmdar2"), else the trailing two words of the
    RAW description, not `gene_names`' pre-shortened one.  `_load_gene_names`
    (snrna_kc_subclusters.py) truncates long descriptions from the front,
    which is right for its own use (marker dot-plot columns, order doesn't
    matter) but wrong here: two of this panel's ten markers are octopamine
    receptor subtypes ("octopamine receptor beta-1R" / "...beta-2R") whose
    only distinguishing text is the trailing "beta-1R"/"beta-2R" -- the
    front-truncated form collapses both to the identical, ambiguous
    "octopamine receptor...".
    """
    if gene_id.lower() in PANEL_SYMBOL:
        return _wrap_label(PANEL_SYMBOL[gene_id.lower()])
    sym = gene_names.get(gene_id, gene_id)
    if " " not in sym:
        return sym
    desc = _raw_gene_descriptions().get(gene_id, sym)
    words = desc.rstrip(".").split()
    label = " ".join(words[-2:]) if len(words) >= 2 else desc
    # Long two-word labels split onto two lines; a length threshold rather
    # than a named label, since any marker can land in this case.
    return _wrap_label(label)
