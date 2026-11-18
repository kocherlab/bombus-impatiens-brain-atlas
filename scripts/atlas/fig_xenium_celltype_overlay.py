#!/usr/bin/env python3
"""Xenium morphology imaging with cell assignments.

Writes the main figure ``fig:xenium_celltype_overlay``
(results/figures/main/xenium_celltype_overlay/) and the Supplementary figure
``fig:xenium_kc_populations`` (results/figures/si/figS_xenium_kc_populations.png).

For the two sections shown in fig:atlas_overview panel D (one queen, one
worker; mid-brain, most
cells), shows per caste

  A  DAPI (nuclear) channel of the whole section, with a boxed Kenyon-cell field
  B  the same section, each Xenium cell boundary filled by its assigned
     cell-type label (the fig:atlas_overview panel D palette and labels)
  C  the boxed field at full resolution, DAPI
  D  the boxed field, DAPI with cell-type-coloured boundaries on top
  E  the four morphology-focus channels of the boxed field

Data (all raw, per crop; nothing is read from the PASTE3-aligned atlas frame):
  * raw per-crop h5ad  pipelines/*/data/output-XETG00151__*.h5ad
      obsm['spatial'] (section um), uns['image'] (morphology_focus, 0.2125 um/px),
      uns['crop_coords']['min'] (pixel origin of the stored image).
  * cell polygons
      data/<run>/output-XETG00151__<slide>__<Region>__*/cell_boundaries.parquet
      (Xenium segmentation, section um, one polygon per cell).
  * labels: hierarchical_vote on bimp_st.paths.CANONICAL_TANGRAM (same call as
    fig_atlas_overview / caste_comparison_tables), joined to the crop by
    (slice_name, cell_id), never positionally.

Run:
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/atlas/fig_xenium_celltype_overlay.py
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import glob
import os
import warnings
from pathlib import Path

import anndata as ad
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata.io import read_elem
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from bimp_st import nine_region_common as NR
from bimp_st.cluster_labels import (
    DISPLAY_COLOR_OVERRIDES,
    DISPLAY_NAME_OVERRIDES,
    get_pooled_color,
    get_pooled_display,
    hierarchical_vote,
)
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED
from bimp_st.si_common import PIXEL_SCALE
from bimp_st.site import get as site_path
from bimp_st.style import add_panel_headers, set_pub_style

warnings.filterwarnings("ignore")

# The checkout holding the raw per-crop h5ads and Xenium outputs (gitignored).
MAIN = site_path("data_repo")
from bimp_st.paths import REPO  # noqa: E402

OUT = REPO / "results/figures/si"
OUT_MAIN = REPO / "results/figures/main/xenium_celltype_overlay"
NAME = "fig_xenium_celltype_overlay"
NAME_CH = "fig_s_xenium_channels"  # not a registered figure
NAME_KC = "figS_xenium_kc_populations"

ZOOM_UM = 250.0  # side of the boxed Kenyon-cell field
DS = 4  # block-mean downsample for the whole-section images
TICK = 6.0  # pt: all text in the figure
CHANNELS = ["DAPI", "ATP1A1/\nCD45/E-cad", "18S", "αSMA/\nVimentin"]
# additive composite: channel index -> display colour (by colour and by the
# OME-TIFF metadata name; identities not yet confirmed against the panel docs)
COMP = [
    (0, (0.0, 0.6, 1.0), "DAPI"),
    (1, (0.0, 1.0, 0.1), "ATP1A1/CD45/E-cad"),
    (2, (1.0, 0.0, 0.8), "18S"),
]
# Zoom windows (lower-left corner, section um) at the homologous calyx location in
# both castes. Chosen by scanning 250 um windows on the raw per-crop h5ad and
# matching, between castes, the fractions of Kenyon-flagged cells in the three
# compartments (in_kc_ic / in_kc_nc / in_kc_oc), the fraction of Kenyon-voted
# cells, the optic-lobe fraction and the position within the Kenyon extent, then
# confirming the best pairs by eye. Queen: IC/NC/OC 0.50/0.29/0.21, 84% Kenyon;
# worker: 0.51/0.27/0.22, 84% Kenyon; neither window touches the optic lobe.
ZOOM_ORIGIN = {"queen": (1124.0, 2568.0), "worker": (717.0, 6309.0)}
LO_P, HI_P, GAMMA = 1.0, 99.5, 1.0


# composite: per-channel (low, high) from tissue pixels, then gamma and gain,
# combined by screen blending
PCT = {0: (1.0, 99.5), 1: (5.0, 99.8), 2: (1.0, 99.9)}
GAINS = {0: (0.8, 1.1), 1: (1.1, 0.8), 2: (1.0, 0.6)}  # ch: (gamma, gain)


def tissue_mask(chs):
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    tot = gaussian_filter(sum(np.asarray(chs[c], np.float32) for c in (0, 1, 2)), 2)
    v = np.log1p(tot)
    return v > threshold_otsu(v)


def tissue_limits(list_of_chs):
    out = {}
    masks = [tissue_mask(c) for c in list_of_chs]
    for ch in (0, 1, 2):
        v = np.concatenate([c[ch][m] for c, m in zip(list_of_chs, masks, strict=False)])
        out[ch] = tuple(np.percentile(v, PCT[ch]))
    return out


def stretch(a, lim, ch):
    lo, hi = lim
    g, gain = GAINS[ch]
    return np.clip(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) ** g * gain, 0, 1)


REGIONS = [
    "lamina",
    "medulla",
    "lobula",
    "kc_oc",
    "kc_ic",
    "kc_nc",
    "al",
    "central_complex",
]
FULL_REGIONS = ["lamina", "medulla", "lobula", "kc_oc", "kc_ic", "kc_nc", "al"]


def load_region_polys(sec, brain):
    """Hand-drawn region polygons of this crop, in section um (the frame of
    obsm['spatial']), straight from the annotation GeoJSON that produced the
    in_* flags (nine_region_common.load_regions, as relabel_nine_regions does).
    One GeoJSON covers a whole capture region (2-4 crops in overlapping local
    coordinates), so only polygon parts that contain >= 10 cells of this crop
    are kept. Validates against the stored in_<region> flags and prints the
    agreement per region. Untraced regions simply have no polygon."""
    from shapely import contains_xy

    slide, section = NR.parse_slide_section(sec["slice_name"])
    src = NR.annotation_sources(brain)[(slide, section)]
    path = src.get("v2") or src.get("old")
    regs = NR.load_regions(path)
    side = pd.read_parquet(
        MAIN
        / "results"
        / "nine_region"
        / "crops"
        / brain
        / f"{sec['slice_name']}.parquet"
    ).set_index("cell_id")
    with h5py.File(sec["h5"], "r") as h:
        cid = np.asarray(read_elem(h["obs"]["cell_id"])).astype(str)
        xy = h["obsm/spatial"][:].astype(float)
    out, agree = {}, {}
    for reg in REGIONS:
        geom = regs.get(reg)
        if geom is None:
            continue
        # one outline per region: union of its polygons (abutting/overlapping
        # hand-drawn pieces lose their shared edge); close <=0.8 um slits and
        # drop interior rings smaller than 100 um^2 (slivers between pieces)
        from shapely.geometry import Polygon as _P

        geom = geom.buffer(0.8).buffer(-0.8)
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        parts = [
            _P(g.exterior, [i for i in g.interiors if _P(i).area >= 100.0])
            for g in parts
        ]
        keep = [g for g in parts if contains_xy(g, xy[:, 0], xy[:, 1]).sum() >= 10]
        if not keep:
            continue
        out[reg] = keep
        inside = np.zeros(len(xy), bool)
        for g in keep:
            inside |= contains_xy(g, xy[:, 0], xy[:, 1])
        flag = side[f"in_{reg}"].reindex(cid).fillna(False).astype(bool).to_numpy()
        agree[reg] = float((inside == flag).mean())
    print(
        f"  {brain} {sec['slice_name']}: {path.parent.name}/{path.name}\n"
        "    polygon-vs-flag agreement: "
        + ", ".join(f"{k} {100 * v:.2f}%" for k, v in agree.items()),
        flush=True,
    )
    return out


def draw_polys(ax, polys, regions, lw=1.0, under=0.6):
    """Exteriors and interiors exactly as traced, region colour over a thin
    dark underlay. No text."""
    import matplotlib.patheffects as pe

    for reg in regions:
        for g in polys.get(reg, []):
            for ring in [g.exterior, *g.interiors]:
                x, y = ring.xy
                ax.plot(
                    x,
                    y,
                    color=NR.REGION_COLORS[reg],
                    lw=lw,
                    solid_joinstyle="round",
                    clip_on=True,
                    zorder=6,
                    path_effects=[
                        pe.withStroke(
                            linewidth=lw * 2.0, foreground="black", alpha=under
                        )
                    ],
                )


def composite(chs, lims):
    """Screen blend of the three stretched channels; lims[ch] = (lo, hi),
    lims.get('x') = optional per-channel extra gain (worker green balance)."""
    prod = 1.0
    extra = lims.get("x", {})
    for ch, col, _ in COMP:
        a = stretch(np.asarray(chs[ch], np.float32), lims[ch], ch)
        a = np.clip(a * extra.get(ch, 1.0), 0, 1)
        prod = prod * (1 - a[..., None] * np.array(col)[None, None, :])
    return np.clip(1 - prod, 0, 1)


CROP_DIRS = {
    "queen": MAIN / "pipelines/wholebrain/queen/data",
    "worker": MAIN / "pipelines/wholebrain/worker/data",
}
RUN_DIRS = {
    "queen": MAIN / "data/20251204__194847___20251204",
    "worker": MAIN / "data/20260115__193720__20260115",
}


# ── labels ───────────────────────────────────────────────────────────────────
def ct_color(name: str) -> str:
    return DISPLAY_COLOR_OVERRIDES.get(name, get_pooled_color(name))


def ct_label(name: str) -> str:
    if name == "Mixed":
        return "Other"
    return DISPLAY_NAME_OVERRIDES.get(name, get_pooled_display(name))


def load_votes() -> pd.DataFrame:
    """(slice_name, cell_id, brain, vote) for all atlas cells."""
    with h5py.File(SPATIAL_COMBINED, "r") as h:
        o = h["obs"]
        sp = pd.DataFrame(
            {
                k: np.asarray(read_elem(o[k])).astype(str)
                for k in ("slice_name", "cell_id", "brain")
            }
        )
    with h5py.File(CANONICAL_TANGRAM, "r") as h:
        o = h["obs"]
        tg = pd.DataFrame(
            {
                k: np.asarray(read_elem(o[k])).astype(str)
                for k in (
                    "leiden",
                    "tangram_cell_type",
                    "brain",
                    "slice_name",
                    "cell_id",
                )
            }
        )
    light = tg.copy()
    light.index = pd.Index(np.arange(len(light)).astype(str))
    a = ad.AnnData(obs=light)
    hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
    )
    # The vote is computed on the Tangram object; join it back on the cell key
    # (slice_name, cell_id), never by position.
    tg_key = pd.Index(tg["slice_name"] + "||" + tg["cell_id"])
    sp_key = pd.Index(sp["slice_name"] + "||" + sp["cell_id"])
    assert not tg_key.has_duplicates and not sp_key.has_duplicates
    pos = tg_key.get_indexer(sp_key)
    assert (pos >= 0).all(), f"{int((pos < 0).sum())} atlas cells not in Tangram"
    assert (sp["brain"].values == tg["brain"].values[pos]).all()
    sp["vote"] = a.obs["cluster_vote"].astype(str).values[pos]
    return sp


def pick_sections(sp: pd.DataFrame) -> dict[str, str]:
    """fig_atlas_overview._select_two_slices, reproduced on obs only (z = rank)."""
    us = np.unique(sp["slice_name"].values)
    brain_of = sp.drop_duplicates("slice_name").set_index("slice_name")["brain"]
    n = sp["slice_name"].value_counts()
    picks = {}
    for b in ("queen", "worker"):
        bs = [s for s in us if brain_of[s] == b]
        mid = bs[len(bs) // 3 : 2 * len(bs) // 3] or bs
        picks[b] = max(mid, key=lambda s: n[s])
    return picks


# ── raw crop ─────────────────────────────────────────────────────────────────
def crop_paths(brain: str, slice_name: str):
    parts = slice_name.split("__")
    slide, region = parts[1], parts[2]
    h5 = CROP_DIRS[brain] / f"{slice_name}.h5ad"
    cap = glob.glob(str(RUN_DIRS[brain] / f"output-XETG00151__{slide}__{region}__*"))
    assert len(cap) == 1, cap
    return h5, Path(cap[0]) / "cell_boundaries.parquet"


def block_mean(img_ds, ds: int) -> np.ndarray:
    """Block-mean downsample of a (H, W) h5py dataset in row chunks."""
    H, W = img_ds.shape
    H2, W2 = H // ds, W // ds
    out = np.empty((H2, W2), np.float32)
    step = 1024 - 1024 % ds
    for r in range(0, H2 * ds, step):
        r1 = min(r + step, H2 * ds)
        blk = img_ds[r:r1, : W2 * ds].astype(np.float32)
        out[r // ds : r1 // ds] = blk.reshape((r1 - r) // ds, ds, W2, ds).mean(
            axis=(1, 3)
        )
    return out


def load_section(brain: str, slice_name: str, votes: pd.DataFrame) -> dict:
    h5, bpq = crop_paths(brain, slice_name)
    with h5py.File(h5, "r") as h:
        cell_id = np.asarray(read_elem(h["obs"]["cell_id"])).astype(str)
        xy = h["obsm/spatial"][:].astype(float)
        cmin = h["uns/crop_coords/min"][:].astype(float)
        dset = h["uns/image"]
        nch = dset.shape[0]
        dapi_ds = block_mean(dset[0], DS)
        ds_ch = {0: dapi_ds, 1: block_mean(dset[1], DS), 2: block_mean(dset[2], DS)}
        shape = dset.shape
    # labels: join on (slice_name, cell_id)
    v = votes[votes["slice_name"] == slice_name]
    assert v["cell_id"].is_unique
    lab = pd.Series(v["vote"].values, index=v["cell_id"].values)
    lbl = lab.reindex(cell_id)
    n_missing = int(lbl.isna().sum())
    # polygons, section um
    b = pd.read_parquet(bpq, columns=["cell_id", "vertex_x", "vertex_y"])
    b["cell_id"] = b["cell_id"].astype(str)
    b = b[b["cell_id"].isin(set(cell_id))]
    ids, first = np.unique(b["cell_id"].values, return_index=True)
    # polygon vertices are contiguous per cell; split in file order
    order = np.argsort(first)
    ids_o = ids[order]
    bounds = np.append(np.sort(first), len(b))
    vx, vy = b["vertex_x"].values, b["vertex_y"].values
    polys = [
        np.column_stack([vx[s:e], vy[s:e]])
        for s, e in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    pl = pd.Series(range(len(polys)), index=ids_o)
    keep = [
        i for i, c in enumerate(cell_id) if c in pl.index and not pd.isna(lbl.iloc[i])
    ]
    cells_poly = [polys[pl[cell_id[i]]] for i in keep]
    cells_lab = lbl.iloc[keep].values.astype(str)
    cells_id = cell_id[keep]
    cells_xy = xy[keep]
    print(
        f"  {brain} {slice_name}: raw {len(cell_id):,} cells, "
        f"{n_missing:,} not in atlas, {len(keep):,} with polygon+label; "
        f"image {shape}",
        flush=True,
    )
    return dict(
        brain=brain,
        slice_name=slice_name,
        h5=h5,
        cmin=cmin,
        xy_all=xy,
        nch=nch,
        dapi_ds=dapi_ds,
        ds_ch=ds_ch,
        polys=cells_poly,
        labs=cells_lab,
        ids=cells_id,
        xy=cells_xy,
        n_raw=len(cell_id),
        n_missing=n_missing,
        shape=shape,
    )


def img_extent_um(cmin, shape_hw, ds=1):
    x0, y0 = cmin * PIXEL_SCALE
    H, W = shape_hw
    return [x0, x0 + W * ds * PIXEL_SCALE, y0 + H * ds * PIXEL_SCALE, y0]


def pick_zoom(sec: dict, size: float) -> tuple[float, float]:
    """Lower-left corner of the size x size um window densest in Kenyon cells
    (but with some non-Kenyon neighbours, so the labels differ in the field)."""
    xy, lab = sec["xy"], sec["labs"]
    kc = lab == "Kenyon cells"
    step = 25.0
    xs = np.arange(xy[:, 0].min(), xy[:, 0].max() - size, step)
    ys = np.arange(xy[:, 1].min(), xy[:, 1].max() - size, step)
    best, bxy = -1, (xs[0], ys[0])
    for x in xs:
        mx = (xy[:, 0] >= x) & (xy[:, 0] < x + size)
        for y in ys:
            m = mx & (xy[:, 1] >= y) & (xy[:, 1] < y + size)
            n = m.sum()
            if n < 600:
                continue
            f = kc[m].mean()
            if not (0.45 <= f <= 0.85):
                continue
            sc = n * (1 - abs(f - 0.65))
            if sc > best:
                best, bxy = sc, (x, y)
    print(f"  zoom window {sec['brain']}: {bxy}, score {best:.0f}", flush=True)
    return bxy


def read_window(sec: dict, x, y, size, ch: int) -> tuple[np.ndarray, list]:
    px0 = int(round(x / PIXEL_SCALE - sec["cmin"][0]))
    py0 = int(round(y / PIXEL_SCALE - sec["cmin"][1]))
    n = int(round(size / PIXEL_SCALE))
    with h5py.File(sec["h5"], "r") as h:
        a = h["uns/image"][ch, py0 : py0 + n, px0 : px0 + n].astype(np.float32)
    ext = [
        (px0 + sec["cmin"][0]) * PIXEL_SCALE,
        (px0 + a.shape[1] + sec["cmin"][0]) * PIXEL_SCALE,
        (py0 + a.shape[0] + sec["cmin"][1]) * PIXEL_SCALE,
        (py0 + sec["cmin"][1]) * PIXEL_SCALE,
    ]
    return a, ext


def norm(a, lo=1.0, hi=99.7, gamma=0.8):
    low, h = np.percentile(a[a > 0], [lo, hi]) if (a > 0).any() else (0, 1)
    return np.clip((a - low) / max(h - low, 1e-6), 0, 1) ** gamma


def style_img_ax(ax):
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


# ── shared drawing helpers ───────────────────────────────────────────────────
FRAME_LW = 2.2  # pt: zoom box and zoom frames, identical
ZOOM_COLOR = "#FFFFFF"  # zoom box / ring / connector: white band ...
ZOOM_EDGE = "#111111"  # ... with a 0.6 pt near-black line on both edges
EDGE_LW = 0.6
PILL = dict(
    boxstyle="round,pad=0.25,rounding_size=0.5", fc="black", ec="none", alpha=0.6
)
FILL_ALPHA = 0.75


def _stroke(lw, alpha=0.7):
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=lw, foreground="black", alpha=alpha)]


def scalebar(ax, length_um, label, below=False, left=False):
    """White bar with dark underlay, bottom-right of every panel. below=True
    puts the label under the bar and the bar higher (clears tissue outlines)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()  # y inverted: y0 > y1
    xr = (x0 + length_um + 0.045 * (x1 - x0)) if left else x1 - 0.045 * (x1 - x0)
    yb = y0 - (0.075 if below else 0.05) * (y0 - y1)
    ax.plot(
        [xr - length_um, xr],
        [yb, yb],
        color="white",
        lw=1.5,
        solid_capstyle="butt",
        clip_on=False,
        zorder=9,
        path_effects=_stroke(3.2),
    )
    if below:
        ax.text(
            xr - length_um / 2,
            yb + 0.012 * (y0 - y1),
            label,
            color="white",
            fontsize=TICK,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=9,
            path_effects=_stroke(2.2, 0.85),
        )
    else:
        ax.text(
            xr - length_um / 2,
            yb - 0.018 * (y0 - y1),
            label,
            color="white",
            fontsize=TICK,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=9,
            path_effects=_stroke(2.2, 0.85),
        )


def _segs_cross(p1, p2, p3, p4):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return (o(p1, p2, p3) * o(p1, p2, p4) < 0) and (o(p3, p4, p1) * o(p3, p4, p2) < 0)


def open_spot_labels(ax, items, pts_frac, text_w, leader=None, dot=None):
    """Direct labels with short leaders. items: [(text, colour, (fx, fy) anchor
    in top-down axes fractions)]. Each label is put at the free spot (clear of
    all cell points, previous labels and the frame) nearest its anchor."""
    import matplotlib.patheffects as pe

    placed, leaders = [], []

    def seg_pts(ax0, ay0, bx, by):
        t = np.linspace(0, 1, 40)
        return np.column_stack([ax0 + (bx - ax0) * t, ay0 + (by - ay0) * t])

    for text, col, (ax_, ay_) in items:
        w = text_w(text)
        best, bc = None, 1e9
        for clr in (0.03, 0.015, 0.0):
            for x0 in np.arange(0.04, 0.96 - w, 0.03):
                for y in np.arange(0.07, 0.93, 0.03):
                    x1 = x0 + w
                    dx = np.maximum(
                        np.maximum(x0 - pts_frac[:, 0], pts_frac[:, 0] - x1), 0
                    )
                    dy = np.maximum(np.abs(pts_frac[:, 1] - y) - 0.035, 0)
                    if np.min(np.hypot(dx, dy)) < clr:
                        continue
                    ex = max(x0 - ax_, 0, ax_ - x1)
                    ey = max(abs(y - ay_) - 0.03, 0)
                    if np.hypot(ex, ey) < 0.07:  # leader must be visible
                        continue
                    if any(
                        not (x1 < px0 - 0.02 or x0 > px1 + 0.02 or abs(y - py) > 0.08)
                        for px0, px1, py in placed
                    ):
                        continue
                    cx_ = 0.5 * (x0 + x1)
                    seg = seg_pts(cx_, y, ax_, ay_)
                    if any(
                        (
                            (seg[:, 0] > px0 - 0.01)
                            & (seg[:, 0] < px1 + 0.01)
                            & (np.abs(seg[:, 1] - py) < 0.045)
                        ).any()
                        for px0, px1, py in placed
                    ):
                        continue
                    if any(
                        (
                            (lp[:, 0] > x0 - 0.01)
                            & (lp[:, 0] < x1 + 0.01)
                            & (np.abs(lp[:, 1] - y) < 0.045)
                        ).any()
                        for lp in leaders
                    ):
                        continue
                    if any(
                        _segs_cross(seg[0], seg[-1], lp[0], lp[-1]) for lp in leaders
                    ):
                        continue
                    c = np.hypot(cx_ - ax_, y - ay_)
                    if c < bc:
                        best, bc = (x0, y), c
            if best is not None:
                break
        if best is None:
            continue
        x0, y = best
        placed.append((x0, x0 + w, y))
        leaders.append(seg_pts(x0 + w / 2, y, ax_, ay_))
        ax.annotate(
            text,
            xy=(ax_, 1 - ay_),
            xycoords="axes fraction",
            xytext=(x0 + w / 2, 1 - y),
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=TICK,
            fontweight="bold",
            color=col,
            zorder=9,
            annotation_clip=False,
            path_effects=[pe.withStroke(linewidth=2.8, foreground="black", alpha=0.85)],
            arrowprops=dict(
                arrowstyle="-",
                lw=0.7,
                color=leader or col,
                shrinkA=0,
                shrinkB=0,
                path_effects=[
                    pe.withStroke(linewidth=1.9, foreground="black", alpha=0.8)
                ],
            ),
        )
        ax.plot(
            [ax_],
            [1 - ay_],
            transform=ax.transAxes,
            marker="o",
            ms=3.4,
            mfc=dot or col,
            mec="black",
            mew=0.5,
            zorder=10,
            clip_on=False,
        )


def tag(ax, text):
    ax.text(
        0.03,
        0.975,
        text,
        transform=ax.transAxes,
        color="white",
        fontsize=TICK,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=PILL,
        zorder=9,
    )


def darker(c, f=0.5):
    r, g, b = mpl.colors.to_rgb(c)
    return (r * f, g * f, b * f, 1.0)


def cell_fill(polys, colors, lw, alpha=FILL_ALPHA, edge=True):
    return PolyCollection(
        polys,
        facecolors=[mpl.colors.to_rgba(c, alpha) for c in colors],
        edgecolors=[darker(c) for c in colors] if edge else "none",
        linewidths=lw,
        zorder=3,
        rasterized=True,
    )


def _fill_or_outline(ax, polys, cols, lw=0.3, outline="#DCDCDC"):
    """Fill cells with a colour; outline (no fill) those whose colour is None."""
    on = [i for i, c in enumerate(cols) if c is not None]
    off = [i for i, c in enumerate(cols) if c is None]
    if off:
        ax.add_collection(
            PolyCollection(
                [polys[i] for i in off],
                facecolors="none",
                edgecolors=mpl.colors.to_rgba(outline, 0.6),
                linewidths=lw,
                zorder=3,
                rasterized=True,
            )
        )
    if on:
        ax.add_collection(cell_fill([polys[i] for i in on], [cols[i] for i in on], lw))


# ── channel tiles (SI) ───────────────────────────────────────────────────────
CH_M, CH_BG, CH_G = 0.06, 0.18, 0.05  # strip margin, caste gap, tile gap


def channel_tile_in(W):
    """Side of one channel tile: eight tiles and a caste gap across ``W``."""
    return (W - 2 * CH_M - CH_BG - 6 * CH_G) / 8


def draw_channel_strip(fig, zooms, zchans, W, H, y_top, letters="AB"):
    """Four single-channel tiles per caste, ``y_top`` inches below the top."""
    M, BG, G = CH_M, CH_BG, CH_G
    tw = channel_tile_in(W)
    heads = []
    for ci, brain in enumerate(("queen", "worker")):
        x0 = M + ci * (4 * tw + 3 * G + BG)
        zx, zy = zooms[brain]
        for k in range(4):
            ax = fig.add_axes(
                [(x0 + k * (tw + G)) / W, 1 - (y_top + tw) / H, tw / W, tw / H]
            )
            style_img_ax(ax)
            a, zext = zchans[brain][k]
            ax.imshow(
                norm(a),
                cmap="gray",
                extent=zext,
                origin="upper",
                interpolation="antialiased",
                aspect="equal",
            )
            ax.set_xlim(zx, zx + ZOOM_UM)
            ax.set_ylim(zy + ZOOM_UM, zy)
            tag(ax, CHANNELS[k])
            if k == 0:
                heads.append((ax, letters[ci], f"{brain.capitalize()} channels"))
                scalebar(ax, 50, "50 µm")
    return heads


def channel_figure(zooms, zchans):
    """Standalone channel strip (not registered; the strip is part of NAME_KC)."""
    W = 7.2
    HDR = 0.2
    H = CH_M + HDR + channel_tile_in(W) + CH_M
    fig = plt.figure(figsize=(W, H))
    heads = draw_channel_strip(fig, zooms, zchans, W, H, CH_M + HDR)
    add_panel_headers(fig, heads, width_frac=1.0)
    fig.savefig(OUT / f"{NAME_CH}.png", dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{NAME_CH}.png")


# ── fine Kenyon cell assignment on the same fields (SI) ──────────────────────
def kc_population_figure(secs, zooms, zchans, lim_zoom, rpts):  # lim_zoom: per brain
    import sys

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str((here / "..").resolve()))
    import kenyon.fig_kc_subtypes as F5  # same palette, order and blend

    slices = {b: secs[b]["slice_name"] for b in secs}

    def load_obj(path, cols):
        a = ad.read_h5ad(path, backed="r")
        o = a.obs[["slice_name", "cell_id"]].astype(str)
        idx = np.flatnonzero(o["slice_name"].isin(set(slices.values())).values)
        pr = a.obsm["tangram_ct_pred"]
        pr = (
            pr
            if isinstance(pr, pd.DataFrame)
            else pd.DataFrame(np.asarray(pr), columns=cols)
        )
        pr = pr[cols].iloc[idx]
        pr = pr.div(pr.sum(axis=1), axis=0).fillna(1.0 / len(cols))
        key = o.iloc[idx]["slice_name"].values + "|" + o.iloc[idx]["cell_id"].values
        assert len(set(key)) == len(key), "duplicate (slice_name, cell_id)"
        return pd.DataFrame(pr.values, index=key, columns=cols)

    pop = load_obj(F5.KC5POP_H5AD, F5.TYPE_ORDER)
    sub = load_obj(F5.LTKC_H5AD, F5.SUB_CLUSTERS)

    W = 7.2
    M, G = 0.06, 0.06
    pw3 = (W - 2 * M - 2 * G) / 3
    HDR, LEG = 0.2, 0.46
    # Row of single-channel tiles (A, B) above the compartment/population grid
    # (C-E): the image channels of the same two fields.
    strip = HDR + channel_tile_in(W) + 0.10
    H = M + strip + HDR + pw3 + G + pw3 + 0.08 + LEG
    fig = plt.figure(figsize=(W, H))
    heads = draw_channel_strip(fig, zooms, zchans, W, H, M + HDR)
    for ri, brain in enumerate(("queen", "worker")):
        sec = secs[brain]
        zx, zy = zooms[brain]
        chans = [c[0] for c in zchans[brain]]
        zext = zchans[brain][0][1]
        rgb = composite(chans, lim_zoom[brain])
        sel = [
            i
            for i, pg in enumerate(sec["polys"])
            if zx - 15 <= pg[:, 0].mean() <= zx + ZOOM_UM + 15
            and zy - 15 <= pg[:, 1].mean() <= zy + ZOOM_UM + 15
        ]
        keys = [slices[brain] + "|" + sec["ids"][i] for i in sel]
        in_pop = np.array([k in pop.index for k in keys])
        in_sub = np.array([k in sub.index for k in keys])
        print(
            f"  KC populations {brain}: {len(sel)} cells in field, "
            f"{int(in_pop.sum())} with a population call "
            f"({100 * in_pop.mean():.1f}%), {int(in_sub.sum())} with a "
            f"subcluster vector",
            flush=True,
        )
        polys = [sec["polys"][i] for i in sel]
        with h5py.File(sec["h5"], "r") as h:
            o = h["obs"]
            cid_raw = np.asarray(read_elem(o["cell_id"])).astype(str)
            nc_flag = np.asarray(read_elem(o["in_kc_nc"])).astype(float) > 0
        nc_ids = set(cid_raw[nc_flag])
        # Cells without a call are outlined, not filled (as the caption says).
        # A #8C8C8C fill would read as the "Other Kenyon cells" population
        # colour (KC_POPULATION_COLORS).
        gray = None
        for ci in range(3):
            ax = fig.add_axes(
                [
                    (M + ci * (pw3 + G)) / W,
                    1 - (M + strip + HDR + ri * (pw3 + G) + pw3) / H,
                    pw3 / W,
                    pw3 / H,
                ]
            )
            style_img_ax(ax)
            ax.imshow(
                rgb,
                extent=zext,
                origin="upper",
                interpolation="antialiased",
                aspect="equal",
            )
            if ci == 0:
                draw_polys(ax, rpts[brain], ["kc_ic", "kc_nc", "kc_oc"], lw=1.0)
            elif ci == 1:
                cols = [
                    F5.TYPE_COLORS[F5.TYPE_ORDER[int(np.argmax(pop.loc[k].values))]]
                    if ok
                    else gray
                    for k, ok in zip(keys, in_pop, strict=False)
                ]
                _fill_or_outline(ax, polys, cols)
            else:
                cols = []
                for k, i in zip(keys, sel, strict=False):
                    if sec["ids"][i] in nc_ids and k in sub.index:
                        cols.append(
                            mpl.colors.to_hex(F5._simplex_rgb(sub.loc[[k]].values)[0])
                        )
                    else:
                        cols.append(gray)
                _fill_or_outline(ax, polys, cols)
            ax.set_xlim(zx, zx + ZOOM_UM)
            ax.set_ylim(zy + ZOOM_UM, zy)
            scalebar(ax, 50, "50 µm")
            if ci == 0:
                tag(ax, brain.capitalize())
            if ri == 0:
                heads.append(
                    (
                        ax,
                        "CDE"[ci],
                        ("Hand labels", "Kenyon populations", "Large-type subclusters")[
                            ci
                        ],
                    )
                )
    add_panel_headers(fig, heads, width_frac=1.0)
    legend_strip(
        fig,
        [
            (
                "Compartments",
                "line",
                [(F5.COMP_COLORS[c], F5.COMP_DISPLAY[c]) for c in F5.COMP_ORDER],
            ),
            (
                "Populations",
                "square",
                [
                    (F5.TYPE_COLORS[t], lbl)
                    for t, lbl in zip(
                        F5.TYPE_ORDER,
                        [
                            "Large-type",
                            "Small-type",
                            "Class II",
                            "Middle-type",
                            "Other",
                        ],
                        strict=False,
                    )
                ],
            ),
            (
                "Subclusters",
                "square",
                [(F5.SUB_COLORS[c], F5.SUB_LABELS[c]) for c in F5.SUB_CLUSTERS]
                # _simplex_rgb paints a cell white when its three subcluster
                # probabilities are equal; pale cells in C are near that point.
                + [("#FFFFFF", "Unresolved (equal probabilities)", "#8C8C8C")],
            ),
        ],
        M,
        H - LEG + 0.02,
        W,
        H,
        M,
        center=True,
    )
    fig.savefig(OUT / f"{NAME_KC}.png", dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{NAME_KC}.png")


# ── main figure (two compact variants) ───────────────────────────────────────
def draw_brain(fig, brain, sec, k, rects, ctx):
    """Four panels per caste row: A section (plain composite + white box),
    B1 image + direct labels, B2 cell types, B3 region outlines. rects: a, b1,
    b2, b3 = (x, ytop, w, h) inches. k = inches per um on the section panel.
    Returns the overlay primitives (inches) for the shared zoom-highlight pass."""
    rpts, zooms, zchans, lim_c = ctx["rpts"], ctx["zooms"], ctx["zchans"], ctx["lim_c"]

    def add(rc):
        x, yt, w, h = rc
        ax = fig.add_axes(
            [x / ctx["W"], 1 - (yt + h) / ctx["H"], w / ctx["W"], h / ctx["H"]]
        )
        style_img_ax(ax)
        return ax

    ax_a, ax_1, ax_2, ax_3 = (add(rects[n]) for n in ("a", "b1", "b2", "b3"))
    xy = sec["xy_all"]
    cx = 0.5 * (xy[:, 0].min() + xy[:, 0].max())
    cy = 0.5 * (xy[:, 1].min() + xy[:, 1].max()) + ctx["ybias"]
    pw_, ph_ = rects["a"][2], rects["a"][3]
    xl = (cx - pw_ / k / 2, cx + pw_ / k / 2)
    yl = (cy + ph_ / k / 2, cy - ph_ / k / 2)
    ext = img_extent_um(sec["cmin"], sec["dapi_ds"].shape, ds=DS)
    ax_a.imshow(
        composite(sec["ds_ch"], lim_c["full"][brain]),
        extent=ext,
        origin="upper",
        interpolation="antialiased",
        aspect="equal",
    )
    ax_a.set_xlim(*xl)
    ax_a.set_ylim(*yl)
    scalebar(ax_a, 1000, "1 mm", below=True, left=True)
    zx, zy = zooms[brain]

    chans = {c: zchans[brain][c][0] for c in range(4)}
    zext = zchans[brain][0][1]
    zrgb = composite(chans, lim_c["zoom"][brain])
    sel = [
        i
        for i, p in enumerate(sec["polys"])
        if zx - 15 <= p[:, 0].mean() <= zx + ZOOM_UM + 15
        and zy - 15 <= p[:, 1].mean() <= zy + ZOOM_UM + 15
    ]
    for ax in (ax_1, ax_2, ax_3):
        ax.imshow(
            zrgb,
            extent=zext,
            origin="upper",
            interpolation="antialiased",
            aspect="equal",
        )
    ax_2.add_collection(
        cell_fill(
            [sec["polys"][i] for i in sel],
            [ct_color(sec["labs"][i]) for i in sel],
            0.3,
            alpha=0.9,
        )
    )
    draw_polys(ax_3, rpts[brain], REGIONS, lw=1.4, under=0.85)
    for ax in (ax_1, ax_2, ax_3):
        ax.set_xlim(zx, zx + ZOOM_UM)
        ax.set_ylim(zy + ZOOM_UM, zy)
    if brain == "queen":  # one 50 um bar for all zoom panels
        scalebar(ax_1, 50, "50 µm")
    for ax, txt in ((ax_1, "Image"), (ax_2, "Cell types"), (ax_3, "Regions")):
        tag(ax, txt)

    # ---- direct labels on B1 ----
    from shapely.geometry import box as sbox

    win = sbox(zx, zy, zx + ZOOM_UM, zy + ZOOM_UM)
    allxy = sec["xy_all"]
    inw = (
        (allxy[:, 0] >= zx)
        & (allxy[:, 0] <= zx + ZOOM_UM)
        & (allxy[:, 1] >= zy)
        & (allxy[:, 1] <= zy + ZOOM_UM)
    )
    pts = np.column_stack(
        [(allxy[inw, 0] - zx) / ZOOM_UM, (allxy[inw, 1] - zy) / ZOOM_UM]
    )
    # keep labels off the scale bar and the bottom edge
    guard = np.array(
        [(x_, y_) for x_ in np.linspace(0.04, 0.96, 24) for y_ in (0.93, 0.97)]
    )
    pts = np.vstack([pts, guard])
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()

    def text_w(t):
        tt = ax_3.text(0, 0, t, fontsize=TICK, fontweight="bold")
        w = tt.get_window_extent(rr).width / ax_3.get_window_extent(rr).width
        tt.remove()
        return w + 0.03

    comp_items = []
    for reg in ("kc_oc", "kc_ic", "kc_nc"):
        cand = [g.intersection(win) for g in rpts[brain].get(reg, [])]
        cand = [c for c in cand if not c.is_empty]
        if cand:
            inter = max(cand, key=lambda c: c.area)
            q = inter.representative_point()
            comp_items.append(
                (
                    NR.DISPLAY[reg],
                    NR.REGION_COLORS[reg],
                    ((q.x - zx) / ZOOM_UM, (q.y - zy) / ZOOM_UM),
                )
            )
    labs_in = np.array([sec["labs"][i] for i in sel])
    xy_in = np.array([sec["xy"][i] for i in sel])
    from scipy.spatial import cKDTree

    tree_all = cKDTree(allxy[inw])
    kcxy = xy_in[labs_in == "Kenyon cells"]
    tree_kc = cKDTree(kcxy) if len(kcxy) else None
    chosen = []
    for want in ("Kenyon cells", "Astrocyte-like glia", "Mixed"):
        m = labs_in == want
        if m.sum() < 1:
            continue
        pp = xy_in[m]
        if want == "Kenyon cells":
            c = pp[np.argmin(np.hypot(*(pp - np.median(pp, axis=0)).T))]
        else:
            d_nn = tree_all.query(pp, k=2)[0][:, 1]
            d_kc = tree_kc.query(pp)[0] if tree_kc is not None else np.full(len(pp), 99)
            ok = (d_nn >= 7.0) & (d_kc >= 20.0)
            if not ok.any() and want == "Mixed":  # few cells: relax separation
                ok = (d_nn >= 4.0) & (d_kc >= 10.0)
            if not ok.any() and want == "Mixed":
                ok = (d_nn >= 3.0) & (d_kc >= 4.0)
            if not ok.any():
                print(f"    no unambiguous '{want}' example: dropped", flush=True)
                continue
            q = pp[ok]
            c = q[np.argmin(np.hypot(*(q - np.median(pp, axis=0)).T))]
        chosen.append((want, c))
    ct_items = [
        (
            ct_label(w).replace(" (all subtypes)", ""),
            ct_color(w),
            ((c[0] - zx) / ZOOM_UM, (c[1] - zy) / ZOOM_UM),
        )
        for w, c in chosen
    ]
    open_spot_labels(ax_3, comp_items, pts, text_w)
    open_spot_labels(ax_2, ct_items, pts, text_w, leader="white", dot="white")
    print(
        f"  {brain}: example cells labelled: "
        + ", ".join(w for w, _ in chosen)
        + f"; {len(sel)} cells in zoom field",
        flush=True,
    )

    # zoom panels: ring colour is the axes face, content clipped inside the band
    for ax in (ax_1, ax_2, ax_3):
        ax.set_facecolor(ZOOM_COLOR)
        fx = FRAME_LW / 72.0 / rects["b1"][2]
        fy = FRAME_LW / 72.0 / rects["b1"][3]
        clip = Rectangle(
            (fx, fy),
            1 - 2 * fx,
            1 - 2 * fy,
            transform=ax.transAxes,
            fc="none",
            ec="none",
        )
        ax.add_patch(clip)
        for ch in list(ax.get_children()):
            if ch is ax.patch or ch is clip or ch in ax.spines.values():
                continue
            if hasattr(ch, "set_clip_path") and hasattr(ch, "set_clip_on"):
                ch.set_clip_on(True)
                ch.set_clip_path(clip)
                ap = getattr(ch, "arrow_patch", None)
                if ap is not None:
                    ap.set_clip_on(True)
                    ap.set_clip_path(clip)

    # box corners in figure inches (for the overlay pass)
    def to_in(ax, xy_):
        d = ax.transData.transform(xy_)
        f = fig.transFigure.inverted().transform(d)
        return f[0] * ctx["W"], (1 - f[1]) * ctx["H"]

    return (
        ax_a,
        ax_1,
        dict(
            box=(to_in(ax_a, (zx, zy)), to_in(ax_a, (zx + ZOOM_UM, zy + ZOOM_UM))),
            rings=[rects["b1"], rects["b2"], rects["b3"]],
        ),
    )


def zoom_overlay(fig, items, W, H):
    """One transparent figure-inch overlay that draws, in this z-order: the two
    connectors (white band, dark casing), then the white box, then the three
    rings, all with a 0.6 pt near-black line on both band edges, so the
    connectors tuck under the box and ring and every join is continuous."""
    ov = fig.add_axes([0, 0, 1, 1], zorder=20)
    ov.set_xlim(0, W)
    ov.set_ylim(H, 0)
    ov.set_xticks([])
    ov.set_yticks([])
    ov.axis("off")
    ov.patch.set_alpha(0)
    ov.set_in_layout(False)
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MP
    from shapely.geometry import Polygon as P
    from shapely.geometry import box as B
    from shapely.ops import unary_union

    bw = FRAME_LW / 72.0  # white band width, inches

    def band(x0, y0, x1, y1):
        return B(x0, y0, x1, y1).difference(B(x0 + bw, y0 + bw, x1 - bw, y1 - bw))

    def quad(p0, p1, w):
        import numpy as _n

        d = _n.array(p1) - _n.array(p0)
        n = _n.array([-d[1], d[0]]) / _n.hypot(*d) * w / 2
        a_, b_ = _n.array(p0), _n.array(p1)
        u = d / _n.hypot(*d) * 0.0
        a_, b_ = a_ - u, b_ + u
        return P([a_ + n, b_ + n, b_ - n, a_ - n])

    def draw(geom):
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in polys:
            verts, codes = [], []
            for ring in [g.exterior, *g.interiors]:
                xy_ = list(ring.coords)
                verts += xy_
                codes += [MP.MOVETO] + [MP.LINETO] * (len(xy_) - 2) + [MP.CLOSEPOLY]
            path = MP(verts, codes)
            ov.add_patch(PathPatch(path, fc=ZOOM_COLOR, ec="none", zorder=2))
            ov.add_patch(
                PathPatch(
                    path,
                    fc="none",
                    ec=ZOOM_EDGE,
                    lw=EDGE_LW,
                    joinstyle="miter",
                    zorder=3,
                )
            )

    for it in items:
        (bx0, by0), (bx1, by1) = it["box"]
        rx, ry, rw, rh = it["rings"][0]
        bx0 -= bw / 2
        by0 -= bw / 2
        bx1 += bw / 2
        by1 += bw / 2  # outer box edge
        # box band centred on the box line: outer = box + half band
        outer = band(bx0, by0, bx1, by1)
        clipbox = B(bx0, ry, rx + rw, ry + rh)
        c1 = quad((bx1 - bw / 2, by0 + bw / 2), (rx + bw / 2, ry + bw / 2), bw)
        c2 = quad((bx1 - bw / 2, by1 - bw / 2), (rx + bw / 2, ry + rh - bw / 2), bw)
        c1, c2 = c1.intersection(clipbox), c2.intersection(clipbox)
        draw(unary_union([outer, c1, c2, band(rx, ry, rx + rw, ry + rh)]))
        for rx, ry, rw, rh in it["rings"][1:]:
            draw(band(rx, ry, rx + rw, ry + rh))


def legend_strip(fig, groups, x0, y0, W, H, M, line_gap=0.02, center=False):
    """Compact legend: one line per group, bold title in a common left column,
    entries packed right after it with fixed small gaps (real fig.legend, so the
    print-scale hook treats the text as legend text). A square entry may carry
    a third element, its edge colour (for a white swatch). center=True centres
    the whole block between the margins instead of starting it at x0."""
    from matplotlib.patches import Patch

    fs = 6.0
    r_ = fig.canvas.get_renderer()
    tws = []
    for title, _, _ in groups:
        t = fig.text(0, 0, title, fontsize=fs, fontweight="bold")
        tws.append(t.get_window_extent(r_).width / fig.dpi)
        t.remove()
    xe = x0 + max(tws) + 0.14
    y = y0
    placed = []
    for (title, kind, ents), _tw in zip(groups, tws, strict=False):
        if kind == "line":
            hs = [Line2D([0], [0], color=c, lw=1.8) for c, _ in ents]
            kw = dict(handlelength=12 / fs)
        elif kind == "square":
            hs = [
                Patch(
                    facecolor=e[0],
                    edgecolor=e[2] if len(e) > 2 else "none",
                    linewidth=0.6,
                )
                for e in ents
            ]
            kw = dict(handlelength=8 / fs, handleheight=8 / fs)
        else:
            hs = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=c,
                    markeredgecolor="none",
                    markersize=8,
                )
                for c, _ in ents
            ]
            kw = dict(handlelength=8 / fs)
        leg = fig.legend(
            hs,
            [e[1] for e in ents],
            loc="upper left",
            ncol=len(ents),
            frameon=False,
            fontsize=fs,
            borderaxespad=0,
            borderpad=0,
            handletextpad=0.4,
            columnspacing=6 / fs,
            bbox_to_anchor=(xe / W, 1 - y / H),
            **kw,
        )
        bb = leg.get_window_extent(r_)
        lh = bb.height / fig.dpi
        ttl = fig.text(
            x0 / W,
            1 - (y + lh / 2) / H,
            title,
            fontsize=fs,
            fontweight="bold",
            va="center",
        )
        placed.append((leg, ttl, bb.x1 / fig.dpi))
        y += lh + line_gap
    if center and placed:
        dx = W / 2 - (x0 + max(r for *_, r in placed)) / 2
        for leg, ttl, _ in placed:
            ax_, ay_ = (
                leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).p0
            )
            leg.set_bbox_to_anchor((ax_ + dx / W, ay_), transform=fig.transFigure)
            ttl.set_x(ttl.get_position()[0] + dx / W)
    return y


def main_figure(secs, ctx):
    """Caste rows, four panels each: A section | B1 image + labels | B2 cell
    types | B3 regions."""
    W = 7.2
    M, G, HDRH, LAB, GG = 0.05, 0.04, 0.17, 0.17, 0.09
    marg, mbot = 25.0, 110.0
    bb = {}
    for b, s_ in secs.items():
        xy = s_["xy_all"]
        bb[b] = (np.ptp(xy[:, 0]) + 2 * marg, np.ptp(xy[:, 1]) + marg + mbot)
    Wm = max(v[0] for v in bb.values())
    Hm = max(v[1] for v in bb.values())
    k = (W - 2 * M - LAB - GG - 2 * G) / (Wm + 3 * Hm)
    wf, hr = k * Wm, k * Hm
    y1 = M + HDRH
    ys = {"queen": y1, "worker": y1 + hr + G}
    H = ys["worker"] + hr + M
    ctx["W"], ctx["H"], ctx["ybias"] = W, H, (mbot - marg) / 2
    x_a = M + LAB
    x_1 = x_a + wf + GG
    panels = {
        b: dict(
            a=(x_a, ys[b], wf, hr),
            b1=(x_1, ys[b], hr, hr),
            b2=(x_1 + hr + G, ys[b], hr, hr),
            b3=(x_1 + 2 * (hr + G), ys[b], hr, hr),
        )
        for b in secs
    }
    fig = plt.figure(figsize=(W, H))
    heads, ov_items = {}, []
    for b in ("queen", "worker"):
        a_ax, b1_ax, ov = draw_brain(fig, b, secs[b], k, panels[b], ctx)
        ov_items.append(ov)
        if b == "queen":
            heads = {"A": a_ax, "B": b1_ax}
        yy = panels[b]["a"][1] + hr / 2
        fig.text(
            (M + 0.07) / W,
            1 - yy / H,
            b.capitalize(),
            rotation=90,
            fontsize=8,
            ha="center",
            va="center",
        )
    zoom_overlay(fig, ov_items, W, H)
    add_panel_headers(
        fig,
        [(heads["A"], "A", "Section"), (heads["B"], "B", "Zoomed field")],
        width_frac=1.0,
    )
    OUT_MAIN.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MAIN / f"{NAME}.png", dpi=300)
    fig.savefig(OUT_MAIN / f"{NAME}.pdf")
    plt.close(fig)
    print(
        "saved",
        OUT_MAIN / f"{NAME}.png",
        f"(W/H {W / H:.2f}, {W:.2f} x {H:.2f} in)",
        flush=True,
    )


def main():
    set_pub_style(font_size=TICK, label_size=TICK, legend_size=TICK)
    OUT.mkdir(parents=True, exist_ok=True)
    votes = load_votes()
    picks = pick_sections(votes)
    print("sections:", picks, flush=True)
    secs = {b: load_section(b, picks[b], votes) for b in ("queen", "worker")}

    zooms, zchans = {}, {}
    for brain, sec in secs.items():
        zooms[brain] = ZOOM_ORIGIN[brain]
        zchans[brain] = [read_window(sec, *zooms[brain], ZOOM_UM, k) for k in range(4)]

    # tissue-only limits, pooled over both castes (identical for queen/worker)
    lim_full = tissue_limits([s_["ds_ch"] for s_ in secs.values()])
    lim_zoom = tissue_limits([{c: zchans[b][c][0] for c in range(3)} for b in secs])
    lim_c = {"full": {}, "zoom": {}}
    for kind, lims, getter in (
        ("full", lim_full, lambda b: secs[b]["ds_ch"]),
        ("zoom", lim_zoom, lambda b: {c: zchans[b][c][0] for c in range(3)}),
    ):
        meds = {}
        for b in secs:
            g = getter(b)
            a = stretch(g[1], lims[1], 1)
            meds[b] = float(np.median(a[a > 0.03]))
        f = meds["queen"] / meds["worker"]
        extra = {1: f} if abs(f - 1) > 0.15 else {}
        lim_c[kind]["queen"] = {**lims, "x": {}}
        lim_c[kind]["worker"] = {**lims, "x": extra}
        print(
            f"  limits ({kind}, tissue pixels, pooled): "
            + "; ".join(
                f"{nm} p{PCT[ch][0]:g}-p{PCT[ch][1]:g} = "
                f"{lims[ch][0]:.0f}-{lims[ch][1]:.0f} (gamma "
                f"{GAINS[ch][0]}, gain {GAINS[ch][1]})"
                for ch, _, nm in COMP
            ),
            flush=True,
        )
        print(
            f"  green median queen {meds['queen']:.3f}, worker "
            f"{meds['worker']:.3f}; worker green extra gain "
            f"{extra.get(1, 1.0):.2f}",
            flush=True,
        )
    rpts = {b: load_region_polys(secs[b], b) for b in secs}
    ctx = dict(rpts=rpts, zooms=zooms, zchans=zchans, lim_c=lim_c)
    # BIMP_OVERLAY_ONLY (comma list of main,channels,kc; default main,kc, the
    # channel strip being part of kc) renders a subset, so
    # one SI figure can be re-rendered without rewriting the others.
    only = set(os.environ.get("BIMP_OVERLAY_ONLY", "main,kc").split(","))
    if "main" in only:
        main_figure(secs, dict(ctx))
    if "channels" in only:
        channel_figure(zooms, zchans)
    if "kc" in only:
        kc_population_figure(secs, zooms, zchans, lim_c["zoom"], rpts)
    for b, s in secs.items():
        print(b, s["slice_name"], "raw", s["n_raw"], "not in atlas", s["n_missing"])
        print(pd.Series(s["labs"]).value_counts().to_string())


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
