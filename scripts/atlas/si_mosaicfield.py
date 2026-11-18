#!/usr/bin/env python3
"""Supplementary figure: MOSAICField nonlinear alignment (consolidated).

MOSAICField is an experimental *alternative* aligner explored alongside the
PASTE3 pipeline that every other analysis uses. This script combines
the schematic, the alignment-pipeline walkthrough, and the 3D reconstruction
comparison into one figure.

Panels:
  A) Chunking schematic -- 30 slices split into 5 groups of <=6 (nonlinear
     registration is only ever done between *adjacent* slices; groups keep
     each chain of composed warps short)
  B) Neural displacement field for one representative adjacent pair --
     deformed grid + magnitude heatmap (um)
  C) Two-colour overlay of the same pair (source magenta, target green),
     affine-only vs after-nonlinear, with a zoom on the most-displaced region
  D) Quantitative check -- adjacent-slice tissue-occupancy Dice overlap,
     affine-only vs nonlinear-warped MOSAICField point clouds vs PASTE3
  E, F) 3D reconstruction, one viewing angle -- MOSAICField (chunked +
     stitched) vs PASTE3, both in micrometres on identical axes

Units. MOSAICField's step 2 (``MOSAICField.pipeline.step2_chain_affine``)
resamples each section image with ``crop_post_tf(..., scale=ps)`` and saves
cells as ``transform(...) * ps`` (ps = 0.2125 um per Xenium image pixel), so
its images, cells and displacement fields are on a 1 um/px grid: a flow value
in px is a distance in um (checked: ``{sid}_cells.npy`` lands 1:1 on tissue
in ``{sid}_img.npy``). PASTE3's ``obsm['spatial_aligned']`` is in 0.2125-um
image pixels (median nearest-neighbour spacing 5.3-5.8 um after * 0.2125,
matching an untransformed section), and PASTE3 transforms are rigid. The
stitched MOSAICField frame is nominal um, but the chained FGW-OT affines
shrink it unevenly along the stack (nearest-neighbour spacing 3.2-5.6 um), so
E draws smaller than F on the same axes; that is the output as produced.

    PYTHONPATH=src:scripts .venv/bin/python scripts/atlas/si_mosaicfield.py

Set BIMP_MOSAIC_OUT to write a preview elsewhere (same file stem, so the
print-scale hook still applies) and BIMP_ALSO_PDF=1 for the vector copy.
"""

import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter, label

sys.path.insert(0, str(Path(__file__).parent))
from bimp_st.si_common import (
    add_panel_headers,
    panel_label_size,
    save,
    set_style,
)

# Printed at full \linewidth (SI tex; print_scale.PRINTED_WIDTH_FRAC = 1.0).
# Fonts are canvas points = printed_pt * FIG_W / 6.67, so tick 6 / label 7 pt
# printed; the print_scale hook then fine-tunes one factor for the whole page.
FIG_W, FIG_H = 11.0, 10.2
_K = FIG_W / 6.67
PANEL_LABEL_SIZE = panel_label_size(FIG_W, width_frac=1.0)
F_TICK, F_LABEL = 6.0 * _K, 7.0 * _K
F_SMALL = 5.5 * _K
ZOOM_ASPECT = 0.55  # zoom height / width (panel C lower row)
D_LABEL_W = 0.085  # figure fraction kept left of panel D for its labels
D_HEIGHT = 0.095  # figure fraction, panel D axes

from bimp_st.utils.mosaicfield_img import (  # noqa: E402
    crop as _crop,
)
from bimp_st.utils.mosaicfield_img import (  # noqa: E402
    enhance as _enhance,
)
from bimp_st.utils.mosaicfield_img import (  # noqa: E402
    load_img as _load_img_base,
)
from bimp_st.utils.mosaicfield_img import (  # noqa: E402
    norm as _norm,
)
from bimp_st.utils.mosaicfield_img import (  # noqa: E402
    tissue_bbox as _tissue_bbox_base,
)
from bimp_st.utils.mosaicfield_img import (  # noqa: E402
    warp_img as _warp_img,
)

# ── Config ────────────────────────────────────────────────────────────
CHUNK_DIR = Path("results/mosaicfield_chunked")  # schematic example chunk
MOSAIC_DIR = Path("results/mosaicfield_noskip")  # full 30-slice pipeline
STITCH_DIR = Path("results/mosaicfield_stitched")  # cached 3D stitch
PASTE3_H5AD = Path("pipelines/wholebrain/queen/data/aligned_clustered.h5ad")
OUTPUT = Path(os.environ.get("BIMP_MOSAIC_OUT", "results/figures/si"))

UM_PER_MF_PX = 1.0  # MOSAICField image/flow grid (see module docstring)
UM_PER_PASTE3_UNIT = 0.2125  # obsm['spatial_aligned'] is in image pixels

ALL_IDS = [
    "1A",
    "1B",
    "2A",
    "2B",
    "3A",
    "3B",
    "4A",
    "4B",
    "5A",
    "5B",
    "6A",
    "6B",
    "7A",
    "7B",
    "8A",
    "2_1A",
    "2_1B",
    "2_2A",
    "2_2B",
    "2_3A",
    "2_3B",
    "2_4A",
    "2_4B",
    "2_5A",
    "2_5B",
    "2_6A",
    "2_6B",
    "2_7A",
    "2_7B",
    "2_8A",
]
N = len(ALL_IDS)
CHUNK_SIZE = 6
CHUNKS = [ALL_IDS[i : i + CHUNK_SIZE] for i in range(0, N, CHUNK_SIZE)]
CHUNK_COLORS = ["#4363D8", "#E6194B", "#3CB44B", "#F58231", "#911EB4"]
EXAMPLE_CHUNK_IDX = 1  # 4A-6B: the pair used in panels B/C/D is drawn from here

# Representative adjacent pair used for panels B, C (and included in D)
PAIR_SRC, PAIR_TGT = "5A", "5B"

# Method colours (panel D) and the depth colour map (panels E/F).
C_AFFINE, C_NONLIN, C_PASTE3 = "#8FB8D8", "#1F5F8B", "#B0703A"
DEPTH_CMAP = "viridis"
INK = "#222222"


# ── Small helpers ────────────────────────────────────────────────────


def _load_img(sid, img_dir=MOSAIC_DIR):
    return _load_img_base(sid, img_dir)


def _load_flow(src, tgt, img_dir=MOSAIC_DIR):
    return np.load(img_dir / f"flow_{src}_to_{tgt}.npy")


def _load_cells(sid, final=False, img_dir=MOSAIC_DIR):
    key = "cells_final" if final else "cells"
    return np.load(img_dir / f"{sid}_{key}.npy")


def _tissue_bbox(imgs, pad_frac=0.03, percentile=10):
    return _tissue_bbox_base(imgs, pad_frac=pad_frac, percentile=percentile)


def _square_bbox(bbox, shape):
    """Grow a (y0, y1, x0, x1) bbox to a square, clipped to the image."""
    y0, y1, x0, x1 = bbox
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    y0 = int(max(0, min(shape[0] - side, cy - side / 2)))
    x0 = int(max(0, min(shape[1] - side, cx - side / 2)))
    return y0, y0 + side, x0, x0 + side


def _tissue_mask(sid, shape, sigma_um=20):
    """Smooth filled tissue silhouette from a section's cell positions.

    ``{sid}_cells.npy`` holds (x, y) in um on the same 1 um/px grid as
    ``{sid}_img.npy`` (row = y, column = x).
    """
    xy = _load_cells(sid, final=False)
    dens = np.zeros(shape, np.float32)
    r = np.clip(xy[:, 1].astype(int), 0, shape[0] - 1)
    c = np.clip(xy[:, 0].astype(int), 0, shape[1] - 1)
    np.add.at(dens, (r, c), 1.0)
    sm = gaussian_filter(dens, sigma_um)
    m = binary_fill_holes(sm > 0.15 * np.percentile(sm[sm > 0], 90))
    m = binary_fill_holes(binary_opening(m, iterations=3))
    lab, n = label(m)
    if n > 1:  # keep the section, drop debris islands
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return m


def _scalebar(ax, length_um, color, *, text=None, frac=(0.06, 0.06), bg=None):
    """Horizontal bar of ``length_um`` at the lower left of an image axes
    whose data units are um (1 um/px); optional label just above, left-set."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()  # imshow: y0 is the bottom (larger value)
    w, h = abs(x1 - x0), abs(y1 - y0)
    bx = min(x0, x1) + frac[0] * w
    by = max(y0, y1) - frac[1] * h
    ax.plot(
        [bx, bx + length_um],
        [by, by],
        color=color,
        lw=1.6,
        solid_capstyle="butt",
        zorder=6,
    )
    if text:
        ax.text(
            bx,
            by - 0.03 * h,
            text,
            ha="left",
            va="bottom",
            fontsize=F_SMALL,
            color=color,
            zorder=6,
        )
        if bg:  # one opaque box behind bar and label, clear of grid lines
            ax.add_patch(
                mpatches.Rectangle(
                    (bx - 0.025 * w, by - 0.115 * h),
                    max(length_um, 0.25 * w) + 0.05 * w,
                    0.15 * h,
                    facecolor=bg,
                    edgecolor="none",
                    zorder=5,
                )
            )


# ── Panel A: chunking schematic ─────────────────────────────────────


# The two slides' crops share IDs ("1A" ... "8A"); the second slide's carry a
# "2_" prefix in the data. Panel A prints the bare ID under a slide bracket, so
# every tile label is the same (legible) size, and leaves a small gap between
# the slides.
SLIDE_GAP = 0.5


def _tile_x(i):
    return i + (SLIDE_GAP if ALL_IDS[i].startswith("2_") else 0.0)


def panel_a(ax):
    box_w, box_h = 0.88, 0.8
    y = 0
    for i, sid in enumerate(ALL_IDS):
        ci = i // CHUNK_SIZE
        color = CHUNK_COLORS[ci % len(CHUNK_COLORS)]
        chunk_end = min((ci + 1) * CHUNK_SIZE, N) - 1
        is_ref = i == chunk_end
        is_ex = ci == EXAMPLE_CHUNK_IDX
        fc = color if is_ref else ("#F5F0FF" if is_ex else "white")
        lw = 1.3 if is_ref else (0.9 if is_ex else 0.6)
        xi = _tile_x(i)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (xi - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.0,rounding_size=0.08",
                facecolor=fc,
                edgecolor=color,
                linewidth=lw,
            )
        )
        ax.text(
            xi,
            y,
            sid.split("_")[-1],
            ha="center",
            va="center",
            fontsize=F_SMALL,
            fontweight="bold" if is_ref else "normal",
            color="white" if is_ref else INK,
        )

    # Slide brackets above the tiles.
    for k, prefix_test in enumerate((False, True)):
        idx = [
            i for i, sid in enumerate(ALL_IDS) if sid.startswith("2_") is prefix_test
        ]
        x0, x1 = _tile_x(idx[0]) - 0.42, _tile_x(idx[-1]) + 0.42
        ty = y + box_h / 2 + 0.18
        ax.plot(
            [x0, x0, x1, x1],
            [ty - 0.1, ty, ty, ty - 0.1],
            color="#555555",
            lw=0.7,
            clip_on=False,
        )
        ax.text(
            (x0 + x1) / 2,
            ty + 0.07,
            f"Slide {k + 1}",
            ha="center",
            va="bottom",
            fontsize=F_TICK,
            color="#333333",
            clip_on=False,
        )

    # Registration arrows (example chunk): thin connectors below the tiles.
    start = EXAMPLE_CHUNK_IDX * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, N)
    ay = y - box_h / 2 - 0.17
    for j in range(start, end - 1):
        ax.annotate(
            "",
            xy=(_tile_x(j + 1) - 0.14, ay),
            xytext=(_tile_x(j) + 0.14, ay),
            arrowprops=dict(
                arrowstyle="-|>",
                color=CHUNK_COLORS[EXAMPLE_CHUNK_IDX],
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=6,
            ),
        )

    # Chunk brackets and labels below.
    for ci, chunk in enumerate(CHUNKS):
        s_ = ci * CHUNK_SIZE
        e = min(s_ + len(chunk), N) - 1
        color = CHUNK_COLORS[ci % len(CHUNK_COLORS)]
        is_ex = ci == EXAMPLE_CHUNK_IDX
        x0, x1 = _tile_x(s_) - 0.42, _tile_x(e) + 0.42
        by = -0.8
        ax.plot(
            [x0, x0, x1, x1],
            [by + 0.1, by, by, by + 0.1],
            color=color,
            lw=1.1 if is_ex else 0.7,
            alpha=1.0 if is_ex else 0.6,
            clip_on=False,
        )
        ax.text(
            (x0 + x1) / 2,
            by - 0.08,
            f"Chunk {ci + 1}",
            ha="center",
            va="top",
            fontsize=F_TICK,
            color=color,
            fontweight="bold" if is_ex else "normal",
            clip_on=False,
        )

    ax.set_xlim(-0.55, _tile_x(N - 1) + 0.55)
    ax.set_ylim(-1.3, 0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


# ── Shared crop for panels B and C ──────────────────────────────────


def _pair_images():
    src = _load_img(PAIR_SRC)
    tgt = _load_img(PAIR_TGT)
    bbox = _square_bbox(
        _tissue_bbox({"s": _norm(src), "t": _norm(tgt)}, pad_frac=0.03), src.shape
    )
    return src, tgt, bbox


# ── Panel B: displacement field ─────────────────────────────────────


def panel_b(ax_grid, ax_mag, cax, src, tgt, bbox, grid_um=100):
    flow = _load_flow(PAIR_SRC, PAIR_TGT)[0] * UM_PER_MF_PX  # (2, H, W): dy, dx
    y0, y1, x0, x1 = bbox
    fl = flow[:, y0:y1, x0:x1]
    h, w = fl.shape[1:]
    mask = _crop(_tissue_mask(PAIR_TGT, tgt.shape), bbox)

    # Deformed grid: dark thin lines over a light tissue silhouette.
    ax_grid.imshow(
        np.where(mask, 1.0, 0.0),
        cmap=mcolors.ListedColormap(["white", "#DCE3EA"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    step = 4  # um between samples along each line
    for yy in range(grid_um // 2, h, grid_um):
        xs = np.arange(0, w, step)
        ax_grid.plot(xs + fl[1, yy, xs], yy + fl[0, yy, xs], color="#2B2B2B", lw=0.45)
    for xx in range(grid_um // 2, w, grid_um):
        ys = np.arange(0, h, step)
        ax_grid.plot(xx + fl[1, ys, xx], ys + fl[0, ys, xx], color="#2B2B2B", lw=0.45)
    ax_grid.set_xlim(0, w)
    ax_grid.set_ylim(h, 0)
    ax_grid.set_title(
        f"Deformed grid ({grid_um} µm)",
        fontsize=F_TICK,
        pad=3,
        color=INK,
        fontweight="normal",
    )
    _scalebar(ax_grid, 500, INK, text="500 µm", bg="white")

    mag = np.hypot(fl[0], fl[1])
    im = ax_mag.imshow(mag, cmap="inferno", interpolation="bilinear", vmin=0)
    ax_mag.contour(
        mask.astype(float), levels=[0.5], colors="white", linewidths=0.4, alpha=0.7
    )
    ax_mag.set_xlim(0, w)
    ax_mag.set_ylim(h, 0)
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks(np.arange(0, float(np.nanmax(mag)) + 1e-9, 20))
    cb.set_label("Displacement (µm)", fontsize=F_TICK, labelpad=2)
    cb.ax.tick_params(labelsize=F_TICK, length=2, pad=1.5, width=0.5)
    cb.outline.set_linewidth(0.4)
    ax_mag.set_title(
        "Displacement magnitude", fontsize=F_TICK, pad=3, color=INK, fontweight="normal"
    )
    _scalebar(ax_mag, 500, "white")
    for a in (ax_grid, ax_mag):
        a.set_xticks([])
        a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_visible(False)
    return mag


# ── Panel C: two-colour overlay ──────────────────────────────────────


def _overlay(a, b):
    """Source magenta, target green; overlap white."""
    return np.clip(np.dstack([a, b, a]), 0, 1)


def panel_c(axes, zoom_axes, src_raw, tgt_raw, bbox, mag, zoom_h=480):
    src_img = _enhance(src_raw)
    tgt_img = _enhance(tgt_raw)
    flow = torch.from_numpy(_load_flow(PAIR_SRC, PAIR_TGT).astype(np.float32))
    warped = _enhance(_warp_img(src_img, flow))

    def prep(img):
        return np.clip(_crop(img, bbox) / 0.7, 0, 1) ** 0.8

    s_c, t_c, w_c = prep(src_img), prep(tgt_img), prep(warped)
    h, w = s_c.shape

    # Zoom window centred on the largest displacement inside the tissue.
    sm = gaussian_filter(mag, 25)
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    # Window as wide as its parent image relative to the zoom row's height,
    # so each zoom spans exactly the width of the overview above it.
    zoom_w = int(round(zoom_h / ZOOM_ASPECT))
    cy = int(np.clip(cy, zoom_h // 2, h - zoom_h // 2))
    cx = int(np.clip(cx, zoom_w // 2, w - zoom_w // 2))
    zy0, zx0 = cy - zoom_h // 2, cx - zoom_w // 2

    for ax, zax, moving, title in (
        (axes[0], zoom_axes[0], s_c, "Affine only"),
        (axes[1], zoom_axes[1], w_c, "After nonlinear"),
    ):
        rgb = _overlay(moving, t_c)
        ax.imshow(rgb, interpolation="bilinear")
        ax.set_title(title, fontsize=F_TICK, pad=3, color=INK, fontweight="normal")
        ax.add_patch(
            mpatches.Rectangle(
                (zx0, zy0),
                zoom_w,
                zoom_h,
                fill=False,
                edgecolor="#FFD54F",
                lw=0.8,
            )
        )
        _scalebar(ax, 500, "white")
        zax.imshow(
            rgb[zy0 : zy0 + zoom_h, zx0 : zx0 + zoom_w], interpolation="bilinear"
        )
        for s_ in zax.spines.values():
            s_.set_edgecolor("#FFD54F")
            s_.set_linewidth(1.0)
        _scalebar(zax, 100, "white", text="100 µm", frac=(0.04, 0.08))
        for a in (ax, zax):
            a.set_xticks([])
            a.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)


# ── Panel D: quantitative alignment check ───────────────────────────


def _occupancy_dice(pts_a, pts_b, bin_frac=1.0 / 40):
    """Dice overlap of tissue-occupancy grids for two adjacent-slice point clouds.

    Bin size is a *fraction of each pair's own bounding-box extent*, not an
    absolute physical unit -- this keeps the metric meaningful even though
    MOSAICField and PASTE3 use different internal coordinate scales (see
    panels E/F caption).
    """
    both = np.vstack([pts_a, pts_b])
    lo = np.percentile(both, 0.5, axis=0)
    hi = np.percentile(both, 99.5, axis=0)
    span = hi - lo
    span[span <= 0] = 1.0
    nbins = max(8, int(round(1.0 / bin_frac)))
    edges = [np.linspace(lo[d], hi[d], nbins + 1) for d in range(2)]

    ha, _, _ = np.histogram2d(pts_a[:, 0], pts_a[:, 1], bins=edges)
    hb, _, _ = np.histogram2d(pts_b[:, 0], pts_b[:, 1], bins=edges)
    ma, mb = ha > 0, hb > 0
    inter = (ma & mb).sum()
    denom = ma.sum() + mb.sum()
    return 2 * inter / denom if denom > 0 else np.nan


def compute_alignment_metric():
    """Adjacent-slice occupancy Dice for affine-only / nonlinear MOSAICField
    vs PASTE3, averaged over all 29 adjacent slice pairs."""
    dice_affine, dice_nl = [], []
    for i in range(len(ALL_IDS) - 1):
        sid_a, sid_b = ALL_IDS[i], ALL_IDS[i + 1]
        try:
            aff_a = _load_cells(sid_a, final=False)
            aff_b = _load_cells(sid_b, final=False)
            nl_a = _load_cells(sid_a, final=True)
            nl_b = _load_cells(sid_b, final=True)
        except FileNotFoundError:
            continue
        dice_affine.append(_occupancy_dice(aff_a, aff_b))
        dice_nl.append(_occupancy_dice(nl_a, nl_b))

    ad = sc.read_h5ad(PASTE3_H5AD)
    xy = np.array(ad.obsm["spatial_aligned"])
    z = ad.obs["z_level"].values.astype(float)
    zs = sorted(np.unique(z))
    dice_paste3 = []
    for i in range(len(zs) - 1):
        pa = xy[z == zs[i]]
        pb = xy[z == zs[i + 1]]
        if len(pa) == 0 or len(pb) == 0:
            continue
        dice_paste3.append(_occupancy_dice(pa, pb))

    return {
        "Affine only\n(MOSAICField)": np.array(dice_affine),
        "Nonlinear\n(MOSAICField)": np.array(dice_nl),
        "PASTE3\n(pairwise OT)": np.array(dice_paste3),
    }


def panel_d(ax, metrics):
    labels = list(metrics.keys())
    means = [np.nanmean(metrics[k]) for k in labels]
    sems = [
        np.nanstd(metrics[k]) / np.sqrt(np.sum(~np.isnan(metrics[k]))) for k in labels
    ]
    colors = [C_AFFINE, C_NONLIN, C_PASTE3]

    # Mean +/- SEM over the per-pair points rather than bars from zero: every
    # value sits in 0.7-0.95, so zero-based bars were mostly empty fill.
    allv = np.concatenate([np.asarray(metrics[k], float) for k in labels])
    # Horizontal strips (one row per method, top to bottom) so the panel fits
    # the short, wide slot under B.
    for yi, k in enumerate(labels):
        vals = metrics[k]
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.44
        ax.scatter(
            vals,
            yi + jitter,
            s=9,
            color=colors[yi],
            alpha=0.85,
            zorder=3,
            linewidths=0,
        )
        ax.errorbar(
            means[yi],
            yi,
            xerr=sems[yi],
            color="black",
            capsize=2.5,
            lw=1.0,
            capthick=1.0,
            marker="D",
            ms=4,
            mfc="white",
            mec="black",
            mew=0.9,
            zorder=5,
        )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=F_TICK)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="x", labelsize=F_TICK)
    ax.set_xlabel("Adjacent-slice tissue-occupancy Dice", fontsize=F_LABEL)
    lo = np.floor((np.nanmin(allv) - 0.01) * 20) / 20
    hi = np.ceil((np.nanmax(allv) + 0.01) * 20) / 20
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(lo, hi + 1e-9, 0.05))
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


# ── Panels E, F: 3D reconstruction, one viewing angle ───────────────


def load_mosaicfield_3d():
    cached = STITCH_DIR / "stitched_coords_3d.npy"
    if not cached.exists():
        raise FileNotFoundError(
            f"{cached} not found -- run scripts/atlas/mosaicfield_stitch.py "
            f"(after the chunked run_mosaicfield.py; run order in its docstring) "
            f"to (re)populate the stitching cache"
        )
    return np.load(cached)  # x, y in nominal um; z = 10 um per section


def load_paste3_3d():
    ad = sc.read_h5ad(PASTE3_H5AD)
    xy = np.array(ad.obsm["spatial_aligned"]) * UM_PER_PASTE3_UNIT
    z = ad.obs["z_level"].values.astype(float)
    return np.column_stack([xy, z])


def _center_um(pts):
    """Median-centre x/y (um, unchanged scale); z kept as section depth."""
    out = pts.astype(float).copy()
    out[:, :2] -= np.median(out[:, :2], axis=0)
    return out


def panel_ef(ax_e, ax_f, mosaic_3d, paste3_3d, elev=18, azim=-58):
    ss_m = max(1, len(mosaic_3d) // 100_000)
    ss_p = max(1, len(paste3_3d) // 100_000)
    m = _center_um(mosaic_3d[::ss_m])
    p = _center_um(paste3_3d[::ss_p])

    # One set of limits for both panels (rounded out to 250 um).
    both = np.vstack([m, p])
    half = np.ceil(np.percentile(np.abs(both[:, :2]), 99.7, axis=0) / 250) * 250
    zmax = float(max(m[:, 2].max(), p[:, 2].max()))
    norm = mcolors.Normalize(0, zmax)

    for ax, pts in ((ax_e, m), (ax_f, p)):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=pts[:, 2],
            cmap=DEPTH_CMAP,
            norm=norm,
            s=0.35,
            alpha=0.5,
            rasterized=True,
            linewidths=0,
            depthshade=False,
        )
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("#D0D0D0")
            axis.pane.set_linewidth(0.4)
            axis.line.set_linewidth(0.5)
            axis.line.set_color("#555555")
        ax.grid(False)
        ax.set_xlim(-half[0], half[0])
        ax.set_ylim(-half[1], half[1])
        ax.set_zlim(zmax, 0)  # anterior (first section) at the top
        ax.set_xticks([-1000, 0, 1000])
        ax.set_yticks([-1000, 0, 1000])
        ax.set_zticks([0, 100, 200])
        ax.tick_params(axis="both", labelsize=F_SMALL, pad=0, length=2)
        ax.tick_params(axis="z", pad=2)
        ax.set_xlabel("x (µm)", fontsize=F_TICK, labelpad=1)
        ax.set_ylabel("y (µm)", fontsize=F_TICK, labelpad=1)
        ax.set_zlabel("Section depth (µm)", fontsize=F_TICK, labelpad=0)
        ax.set_box_aspect((half[0], half[1], 0.75 * max(half)), zoom=1.26)
        ax.view_init(elev=elev, azim=azim)
        ax.patch.set_visible(False)


# ── Main figure ──────────────────────────────────────────────────────


def plot_figure():
    set_style()
    plt.rcParams.update(
        {
            "font.size": F_TICK,
            "axes.titlesize": F_TICK,
            "axes.labelsize": F_LABEL,
            "xtick.labelsize": F_TICK,
            "ytick.labelsize": F_TICK,
            "axes.linewidth": 0.6,
        }
    )
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(
        3,
        1,
        figure=fig,
        hspace=0.22,
        height_ratios=[0.12, 0.47, 0.41],
        left=0.035,
        right=0.975,
        top=0.975,
        bottom=0.02,
    )
    ax_a = fig.add_subplot(gs[0])

    # Row 2, left: B (grid | magnitude | colorbar) over D.
    # Row 2, right: C (affine | nonlinear) over their zooms.
    gs_bc = GridSpecFromSubplotSpec(
        2,
        6,
        subplot_spec=gs[1],
        wspace=0.06,
        hspace=0.12,
        width_ratios=[1, 1, 0.05, 0.22, 1, 1],
        height_ratios=[1, 0.62],
    )
    ax_grid = fig.add_subplot(gs_bc[0, 0])
    ax_mag = fig.add_subplot(gs_bc[0, 1])
    ax_cb = fig.add_subplot(gs_bc[0, 2])
    ax_c1 = fig.add_subplot(gs_bc[0, 4])
    ax_c2 = fig.add_subplot(gs_bc[0, 5])
    ax_z1 = fig.add_subplot(gs_bc[1, 4])
    ax_z2 = fig.add_subplot(gs_bc[1, 5])
    spec_d = gs_bc[1, 0:3]
    ax_d = fig.add_subplot(spec_d)

    # Row 3: E | F, the full width.
    gs_ef = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.0)
    ax_e = fig.add_subplot(gs_ef[0, 0], projection="3d")
    ax_f = fig.add_subplot(gs_ef[0, 1], projection="3d")

    print("Panel A: chunking schematic...", flush=True)
    panel_a(ax_a)

    src, tgt, bbox = _pair_images()
    print("Panel B: displacement field...", flush=True)
    mag = panel_b(ax_grid, ax_mag, ax_cb, src, tgt, bbox)
    print(f"    max displacement in crop: {np.nanmax(mag):.1f} um", flush=True)

    print("Panel C: two-colour overlay...", flush=True)
    panel_c((ax_c1, ax_c2), (ax_z1, ax_z2), src, tgt, bbox, mag)
    for a_ in (ax_grid, ax_mag, ax_c1, ax_c2, ax_z1, ax_z2):
        a_.set_anchor("N")

    print("Panel D: quantitative alignment metric...", flush=True)
    metrics = compute_alignment_metric()
    for k, v in metrics.items():
        print(
            f"    {k.strip().splitlines()[0]}: Dice = {np.nanmean(v):.3f} "
            f"+/- {np.nanstd(v):.3f} (n={np.sum(~np.isnan(v))})",
            flush=True,
        )
    panel_d(ax_d, metrics)

    print("Panels E/F: 3D reconstruction comparison...", flush=True)
    panel_ef(ax_e, ax_f, load_mosaicfield_3d(), load_paste3_3d())

    fig.canvas.draw()
    pm, pc = ax_mag.get_position(), ax_cb.get_position()
    ax_cb.set_position([pc.x0, pm.y0, pc.width, pm.height])
    # D: leave room on the left for its method labels; start the axes below
    # the panel header, end at the magnitude image's right edge.
    # Each zoom spans its parent's width, just below it.
    fw, fh = fig.get_size_inches()
    for parent, zax in ((ax_c1, ax_z1), (ax_c2, ax_z2)):
        pp = parent.get_position()
        zh = pp.width * ZOOM_ASPECT * fw / fh
        zax.set_position([pp.x0, pp.y0 - 0.012 - zh, pp.width, zh])
    zp = ax_z1.get_position()
    pdd = ax_d.get_position()
    d_x0 = pdd.x0 + D_LABEL_W
    d_top = pm.y0 - 0.055
    ax_d.set_position([d_x0, d_top - D_HEIGHT, pm.x1 - d_x0, D_HEIGHT])
    # Invisible anchor so D's header sits level with the zoom row's top.
    ax_dh = fig.add_axes([pdd.x0, d_top + 0.012, 0.01, 0.001])
    ax_dh.set_xticks([])
    ax_dh.set_yticks([])
    ax_dh.axis("off")
    # 3D boxes: shift left and enlarge so the stacks fill their cells.
    for ax_, dx in ((ax_e, -0.07), (ax_f, -0.05)):
        p_ = ax_.get_position()
        ax_.set_position([p_.x0 + dx, p_.y0 + 0.004, p_.width + 0.06, p_.height + 0.02])

    add_panel_headers(fig, [(ax_a, "A", "Chunked alignment")], width_frac=1.0)
    add_panel_headers(
        fig,
        [
            (ax_grid, "B", "Displacement field, 5A→5B"),
            (ax_c1, "C", "Adjacent-section overlay"),
        ],
        width_frac=1.0,
        dy_pts=2.6 * F_TICK,
    )
    add_panel_headers(
        fig, [(ax_dh, "D", "Alignment overlap")], width_frac=1.0, dy_pts=0.0
    )
    add_panel_headers(
        fig,
        [(gs_ef[0, 0], "E", "MOSAICField"), (gs_ef[0, 1], "F", "PASTE3")],
        width_frac=1.0,
        dy_pts=0.0,
    )
    # Key for the overlay colours, under the zooms.
    fig.text(
        (zp.x0 + ax_z2.get_position().x1) / 2,
        zp.y0 - 0.008,
        f"magenta, {PAIR_SRC}; green, {PAIR_TGT}; white, overlap",
        ha="center",
        va="top",
        fontsize=F_SMALL,
        color=INK,
    )

    # Tight bbox ignores 3D axis labels; a white backing rectangle reserves a
    # consistent margin round the whole canvas.
    fig.add_artist(
        mpatches.Rectangle(
            (-0.008, -0.012),
            1.016,
            1.022,
            transform=fig.transFigure,
            zorder=-100,
            facecolor="white",
            edgecolor="none",
        )
    )
    save(fig, OUTPUT, "figS_mosaicfield")


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if not MOSAIC_DIR.exists():
            print(f"ERROR: {MOSAIC_DIR} not found", flush=True)
            sys.exit(1)
        plot_figure()
        print("\nDone.", flush=True)
