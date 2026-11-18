"""
Multi-modal Tachykinin comparison — worker brain: microCT, Xenium, MALDI.

Every panel is displayed in a window of the same physical size (2*HALF_W_UM by
2*HALF_H_UM) and with the same 500 µm scale bar. Each dataset is centered on
its brain, then rendered with native pixel/voxel sizes:
  - µCT:    4.53 µm / voxel   (ImageJ metadata: unit=µm, XRes=220724/1e6)
  - Xenium: native µm coords  (Xenium Onboard Analysis)
  - MALDI:  bimp_st.maldi map of the Tk2 readout (40 µm bins, summed over
            sections), the same map as the main MALDI figure

All axes use µm, so the 500 µm scale bar is identical on screen across panels.

Output: results/figures/si/figS_multimodal_tk.png
"""

import os
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports


import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import LinearSegmentedColormap

# Text is normalised to the house printed sizes at save time; this script
# reaches neither bimp_st.style.set_pub_style() nor scripts/_si_common,
# so the hook is installed here.
from bimp_st.style import set_pub_style  # noqa: E402

# Without it the figure gets raw matplotlib defaults (normal-weight titles,
# top/right spines, larger fonts) unlike every neighbouring figure.
set_pub_style()
from bimp_st.print_scale import install as _install_print_scale  # noqa: E402

_install_print_scale()

# =============================================================================
# CONFIG
# =============================================================================
from bimp_st import maldi  # noqa: E402
from bimp_st.paths import (
    REPO,  # noqa: E402  # resolved from this file, so a worktree
)

# writes to its own results/ rather than the main checkout's
XENIUM_H5AD = REPO / "pipelines/wholebrain/worker/data/aligned_clustered.h5ad"
MICROCT_TIF = REPO / "data/microct/Z1224-18TF_Segmentation-Interpolated.tif"
# Write straight to the SI figure name (the registry's target,
# figures/figS_multimodal_tk). The older output path
# results/figures/multimodal/tk_xenium_maldi_microct.png is not registered.
# BIMP_MULTIMODAL_OUT renders to a scratch dir (render-identity checks).
OUT_DIR = _Path(os.environ.get("BIMP_MULTIMODAL_OUT") or REPO / "results/figures/si")
OUT_DIR.mkdir(parents=True, exist_ok=True)

XENIUM_TK_VAR = "BIMP_vNP3.1_07029|MGEN_v3.1_10142"
XENIUM_PIXEL_UM = 0.2125  # 10X Xenium default px pitch (obsm['spatial']
# is in pixels, not µm — multiply to convert)

MICROCT_VOXEL_UM = 4.5305  # ImageJ metadata: pitch = 1 / (220724/1e6)
MICROCT_CHANNEL = 0  # ch0 = full segmentation

# The MALDI map, its orientation and its smoothing come from bimp_st.maldi, the
# loader of the main MALDI figure, so the two figures draw the same map.

# ---- Panel geometry ---------------------------------------------------------
# Every panel shows a 2*HALF_W x 2*HALF_H µm window, so all three share one
# scale; the window is only as large as the µCT brain (the tallest of the
# three) needs. Each window is framed on its own tissue (WINDOW_CENTER_UM, in
# the centred data coordinates the loaders return, measured off a render):
# the modalities are not registered to one another, so this moves nothing
# relative to anything else.
HALF_W_UM = 1400.0
HALF_H_UM = 1980.0
WINDOW_CENTER_UM = {
    "uct": (160.0, 310.0),
    "xenium": (610.0, -58.0),
    "maldi": (0.0, 0.0),  # load_maldi centres the map on its tissue
}
SCALEBAR_UM = 500.0

# ---- Display clipping -------------------------------------------------------
# The Xenium panel is clipped to these percentiles of its non-zero values.
CLIP_LO_PCT = 1.0
CLIP_HI_PCT = 99.0

# ---- MALDI display clipping -------------------------------------------------
# The MALDI panel does not share the Xenium percentiles: everything below vmin
# is left dark, so this panel shows the upper intensity range of the map the
# main MALDI figure draws in full (the caption says so).
MALDI_CLIP_LO_PCT = 60.0
MALDI_CLIP_HI_PCT = 99.0

# ---- Colors + style ---------------------------------------------------------
DPI = 300
BG = "#000000"  # pure black for plot areas
FG = "#000000"  # black text for white figure background
BAR_FG = "#FFFFFF"  # in-panel annotation (scale bar) on the black panels
FIG_BG = "#ffffff"  # white figure/margin background


def make_tk_cmap():
    colors = [
        (0, 0, 0),
        (0.15, 0.0, 0.2),
        (0.55, 0.05, 0.4),
        (0.9, 0.25, 0.35),
        (1.0, 0.7, 0.2),
        (1.0, 1.0, 0.9),
    ]
    return LinearSegmentedColormap.from_list("tk", colors)


TK_CMAP = make_tk_cmap()


# =============================================================================
# ORIENTATION
# =============================================================================
def rotate_xy(x, y, rot90=0, rot_deg=0.0, flip_x=False, flip_y=False):
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    for _ in range(rot90 % 4):
        x, y = -y, x
    if rot_deg:
        theta = np.deg2rad(rot_deg)
        c, s = np.cos(theta), np.sin(theta)
        x, y = c * x - s * y, s * x + c * y
    return x, y


# =============================================================================
# DATA LOADERS  (always return data centered at (0,0) in µm)
# =============================================================================
def load_xenium():
    """Return (x_um, y_um, tk_vals), centered at (0,0). Converts px→µm."""
    print(f"Loading Xenium {XENIUM_H5AD.name}...")
    a = ad.read_h5ad(XENIUM_H5AD)
    coords_px = a.obsm["spatial"]  # in PIXELS
    x = coords_px[:, 0] * XENIUM_PIXEL_UM  # → µm
    y = coords_px[:, 1] * XENIUM_PIXEL_UM
    gene_idx = list(a.var_names).index(XENIUM_TK_VAR)
    X = a.X
    vals = (
        X[:, gene_idx].toarray().flatten()
        if hasattr(X, "toarray")
        else np.asarray(X[:, gene_idx]).flatten()
    )
    # Center on the median of Tk-expressing cells (density peak of the brain)
    mask = vals > 0
    cx = float(np.median(x[mask])) if mask.any() else float(np.median(x))
    cy = float(np.median(y[mask])) if mask.any() else float(np.median(y))
    print(
        f"  n_cells={len(vals)}, Tk>0: {int(mask.sum())}, "
        f"px-span x={coords_px[:, 0].max() - coords_px[:, 0].min():.0f} "
        f"→ {XENIUM_PIXEL_UM * (coords_px[:, 0].max() - coords_px[:, 0].min()):.0f} "
        "µm, "
        f"center=({cx:.0f}, {cy:.0f}) µm"
    )
    return x - cx, y - cy, vals


def load_maldi():
    """Worker Tk2 map from bimp_st.maldi, centred on its tissue bounding box.
    Returns (map with NaN off tissue, extent_um)."""
    img, ext, column = maldi.peptide_map("worker", "Tk")
    # plot_maldi draws row 0 at the low-y edge of an ascending axis; flip so the
    # map reads the same way up as in the main MALDI figure.
    img = np.flipud(img)
    rows, cols = np.where(np.isfinite(img))
    um = maldi.BIN_PX * maldi.UM_PER_PX
    cy_px = (rows.min() + rows.max() + 1) / 2
    cx_px = (cols.min() + cols.max() + 1) / 2
    h_um, w_um = img.shape[0] * um, img.shape[1] * um
    extent = (-cx_px * um, w_um - cx_px * um, -cy_px * um, h_um - cy_px * um)
    print(f"  MALDI {column}: map {img.shape}, {int(np.isfinite(img).sum())} bins")
    return img, extent


def load_microct_mip():
    """Return (rgba, extent_um) — depth-attenuated axial MIP of channel 0.

    For each (y,x) pixel, takes the max along Z and tints by the depth of
    that max voxel. Surface-level bright features stay opaque; deep-interior
    maxima fade slightly → visual 3D cue from a top-down view.
    """
    print(f"Loading microCT {MICROCT_TIF.name} (full resolution)...")
    with tifffile.TiffFile(str(MICROCT_TIF)) as tif:
        vol = tif.series[0].asarray(out="memmap")  # (Z, C, Y, X)
    ch0 = np.asarray(vol[:, 0])  # (Z, Y, X)

    mip = ch0.max(axis=0).astype(np.float32)  # max projection
    z_of_max = ch0.argmax(axis=0).astype(np.float32)  # depth of max
    nz = ch0.shape[0]

    # Depth attenuation: surface (z=0) → α=1.0; deepest (z=nz) → α=0.4
    depth_norm = z_of_max / nz
    attenuation = 1.0 - 0.6 * depth_norm  # 1.0 → 0.4

    # Horizontal flip
    mip = np.fliplr(mip)
    attenuation = np.fliplr(attenuation)

    # Center on non-zero bbox
    mask = mip > 0
    if mask.any():
        rows, cols = np.where(mask)
        cy_px = (rows.min() + rows.max()) / 2
        cx_px = (cols.min() + cols.max()) / 2
    else:
        cy_px, cx_px = mip.shape[0] / 2, mip.shape[1] / 2
    w_um = mip.shape[1] * MICROCT_VOXEL_UM
    h_um = mip.shape[0] * MICROCT_VOXEL_UM
    extent_um = (
        -cx_px * MICROCT_VOXEL_UM,
        w_um - cx_px * MICROCT_VOXEL_UM,
        -cy_px * MICROCT_VOXEL_UM,
        h_um - cy_px * MICROCT_VOXEL_UM,
    )
    print(
        f"  shape={mip.shape} @ {MICROCT_VOXEL_UM:.2f} µm, MIP "
        f"range=[{mip.min():.0f}, {mip.max():.0f}]"
    )
    return mip, attenuation, extent_um


# =============================================================================
# PLOTTING
# =============================================================================
def setup_panel(ax, key):
    """Fix the shared-size window, framed on this modality's tissue. The
    modality name is drawn as the panel tag (add_panel_headers in main())."""
    cx, cy = WINDOW_CENTER_UM[key]
    ax.set_xlim(cx - HALF_W_UM, cx + HALF_W_UM)
    ax.set_ylim(cy - HALF_H_UM, cy + HALF_H_UM)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FG)


def add_scalebar(ax, length_um=SCALEBAR_UM, color=BAR_FG):
    # Bottom-left of the window; the panels are black, so the bar is light.
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    sx = x0 + 0.07 * (x1 - x0)
    sy = y0 + 0.04 * (y1 - y0)
    ax.plot(
        [sx, sx + length_um],
        [sy, sy],
        color=color,
        linewidth=2,
        solid_capstyle="butt",
        zorder=10,
    )
    ax.text(
        sx + length_um / 2,
        sy + 0.015 * (y1 - y0),
        f"{length_um:.0f} µm",
        color=color,
        fontsize=6.5,
        ha="center",
        va="bottom",
        zorder=10,
    )


def add_colorbar(fig, cax, cmap, vmin, vmax, label):
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    cb.ax.tick_params(colors=FG, labelsize=6.5, length=2, pad=1.5)
    cb.set_label(label, color=FG, fontsize=7, labelpad=2)
    cb.outline.set_edgecolor(FG)
    cb.outline.set_linewidth(0.5)


def plot_microct(ax, img, attenuation, extent_um, vmin=50, vmax=250):
    """Plot depth-attenuated µCT MIP — surface features opaque, deep ones faded."""
    cmap = plt.get_cmap("bone")
    normed = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(normed)
    # Apply depth attenuation to alpha channel (surface opaque, deep faded)
    rgba[..., 3] = np.where(img > vmin, attenuation, 0.0)
    ax.imshow(
        rgba,
        interpolation="bilinear",
        extent=(extent_um[0], extent_um[1], extent_um[3], extent_um[2]),
        origin="upper",
    )


def plot_xenium(ax, x, y, vals):
    # Draw dim zeros first, then brighter positives on top
    order = np.argsort(vals)
    x, y, v = x[order], y[order], vals[order]
    pos = v[v > 0]
    if len(pos):
        vmin = float(np.percentile(pos, CLIP_LO_PCT))
        vmax = float(np.percentile(pos, CLIP_HI_PCT))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    print(
        f"  Xenium display: {len(pos):,} Tk+ cells, "
        f"clim=[{vmin:.2f}, {vmax:.2f}], max={v.max():.0f}"
    )
    ax.scatter(
        x,
        y,
        c=v,
        cmap=TK_CMAP,
        s=0.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
    return vmin, vmax


def plot_maldi(ax, mip, extent, cmap):
    """Render the MALDI map — viridis + clim does all the work.
    No alpha tricks: viridis minimum is dark purple (near-invisible on dark BG),
    viridis maximum is bright yellow (focal MB Tk expression).
    """
    # Clip to MALDI_CLIP_LO_PCT/MALDI_CLIP_HI_PCT of the non-zero values
    # (see the MALDI display-clipping note above).
    nz = mip[np.isfinite(mip) & (mip > 0)]
    if nz.size > 0:
        vmin = float(np.percentile(nz, MALDI_CLIP_LO_PCT))
        vmax = float(np.percentile(nz, MALDI_CLIP_HI_PCT))
    else:
        vmin, vmax = 0.0, float(mip.max())

    cmap_copy = cmap.copy()
    cmap_copy.set_bad(BG)
    mip_masked = np.ma.masked_where(~(mip >= vmin), mip)

    n_above = int((mip >= vmin).sum())
    n_tissue = int(nz.size)
    print(
        f"  MALDI display: {n_tissue} tissue bins, {n_above} above vmin={vmin:.2f}, "
        f"clim=[{vmin:.2f}, {vmax:.2f}], max={np.nanmax(mip):.2f}"
    )

    ax.imshow(
        mip_masked,
        cmap=cmap_copy,
        vmin=vmin,
        vmax=vmax,
        extent=(extent[0], extent[1], extent[3], extent[2]),
        origin="upper",
        interpolation="nearest",
    )
    return vmin, vmax


# =============================================================================
# MAIN
# =============================================================================
def main():
    xen_x, xen_y, xen_vals = load_xenium()
    mal_mip, mal_extent = load_maldi()
    ct_img, ct_atten, ct_extent = load_microct_mip()

    # Use viridis for MALDI (matches collaborator's napari style)
    MALDI_CMAP = plt.get_cmap("viridis")

    # Fixed layout (inches): three equal panels of the shared window aspect,
    # each with a colorbar slot underneath (left empty under µCT), so every
    # panel is drawn at the same size and the 500 µm bars match on the page.
    W, M, G = 7.2, 0.05, 0.08
    pw = (W - 2 * M - 2 * G) / 3
    ph = pw * HALF_H_UM / HALF_W_UM
    HDR, CB_GAP, CB_H, CB_LAB = 0.24, 0.07, 0.07, 0.36
    H = HDR + ph + CB_GAP + CB_H + CB_LAB
    fig = plt.figure(figsize=(W, H), facecolor=FIG_BG)
    axes, caxes = [], []
    for i in range(3):
        x = M + i * (pw + G)
        axes.append(fig.add_axes([x / W, 1 - (HDR + ph) / H, pw / W, ph / H]))
        caxes.append(
            fig.add_axes(
                [
                    (x + 0.12 * pw) / W,
                    1 - (HDR + ph + CB_GAP + CB_H) / H,
                    0.76 * pw / W,
                    CB_H / H,
                ]
            )
        )
    caxes[0].set_visible(False)

    # --- Panel 1: µCT (axial MIP, contrast 50-250) ---
    plot_microct(axes[0], ct_img, ct_atten, ct_extent, vmin=0, vmax=255)
    setup_panel(axes[0], "uct")
    add_scalebar(axes[0])

    # --- Panel 2: Xenium Tk ---
    xen_vmin, xen_vmax = plot_xenium(axes[1], xen_x, xen_y, xen_vals)
    setup_panel(axes[1], "xenium")
    # Not raw counts: aligned_clustered.h5ad carries uns['log1p'], so X is
    # normalized + log1p (positives run ~3.5-7), not transcripts per cell.
    add_colorbar(
        fig, caxes[1], TK_CMAP, xen_vmin, xen_vmax, "normalized expression (log1p)"
    )
    add_scalebar(axes[1])

    # --- Panel 3: MALDI Tk v3 ---
    mal_vmin, mal_vmax = plot_maldi(axes[2], mal_mip, mal_extent, MALDI_CMAP)
    setup_panel(axes[2], "maldi")
    add_colorbar(fig, caxes[2], MALDI_CMAP, mal_vmin, mal_vmax, "intensity (a.u.)")
    add_scalebar(axes[2])

    # --- Sanity check: axes are identical ---
    # Same window size and the same drawn size are both needed for the scale
    # bars to agree on the page.
    fig.canvas.draw()
    widths = []
    for ax, name in zip(axes, ["µCT", "Xenium", "MALDI"], strict=False):
        bb = ax.get_window_extent()
        widths.append(bb.width)
        print(
            f"  {name} xlim={ax.get_xlim()}  ylim={ax.get_ylim()}  "
            f"drawn={bb.width:.1f}x{bb.height:.1f} px"
        )
    if max(widths) - min(widths) > 1.0:
        print(
            f"  WARNING: panel widths differ by "
            f"{max(widths) - min(widths):.1f} px -> the three 500 µm scale "
            f"bars are NOT the same length on the page"
        )

    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [
            (axes[0], "A", "µCT"),
            (axes[1], "B", "Xenium, Tk mRNA"),
            (axes[2], "C", "MALDI, Tk peptide"),
        ],
        width_frac=1.0,
    )

    out = OUT_DIR / "figS_multimodal_tk.png"
    fig.savefig(out, dpi=DPI, facecolor=FIG_BG, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
