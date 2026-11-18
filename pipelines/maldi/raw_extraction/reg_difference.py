#!/usr/bin/env python
"""Sum-over-z intensity maps for sNPF_v1 & Tk2, register the two brains
(queen <- worker) rigidly (flip + rotation + translation via silhouette), and
show the caste difference as a diverging overlay, in the house figure style.

Run with bimp-st env (anndata, scipy, skimage).
"""

import os

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from skimage.registration import phase_cross_correlation

FIG_BG, FG, DPI = "#ffffff", "#000000", 300
UM_PER_PX, BIN_PX = 20.0, 2.0  # 40 µm bins
SCALEBAR_UM = 1000.0
FLOOR = 0.18  # display floor: hide relative intensity < this
VIR = plt.get_cmap("viridis").copy()
VIR.set_bad(FIG_BG)
# Diverging map ordered for a (queen − worker) metric: NEGATIVE (worker-enriched)
# = blue, POSITIVE (queen-enriched) = gold (house convention: gold = queen).
CASTE = LinearSegmentedColormap.from_list(
    "caste_div",
    ["#2A5FAA", "#4A90D9", "#C0D8F0", "#FFFFFF", "#F5E6C0", "#DAA520", "#B8860B"],
    N=256,
)
CASTE.set_bad(FIG_BG)
PEPTIDES = [("sNPF_v1", "sNPF v1"), ("Tk2", "Tk2")]
OUT = "ion_out/density"
os.makedirs(OUT, exist_ok=True)

# ---- common grid ----
data = {}
allc = []
for sid in ("princeton_80524", "princeton_80642"):
    a = ad.read_h5ad(f"ion_out/aligned_panel_{sid}.h5ad")
    c = np.asarray(a.obsm["spatial"])
    c = c - c.mean(0)
    data[sid] = (c, {p: a.obs[p].to_numpy(float) for p, _ in PEPTIDES})
    allc.append(c)
R = np.abs(np.concatenate(allc)).max() * 1.1
edges = np.arange(-R, R + BIN_PX, BIN_PX)


def rasterize(c, vals):
    s, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=[edges, edges], weights=vals)
    return s.T  # SUM over z (all spots in bin)


def occ(c):
    n, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=[edges, edges])
    return ndimage.gaussian_filter((n.T > 0).astype(float), 1.0)


imgs = {
    sid: {p: rasterize(c, v[p]) for p, _ in PEPTIDES} for sid, (c, v) in data.items()
}
occs = {sid: occ(c) for sid, (c, _) in data.items()}

# ---- register worker (80642) onto queen (80524) ----
q, w = "princeton_80524", "princeton_80642"
oq = occs[q]
best = None
for flip in (False, True):
    ow0 = occs[w][:, ::-1] if flip else occs[w]
    for ang in range(0, 360, 3):
        owr = ndimage.rotate(ow0, ang, reshape=False, order=1)
        sh, _, _ = phase_cross_correlation(oq, owr, upsample_factor=4)
        ows = ndimage.shift(owr, sh, order=1)
        a_, b_ = (oq > 0.2), (ows > 0.2)
        iou = (a_ & b_).sum() / max((a_ | b_).sum(), 1)
        if best is None or iou > best[0]:
            best = (iou, flip, ang, sh)
iou, flip, ang, sh = best
print(f"registration: flip={flip} angle={ang} shift={sh.round(1)} IoU={iou:.3f}")


def _xf(arr):  # rigid worker->queen transform (flip+rot+shift)
    a = arr[:, ::-1] if flip else arr
    return ndimage.shift(ndimage.rotate(a, ang, reshape=False, order=1), sh, order=1)


# EDGE-PRESERVING registration of the worker intensity: weight by the worker
# tissue mask and divide by the transported mask, so boundary values are NOT
# diluted by the zero background (this is what caused the spurious queen-gold rim).
wm0 = (occs[w] > 0.2).astype(float)
wm0_reg = _xf(wm0)
w_reg = {
    p: np.where(
        wm0_reg > 0.3,
        _xf(np.nan_to_num(imgs[w][p]) * wm0) / np.maximum(wm0_reg, 1e-9),
        np.nan,
    )
    for p, _ in PEPTIDES
}
occ_w_reg = _xf(occs[w])

# ---- display rotation (RIGID: rotation only, scale preserved -> scalebar valid) ----
ROT_DEG = -20.0  # negative = true CCW with origin="lower"
SMOOTH = 1.6  # bins; matched smoothing so queen/worker look equally smooth


def smooth_nan(img, sigma):
    m = np.isfinite(img)
    sa = ndimage.gaussian_filter(np.where(m, img, 0.0), sigma)
    sm = ndimage.gaussian_filter(m.astype(float), sigma)
    out = np.where(sm > 0.05, sa / np.maximum(sm, 1e-9), np.nan)
    out[~m] = np.nan
    return out


def disp(img, mask):
    # EDGE-PRESERVING display rotation: divide by the rotated mask so tissue
    # values aren't diluted at the boundary (identical treatment for both samples).
    ar = ndimage.rotate(np.nan_to_num(img) * mask, ROT_DEG, reshape=True, order=1)
    mr = ndimage.rotate(mask.astype(float), ROT_DEG, reshape=True, order=1)
    out = np.where(mr > 0.5, ar / np.maximum(mr, 1e-9), np.nan)
    return smooth_nan(out, SMOOTH)  # equal smoothing for both samples


# ---- figure: rows=peptide, cols=[queen, worker(reg), difference] ----
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(
    2, 3, figsize=(11.5, 8.4), facecolor=FIG_BG, constrained_layout=True
)
SB = SCALEBAR_UM / UM_PER_PX / BIN_PX  # scalebar length in bins (rotation-invariant)


def style(ax, title, shape, bar=False):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=FG, fontsize=11, fontweight="bold", pad=4)
    if bar:  # real 1000 µm bar, only on the queen (native-frame) panels
        x0, y0 = 0.07 * shape[1], 0.10 * shape[0]
        ax.plot([x0, x0 + SB], [y0, y0], color=FG, lw=2.5, solid_capstyle="butt")
        ax.text(
            x0 + SB / 2,
            y0 + 0.025 * shape[0],
            f"{SCALEBAR_UM:.0f} µm",
            color=FG,
            fontsize=7,
            ha="center",
            va="bottom",
        )


# tighter tissue masks: trim the sparse, dark-blue low-occupancy rim (esp. the
# left tail) so the displayed silhouettes are clean.
mq = ndimage.binary_erosion(oq > 0.45, iterations=1)
mw = ndimage.binary_erosion(occ_w_reg > 0.45, iterations=1)


def letter(ax, L):
    ax.text(
        0.02,
        0.98,
        L,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color=FG,
    )


for r, (pep, pshort) in enumerate(PEPTIDES):
    qi, wi = imgs[q][pep], w_reg[pep]
    # MSI best practice: per-spot TIC-norm (already) + Σz + per-sample MEDIAN scaling
    # -> "× sample median", a run-independent relative unit that IS comparable across
    # the two acquisitions, so a shared per-peptide color scale is valid.
    qd = disp(qi / np.median(qi[mq]), mq)
    wd = disp(wi / np.median(wi[mw]), mw)
    vmax = np.nanpercentile(
        np.concatenate([qd[np.isfinite(qd)], wd[np.isfinite(wd)]]), 99
    )
    for ci, (img, lab) in enumerate([(qd, "queen (80524)"), (wd, "worker (80642)")]):
        ax = axes[r, ci]
        di = img.copy()
        di[di < FLOOR * vmax] = np.nan  # raised floor
        im = ax.imshow(
            di,
            origin="lower",
            cmap=VIR,
            vmin=FLOOR * vmax,
            vmax=vmax,
            aspect="equal",
            interpolation="bilinear",
        )
        style(ax, f"{pshort} — {lab}", img.shape, bar=True)
        letter(ax, chr(65 + r * 3 + ci))
        if ci == 1:
            cb = fig.colorbar(im, ax=axes[r, :2], fraction=0.04, pad=0.02)
            cb.set_label(
                f"{pshort}  (× sample median, TIC-norm.)", color=FG, fontsize=8
            )
            cb.ax.tick_params(colors=FG, labelsize=7)
            cb.outline.set_edgecolor(FG)
    # difference of the comparable median-normalized maps (run-independent), on the
    # eroded intersection of the two silhouettes.
    # the two brains differ in size/shape -> a rigid overlay can't match the rim,
    # so compare only the well-overlapping INTERIOR (erode the intersection) and
    # feather the edge to match the other maps.
    inter = ndimage.binary_erosion(np.isfinite(qd) & np.isfinite(wd), iterations=6)
    shared = ndimage.gaussian_filter(inter.astype(float), 2.0) > 0.5
    dd = np.where(shared, np.nan_to_num(qd) - np.nan_to_num(wd), np.nan)
    dd = smooth_nan(dd, 1.0)
    vl = np.nanpercentile(np.abs(dd[np.isfinite(dd)]), 98)
    axd = axes[r, 2]
    imd = axd.imshow(
        dd,
        origin="lower",
        cmap=CASTE,
        vmin=-vl,
        vmax=vl,
        aspect="equal",
        interpolation="bilinear",
    )
    style(axd, f"{pshort} — difference", dd.shape, bar=True)
    letter(axd, chr(65 + r * 3 + 2))
    cb = fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.02)
    cb.set_label("queen − worker  (× median)", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=7)
    cb.outline.set_edgecolor(FG)

fig.savefig(
    f"{OUT}/density_diff_manuscript.png", dpi=DPI, facecolor=FIG_BG, bbox_inches="tight"
)
print(
    f"wrote {OUT}/density_diff_manuscript.png (rot {ROT_DEG}deg CCW, per-map relative)"
)
