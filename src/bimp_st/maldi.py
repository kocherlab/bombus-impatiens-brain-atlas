"""MALDI-MSI peptide maps: the one loader every MALDI panel reads.

Pipeline: the
Princeton extraction takes the most intense centroided peak within +/-15 ppm of
each readout mass per pixel and divides by the pixel's total ion count; PASTE3
aligns each brain's sections into ``aligned_panel_princeton_<sample>.h5ad``
(``pipelines/maldi/raw_extraction``). Here the aligned pixels are
summed over sections in 40 um bins, the worker brain is placed on the queen
frame by a pinned rigid transform, each map is divided by its median over
tissue, rotated upright for display and smoothed with a one-bin Gaussian.

Readouts are the single-peak columns the figures use: ``Tk2`` (mature amidated
tachykinin, [M+H]+ m/z 993.4757) and ``sNPF_v1`` ([M+H]+ m/z 974.5891).

MALDI is qualitative here: these maps show where a peptide is, not how
much of it a caste has.
"""

from __future__ import annotations

import numpy as np

from bimp_st.paths import REPO

PRINCETON_DIR = REPO / "pipelines/maldi/raw_extraction/ion_out"
SAMPLES = {"queen": "80524", "worker": "80642"}

#: Readout column of the aligned panel for each peptide, and its [M+H]+ mass.
PEPTIDE_COLUMN = {"Tk": "Tk2", "sNPF": "sNPF_v1"}
PEPTIDE_MZ = {"Tk2": 993.4757, "sNPF_v1": 974.5891}

UM_PER_PX = 20.0  # acquisition pitch
BIN_PX = 2.0  # display bins of 2 x 2 pixels = 40 um

# Worker -> queen rigid placement. Thaw-mounted sections are not mirrored, so the
# worker map is only rotated (then shifted by phase correlation, as before).
# The angle is the best unmirrored silhouette overlap; registration_search()
# re-runs that search and princeton_maps() asserts it still lands here.
WORKER_FLIP = False
WORKER_ANGLE_DEG = 33
ANGLE_STEP_DEG = 3  # search grid of the assertion

DISPLAY_ROT_DEG = 250.0  # upright display, matching the Xenium panels

# Mask thresholds, all on 0-1 smoothed occupancy maps of the 40 um grid.
OCC_SIGMA = 1.0  # Gaussian (bins) applied to the binary pixel occupancy
IOU_LEVEL = 0.2  # occupancy level defining a silhouette for the overlap score
WARP_KEEP = 0.3  # keep a warped worker bin if its tissue weight is above this
TISSUE_CORE = 0.45  # occupancy level for the median-normalization tissue mask
DISPLAY_KEEP = 0.5  # keep a display-rotated bin if its mask weight is above this
SMOOTH_SUPPORT = 0.05  # NaN-aware smoothing keeps bins with this much support

_CACHE: dict = {}


def load_panel(sample_key: str):
    """(centred pixel coordinates, obs) of one brain's aligned panel."""
    import anndata as ad

    path = PRINCETON_DIR / f"aligned_panel_princeton_{SAMPLES[sample_key]}.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: ion_out/ is gitignored; link it from the main "
            "checkout (pipelines/maldi/raw_extraction/README.md; "
            "md5s in ion_out.md5)"
        )
    a = ad.read_h5ad(path)
    c = np.asarray(a.obsm["spatial"])
    return c - c.mean(0), a.obs


def smooth_nan(im, sigma):
    """Gaussian smoothing that ignores NaN (off-tissue) bins."""
    from scipy import ndimage

    m = np.isfinite(im)
    sm = ndimage.gaussian_filter(m.astype(float), sigma)
    out = np.where(
        sm > SMOOTH_SUPPORT,
        ndimage.gaussian_filter(np.where(m, im, 0.0), sigma) / np.maximum(sm, 1e-9),
        np.nan,
    )
    out[~m] = np.nan
    return out


def _grids():
    """Shared 40 um grid, per-brain pixel coordinates and smoothed occupancy."""
    from scipy import ndimage

    if "grid" in _CACHE:
        return _CACHE["grid"]
    coords, obs = {}, {}
    for key in SAMPLES:
        coords[key], obs[key] = load_panel(key)
    R = np.abs(np.concatenate(list(coords.values()))).max() * 1.1
    edges = np.arange(-R, R + BIN_PX, BIN_PX)

    def occ(c):
        hist = np.histogram2d(c[:, 0], c[:, 1], bins=[edges, edges])[0].T > 0
        return ndimage.gaussian_filter(hist.astype(float), OCC_SIGMA)

    occs = {k: occ(c) for k, c in coords.items()}
    _CACHE["grid"] = (edges, coords, obs, occs)
    return _CACHE["grid"]


def _place(oq, ow, flip, ang):
    """Rotate (and optionally mirror) the worker occupancy, shift it onto the
    queen's by phase correlation; returns (iou, shift)."""
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    ow0 = ow[:, ::-1] if flip else ow
    owr = ndimage.rotate(ow0, ang, reshape=False, order=1)
    sh, _, _ = phase_cross_correlation(oq, owr, upsample_factor=4)
    ows = ndimage.shift(owr, sh, order=1)
    a, b = oq > IOU_LEVEL, ows > IOU_LEVEL
    return (a & b).sum() / max((a | b).sum(), 1), sh


def registration_search():
    """Silhouette IoU of every (flip, angle) on the ANGLE_STEP_DEG grid, best
    first: [(iou, flip, angle), ...]."""
    _, _, _, occs = _grids()
    res = [
        (float(_place(occs["queen"], occs["worker"], flip, ang)[0]), flip, ang)
        for flip in (False, True)
        for ang in range(0, 360, ANGLE_STEP_DEG)
    ]
    return sorted(res, key=lambda r: -r[0])


def check_registration():
    """Assert the pinned worker angle is still the best unmirrored placement,
    and print the mirrored runner-up (a near tie, so it is reported, not
    enforced)."""
    res = registration_search()
    unflipped = [r for r in res if not r[1]]
    mirrored = [r for r in res if r[1]]
    best = unflipped[0]
    print(
        f"  MALDI worker->queen: pinned {WORKER_ANGLE_DEG} deg (no mirror); "
        f"best unmirrored {best[2]} deg IoU {best[0]:.3f}; "
        f"best mirrored {mirrored[0][2]} deg IoU {mirrored[0][0]:.3f}",
        flush=True,
    )
    assert best[2] == WORKER_ANGLE_DEG and not WORKER_FLIP, (
        f"pinned worker registration ({WORKER_ANGLE_DEG} deg) is no longer the "
        f"best unmirrored placement ({best[2]} deg); re-check before re-pinning"
    )


def princeton_maps(column: str):
    """Upright, median-normalized maps of one readout column for both brains.

    Returns (queen_map, worker_map, vmax, extent_um): NaN off tissue, the worker
    placed on the queen frame, vmax the 99th percentile over both, extent
    (0, width, 0, height) in micrometres. Cached per column.
    """
    from scipy import ndimage

    if column in _CACHE:
        return _CACHE[column]
    if "checked" not in _CACHE:
        check_registration()
        _CACHE["checked"] = True
    edges, coords, obs, occs = _grids()

    def rast(c, v):
        return np.histogram2d(c[:, 0], c[:, 1], bins=[edges, edges], weights=v)[0].T

    qi = rast(coords["queen"], obs["queen"][column].to_numpy(float))
    wi = rast(coords["worker"], obs["worker"][column].to_numpy(float))
    oq, ow = occs["queen"], occs["worker"]
    _, sh = _place(oq, ow, WORKER_FLIP, WORKER_ANGLE_DEG)

    def _xf(a):
        a = a[:, ::-1] if WORKER_FLIP else a
        rotated = ndimage.rotate(a, WORKER_ANGLE_DEG, reshape=False, order=1)
        return ndimage.shift(rotated, sh, order=1)

    wm0 = (ow > IOU_LEVEL).astype(float)
    wm0r = _xf(wm0)
    wi = np.where(
        wm0r > WARP_KEEP, _xf(np.nan_to_num(wi) * wm0) / np.maximum(wm0r, 1e-9), np.nan
    )
    ow = _xf(ow)
    mq = ndimage.binary_erosion(oq > TISSUE_CORE, iterations=1)
    mw = ndimage.binary_erosion(ow > TISSUE_CORE, iterations=1)

    def disp(im, mask):
        ar = ndimage.rotate(
            np.nan_to_num(im) * mask, DISPLAY_ROT_DEG, reshape=True, order=1
        )
        mr = ndimage.rotate(mask.astype(float), DISPLAY_ROT_DEG, reshape=True, order=1)
        return smooth_nan(
            np.where(mr > DISPLAY_KEEP, ar / np.maximum(mr, 1e-9), np.nan), 1.0
        )

    qd = disp(qi / np.median(qi[mq]), mq)
    wd = disp(wi / np.median(wi[mw]), mw)
    vmax = np.nanpercentile(
        np.concatenate([qd[np.isfinite(qd)], wd[np.isfinite(wd)]]), 99
    )
    H, W = qd.shape
    um = BIN_PX * UM_PER_PX
    ext = (0.0, W * um, 0.0, H * um)
    _CACHE[column] = (qd, wd, float(vmax), ext)
    return _CACHE[column]


def peptide_map(sample_key: str, peptide: str):
    """(map, extent_um, column) of one brain for 'Tk' or 'sNPF'."""
    column = PEPTIDE_COLUMN[peptide]
    qd, wd, _, ext = princeton_maps(column)
    return (qd if sample_key == "queen" else wd), ext, column
