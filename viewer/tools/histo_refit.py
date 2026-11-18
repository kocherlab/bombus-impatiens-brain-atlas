"""Per-section registration of Xenium DAPI pages onto the current atlas frame.

Why this exists
---------------
``export_histology.py`` assumes one constant shift per brain
(``image_px = spatial_aligned + real_center``). That holds only when the
``aligned_images.tif`` pages and the atlas coordinates come from the same PASTE3
run. The worker's stack was written 2026-02-02 and its atlas re-aligned
2026-04-08, so every page carries February's per-section rigid transform while
the cells carry April's; PASTE3's transforms were never saved. This module fits,
for each page separately, the rigid map from the April aligned frame to that
page's pixel frame and resamples the page into the (shifted) aligned frame, so
everything downstream (mask, crop, ``bounds``) is unchanged.

Model per section ``image_px = R(theta) (a - mu) + mu + rc + delta`` with
``a`` the atlas ``spatial_aligned`` position (full-resolution px), ``mu`` the
section's mean cell position, ``rc`` the brain's ``real_center`` (from slice 0)
and ``(theta, delta)`` the fitted rotation and residual shift. Fit by an angle
sweep of FFT cross-correlation between a cell-density raster and the page's
log-DAPI (coarse over the full circle, then fine, sub-pixel by parabola).
Independent residuals come from tiling the section after the fit and
cross-correlating each tile (see ``local_residuals``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

PIXEL_UM = 0.2125
WORKERS = 4


def rot(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------------
# raw crop -> atlas (exact point fit)
# ----------------------------------------------------------------------------


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> dict:
    """Kabsch/Procrustes ``dst = R (src - ms) + md``; reports det, scale and
    residuals."""
    ms, md = src.mean(0), dst.mean(0)
    P, Q = src - ms, dst - md
    U, S, Vt = np.linalg.svd(P.T @ Q)
    R = Vt.T @ U.T
    pred = P @ R.T + md
    res = np.linalg.norm(pred - dst, axis=1)
    return dict(
        theta_deg=float(math.degrees(math.atan2(R[1, 0], R[0, 0]))),
        det=float(np.linalg.det(R)),
        scale=float(S.sum() / max((P**2).sum(), 1e-12)),
        src_centre=ms.tolist(),
        dst_centre=md.tolist(),
        resid_median_px=float(np.median(res)),
        resid_p95_px=float(np.percentile(res, 95)),
        resid_p95_um=float(np.percentile(res, 95) * PIXEL_UM),
        n=int(len(src)),
    )


# ----------------------------------------------------------------------------
# cross-correlation machinery
# ----------------------------------------------------------------------------


def prep_image(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Zero-mean smoothed log-DAPI, zero where the page is canvas padding.

    Normalised convolution (blur of the data over blur of the validity mask), so
    the padding edge does not bleed a dark halo into the tissue.
    """
    valid = (img > 0).astype(np.float32)
    num = ndi.gaussian_filter(np.log1p(img) * valid, sigma_px)
    den = ndi.gaussian_filter(valid, sigma_px)
    im = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    v = valid > 0
    if v.any():
        im[v] -= im[v].mean()
    im[~v] = 0
    return im


def density(
    xy: np.ndarray,
    theta: float,
    mu: np.ndarray,
    rc: np.ndarray,
    f: int,
    shape: tuple[int, int],
    sigma_px: float,
) -> np.ndarray:
    """Cell-density raster of atlas cells moved by rotation ``theta`` about ``mu``."""
    p = (xy - mu) @ rot(theta).T + mu + rc
    ix = np.floor((p[:, 0] + 0.5) / f).astype(np.int64)
    iy = np.floor((p[:, 1] + 0.5) / f).astype(np.int64)
    h, w = shape
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    d = (
        np.bincount(iy[ok] * w + ix[ok], minlength=h * w)
        .reshape(h, w)
        .astype(np.float32)
    )
    return ndi.gaussian_filter(d, sigma_px) if sigma_px > 0 else d


class Correlator:
    """Cross-correlation of fixed image ``img`` against moving rasters, via FFT.

    ``surface(D, win)[dr+win, dc+win] = sum_x img[x + (dr, dc)] * D[x]``: the score of
    shifting the density by ``(dr, dc)`` working pixels onto the image.
    """

    def __init__(self, img: np.ndarray):
        self.shape = img.shape
        self.P = tuple(sfft.next_fast_len(2 * n) for n in img.shape)
        self.FI = sfft.rfft2(img, s=self.P, workers=WORKERS)

    def surface(self, D: np.ndarray, win: int) -> np.ndarray:
        FD = sfft.rfft2(D, s=self.P, workers=WORKERS)
        c = sfft.irfft2(self.FI * np.conj(FD), s=self.P, workers=WORKERS)
        ri = np.arange(-win, win + 1) % self.P[0]
        ci = np.arange(-win, win + 1) % self.P[1]
        return c[np.ix_(ri, ci)]


def peak(surf: np.ndarray) -> tuple[float, float, float, bool]:
    """(dr, dc, value, on_edge) of the maximum, sub-pixel by parabola per axis."""
    k = np.unravel_index(int(np.argmax(surf)), surf.shape)
    off = []
    for ax in (0, 1):
        lo, hi = list(k), list(k)
        lo[ax] -= 1
        hi[ax] += 1
        if lo[ax] < 0 or hi[ax] >= surf.shape[ax]:
            off.append(0.0)
            continue
        y0, y1, y2 = surf[tuple(lo)], surf[k], surf[tuple(hi)]
        den = y0 - 2 * y1 + y2
        off.append(float(0.5 * (y0 - y2) / den) if den != 0 else 0.0)
    c = surf.shape[0] // 2, surf.shape[1] // 2
    edge = bool(k[0] in (0, surf.shape[0] - 1) or k[1] in (0, surf.shape[1] - 1))
    return k[0] + off[0] - c[0], k[1] + off[1] - c[1], float(surf[k]), edge


def block_down(img: np.ndarray, k: int) -> np.ndarray:
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    return img[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _parabola(xs, ys) -> float:
    """Vertex of the parabola through the best sample and its two neighbours."""
    j = int(np.argmax(ys))
    if j == 0 or j == len(ys) - 1:
        return float(xs[j])
    y0, y1, y2 = ys[j - 1], ys[j], ys[j + 1]
    den = y0 - 2 * y1 + y2
    return (
        float(xs[j] + (0.5 * (y0 - y2) / den) * (xs[1] - xs[0]))
        if den != 0
        else float(xs[j])
    )


# ----------------------------------------------------------------------------
# which page is which section
# ----------------------------------------------------------------------------


def screen_scores(
    pages: dict,
    sec_xy: list,
    rc: np.ndarray,
    f: int,
    step: float = 2.0,
    blur_um: float = 40.0,
    search_px: float = 2000.0,
    k: int = 9,
) -> np.ndarray:
    """Coarse match score of every page against every section's cells.

    ``pages`` maps page index -> working-resolution page. Returns ``S[p, s]``: the
    best (over rotation and shift) correlation of page ``p`` with the cell density
    of section ``s``, divided by that density's norm so sections with more cells do
    not win by size. Heavily blurred and at ~``f*k*0.2125`` um/px, so the whole
    page x section grid costs minutes.
    """
    fc = f * k
    sig = blur_um / (PIXEL_UM * fc)
    keys = list(pages)
    ims = [prep_image(block_down(pages[p], k), sig) for p in keys]
    shape = ims[0].shape
    P = tuple(sfft.next_fast_len(2 * n) for n in shape)
    FI = np.stack([sfft.rfft2(im, s=P, workers=WORKERS) for im in ims])
    win = int(math.ceil(search_px / fc))
    ri = (np.arange(-win, win + 1) % P[0])[:, None]
    ci = (np.arange(-win, win + 1) % P[1])[None, :]
    thetas = np.arange(-180.0, 180.0, step)
    S = np.zeros((len(keys), len(sec_xy)))
    for s, xy in enumerate(sec_xy):
        mu = xy.mean(0)
        best = np.full(len(keys), -np.inf)
        for th in thetas:
            FD = sfft.rfft2(
                density(xy, th, mu, rc, fc, shape, sig), s=P, workers=WORKERS
            )
            c = sfft.irfft2(FI * np.conj(FD)[None], s=P, axes=(-2, -1), workers=WORKERS)
            best = np.maximum(best, c[:, ri, ci].reshape(len(keys), -1).max(axis=1))
        S[:, s] = best / max(
            np.linalg.norm(density(xy, 0.0, mu, rc, fc, shape, sig)), 1e-9
        )
    return S


# ----------------------------------------------------------------------------
# the per-section fit
# ----------------------------------------------------------------------------


def fit_page(
    img9: np.ndarray,
    xy: np.ndarray,
    rc: np.ndarray,
    f: int,
    search_px: float = 2000.0,
    coarse_step: float = 1.5,
) -> dict:
    """Fit one page. ``img9`` is the working-resolution page (block ``f``)."""
    mu = xy.mean(0)
    k = 3
    fc = f * k

    # ---- A: coarse, heavy blur, full circle
    ic = block_down(img9, k)
    sig_c = 30.0 / (PIXEL_UM * fc)
    Ic = prep_image(ic, sig_c)
    cc = Correlator(Ic)
    win_c = int(math.ceil(search_px / fc))
    thetas = np.arange(-180.0, 180.0, coarse_step)
    sc = np.empty(len(thetas))
    for j, th in enumerate(thetas):
        D = density(xy, th, mu, rc, fc, Ic.shape, sig_c)
        sc[j] = peak(cc.surface(D, win_c))[2]
    jb = int(np.argmax(sc))
    th0 = float(thetas[jb])
    far = np.abs(((thetas - th0 + 180) % 360) - 180) > 4.0
    second = float(sc[far].max())
    coarse_sharp = float(sc[jb] / second) if second > 0 else float("inf")

    # ---- B: working resolution, +-3 deg then +-0.75 deg
    img = prep_image(img9, 1.0)
    cw = Correlator(img)
    win = int(math.ceil(search_px / f))

    def sweep(centre, half, step):
        ts = np.arange(centre - half, centre + half + 1e-9, step)
        ys = np.array(
            [
                peak(cw.surface(density(xy, t, mu, rc, f, img.shape, 1.0), win))[2]
                for t in ts
            ]
        )
        return ts, ys

    ts, ys = sweep(th0, 3.0, 0.5)
    th1 = _parabola(ts, ys)
    ts2, ys2 = sweep(th1, 0.75, 0.15)
    th2 = _parabola(ts2, ys2)
    surf = cw.surface(density(xy, th2, mu, rc, f, img.shape, 1.0), win)
    dr, dc, val, edge = peak(surf)
    delta = np.array([dc, dr]) * f  # full-resolution px, (x, y)
    return dict(
        theta_deg=float(th2),
        delta_px=delta.tolist(),
        delta_um=(delta * PIXEL_UM).tolist(),
        mu=mu.tolist(),
        coarse_theta_deg=th0,
        coarse_sharpness=coarse_sharp,
        peak_on_window_edge=edge,
        peak_value=val,
    )


def warp_to_frame(img9: np.ndarray, fit: dict, rc: np.ndarray, f: int) -> np.ndarray:
    """Resample a page into the atlas-aligned frame (+rc), same canvas as the page."""
    mu = np.asarray(fit["mu"])
    R = rot(fit["theta_deg"])
    delta = np.asarray(fit["delta_px"])
    kk = mu + rc

    def src(c, r):  # working px in the output canvas -> working px in the page
        pout = np.array([(c + 0.5) * f - 0.5, (r + 0.5) * f - 0.5])
        pin = R @ (pout - kk) + kk + delta
        return (pin + 0.5) / f - 0.5

    o = src(0, 0)
    a = src(1, 0) - o  # per output column
    b = src(0, 1) - o  # per output row
    mat = np.array([[b[1], a[1]], [b[0], a[0]]])
    return ndi.affine_transform(
        img9,
        mat,
        offset=[o[1], o[0]],
        output_shape=img9.shape,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


# ----------------------------------------------------------------------------
# independent residuals
# ----------------------------------------------------------------------------


def local_residuals(
    warped: np.ndarray,
    xy: np.ndarray,
    rc: np.ndarray,
    f: int,
    tile_um: float = 300.0,
    search_um: float = 60.0,
    min_cells: int = 150,
    hp_um: float = 60.0,
) -> dict:
    """Residual shift of ~300 um tiles of the warped page against the April cells.

    Both fields are high-passed so tissue-scale contrast, not brightness, drives
    the match; a section registered rigidly shows ~0 everywhere, a wrong rotation
    or a bent section shows a pattern. Shifts are image minus cells, in um.
    """
    px_um = PIXEL_UM * f
    img = prep_image(warped, 1.0)
    D = density(xy, 0.0, np.zeros(2), rc, f, img.shape, 1.0)
    s_hp = hp_um / px_um
    valid = (warped > 0).astype(np.float32)
    Ih = img - ndi.gaussian_filter(img, s_hp)
    Dh = D - ndi.gaussian_filter(D, s_hp)
    Ih *= valid
    T = max(8, int(round(tile_um / px_um)))
    s = max(3, int(round(search_um / px_um)))
    p = np.floor((xy + rc + 0.5) / f).astype(np.int64)
    r_lo, r_hi = max(0, p[:, 1].min()), min(img.shape[0], p[:, 1].max() + 1)
    c_lo, c_hi = max(0, p[:, 0].min()), min(img.shape[1], p[:, 0].max() + 1)
    Ip = np.pad(Ih, s)
    out = []
    for r0 in range(r_lo, r_hi - T // 2, T):
        for c0 in range(c_lo, c_hi - T // 2, T):
            r1, c1 = min(r0 + T, img.shape[0]), min(c0 + T, img.shape[1])
            inside = (p[:, 1] >= r0) & (p[:, 1] < r1) & (p[:, 0] >= c0) & (p[:, 0] < c1)
            if inside.sum() < min_cells or valid[r0:r1, c0:c1].mean() < 0.7:
                continue
            Dt = Dh[r0:r1, c0:c1]
            It = Ip[r0 : r1 + 2 * s, c0 : c1 + 2 * s]
            if It.shape[0] < Dt.shape[0] + 2 * s or It.shape[1] < Dt.shape[1] + 2 * s:
                continue
            corr = fftconvolve(It, Dt[::-1, ::-1], mode="valid")
            dr, dc, val, edge = peak(corr)
            sh = np.array([dc, dr]) * px_um
            out.append(
                dict(
                    r0=int(r0),
                    c0=int(c0),
                    n_cells=int(inside.sum()),
                    shift_um=[float(sh[0]), float(sh[1])],
                    on_edge=edge,
                )
            )
    if not out:
        return dict(n_tiles=0)
    sh = np.array([o["shift_um"] for o in out])
    mag = np.linalg.norm(sh, axis=1)
    use = np.array([not o["on_edge"] for o in out])
    ms = mag[use] if use.any() else mag
    return dict(
        n_tiles=len(out),
        n_tiles_at_search_edge=int((~use).sum()),
        tile_um=tile_um,
        search_um=search_um,
        median_um=float(np.median(ms)),
        rms_um=float(np.sqrt((ms**2).mean())),
        p90_um=float(np.percentile(ms, 90)),
        max_um=float(ms.max()),
        mean_shift_um=[
            float(sh[use, 0].mean()) if use.any() else 0.0,
            float(sh[use, 1].mean()) if use.any() else 0.0,
        ],
        median_px_full=float(np.median(ms) / PIXEL_UM),
        rms_px_full=float(np.sqrt((ms**2).mean()) / PIXEL_UM),
        tiles=out,
    )
