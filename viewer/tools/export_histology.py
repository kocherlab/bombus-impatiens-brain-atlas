#!/usr/bin/env python
"""Turn the PASTE3-aligned Xenium DAPI stacks into small web images for the atlas.

Reads each brain's ``aligned_images.tif`` (one uint16 DAPI page per section,
~20 GB per brain) a page at a time, registers it to the 3D viewer's coordinate
frame, cleans it up and writes lossy WebP plus a manifest the viewer reads.

Run it on the cluster, where the stacks and the atlas h5ad live
---------------------------------------------------------------

From the repo root::

    # quick test: sections 0 and 1 of each brain, into histo_dryrun/
    PYTHONPATH=src .venv/bin/python viewer/tools/export_histology.py --dry-run

    # everything, into viewer/pages/data/histo/
    PYTHONPATH=src .venv/bin/python viewer/tools/export_histology.py

Needs numpy, scipy, h5py, tifffile and Pillow built with WebP (all in the
shared ``.venv``; the script checks WebP support before reading anything).
Memory: one full-resolution page is ~0.75 GB of uint16 and the downsampled
stack is ~0.5 GB, so 16 GB is comfortable. It is I/O bound (every byte of each
page is read once, ~41 GB in all), roughly 10-30 min. As a SLURM job::

    sbatch $BIMP_SBATCH_CPU -J histo -c 4 --mem=16G -t 01:30:00 -o histo_%j.log \\
      --wrap "PYTHONPATH=src .venv/bin/python viewer/tools/export_histology.py"

Then copy ``viewer/pages/data/histo/`` to the machine that builds the site.
Read the per-section registration table it prints (also written to
``histo/registration.json``) before trusting a section: see *Checks* below.

Inputs
------
* ``pipelines/wholebrain/queen/figures/alignment_validation/aligned_images.tif``
  (queen, 30 pages, 21637 x 17095) and
  ``pipelines/wholebrain/worker/figures/alignment_validation/aligned_images.tif``
  (worker, 35 pages, 18470 x 15000). NOT ``pipelines/wholebrain/queen/data/
  aligned_images.tif`` -- that one is partial.
* The atlas object the viewer is exported from (``TANGRAM_REL`` in
  ``viewer/export_data.py``, read from that file so the two cannot drift):
  ``obsm/spatial_aligned``, ``obs/brain``, ``obs/slice_name``, ``obs/z_level``.
* Each brain's slice-0 raw crop h5ad (``pipelines/<brain>/data/<slice_name>.h5ad``),
  to recover the image offset (below).

How an image pixel maps to the viewer
-------------------------------------
``analysis_workflow.py align --align-images`` works in Xenium pixels
(0.2125 um/px, crop-relative: ``raw_um / 0.2125 - uns['crop_coords']['min']``).
For the cells it writes ``aligned = (px - c_i) @ R_i.T`` (slice 0: ``px - c_0``);
for the images it warps each crop with the same rotation and then translates by
``real_center = translations[0][0]`` (``align_images_with_paste``), so that,
for EVERY section of a brain,

    image_px (col, row)  =  spatial_aligned (x, y)  +  real_center

with one constant ``real_center`` per brain. PASTE3's transforms were never
saved, so ``real_center`` is recovered here:

1. ``cells`` (preferred): slice 0's page is the untransformed crop, so for each
   slice-0 cell ``real_center = raw_crop_px - spatial_aligned``, joining the raw
   crop h5ad to the atlas by cell id (never positionally: QC drops cells). The
   median is used, and the spread must be ~0 px or the method is rejected.
2. ``image`` (fallback, and always computed as a cross-check): FFT
   cross-correlation of a cell-density raster built from ``spatial_aligned``
   against the downsampled DAPI, summed over the sections being exported.

``--offset-queen X,Y`` / ``--offset-worker X,Y`` (pixels) override both.

Per-section model (default for the worker; ``--per-section``)
-------------------------------------------------------------
The one-shift-per-brain model needs the stack and the atlas to come from the same
PASTE3 run. The worker's ``aligned_images.tif`` (2026-02-02) predates its atlas
(re-aligned 2026-04-08) and PASTE3's transforms were never saved, so its pages
carry older per-section rotations and shifts than the cells, and their order need
not be the atlas's z rank. ``histo_refit.py`` therefore (1) matches every page to
every section's cells (coarse screen) and assigns pages to sections one-to-one,
(2) fits a rigid April-aligned -> page transform per section (angle sweep of FFT
cross-correlation, sub-pixel), (3) resamples each page into the aligned frame
(``+ real_center``), after which everything below is unchanged, and (4) measures
residuals independently with 300 um tiles. Sections whose fit is poor are not
written to the manifest; registration.json records every section (its kept flag,
``flag_reasons``, raw->April fit, April->page fit, local residuals, old-model
scores) plus the stack / atlas dates. The queen keeps the global-shift model.

To reproduce (from a checkout or worktree whose ``results/``, ``data/`` and
``pipelines/`` are the shared copies and whose ``.venv`` is the shared venv; the
fit is deterministic, so a rerun gives the same angles and shifts to 0.01)::

    sbatch -J histo -c 4 --mem=16G -t 01:30:00 -o histo_%j.log --wrap \\
      "PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \\
       .venv/bin/python -u viewer/tools/export_histology.py"

About 16 min: the worker's page-to-section screen is ~2 min, each fit ~12 s.
Pages 6-30 are worker sections 8-32 and pages 31-32 are sections 6-7; a rerun
leaves the queen byte-identical.

The viewer frame replicates ``export_data.py`` (``--pages`` mode, all cells):
``x_um = spatial_aligned * 0.2125``; each caste is centred on the midpoint of
its 0.5/99.5-percentile bounding box; ``scale = max(p99.8|x|, p99.8|y|) / 0.9``
pooled over both castes; ``z = (z_level - (zmin + zmax) / 2) / scale`` pooled.
If that export ever changes, change ``viewer_frame()`` with it.

Output (the format the viewer is built against -- note any change here)
-------------------------------------------------------------------------
``histo/{caste}/{i:02d}.webp`` (long edge ~1024), ``histo/{caste}/{i:02d}-s.webp``
(~512), and ``histo/manifest.json``::

    {"version": 1,
     "castes": {"queen": {"sections": [
         {"i": 0, "z": <viewer z>, "bounds": [x0, y0, x1, y1], "file": "queen/00.webp",
          "small": "queen/00-s.webp", "w": 1024, "h": 812}, ...]},
                "worker": {...}},
     "frame": {...}}

* ``i`` is the section index = TIFF page = rank of the section's ``z_level``.
  A page where no tissue is found is skipped, so ``i`` need not be contiguous.
* ``bounds`` are in viewer coordinates and describe the image's OUTER EDGES:
  the left edge of column 0 is at x0, the right edge of the last column at x1,
  the top edge of row 0 at y0 and the bottom edge of the last row at y1. Image
  row index increases with viewer y (row 0 is at y0, the smaller y); any y flip
  the viewer applies to the cells applies to the image too. ``w``/``h`` are the
  large image's pixel size; the small one has the same bounds.
* Pixels outside the tissue are fully transparent AND black (gray is
  premultiplied by alpha, so ignoring alpha is safe); ``--background black``
  drops the alpha channel.
* ``frame`` is ADDITIVE (not in the original brief): the offsets, centres and
  scale used, for provenance. Readers may ignore it.

Re-running with ``--castes queen`` keeps the worker entry already in the
manifest, and vice versa.

Checks
------
For every section it prints the mean DAPI under that section's cells relative
to the mean over the section's tissue (``ratio``; registered sections sit well
above 1) and the fraction of cells that land on the tissue mask (``on``). A
section below ``--warn-ratio`` / ``--warn-on`` is flagged ``CHECK`` -- look at
it before shipping. The two offset estimates are also compared; they should
agree to within a few working pixels.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parent
REPO = APP.parent

PIXEL_UM = 0.2125  # Xenium pixel size, as in analysis_workflow.py / export_data.py

BRAINS = {
    "queen": dict(code="queen", dir="pipelines/wholebrain/queen"),
    "worker": dict(code="worker", dir="pipelines/wholebrain/worker"),
}
STACK_REL = "figures/alignment_validation/aligned_images.tif"
TANGRAM_FALLBACK = "results/tangram/combined_clustered_colfix_tangram.h5ad"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# atlas object
# ----------------------------------------------------------------------------


def tangram_rel() -> str:
    """TANGRAM_REL from export_data.py, parsed (importing it would run its setup)."""
    try:
        tree = ast.parse((APP / "export_data.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "TANGRAM_REL" for t in node.targets
            ):
                return ast.literal_eval(node.value)
    except (OSError, SyntaxError, ValueError):
        pass
    log(f"  could not read TANGRAM_REL from export_data.py; using {TANGRAM_FALLBACK}")
    return TANGRAM_FALLBACK


def _decode(a) -> np.ndarray:
    return np.array(
        [x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object
    )


def read_obs_column(f, name: str) -> np.ndarray:
    """An obs column as a plain array, for both anndata categorical layouts."""
    g = f["obs"]
    node = g[name]
    if hasattr(node, "keys"):  # anndata >= 0.8 categorical group
        cats = _decode(node["categories"][:])
        codes = node["codes"][:]
        out = np.empty(len(codes), dtype=object)
        ok = codes >= 0
        out[ok] = cats[codes[ok]]
        out[~ok] = None
        return out
    vals = node[:]
    cat_key = node.attrs.get("categories", None)
    if cat_key is not None:  # anndata < 0.8: codes + reference into obs/__categories
        cats = _decode(f[cat_key][:])
        return cats[vals]
    if vals.dtype.kind in "SO":
        return _decode(vals)
    return vals


def obs_index(f) -> np.ndarray:
    key = f["obs"].attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    return _decode(f["obs"][key][:])


def load_atlas(path: Path) -> dict:
    import h5py

    log(f"atlas: {path}")
    with h5py.File(path, "r") as f:
        xy = f["obsm/spatial_aligned"][:].astype(np.float64)
        brain = read_obs_column(f, "brain")
        slice_name = read_obs_column(f, "slice_name")
        z = f["obs/z_level"][:].astype(np.float64)
        ids = read_obs_column(f, "cell_id") if "cell_id" in f["obs"] else obs_index(f)
    log(f"  {len(xy):,} cells; brains {sorted(set(brain))}")
    return dict(xy=xy, brain=brain, slice_name=slice_name, z=z, cell_id=ids)


def viewer_frame(atlas: dict) -> dict:
    """Replicates viewer/export_data.py (--pages: every cell)."""
    x = atlas["xy"][:, 0] * PIXEL_UM
    y = atlas["xy"][:, 1] * PIXEL_UM
    centers = {}
    xc, yc = x.copy(), y.copy()
    for caste, b in BRAINS.items():
        m = atlas["brain"] == b["code"]
        if not m.any():
            raise SystemExit(f"no cells with obs/brain == {b['code']!r}")
        xlo, xhi = np.percentile(x[m], [0.5, 99.5])
        ylo, yhi = np.percentile(y[m], [0.5, 99.5])
        centers[caste] = [float((xlo + xhi) / 2), float((ylo + yhi) / 2)]
        xc[m] -= centers[caste][0]
        yc[m] -= centers[caste][1]
    half_x = np.percentile(np.abs(xc), 99.8)
    half_y = np.percentile(np.abs(yc), 99.8)
    scale = float(max(half_x, half_y)) / 0.9
    z = atlas["z"]
    zmid = float((z.min() + z.max()) / 2)
    centres = ", ".join(f"{k} ({v[0]:.1f}, {v[1]:.1f}) um" for k, v in centers.items())
    log(
        f"  viewer frame: scale {scale:.2f} um/unit, z mid {zmid:.1f} um, "
        f"centres {centres}"
    )
    return dict(pixel_um=PIXEL_UM, scale_um=scale, z_mid_um=zmid, center_um=centers)


def sections_of(atlas: dict, caste: str) -> list[dict]:
    """Sections of one brain in z order: [{slice_name, z_level}]; index = TIFF page."""
    m = atlas["brain"] == BRAINS[caste]["code"]
    names = atlas["slice_name"][m]
    zs = atlas["z"][m]
    out = []
    for nm in sorted(set(names), key=lambda s: float(np.median(zs[names == s]))):
        zz = np.unique(zs[names == nm])
        if len(zz) != 1:
            log(
                f"  WARNING {caste} {nm}: {len(zz)} distinct z_level values "
                f"({zz.min():g}..{zz.max():g}); placing the image at their median"
            )
        out.append(dict(slice_name=nm, z_level=float(np.median(zs[names == nm]))))
    if len({s["z_level"] for s in out}) != len(out):
        raise SystemExit(f"{caste}: two sections share a z_level; cannot order them")
    return out


# ----------------------------------------------------------------------------
# stack reading
# ----------------------------------------------------------------------------


class Stack:
    """Page-at-a-time access to an aligned_images.tif without loading it."""

    def __init__(self, path: Path):
        import tifffile

        self.path = path
        self.tif = tifffile.TiffFile(path)
        self.mm = None
        try:
            mm = tifffile.memmap(path, mode="r")
            self.mm = mm if mm.ndim == 3 else mm[None]
            self.n, self.h, self.w = self.mm.shape
            how = "memmap"
        except Exception as e:  # compressed / non-contiguous: fall back to whole pages
            p = self.tif.pages[0]
            self.n, (self.h, self.w) = len(self.tif.pages), p.shape[-2:]
            how = f"page reads ({type(e).__name__}: {e})"
        log(
            f"  stack {path.name}: {self.n} pages x {self.w} x {self.h}, "
            f"{path.stat().st_size / 1e9:.1f} GB, via {how}"
        )

    def block_mean(self, i: int, f: int) -> np.ndarray:
        """Page i, f x f block-averaged (anti-aliased), float32. Strips of rows."""
        hh, ww = self.h // f, self.w // f
        out = np.empty((hh, ww), np.float32)
        page = self.mm[i] if self.mm is not None else self.tif.pages[i].asarray()
        rows = max(1, 2048 // f)  # output rows per strip
        for r0 in range(0, hh, rows):
            r1 = min(hh, r0 + rows)
            blk = np.asarray(page[r0 * f : r1 * f, : ww * f], dtype=np.float32)
            out[r0:r1] = blk.reshape(r1 - r0, f, ww, f).mean(axis=(1, 3))
        del page
        return out


# ----------------------------------------------------------------------------
# image processing
# ----------------------------------------------------------------------------


def otsu(v: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(v, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    return float(mids[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])


def tissue_mask(
    img: np.ndarray,
    px_um: float,
    sigma_um: float = 30.0,
    frac: float = 0.5,
    close_um: float = 50.0,
) -> np.ndarray:
    """Tissue = heavily smoothed log-DAPI above a lenient threshold, holes filled.

    Deliberately inclusive: bee-brain DAPI is a nuclei-dense rind around dim,
    nuclei-poor neuropil, and a plain Otsu cut keeps the rind and drops the
    neuropil. The threshold sits ``frac`` of the way from the slide background
    (5th percentile) up to Otsu; wide smoothing and closing bridge the neuropil.
    """
    from scipy import ndimage as ndi

    valid = img > 0  # zero = canvas padding from the alignment warp
    if valid.sum() < 100:
        return np.zeros_like(valid)
    sm = ndi.gaussian_filter(np.log1p(img), sigma=max(1.0, sigma_um / px_um))
    bg = float(np.percentile(sm[valid], 5))
    t = bg + frac * (otsu(sm[valid]) - bg)
    m = (sm > t) & valid
    # round closing (distance transforms, not an iterated 3x3, which squares corners
    # off)
    r = max(1.0, close_um / px_um)
    pad = int(r) + 2
    grown = ndi.distance_transform_edt(~np.pad(m, pad)) <= r
    m = (ndi.distance_transform_edt(grown) > r)[pad:-pad, pad:-pad]
    m = ndi.binary_fill_holes(m) & valid
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= 0.02 * sizes.max()) + 1
        m = np.isin(lab, keep)
    return m


def clean(
    img: np.ndarray, mask: np.ndarray, px_um: float, a: argparse.Namespace
) -> np.ndarray:
    """Background subtraction, percentile stretch over tissue, gamma -> [0, 1]."""
    from scipy import ndimage as ndi

    out = img.astype(np.float32)
    if a.bg_strength > 0:
        bg = ndi.gaussian_filter(out, sigma=a.bg_sigma_um / px_um)
        out = np.clip(out - a.bg_strength * bg, 0, None)
    vals = out[mask] if mask.sum() > 100 else out[out > 0]
    if vals.size == 0:
        return np.zeros_like(out)
    lo, hi = np.percentile(vals, [a.pct_lo, a.pct_hi])
    out = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1) ** a.gamma
    return out


# ----------------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------------


def offset_from_cells(
    atlas: dict, caste: str, sec0: dict
) -> tuple[np.ndarray | None, str]:
    """real_center from slice-0 cells: raw crop px - aligned px, joined by cell id."""
    import h5py

    raw = REPO / BRAINS[caste]["dir"] / "data" / f"{sec0['slice_name']}.h5ad"
    if not raw.exists():
        return None, f"raw slice-0 h5ad not found: {raw}"
    with h5py.File(raw, "r") as f:
        sp = f["obsm/spatial"][:].astype(np.float64)
        # align subtracts the crop origin only when the key exists; mirror that
        cmin = (
            np.asarray(f["uns/crop_coords/min"][:], dtype=np.float64)
            if "uns/crop_coords/min" in f
            else np.zeros(2)
        )
        rid = read_obs_column(f, "cell_id") if "cell_id" in f["obs"] else obs_index(f)
    raw_px = sp / PIXEL_UM - cmin  # as analysis_workflow.py align does
    m = (atlas["brain"] == BRAINS[caste]["code"]) & (
        atlas["slice_name"] == sec0["slice_name"]
    )
    aid, axy = atlas["cell_id"][m], atlas["xy"][m]
    pos = {k: j for j, k in enumerate(rid)}
    if len(pos) != len(rid):
        return None, f"{raw.name}: duplicate cell ids in the raw crop"
    # the atlas index may carry an obs_names_make_unique "-N" suffix on top of
    # Xenium's own "-1": try the ids as they are, then with one suffix stripped
    best = None
    for ids in (
        aid,
        np.array(
            [
                str(k).rsplit("-", 1)[0]
                if str(k).rsplit("-", 1)[-1].isdigit()
                else str(k)
                for k in aid
            ],
            dtype=object,
        ),
    ):
        jr_try = np.array([pos.get(k, -1) for k in ids])
        if best is None or (jr_try >= 0).sum() > (best >= 0).sum():
            best = jr_try
    jr = best
    ok = jr >= 0
    n_ok = int(ok.sum())
    if n_ok < max(100, 0.5 * len(aid)):
        return None, (
            f"only {n_ok:,}/{len(aid):,} slice-0 atlas cells matched a raw "
            f"cell id (atlas ids look like {aid[:2].tolist()}, raw {rid[:2].tolist()})"
        )
    d = raw_px[jr[ok]] - axy[ok]
    off = np.median(d, axis=0)
    spread = float(np.percentile(np.linalg.norm(d - off, axis=1), 95))
    msg = f"{n_ok:,}/{len(aid):,} cells matched, p95 residual {spread:.3f} px"
    if spread > 2.0:
        return None, msg + " -- NOT a constant offset; rejecting"
    return off, msg


def raw_atlas_pairs(atlas: dict, caste: str, sec: dict):
    """(raw crop px, atlas aligned px, n atlas cells) for one section, joined by cell
    id.

    Same join as ``offset_from_cells`` (ids as they are, else one ``-N`` suffix
    stripped), for any section. ``(None, None, note)`` when it cannot be made.
    """
    import h5py

    raw = REPO / BRAINS[caste]["dir"] / "data" / f"{sec['slice_name']}.h5ad"
    if not raw.exists():
        return None, None, f"raw h5ad not found: {raw}"
    with h5py.File(raw, "r") as f:
        sp = f["obsm/spatial"][:].astype(np.float64)
        cmin = (
            np.asarray(f["uns/crop_coords/min"][:], dtype=np.float64)
            if "uns/crop_coords/min" in f
            else np.zeros(2)
        )
        rid = read_obs_column(f, "cell_id") if "cell_id" in f["obs"] else obs_index(f)
    raw_px = sp / PIXEL_UM - cmin
    m = (atlas["brain"] == BRAINS[caste]["code"]) & (
        atlas["slice_name"] == sec["slice_name"]
    )
    aid, axy = atlas["cell_id"][m], atlas["xy"][m]
    pos = {k: j for j, k in enumerate(rid)}
    if len(pos) != len(rid):
        return None, None, f"{raw.name}: duplicate cell ids in the raw crop"
    best = None
    for ids in (
        aid,
        np.array(
            [
                str(k).rsplit("-", 1)[0]
                if str(k).rsplit("-", 1)[-1].isdigit()
                else str(k)
                for k in aid
            ],
            dtype=object,
        ),
    ):
        jr_try = np.array([pos.get(k, -1) for k in ids])
        if best is None or (jr_try >= 0).sum() > (best >= 0).sum():
            best = jr_try
    ok = best >= 0
    if ok.sum() < max(100, 0.5 * len(aid)):
        return (
            None,
            None,
            f"only {int(ok.sum()):,}/{len(aid):,} atlas cells matched a raw cell id",
        )
    return raw_px[best[ok]], axy[ok], f"{int(ok.sum()):,}/{len(aid):,} cells matched"


def refit_caste(
    atlas: dict, caste: str, secs: list, imgs: dict, f: int, rc: np.ndarray
):
    """Per-page fit of the DAPI stack onto the atlas frame (see histo_refit.py).

    Returns ``(warped, fits)``: each page resampled into the atlas-aligned frame
    (+ ``rc``, the same convention the global-shift model uses) and, per page, the
    parameters and residuals that go into registration.json.
    """
    import histo_refit as hr
    from scipy.optimize import linear_sum_assignment

    m_b = atlas["brain"] == BRAINS[caste]["code"]
    sec_xy = [atlas["xy"][m_b & (atlas["slice_name"] == s["slice_name"])] for s in secs]
    warped, fits = {}, {}
    # Which page is which section is itself unknown: the stack was written before the
    # atlas was re-aligned and its page order need not be the atlas's z rank. Match
    # every page against every section's cells, then solve the one-to-one assignment.
    t0 = time.time()
    pages = list(imgs)
    S = hr.screen_scores(imgs, sec_xy, rc, f)
    r_, c_ = linear_sum_assignment(-np.log(np.maximum(S, 1e-9)))
    assign = {pages[r]: int(c) for r, c in zip(r_, c_, strict=True)}
    log(
        f"  page -> section screen ({len(pages)} pages x {len(secs)} "
        f"sections, {time.time() - t0:.0f}s)"
    )
    margins = {}
    for r, p in enumerate(pages):
        j = assign[p]
        others = np.delete(S[r], j)
        margins[p] = float(S[r, j] / others.max()) if others.size else float("inf")
        z_rank = "" if j == p else f"  <-- z-rank section is {p}"
        log(
            f"    page {p:2d} -> section {j:2d} ({secs[j]['slice_name'][-22:]})  "
            f"margin {margins[p]:.2f}x over the next best{z_rank}"
        )
    log(
        f"  per-section fit onto the atlas frame ({len(imgs)} page(s)); rc = "
        f"({rc[0]:.2f}, {rc[1]:.2f}) px"
    )
    log(
        f"  {'pg':>2} {'sec':>3} {'slice':<26} {'theta':>7} "
        f"{'dx,dy (um)':>13} {'coarse':>6} {'loc med':>7} "
        f"{'rms':>5} {'max':>5} {'tiles':>5}  raw->atlas p95"
    )
    for r, (page, img) in enumerate(imgs.items()):
        t0 = time.time()
        i = assign[page]
        sec = secs[i]
        xy = sec_xy[i]
        rec = dict(
            page=int(page),
            mapping=dict(
                page=int(page),
                z_rank_section=int(page),
                assigned_section=i,
                matches_z_rank=bool(i == page),
                screen_score=float(S[r, i]),
                margin_over_next_best=margins[page],
                next_best_section=int(
                    np.delete(np.arange(len(secs)), i)[np.argmax(np.delete(S[r], i))]
                ),
            ),
        )
        raw_px, axy, note = raw_atlas_pairs(atlas, caste, sec)
        if raw_px is not None:
            rec["raw_to_april"] = dict(hr.fit_rigid(raw_px, axy), note=note)
        else:
            rec["raw_to_april"] = dict(note=note)
        fit = hr.fit_page(img, xy, rc, f)
        w = hr.warp_to_frame(img, fit, rc, f)
        loc = hr.local_residuals(w, xy, rc, f)
        rec["april_to_image"] = fit
        rec["local_residual"] = loc
        warped[i] = w
        fits[i] = rec
        r2a = rec["raw_to_april"]
        log(
            f"  {page:2d} {i:3d} {sec['slice_name'][-26:]:<26} {fit['theta_deg']:7.2f} "
            f"{fit['delta_um'][0]:6.0f},{fit['delta_um'][1]:6.0f} "
            f"{fit['coarse_sharpness']:6.2f} "
            f"{loc.get('median_um', float('nan')):7.1f} "
            f"{loc.get('rms_um', float('nan')):5.1f} "
            f"{loc.get('max_um', float('nan')):5.1f} {loc.get('n_tiles', 0):5d}  "
            f"{r2a.get('resid_p95_um', float('nan')):.3f} um ({time.time() - t0:.0f}s)"
        )
    return warped, fits


def fit_flags(rec: dict, ratio: float, on: float, a: argparse.Namespace) -> list[str]:
    """Reasons a per-section fit should not ship (empty = it passes)."""
    why = []
    fit, loc, r2a = rec["april_to_image"], rec["local_residual"], rec["raw_to_april"]
    if ratio < a.warn_ratio:
        why.append(f"DAPI under cells {ratio:.2f}x tissue mean < {a.warn_ratio}")
    if on < a.warn_on:
        why.append(f"only {on:.0%} of cells on tissue < {a.warn_on:.0%}")
    mp = rec["mapping"]
    if mp["margin_over_next_best"] < a.min_margin:
        why.append(
            f"page matches section {mp['assigned_section']} "
            f"only {mp['margin_over_next_best']:.2f}x "
            f"better than section {mp['next_best_section']} (< {a.min_margin})"
        )
    if fit["peak_on_window_edge"]:
        why.append("xcorr peak on the search-window edge")
    if fit["coarse_sharpness"] < a.min_sharpness:
        why.append(
            f"angle peak not unique (sharpness {fit['coarse_sharpness']:.2f} "
            f"< {a.min_sharpness})"
        )
    if loc.get("n_tiles", 0) < a.min_tiles:
        why.append(f"only {loc.get('n_tiles', 0)} verification tiles < {a.min_tiles}")
    elif loc["rms_um"] > a.max_local_rms_um:
        why.append(
            f"local residual rms {loc['rms_um']:.1f} um > {a.max_local_rms_um:g} um"
        )
    if "det" not in r2a:
        why.append("raw crop could not be joined to the atlas: " + r2a.get("note", ""))
    elif r2a["det"] < 0 or r2a["resid_p95_um"] > 2.0:
        why.append(
            f"raw->atlas is not rigid (det {r2a['det']:.2f}, "
            f"p95 {r2a['resid_p95_um']:.2f} um)"
        )
    return why


def density_raster(xy_px: np.ndarray, f: int, origin: np.ndarray, shape) -> np.ndarray:
    idx = np.floor((xy_px - origin) / f).astype(np.int64)
    h, w = shape
    ok = (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
    d = np.zeros(shape, np.float32)
    np.add.at(d, (idx[ok, 1], idx[ok, 0]), 1.0)
    return d


def offset_from_image(
    atlas, caste, secs, imgs: dict, f: int
) -> tuple[np.ndarray, float]:
    """real_center by cross-correlating cell density with DAPI over the given pages.

    With the density raster D anchored at aligned-px ``origin`` and the image I at
    working scale, a cell at aligned a sits at D[(a - origin)/f] and at
    I[(a + rc)/f]; the correlation peak is the shift t = (rc + origin)/f.
    """
    from scipy import ndimage as ndi
    from scipy.signal import fftconvolve

    m_b = atlas["brain"] == BRAINS[caste]["code"]
    # robust extent: a stray far-off cell must not blow the FFT up
    lo, hi = np.percentile(atlas["xy"][m_b], [0.05, 99.95], axis=0)
    pad = 0.05 * (hi - lo) + f
    origin = lo - pad
    span = hi + pad - origin
    dshape = (int(math.ceil(span[1] / f)) + 1, int(math.ceil(span[0] / f)) + 1)
    acc = None
    for i, img in imgs.items():
        m = m_b & (atlas["slice_name"] == secs[i]["slice_name"])
        d = ndi.gaussian_filter(density_raster(atlas["xy"][m], f, origin, dshape), 1.0)
        d -= d.mean()
        im = np.log1p(img)
        im = ndi.gaussian_filter(im, 1.0)
        im -= im[img > 0].mean() if (img > 0).any() else 0
        im[img <= 0] = 0
        c = fftconvolve(im, d[::-1, ::-1], mode="full")
        acc = c if acc is None else acc + c
    k = np.unravel_index(int(np.argmax(acc)), acc.shape)
    # sub-pixel parabola per axis
    sub = []
    for ax in (0, 1):
        kk = list(k)
        lo, hi = kk.copy(), kk.copy()
        lo[ax] -= 1
        hi[ax] += 1
        if lo[ax] < 0 or hi[ax] >= acc.shape[ax]:
            sub.append(0.0)
            continue
        y0, y1, y2 = acc[tuple(lo)], acc[k], acc[tuple(hi)]
        den = y0 - 2 * y1 + y2
        sub.append(float(0.5 * (y0 - y2) / den) if den != 0 else 0.0)
    t_row = k[0] + sub[0] - (dshape[0] - 1)
    t_col = k[1] + sub[1] - (dshape[1] - 1)
    # density bin k is centred at origin + (k + .5) f; image block j at (j + .5) f - .5
    rc = np.array([t_col, t_row]) * f - origin - 0.5
    # peak sharpness: peak vs the 99th percentile of the correlation surface
    sharp = float(acc[k] / max(np.percentile(acc, 99.9), 1e-9))
    return rc, sharp


def section_score(img, mask, xy_px, rc, f) -> tuple[float, float]:
    """(mean DAPI under cells / mean DAPI over tissue, fraction of cells on tissue)."""
    h, w = img.shape
    p = np.floor((xy_px + rc) / f).astype(np.int64)
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    if ok.sum() == 0 or mask.sum() == 0:
        return 0.0, 0.0
    under = img[p[ok, 1], p[ok, 0]]
    on = mask[p[ok, 1], p[ok, 0]].sum() / len(p)
    return float(under.mean() / max(img[mask].mean(), 1e-9)), float(on)


def write_preview(path: Path, img, mask, cell_xy, rc, f, a) -> None:
    """Whole working-res page: DAPI gray, tissue-mask outline red, ~3k cells cyan."""
    from PIL import Image
    from scipy import ndimage as ndi

    g = clean(img, mask if mask.any() else img > 0, PIXEL_UM * f, a)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    edge = mask & ~ndi.binary_erosion(mask, iterations=2)
    rgb[edge] = (255, 40, 40)
    rng = np.random.default_rng(0)
    pick = (
        cell_xy[rng.choice(len(cell_xy), min(3000, len(cell_xy)), replace=False)]
        if len(cell_xy)
        else cell_xy
    )
    p = np.floor((pick + rc) / f).astype(int)
    ok = (
        (p[:, 0] >= 0)
        & (p[:, 0] < img.shape[1])
        & (p[:, 1] >= 0)
        & (p[:, 1] < img.shape[0])
    )
    rgb[p[ok, 1], p[ok, 0]] = (0, 230, 255)
    im = Image.fromarray(rgb, "RGB")
    k = min(1.0, 1400 / max(im.size))
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


# ----------------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------------


def to_pil(gray01: np.ndarray, alpha01: np.ndarray, background: str):
    from PIL import Image

    # gray is premultiplied by alpha, so a reader that ignores alpha still sees black
    # outside
    g = np.round(gray01 * alpha01 * 255).astype(np.uint8)
    if background == "black":
        return Image.fromarray(
            np.round(gray01 * alpha01 * 255).astype(np.uint8), "L"
        ).convert("RGB")
    a = np.round(alpha01 * 255).astype(np.uint8)
    return Image.merge(
        "RGBA", [Image.fromarray(g, "L")] * 3 + [Image.fromarray(a, "L")]
    )


def encode(img, path: Path, quality: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(
        path, format="WEBP", quality=int(quality), method=4
    )  # method 6: ~50x slower here, ~5% smaller
    return path.stat().st_size


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def parse_xy(s: str | None):
    if not s:
        return None
    x, y = (float(v) for v in s.split(","))
    return np.array([x, y])


def main() -> None:
    global REPO
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--repo",
        type=Path,
        default=REPO,
        help="repo root holding pipelines/ and results/ (default: this checkout)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=APP / "pages" / "data" / "histo",
        help="output dir (the histo/ folder); default viewer/pages/data/histo",
    )
    ap.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help="atlas h5ad (default: export_data.py's TANGRAM_REL under the repo)",
    )
    ap.add_argument("--stack-queen", type=Path, default=None)
    ap.add_argument("--stack-worker", type=Path, default=None)
    ap.add_argument(
        "--castes", nargs="+", default=["queen", "worker"], choices=list(BRAINS)
    )
    ap.add_argument(
        "--sections",
        nargs="+",
        type=int,
        default=None,
        help="only these section indices (TIFF pages)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="sections 0 1 only (unless --sections), written to <out>_dryrun/",
    )
    ap.add_argument(
        "--registration",
        choices=["auto", "cells", "image"],
        default="auto",
        help="how to recover real_center (auto: cells, else image)",
    )
    ap.add_argument("--offset-queen", default=None, help="X,Y px: override real_center")
    ap.add_argument(
        "--offset-worker", default=None, help="X,Y px: override real_center"
    )
    ap.add_argument("--long-edge", type=int, default=1024)
    ap.add_argument("--small-edge", type=int, default=512)
    ap.add_argument("--quality", type=int, default=74)
    ap.add_argument("--min-quality", type=int, default=55)
    ap.add_argument(
        "--min-edge",
        type=int,
        default=640,
        help="smallest long edge the budget may shrink the large images to",
    )
    ap.add_argument(
        "--budget-mb",
        type=float,
        default=3.0,
        help="target for the large images per caste; quality steps down to fit",
    )
    ap.add_argument("--margin-um", type=float, default=40.0)
    ap.add_argument("--pct-lo", type=float, default=0.5)
    ap.add_argument("--pct-hi", type=float, default=99.7)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument(
        "--bg-strength",
        type=float,
        default=0.6,
        help="fraction of a wide Gaussian background to subtract (0 = off)",
    )
    ap.add_argument("--bg-sigma-um", type=float, default=60.0)
    ap.add_argument("--background", choices=["alpha", "black"], default="alpha")
    ap.add_argument(
        "--mask-sigma-um", type=float, default=30.0, help="tissue-mask smoothing"
    )
    ap.add_argument(
        "--mask-frac",
        type=float,
        default=0.5,
        help="tissue threshold, 0 = slide background .. 1 = Otsu (lower = more "
        "inclusive)",
    )
    ap.add_argument(
        "--mask-close-um", type=float, default=50.0, help="tissue-mask closing radius"
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="also write preview/{caste}_NN.png: DAPI, mask outline, cells (on "
        "with --dry-run)",
    )
    ap.add_argument("--warn-ratio", type=float, default=1.15)
    ap.add_argument("--warn-on", type=float, default=0.6)
    ap.add_argument(
        "--per-section",
        nargs="*",
        default=["worker"],
        choices=list(BRAINS),
        help="castes whose pages are fitted onto the atlas one section at a time "
        "(histo_refit.py) instead of one shift per brain; a failing section is "
        "left out of the manifest. Default: worker (its stack predates its atlas)",
    )
    ap.add_argument(
        "--max-local-rms-um",
        type=float,
        default=15.0,
        help="per-section fit fails above this rms of the 300 um tile residuals",
    )
    ap.add_argument(
        "--min-sharpness",
        type=float,
        default=1.03,
        help="per-section fit fails when the best angle does not beat the next by "
        "this factor",
    )
    ap.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="per-section fit fails when its page matches the assigned section "
        "less than "
        "this factor better than the runner-up at the coarse screen. Off by default: "
        "neighbouring sections look alike at that scale (margins of 1.03x are normal "
        "for a perfect fit), so the fitted ratio / residuals decide instead",
    )
    ap.add_argument(
        "--min-tiles",
        type=int,
        default=4,
        help="per-section fit fails with fewer verification tiles than this",
    )
    a = ap.parse_args()
    REPO = a.repo.resolve()

    from PIL import Image, features

    if not features.check("webp"):
        raise SystemExit(
            "this Pillow has no WebP support (PIL.features.check('webp') is False)"
        )
    if a.dry_run:
        a.sections = a.sections if a.sections is not None else [0, 1]
        a.out = a.out.with_name(a.out.name + "_dryrun")
        a.preview = True
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    log(f"output: {out}")

    atlas_path = a.atlas or (REPO / tangram_rel())
    if not atlas_path.exists():
        raise SystemExit(f"atlas h5ad not found: {atlas_path}")
    atlas = load_atlas(atlas_path)
    frame = viewer_frame(atlas)
    frame["atlas"] = (
        str(atlas_path.relative_to(REPO))
        if atlas_path.is_relative_to(REPO)
        else str(atlas_path)
    )
    frame["real_center_px"] = {}

    man_path = out / "manifest.json"
    manifest = {"version": 1, "castes": {}}
    if man_path.exists():
        try:
            old = json.loads(man_path.read_text())
            if old.get("version") == 1:
                manifest["castes"] = {
                    k: v for k, v in old.get("castes", {}).items() if k not in a.castes
                }
                frame["real_center_px"] = {
                    k: v
                    for k, v in old.get("frame", {}).get("real_center_px", {}).items()
                    if k not in a.castes
                }
        except (OSError, ValueError):
            pass
    reg_report = {}
    sizes = {}
    any_warn = False

    for caste in a.castes:
        log("")
        log(f"===== {caste} =====")
        secs = sections_of(atlas, caste)
        stack_path = getattr(a, f"stack_{caste}") or (
            REPO / BRAINS[caste]["dir"] / STACK_REL
        )
        if not stack_path.exists():
            raise SystemExit(f"{caste}: stack not found: {stack_path}")
        st = Stack(stack_path)
        if st.n != len(secs):
            raise SystemExit(
                f"{caste}: {st.n} TIFF pages but {len(secs)} sections in the atlas; "
                "the page <-> section mapping would be a guess. Stopping."
            )
        want = a.sections if a.sections is not None else list(range(st.n))
        bad = [i for i in want if not 0 <= i < st.n]
        if bad:
            raise SystemExit(f"{caste}: sections {bad} out of range 0..{st.n - 1}")

        # working scale: ~2x the large output, anti-aliased by block mean
        f = max(1, max(st.h, st.w) // (2 * a.long_edge))
        px_um = PIXEL_UM * f
        log(f"  working downsample x{f} ({px_um:.2f} um/px), sections {want}")

        imgs = {}
        for i in want:
            t0 = time.time()
            imgs[i] = st.block_mean(i, f)
            nz = imgs[i][imgs[i] > 0]
            log(
                f"  read page {i:2d} "
                f"({secs[i]['slice_name']}): {imgs[i].shape[1]}x{imgs[i].shape[0]}, "
                f"nonzero {nz.size / imgs[i].size:.0%}, "
                f"p99 {np.percentile(nz, 99) if nz.size else 0:.0f}, "
                f"{time.time() - t0:.1f}s"
            )

        # ---- registration
        override = parse_xy(getattr(a, f"offset_{caste}"))
        rc_cells, why = (None, "skipped")
        if override is None and a.registration in ("auto", "cells"):
            rc_cells, why = offset_from_cells(atlas, caste, secs[0])
            rc_txt = (
                "%.2f, %.2f px" % tuple(rc_cells)
                if rc_cells is not None
                else "unavailable"
            )
            log(f"  offset (cells): {rc_txt} -- {why}")
        per_sec = caste in (a.per_section or [])
        fits, old_scores = {}, {}
        if per_sec:
            rc0 = override if override is not None else rc_cells
            if rc0 is None:
                raise SystemExit(
                    f"{caste}: per-section fit needs real_center from slice 0 "
                    f"(cells: {why}); pass --offset-{caste}"
                )
            # score the pages as the old one-shift model would have placed them
            m_b0 = atlas["brain"] == BRAINS[caste]["code"]
            for i, img in imgs.items():
                mk = tissue_mask(
                    img, px_um, a.mask_sigma_um, a.mask_frac, a.mask_close_um
                )
                cxy = atlas["xy"][m_b0 & (atlas["slice_name"] == secs[i]["slice_name"])]
                old_scores[i] = (
                    section_score(img, mk, cxy, rc0, f) if mk.any() else (0.0, 0.0)
                )
            imgs, fits = refit_caste(atlas, caste, secs, imgs, f, rc0)
            want = sorted(
                imgs
            )  # now keyed by SECTION (the page a section was matched to)
        rc_img, sharp = offset_from_image(atlas, caste, secs, imgs, f)
        log(
            f"  offset (image xcorr over {len(imgs)} "
            f"section(s)): {rc_img[0]:.1f}, {rc_img[1]:.1f} px, "
            f"peak/p99.9 {sharp:.2f}" + (" [on the refitted pages]" if per_sec else "")
        )
        if override is not None:
            rc, method = override, "override"
        elif rc_cells is not None:
            rc, method = rc_cells, "cells"
        elif a.registration == "cells":
            raise SystemExit(f"{caste}: --registration cells but {why}")
        else:
            rc, method = rc_img, "image"
            log(
                "  WARNING: using the image cross-correlation offset; check "
                "the scores below"
            )
            any_warn = True
        if rc_cells is not None:
            dd = float(np.linalg.norm(rc_cells - rc_img))
            flag = "" if dd <= 3 * f else "  <-- DISAGREE: inspect before trusting"
            any_warn |= bool(flag)
            log(
                f"  cells vs image offsets differ by {dd:.1f} px "
                f"({dd / f:.1f} working px){flag}"
            )
        log(f"  using real_center = ({rc[0]:.2f}, {rc[1]:.2f}) px [{method}]")
        frame["real_center_px"][caste] = [float(rc[0]), float(rc[1])]
        reg_report[caste] = dict(
            method=method,
            real_center_px=[float(v) for v in rc],
            cells=None if rc_cells is None else [float(v) for v in rc_cells],
            cells_note=why,
            image=[float(v) for v in rc_img],
            image_peak_sharpness=sharp,
            sections=[],
        )
        if per_sec:

            def _mt(p):
                p = Path(p)
                return (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
                    if p.exists()
                    else None
                )

            bdir = REPO / BRAINS[caste]["dir"]
            reg_report[caste].update(
                model="per_section",
                provenance=dict(
                    reason=(
                        "the DAPI stack predates the atlas it is drawn "
                        "against, so its pages carry an "
                        "older per-section PASTE3 transform than the cells; "
                        "PASTE3's transforms were "
                        "never saved, so each page is fitted onto the atlas "
                        "frame separately"
                    ),
                    stack=str(stack_path),
                    stack_mtime=_mt(stack_path),
                    atlas=str(atlas_path),
                    atlas_mtime=_mt(atlas_path),
                    brain_aligned_h5ad={
                        n: _mt(bdir / "data" / n)
                        for n in ("aligned.h5ad", "aligned_clustered.h5ad")
                    },
                    fit=(
                        "rigid April-aligned -> page: angle sweep (1.5 deg "
                        "over the full circle at "
                        f"{PIXEL_UM * f * 3:.1f} um/px, then 0.5 and 0.15 "
                        f"deg at {px_um:.2f} um/px) of FFT "
                        "cross-correlation between the section's cell "
                        "density and log-DAPI, sub-pixel by "
                        "parabola; page resampled bilinearly into the aligned frame"
                    ),
                    residual=(
                        "local_residual: 300 um tiles of the resampled page "
                        "cross-correlated with the "
                        "April cell density after high-pass, shift in um "
                        "(image minus cells)"
                    ),
                    raw_to_april=(
                        "Kabsch fit of the raw crop h5ad to the atlas, joined by "
                        "cell id"
                    ),
                    thresholds=dict(
                        warn_ratio=a.warn_ratio,
                        warn_on=a.warn_on,
                        max_local_rms_um=a.max_local_rms_um,
                        min_sharpness=a.min_sharpness,
                        min_tiles=a.min_tiles,
                        min_margin=a.min_margin,
                    ),
                    page_to_section=(
                        "each page is matched to every section's cells (coarse, "
                        f"{PIXEL_UM * f * 9:.0f} um/px, heavily blurred) and "
                        "the pages are "
                        "assigned one-to-one to sections by linear_sum_assignment; the "
                        "stack's page order is NOT assumed to be the atlas z rank"
                    ),
                ),
            )
            frame.setdefault("registration_model", {})[caste] = "per_section"

        # ---- per section: mask, score, clean, crop, resize
        cx, cy = frame["center_um"][caste]
        scale, zmid = frame["scale_um"], frame["z_mid_um"]
        m_b = atlas["brain"] == BRAINS[caste]["code"]
        rendered = []
        log(
            f"  {'i':>2}  {'slice':<34} {'ratio':>6} {'on':>5} "
            f" {'crop (work px)':<22} out"
        )
        for i in want:
            img = imgs[i]
            mask = tissue_mask(
                img, px_um, a.mask_sigma_um, a.mask_frac, a.mask_close_um
            )
            cell_xy = atlas["xy"][m_b & (atlas["slice_name"] == secs[i]["slice_name"])]
            ratio, on = section_score(img, mask, cell_xy, rc, f)
            warn = ratio < a.warn_ratio or on < a.warn_on or mask.sum() == 0
            reasons, extra = [], {}
            if per_sec:
                reasons = fit_flags(fits[i], ratio, on, a)
                warn = warn or bool(reasons)
                pg = fits[i]["page"]
                extra = dict(
                    model="per_section",
                    page=pg,
                    mapping=fits[i]["mapping"],
                    old_model_ratio=round(old_scores[pg][0], 3),
                    old_model_on_tissue=round(old_scores[pg][1], 3),
                    raw_to_april=fits[i]["raw_to_april"],
                    april_to_image=fits[i]["april_to_image"],
                    local_residual=fits[i]["local_residual"],
                    shipped=not warn,
                    flag_reasons=reasons,
                )
            any_warn |= warn
            if mask.sum() == 0:
                log(
                    f"  {i:2d}  {secs[i]['slice_name'][-34:]:<34}  no tissue "
                    "found -- skipped  CHECK"
                )
                if per_sec:
                    reg_report[caste]["sections"].append(
                        dict(
                            i=i,
                            slice_name=secs[i]["slice_name"],
                            z_level=secs[i]["z_level"],
                            ratio=round(ratio, 3),
                            on_tissue=round(on, 3),
                            check=True,
                            **extra,
                        )
                    )
                continue
            if per_sec and warn:
                log(
                    f"  {i:2d}  {secs[i]['slice_name'][-34:]:<34} "
                    f"{ratio:6.2f} {on:5.0%}  NOT SHIPPED: " + "; ".join(reasons)
                )
                reg_report[caste]["sections"].append(
                    dict(
                        i=i,
                        slice_name=secs[i]["slice_name"],
                        z_level=secs[i]["z_level"],
                        ratio=round(ratio, 3),
                        on_tissue=round(on, 3),
                        check=True,
                        **extra,
                    )
                )
                if a.preview:
                    write_preview(
                        out / "preview" / f"{caste}_{i:02d}.png",
                        img,
                        mask,
                        cell_xy,
                        rc,
                        f,
                        a,
                    )
                continue
            from scipy import ndimage as ndi

            marg = max(1, round(a.margin_um / px_um))
            rr, cc = np.nonzero(mask)
            r0, r1 = max(0, rr.min() - marg), min(img.shape[0], rr.max() + 1 + marg)
            c0, c1 = max(0, cc.min() - marg), min(img.shape[1], cc.max() + 1 + marg)
            g = clean(img, mask, px_um, a)[r0:r1, c0:c1]
            alpha = ndi.gaussian_filter(
                ndi.binary_dilation(mask, iterations=max(1, marg // 3)).astype(
                    np.float32
                ),
                sigma=max(1.0, marg / 4),
            )[r0:r1, c0:c1]
            alpha = np.clip(alpha * 1.6, 0, 1)  # feathered edge, solid inside

            hh, ww = g.shape
            k = min(1.0, a.long_edge / max(hh, ww))  # never upsample
            W, H = max(1, round(ww * k)), max(1, round(hh * k))
            ks = min(1.0, a.small_edge / max(W, H))
            Ws, Hs = max(1, round(W * ks)), max(1, round(H * ks))

            def rs(arr, size):
                return np.asarray(
                    Image.fromarray(arr.astype(np.float32), "F").resize(
                        size, Image.LANCZOS
                    )
                ).clip(0, 1)

            gl, al = rs(g, (W, H)), rs(alpha, (W, H))
            gs, as_ = rs(gl, (Ws, Hs)), rs(al, (Ws, Hs))

            # outer edges of the crop, full-res px -> aligned px -> um -> viewer
            X0, X1 = c0 * f - 0.5, c1 * f - 0.5
            Y0, Y1 = r0 * f - 0.5, r1 * f - 0.5
            bx = [((X - rc[0]) * PIXEL_UM - cx) / scale for X in (X0, X1)]
            by = [((Y - rc[1]) * PIXEL_UM - cy) / scale for Y in (Y0, Y1)]
            z = (secs[i]["z_level"] - zmid) / scale
            entry = {
                "i": i,
                "z": round(z, 6),
                "bounds": [
                    round(bx[0], 6),
                    round(by[0], 6),
                    round(bx[1], 6),
                    round(by[1], 6),
                ],
                "file": f"{caste}/{i:02d}.webp",
                "small": f"{caste}/{i:02d}-s.webp",
                "w": W,
                "h": H,
            }
            rendered.append(
                (entry, to_pil(gl, al, a.background), to_pil(gs, as_, a.background))
            )
            reg_report[caste]["sections"].append(
                dict(
                    i=i,
                    slice_name=secs[i]["slice_name"],
                    z_level=secs[i]["z_level"],
                    ratio=round(ratio, 3),
                    on_tissue=round(on, 3),
                    check=bool(warn),
                    crop_work_px=[int(c0), int(r0), int(c1), int(r1)],
                    work_downsample=f,
                    **extra,
                )
            )
            log(
                f"  {i:2d}  {secs[i]['slice_name'][-34:]:<34} {ratio:6.2f} {on:5.0%}  "
                f"{f'{c1 - c0}x{r1 - r0} @({c0},{r0})':<22} {W}x{H}"
                f"{'  CHECK' if warn else ''}"
            )
            if a.preview:
                write_preview(
                    out / "preview" / f"{caste}_{i:02d}.png",
                    img,
                    mask,
                    cell_xy,
                    rc,
                    f,
                    a,
                )
        del imgs

        # ---- encode to fit the per-caste budget: first step quality down to
        # --min-quality, then shrink the large images (bounds are unchanged: same
        # extent, fewer pixels) down to --min-edge on the long side.
        if per_sec and a.sections is None:
            # sections are renumbered / left out by the fit: drop images from an earlier
            # run
            for old in (out / caste).glob("*.webp"):
                old.unlink()
        budget = a.budget_mb * 1e6 * (len(rendered) / st.n)  # pro-rata for partial runs
        q = a.quality
        while True:
            big = sum(encode(im, out / e["file"], q) for e, im, _ in rendered)
            if big <= budget:
                break
            long_now = max(max(im.size) for _, im, _ in rendered)
            if q > a.min_quality:
                log(
                    f"  large images {big / 1e6:.2f} MB > {budget / 1e6:.2f} "
                    f"MB at q{q}; lowering quality"
                )
                q = max(a.min_quality, q - 6)
            elif long_now * 0.88 >= a.min_edge:
                log(
                    f"  large images {big / 1e6:.2f} MB > {budget / 1e6:.2f} "
                    f"MB at q{q}, "
                    f"{long_now}px; shrinking"
                )
                for j, (e, im, ims) in enumerate(rendered):
                    im = im.resize(
                        (
                            max(1, round(im.width * 0.88)),
                            max(1, round(im.height * 0.88)),
                        ),
                        Image.LANCZOS,
                    )
                    rendered[j] = (dict(e, w=im.width, h=im.height), im, ims)
            else:
                break
        small = sum(encode(ims, out / e["small"], q) for e, _, ims in rendered)
        if big > budget:
            log(
                f"  WARNING: {big / 1e6:.2f} MB still over budget at q{q} "
                "and the minimum size"
            )
        sizes[caste] = dict(
            n=len(rendered),
            large=big,
            small=small,
            quality=q,
            long_edge=max(max(im.size) for _, im, _ in rendered) if rendered else 0,
        )
        reg_report[caste]["quality"] = q
        manifest["castes"][caste] = {
            "sections": sorted((e for e, _, _ in rendered), key=lambda e: e["i"])
        }

    manifest["frame"] = frame
    man_path.write_text(json.dumps(manifest, indent=1))
    reg_path = out / "registration.json"
    if (
        reg_path.exists()
    ):  # a --castes run keeps the other castes' report, as the manifest does
        try:
            reg_report = {
                **{
                    k: v
                    for k, v in json.loads(reg_path.read_text()).items()
                    if k not in a.castes
                },
                **reg_report,
            }
        except (OSError, ValueError):
            pass
    reg_path.write_text(json.dumps(reg_report, indent=1))

    log("")
    log("===== sizes =====")
    for caste, s in sizes.items():
        log(
            f"  {caste:<7} {s['n']:2d} sections  large {s['large'] / 1e6:5.2f} MB "
            f"(avg {s['large'] / max(s['n'], 1) / 1e3:4.0f} kB, {s['long_edge']}px)  "
            f"small {s['small'] / 1e6:5.2f} MB  q{s['quality']}"
        )
    tot = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"  total on disk {tot / 1e6:.2f} MB -> {out}")
    if any_warn:
        log(
            "  Some sections or offsets are flagged CHECK / DISAGREE above "
            "-- look at them "
            "(registration.json) before shipping."
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
