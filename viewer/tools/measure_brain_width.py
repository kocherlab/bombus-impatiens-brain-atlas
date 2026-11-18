"""Measure brain width from the atlas's own raw sections, in micrometres.

The aligned atlas (aligned_clustered.h5ad) is a PASTE3 registration frame -- its
coordinates are Xenium PIXELS, i.e. the micron values divided by 0.2125 --
so widths must come from the raw per-crop h5ad files, where obsm['spatial'] is
section microns. Each crop file (one slice_name) is one brain section; a Xenium
capture region holds 2-4 of them in overlapping coordinates, so this measures
per crop and never pools a region. The brain's width is its widest section.

    PYTHONPATH=src .venv/bin/python viewer/tools/measure_brain_width.py
    # + DAPI cross-check:
    PYTHONPATH=src .venv/bin/python viewer/tools/measure_brain_width.py --dapi

Width is the extent along the section's long principal axis (0.5th-99.5th
percentile of cell centroids projected onto it, so a few stray cells do not
stretch it). Sections are not square to the slide axes, so the plain x / y
extents (also printed) under-read a tilted section; a bee brain is much wider
than tall, so its long axis is left-right. Also printed: ``full`` (the same at
the 0.05th-99.95th percentiles) and ``maxcal``, the widest extent in ANY direction -- an upper
bound that picks corner-to-corner diagonals on blocky shapes.

--dapi repeats the measurement on the PASTE3-aligned DAPI stacks
(figures/alignment_validation/aligned_images.tif): tissue mask -> same caliper,
pixels times 0.2125 um. The alignment only rotates and translates the images, so
this independently checks both the micron scale and the cell-based number
(cells sit inside the tissue, so DAPI should read equal or slightly wider).
It reads every 8th row and column of each page (~6 GB of I/O per brain).
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PIXEL_UM = 0.2125
BRAINS = {
    "queen": "pipelines/wholebrain/queen",
    "worker": "pipelines/wholebrain/worker",
}
CROP_GLOB = "data/output-XETG00151__*.h5ad"
STACK = "figures/alignment_validation/aligned_images.tif"
ANGLES = np.deg2rad(np.arange(0, 180, 1.0))
DIRS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)


def principal(xy: np.ndarray, lo: float, hi: float) -> tuple[float, float, float]:
    """(extent along the long principal axis, extent across it, axis angle deg)."""
    c = xy - xy.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(c.T))
    major, minor = evecs[:, 1], evecs[:, 0]
    ext = lambda v: float(np.diff(np.percentile(c @ v, [lo, hi]))[0])  # noqa: E731
    return (
        ext(major),
        ext(minor),
        float(np.rad2deg(np.arctan2(major[1], major[0])) % 180),
    )


def caliper(xy: np.ndarray, lo: float, hi: float) -> tuple[float, float, float]:
    """(widest extent, extent perpendicular to it, angle deg) over 1-degree steps."""
    proj = xy @ DIRS.T
    ext = np.diff(np.percentile(proj, [lo, hi], axis=0), axis=0)[0]
    k = int(np.argmax(ext))
    return float(ext[k]), float(ext[(k + 90) % 180]), float(np.rad2deg(ANGLES[k]))


def crops(brain_dir: Path) -> list[Path]:
    out = []
    for p in sorted(glob.glob(str(brain_dir / CROP_GLOB))):
        name = Path(p).stem
        if any(t in name for t in ("aligned", "_clustered", "_labeled", "test_")):
            continue
        out.append(Path(p))
    return out


def measure_cells(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        xy = f["obsm/spatial"][:, :2].astype(np.float64)
        has_crop = "uns/crop_coords" in f
    w, h, ang = principal(xy, 0.5, 99.5)
    wf, _, _ = principal(xy, 0.05, 99.95)
    wmax, _, _ = caliper(xy, 0.5, 99.5)
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
    return dict(
        name=path.stem,
        n=len(xy),
        width=w,
        height=h,
        angle=ang,
        width_full=wf,
        width_max=wmax,
        x=float(hi[0] - lo[0]),
        y=float(hi[1] - lo[1]),
        crop=has_crop,
    )


def tissue_xy(page: np.ndarray, step: int) -> np.ndarray:
    """Tissue-pixel centres (um) of a strided DAPI page: smoothed log, Otsu, specks
    dropped."""
    from scipy import ndimage as ndi

    img = page.astype(np.float32)
    valid = img > 0
    if valid.sum() < 100:
        return np.zeros((0, 2))
    sm = ndi.gaussian_filter(np.log1p(img), 10.0 / (PIXEL_UM * step))
    v = sm[valid]
    hist, edges = np.histogram(v, 256)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids)
    t = mids[
        np.argmax(
            w0 * w1 * (m0 / np.maximum(w0, 1) - (m0[-1] - m0) / np.maximum(w1, 1)) ** 2
        )
    ]
    m = (sm > t) & valid
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        m = np.isin(lab, np.flatnonzero(sizes >= 0.02 * sizes.max()) + 1)
    r, c = np.nonzero(m)
    return np.stack([c, r], axis=1) * (step * PIXEL_UM)


def measure_dapi(stack: Path, step: int = 8) -> list[dict]:
    import tifffile

    try:
        mm = tifffile.memmap(stack, mode="r")
        pages = (mm[i, ::step, ::step] for i in range(mm.shape[0]))
    except Exception:  # not memmappable: whole pages, one at a time
        tif = tifffile.TiffFile(stack)
        pages = (p.asarray()[::step, ::step] for p in tif.pages)
    out = []
    for i, pg in enumerate(pages):
        xy = tissue_xy(np.asarray(pg), step)
        if len(xy) < 50:
            out.append(dict(page=i, width=float("nan"), width_full=float("nan")))
            continue
        w, h, ang = principal(xy, 0.5, 99.5)
        wf, _, _ = principal(xy, 0.05, 99.95)
        out.append(dict(page=i, width=w, height=h, width_full=wf))
    return out


def slice_order(name: str) -> tuple:
    """analysis_workflow.get_slice_sort_key: the order the stack's pages are in."""
    m = re.search(r"_(2_)?(\d)([AB])(?:_clustered)?$", name)
    if not m:
        return (999, 999, 999)
    return (1 if m.group(1) else 0, int(m.group(2)), 0 if m.group(3) == "A" else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument(
        "--dapi", action="store_true", help="cross-check on the aligned DAPI stacks"
    )
    ap.add_argument(
        "--all", action="store_true", help="print every section, not just the top 3"
    )
    a = ap.parse_args()

    summary = []
    for brain, rel in BRAINS.items():
        bdir = a.repo / rel
        files = crops(bdir)
        print(f"\n== {brain}: {len(files)} crops in {bdir / CROP_GLOB}")
        if not files:
            print("   none found")
            continue
        rows = [measure_cells(p) for p in files]
        no_crop = [r["name"] for r in rows if not r["crop"]]
        if no_crop:
            print(
                f"   WARNING: {len(no_crop)} file(s) lack uns/crop_coords -- "
                "may be whole regions, "
                f"not single sections: {no_crop[:3]}"
            )
        rows.sort(key=lambda r: -r["width"])
        print(
            f"   {'crop':<44} {'cells':>7} {'width':>7} {'perp':>7} {'@deg':>5} "
            f"{'x':>7} {'y':>7} {'full':>7} {'maxcal':>7}  (um)"
        )
        for r in rows if a.all else rows[:3]:
            print(
                f"   {r['name']:<44} {r['n']:7,d} {r['width']:7.0f} "
                f"{r['height']:7.0f} {r['angle']:5.0f} "
                f"{r['x']:7.0f} {r['y']:7.0f} {r['width_full']:7.0f} "
                f"{r['width_max']:7.0f}"
            )
        top = rows[0]
        med = float(np.median([r["width"] for r in rows]))
        print(f"   median section width {med:,.0f} um")
        line = (
            f"{brain}: max left-right width {top['width'] / 1000:.2f} mm "
            f"({top['width_full'] / 1000:.2f} mm at 0.05-99.95%), crop {top['name']}"
        )

        if a.dapi:
            stack = bdir / STACK
            if not stack.exists():
                print(f"   DAPI: stack not found: {stack}")
            else:
                print(f"   DAPI: reading {stack} (every 8th px)...", flush=True)
                d = measure_dapi(stack)
                names = sorted((r["name"] for r in rows), key=lambda s: slice_order(s))
                by_name = {r["name"]: r for r in rows}
                if len(d) != len(names):
                    print(
                        f"   DAPI: {len(d)} pages vs {len(names)} crops -- "
                        "per-page matching skipped"
                    )
                    names = [None] * len(d)
                good = [x for x in d if np.isfinite(x["width"])]
                dt = max(good, key=lambda x: x["width"])
                nm = names[dt["page"]]
                print(
                    f"   DAPI widest page {dt['page']} "
                    f"({nm or '?'}): {dt['width']:,.0f} um "
                    f"(0.05-99.95%: {dt['width_full']:,.0f}); "
                    f"median {np.median([x['width'] for x in good]):,.0f} um"
                )
                if nm is not None:
                    diffs = [
                        x["width_full"] - by_name[names[x["page"]]]["width_full"]
                        for x in good
                    ]
                    print(
                        "   DAPI - cells, same section (0.05-99.95%): "
                        f"median {np.median(diffs):+.0f} um, "
                        f"range {min(diffs):+.0f}..{max(diffs):+.0f}"
                    )
                line += f"; DAPI widest {dt['width'] / 1000:.2f} mm"
        summary.append(line)

    print("\n== summary")
    for s in summary:
        print("  " + s)


if __name__ == "__main__":
    main()
