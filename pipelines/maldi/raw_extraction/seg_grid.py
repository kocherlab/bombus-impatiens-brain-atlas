#!/usr/bin/env python
"""Auto-segment a gridded MSI slide into one region per tissue section using
connected-component labeling (pure numpy label-propagation, no scipy). Names
each section by its grid position (r{row}c{col}) from its centroid. Emits the
SAME label-TIFF + names-JSON as manual drawing,
so import_regions.py works unchanged. Also writes an annotated overlay PNG for QC.

Usage: seg_grid.py <ionimg_h5> <sample> <outdir> [min_area]
"""

import json
import os
import sys

import h5py
import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

H5, SAMPLE, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
MIN_AREA = int(sys.argv[4]) if len(sys.argv) > 4 else 800  # drop specks < this many px

os.makedirs(OUTDIR, exist_ok=True)
with h5py.File(H5, "r") as h:
    tic = h["TIC"][:]
mask = np.isfinite(tic) & (tic > 0)
Hh, Ww = mask.shape
BIG = Hh * Ww + 1

# --- connected components via iterative neighbor-min label propagation ---
lab = np.where(mask, np.arange(Hh * Ww).reshape(Hh, Ww), BIG).astype(np.int64)
for _ in range(10000):
    prev = lab
    m = lab.copy()
    m[1:, :] = np.minimum(m[1:, :], lab[:-1, :])  # from above
    m[:-1, :] = np.minimum(m[:-1, :], lab[1:, :])  # from below
    m[:, 1:] = np.minimum(m[:, 1:], lab[:, :-1])  # from left
    m[:, :-1] = np.minimum(m[:, :-1], lab[:, 1:])  # from right
    m[~mask] = BIG  # keep background fixed
    lab = m
    if np.array_equal(lab, prev):
        break

# --- gather components, filter by area ---
flat = lab[mask]
uniq, counts = np.unique(flat, return_counts=True)
comps = [
    (u, c) for u, c in zip(uniq, counts, strict=True) if u != BIG and c >= MIN_AREA
]
print(
    f"connected components: {len(uniq)} raw, {len(comps)} after area>={MIN_AREA}",
    flush=True,
)

# centroids + bboxes
ys, xs = np.where(mask)
labvals = lab[ys, xs]
info = []
for u, c in comps:
    sel = labvals == u
    cy, cx = ys[sel].mean(), xs[sel].mean()
    info.append(
        {
            "old": u,
            "area": int(c),
            "cy": cy,
            "cx": cx,
            "y0": ys[sel].min(),
            "y1": ys[sel].max() + 1,
            "x0": xs[sel].min(),
            "x1": xs[sel].max() + 1,
        }
    )

# --- cluster centroids into grid rows/cols by gap-splitting ---
med_h = np.median([d["y1"] - d["y0"] for d in info])
med_w = np.median([d["x1"] - d["x0"] for d in info])


def cluster(vals, gap):
    order = np.argsort(vals)
    g = 0
    grp = {}
    grp[order[0]] = 0
    for i in range(1, len(order)):
        if vals[order[i]] - vals[order[i - 1]] > gap:
            g += 1
        grp[order[i]] = g
    return [grp[i] for i in range(len(vals))]


rows = cluster(np.array([d["cy"] for d in info]), 0.5 * med_h)
# Cluster columns PER ROW (rows can be offset, so a global column binning
# misassigns -> two distinct sections collide on one r#c# name).
cols = [0] * len(info)
for r in sorted(set(rows)):
    idxs = [i for i in range(len(info)) if rows[i] == r]
    subcols = cluster(np.array([info[i]["cx"] for i in idxs]), 0.5 * med_w)
    for j, i in enumerate(idxs):
        cols[i] = subcols[j]
nrows = max(rows) + 1
ncols = max(cols) + 1
print(
    f"grid layout: {nrows} rows x up to {ncols} cols; {len(info)} tissue sections",
    flush=True,
)

# --- build relabeled image + names (guaranteed unique) ---
labels = np.zeros((Hh, Ww), dtype=np.int32)
names = {}
old_to_new = {}
used = {}
for i, d in enumerate(info):
    rid = i + 1
    old_to_new[d["old"]] = rid
    base = f"r{rows[i] + 1}c{cols[i] + 1}"
    n = used.get(base, 0) + 1
    used[base] = n
    names[rid] = base if n == 1 else f"{base}_{n}"  # safety net for any collision
remap = np.zeros(int(lab.max()) + 2, dtype=np.int32)
for old, new in old_to_new.items():
    remap[int(old)] = new
labels[mask] = remap[lab[mask]]

lab_path = os.path.join(OUTDIR, f"labels_grid_{SAMPLE}.tif")
json_path = os.path.join(OUTDIR, f"regions_grid_{SAMPLE}.json")
tifffile.imwrite(lab_path, labels)
json.dump({str(k): v for k, v in names.items()}, open(json_path, "w"), indent=2)
areas = [d["area"] for d in info]
print(
    f"section areas px: min {min(areas)}, median {int(np.median(areas))}, "
    f"max {max(areas)}",
    flush=True,
)

# --- QC overlay ---
fig, ax = plt.subplots(figsize=(18, 8))
ax.imshow(np.where(mask, np.log1p(tic), np.nan), cmap="gray", interpolation="nearest")
for i, d in enumerate(info):
    ax.add_patch(
        Rectangle(
            (d["x0"], d["y0"]),
            d["x1"] - d["x0"],
            d["y1"] - d["y0"],
            fill=False,
            edgecolor="red",
            linewidth=0.6,
        )
    )
    ax.text(d["x0"] + 2, d["y0"] + 14, names[i + 1], color="yellow", fontsize=5)
ax.set_title(f"{SAMPLE}: {len(info)} auto-detected sections ({nrows}×{ncols} grid)")
ax.set_xlabel("x")
ax.set_ylabel("y")
png_path = os.path.join(OUTDIR, f"grid_regions_{SAMPLE}.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"WROTE:\n  {lab_path}\n  {json_path}\n  {png_path}", flush=True)
