#!/usr/bin/env python
"""Turn the wide panel parquet + seg_grid regions into per-section slice_###.h5ad
for maldi_alignment.py (bimp-st recipe).

Each slice: X = TIC-normalized ALIGN channels (the 'other' neuropeptides);
obsm['spatial'] = section-centered (x,y); obs carries region, sample, TIC, and
the collaborator readouts sNPF_v1 / Tk2 (TIC-normalized). Files named in serial
grid order so _load_slices() (sorted slice_*.h5ad) stacks them in order.

Run with bimp-st's env (has anndata):
  ../../../.venv/bin/python make_slices.py \
     ion_out/panel_princeton_80524.parquet \
     ion_out/regions/pixel_regions_princeton_80524.parquet \
     channel_spec.json princeton_80524 ion_out/slices_princeton_80524
"""

import json
import os
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd

PANEL, PIXREG, SPEC, SAMPLE, OUTDIR = sys.argv[1:6]
MAX_SPOTS = int(sys.argv[6]) if len(sys.argv) > 6 else 0  # 0 = no cap
os.makedirs(OUTDIR, exist_ok=True)
_rng = np.random.default_rng(0)

spec = json.load(open(SPEC))
align_ch = [c["name"] for c in spec["channels"] if c["role"] == "align"]
readouts = [c["name"] for c in spec["channels"] if c["role"] == "readout"]

wide = pd.read_parquet(PANEL)
pr = pd.read_parquet(PIXREG)[["px", "py", "region_id", "region_name"]]
df = wide.merge(pr, on=["px", "py"], how="inner")
df = df[df["region_id"] != 0]


def grid_key(n):
    m = re.match(r"r(\d+)c(\d+)", str(n))
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)


regions = sorted(df["region_name"].unique(), key=grid_key)

n_written = 0
for idx, rn in enumerate(regions):
    g = df[df["region_name"] == rn]
    if MAX_SPOTS and len(g) > MAX_SPOTS:
        g = g.iloc[_rng.choice(len(g), MAX_SPOTS, replace=False)]
    tic = g["TIC"].to_numpy(np.float64)
    safe_tic = np.where(tic > 0, tic, 1.0)
    Xa = g[align_ch].to_numpy(np.float64) / safe_tic[:, None]  # TIC-normalized
    coords = g[["px", "py"]].to_numpy(np.float64)
    coords = coords - coords.mean(axis=0)
    a = ad.AnnData(X=Xa.astype(np.float32))
    a.var_names = align_ch
    a.obsm["spatial"] = coords
    a.obs["region"] = rn
    a.obs["region_id"] = int(g["region_id"].iloc[0])
    a.obs["sample"] = SAMPLE
    a.obs["px"] = g["px"].to_numpy()
    a.obs["py"] = g["py"].to_numpy()
    a.obs["TIC"] = tic
    for r in readouts:  # collaborator readouts
        a.obs[r] = g[r].to_numpy(np.float64) / safe_tic
    p = os.path.join(OUTDIR, f"slice_{idx:03d}.h5ad")
    a.write_h5ad(p)
    n_written += 1
print(f"wrote {n_written} slices to {OUTDIR}/ (slice_000..{n_written - 1:03d})")
print(
    f"  X = {len(align_ch)} align channels (TIC-normalized); readouts in "
    f"obs: {readouts}"
)
