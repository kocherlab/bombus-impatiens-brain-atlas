#!/usr/bin/env python
"""Extract the neuropeptide panel (+ collaborator sNPF/Tk readouts) per pixel
from the .ibd, at ±ppm, max-collapsing each channel's fragment windows. Outputs
a wide parquet (px, py, TIC, + one column per channel) for slice-building.

Usage: extract_panel.py <offsets.parquet> <ibd> <channel_spec.json> <sample>
           <out.parquet> [nworkers] [stage_dir]
"""

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

OFFS, IBD, SPEC, SAMPLE, OUT = sys.argv[1:6]
NWORK = int(sys.argv[6]) if len(sys.argv) > 6 else 64
STAGE = sys.argv[7] if len(sys.argv) > 7 else ""

spec = json.load(open(SPEC))
PPM = spec["ppm"]
chan_names = [c["name"] for c in spec["channels"]]
n_chan = len(chan_names)
# flatten targets: per target m/z -> channel index
tgt_mz, tgt_chan = [], []
for ci, c in enumerate(spec["channels"]):
    for mz in c["mz"]:
        tgt_mz.append(mz)
        tgt_chan.append(ci)
tgt_mz = np.array(tgt_mz, dtype=np.float64)
tgt_chan = np.array(tgt_chan, dtype=np.int32)
lo = tgt_mz * (1 - PPM * 1e-6)
hi = tgt_mz * (1 + PPM * 1e-6)
print(f"{SAMPLE}: {n_chan} channels, {len(tgt_mz)} target m/z, ±{PPM} ppm", flush=True)

if STAGE:
    os.makedirs(STAGE, exist_ok=True)
    local = os.path.join(STAGE, os.path.basename(IBD))
    if not os.path.exists(local) or os.path.getsize(local) != os.path.getsize(IBD):
        import shutil

        t = time.time()
        shutil.copy(IBD, local)
        print(f"staged ibd in {time.time() - t:.0f}s", flush=True)
    IBD = local

df = pd.read_parquet(OFFS)
px = df["px"].to_numpy(np.int32)
py = df["py"].to_numpy(np.int32)
mz_off = df["mz_off"].to_numpy(np.int64)
mz_len = df["mz_len"].to_numpy(np.int64)
int_off = df["int_off"].to_numpy(np.int64)
int_len = df["int_len"].to_numpy(np.int64)
N = len(px)
fsize = os.path.getsize(IBD)
reach = mz_off + mz_len * 8 <= fsize
print(f"{N:,} pixels, reachable {reach.sum():,}", flush=True)

G = dict(
    mz_off=mz_off,
    mz_len=mz_len,
    int_off=int_off,
    int_len=int_len,
    reach=reach,
    lo=lo,
    hi=hi,
    tgt_chan=tgt_chan,
    n_chan=n_chan,
    IBD=IBD,
)


def work(rng):
    a, b = rng
    g = G
    f = open(g["IBD"], "rb")
    out = np.zeros((b - a, g["n_chan"] + 1), dtype=np.float32)  # last col = TIC
    lo_, hi_, tc = g["lo"], g["hi"], g["tgt_chan"]
    mo, ml, io_, il, rc = (
        g["mz_off"],
        g["mz_len"],
        g["int_off"],
        g["int_len"],
        g["reach"],
    )
    for k, i in enumerate(range(a, b)):
        L = ml[i]
        if not rc[i] or L == 0:
            continue
        f.seek(int(mo[i]))
        mz = np.frombuffer(f.read(int(L) * 8), dtype="<f8")
        f.seek(int(io_[i]))
        inten = np.frombuffer(f.read(int(il[i]) * 4), dtype="<f4")
        out[k, g["n_chan"]] = inten.sum()  # TIC
        ilo = np.searchsorted(mz, lo_, "left")
        ihi = np.searchsorted(mz, hi_, "right")
        ne = np.nonzero(ihi > ilo)[0]  # targets with peaks
        for t in ne:
            v = inten[ilo[t] : ihi[t]].max()
            c = tc[t]
            if v > out[k, c]:
                out[k, c] = v  # max-collapse
    f.close()
    return a, out


t0 = time.time()
chunk = (N // (NWORK * 8)) + 1
bounds = [(i, min(i + chunk, N)) for i in range(0, N, chunk)]
vals = np.zeros((N, n_chan + 1), dtype=np.float32)
with mp.Pool(NWORK) as pool:
    done = 0
    for a, out in pool.imap_unordered(work, bounds):
        vals[a : a + len(out)] = out
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(bounds)} ({time.time() - t0:.0f}s)", flush=True)
print(f"extracted in {time.time() - t0:.0f}s", flush=True)

wide = pd.DataFrame({"px": px, "py": py})
for ci, name in enumerate(chan_names):
    wide[name] = vals[:, ci]
wide["TIC"] = vals[:, n_chan]
wide.to_parquet(OUT, index=False)
nz = {n: int((vals[:, i] > 0).sum()) for i, n in enumerate(chan_names)}
print(f"wrote {OUT}  ({wide.shape[0]:,} px x {wide.shape[1]} cols)", flush=True)
print(
    f"channels with >0 signal: {sum(v > 0 for v in nz.values())}/{n_chan}", flush=True
)
print(
    f"  sNPF_v1 nonzero: {nz.get('sNPF_v1')}, Tk2 nonzero: {nz.get('Tk2')}", flush=True
)
