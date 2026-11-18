#!/usr/bin/env python
"""Parallel per-pixel ion-image extraction for target peptides.
Reads m/z + intensity from the .ibd using a cached parquet offset table.
Caches results as parquet (feature table) + h5 (ion arrays) + PNG.

Usage: slurm_extract.py <offsets.parquet> <ibd> <sample> <ppm> <nworkers> [stage_dir]
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

OFFS, IBD, SAMPLE = sys.argv[1], sys.argv[2], sys.argv[3]
PPM = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
NWORK = int(sys.argv[5]) if len(sys.argv) > 5 else 64
STAGE = sys.argv[6] if len(sys.argv) > 6 else ""  # node-local dir, optional

TARGETS = [("sNPF_v1", 974.5891), ("Tk2", 993.4757)]
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ion_out")
os.makedirs(OUTDIR, exist_ok=True)

# Optionally stage the ibd to node-local fast storage.
if STAGE:
    os.makedirs(STAGE, exist_ok=True)
    local = os.path.join(STAGE, os.path.basename(IBD))
    if not os.path.exists(local) or os.path.getsize(local) != os.path.getsize(IBD):
        t = time.time()
        import shutil

        shutil.copy(IBD, local)
        print(
            f"staged ibd -> {local} in {time.time() - t:.0f}s "
            f"({os.path.getsize(local) / 1e9:.0f} GB)",
            flush=True,
        )
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
reachable = mz_off + mz_len * 8 <= fsize
print(
    f"{SAMPLE}: {N:,} pixels | ibd {fsize / 1e9:.1f} GB | reachable "
    f"{reachable.sum():,} ({100 * reachable.mean():.1f}%) | ppm ±{PPM} | "
    f"workers {NWORK}",
    flush=True,
)

tgt = np.array([t[1] for t in TARGETS])
lo, hi = tgt * (1 - PPM * 1e-6), tgt * (1 + PPM * 1e-6)

# Share read-only arrays with workers via globals (fork).
G = dict(
    mz_off=mz_off,
    mz_len=mz_len,
    int_off=int_off,
    int_len=int_len,
    reachable=reachable,
    lo=lo,
    hi=hi,
    IBD=IBD,
    nt=len(TARGETS),
)


def work(rng):
    a, b = rng
    g = G
    f = open(g["IBD"], "rb")
    out = np.zeros((b - a, g["nt"] + 1), dtype=np.float64)  # +1 for TIC (last col)
    mo, ml, io_, il, rc, lo_, hi_ = (
        g["mz_off"],
        g["mz_len"],
        g["int_off"],
        g["int_len"],
        g["reachable"],
        g["lo"],
        g["hi"],
    )
    for k, i in enumerate(range(a, b)):
        L = ml[i]
        if not rc[i] or L == 0:
            continue
        f.seek(int(mo[i]))
        mzs = np.frombuffer(f.read(int(L) * 8), dtype="<f8")
        f.seek(int(io_[i]))
        full_int = np.frombuffer(f.read(int(il[i]) * 4), dtype="<f4")
        out[k, g["nt"]] = full_int.sum()  # TIC
        for t in range(g["nt"]):
            i0 = np.searchsorted(mzs, lo_[t])
            i1 = np.searchsorted(mzs, hi_[t])
            if i1 > i0:
                out[k, t] = full_int[i0:i1].sum()
    f.close()
    return a, out


t0 = time.time()
chunk = (N // (NWORK * 8)) + 1
bounds = [(i, min(i + chunk, N)) for i in range(0, N, chunk)]
inten = np.zeros((N, len(TARGETS) + 1))  # last column = TIC
with mp.Pool(NWORK) as pool:
    done = 0
    for a, out in pool.imap_unordered(work, bounds):
        inten[a : a + len(out)] = out
        done += 1
        if done % 50 == 0:
            print(
                f"  {done}/{len(bounds)} chunks  ({time.time() - t0:.0f}s)", flush=True
            )
print(f"extracted {N:,} pixels in {time.time() - t0:.0f}s", flush=True)

# ---- cache: feature parquet ----
feat = pd.DataFrame({"px": px, "py": py})
for t, (name, _m) in enumerate(TARGETS):
    feat[name] = inten[:, t]
feat["TIC"] = inten[:, len(TARGETS)]
fpq = f"{OUTDIR}/features_{SAMPLE}_ppm{int(PPM)}.parquet"
feat.to_parquet(fpq, index=False)

# ---- cache: ion arrays h5 ----
import h5py

W, H = int(px.max()), int(py.max())
imgs = []
h5p = f"{OUTDIR}/ionimg_{SAMPLE}_ppm{int(PPM)}.h5"
with h5py.File(h5p, "w") as h:
    h.attrs["ppm"] = PPM
    h.attrs["sample"] = SAMPLE
    for t, (name, m) in enumerate(TARGETS):
        img = np.full((H, W), np.nan, np.float32)
        img[py - 1, px - 1] = inten[:, t]
        imgs.append(img)
        d = h.create_dataset(name, data=img, compression="gzip", compression_opts=1)
        d.attrs["mz"] = m
        nz = inten[:, t][inten[:, t] > 0]
        print(
            f"{name}: nonzero {len(nz):,}/{N:,} ({100 * len(nz) / N:.1f}%)  "
            f"max {inten[:, t].max():.0f}  "
            f"median(nz) {np.median(nz) if len(nz) else 0:.0f}",
            flush=True,
        )
    tic_img = np.full((H, W), np.nan, np.float32)
    tic_img[py - 1, px - 1] = inten[:, len(TARGETS)]
    h.create_dataset("TIC", data=tic_img, compression="gzip", compression_opts=1)

# ---- plot ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(TARGETS), 1, figsize=(16, 9), constrained_layout=True)
for ax, img, (name, m) in zip(np.atleast_1d(axes), imgs, TARGETS, strict=True):
    fin = img[np.isfinite(img)]
    vmax = np.percentile(fin[fin > 0], 99.5) if (fin > 0).any() else 1
    im = ax.imshow(
        img,
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )
    ax.set_title(f"{name}  (m/z {m}, ±{PPM:.0f} ppm)  —  {SAMPLE}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01, label="intensity (a.u.)")
png = f"{OUTDIR}/ionimg_{SAMPLE}_ppm{int(PPM)}.png"
fig.savefig(png, dpi=150)
print(f"WROTE:\n  {fpq}\n  {h5p}\n  {png}\nTOTAL {time.time() - t0:.0f}s", flush=True)
