"""Quantitative accuracy validation of the PASTE3 3D reconstruction.

Visual inspection and centroid drift check the PASTE3 stack qualitatively,
and the adjacent-slice tissue-occupancy Dice in the MOSAICField SI figure
describes a different method. This script computes
accuracy numbers for the PASTE3 alignment, for both brains, from data already
on disk.

What is computed
----------------
A. Adjacent-section tissue-occupancy Dice on the aligned coordinates, per
   brain, reusing `scripts/atlas/si_mosaicfield.py::_occupancy_dice` verbatim so
   the number is directly comparable to the 0.832 already in the SI.  Also a
   fixed-micron-grid version, which is the one that supports cross-condition
   comparisons (the bbox-relative binning changes physical resolution as a
   function of how well aligned the pair is).

B. A baseline ladder on the fixed grid: crop-frame native, per-slice
   centroid-centred, PASTE3, and centroid-centred + brute-force optimal
   rotation (the ceiling any rigid method could reach on this pair).

C. Anatomical-label agreement between adjacent sections, using the six
   hand-drawn region flags.  These are drawn per-slice on native coordinates
   by the `label` rule, which runs *before* `align` in the Snakemake DAG, so
   they are not an input to the registration and this is a non-circular test.
   Reported per flag (Dice) and in aggregate, with the denominator restricted
   to bins that carry a flag in at least one of the two sections.

D. A delta-z ladder (10/20/30/40 um) for both metrics.  Perfect registration
   does not give agreement 1.0 -- sections are 10 um apart and boundaries
   genuinely move -- so the ladder is extrapolated to delta-z = 0 to separate
   the registration-limited intercept from the real-anatomy slope.

E. Accumulated drift.  Pairwise metrics cannot see accumulation by
   construction (every pair can register well while each contributes a degree
   of rotation error and the stack corkscrews), so the test is non-pairwise:
   the six region centroids of each section are Procrustes-fitted onto
   section 0's configuration and the extracted rotation angle theta(z) is
   tested for a monotone ramp.

Coordinate units
----------------
`obsm['spatial_aligned']` is in Xenium *pixels* (0.2125 um each), so per-slice
extents there are 1 / 0.2125 times the same slices' native-micron extents in
the pre-alignment crops.  Everything here is converted to microns
so bin sizes are physically interpretable.  Note that in the aligned objects
`obsm['spatial']` was overwritten with the aligned coordinates (the two arrays
are byte-identical), so the pre-alignment coordinates used for the baseline
ladder are pulled from the per-slice crop h5ads and matched by `cell_id`.

Run:  PYTHONPATH=src .venv/bin/python scripts/atlas/paste3_registration_accuracy.py
"""

from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from scipy import stats

faulthandler.enable()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (
    STATISTICAL_REGIONS,  # noqa: E402
    attach_region_flags,  # noqa: E402
)

REPO = Path(__file__).resolve().parents[2]
COMBINED = REPO / "pipelines/combined_atlas/data/combined_clustered.h5ad"
CROP_DIRS = {
    "queen": REPO / "pipelines/wholebrain/queen/data",
    "worker": REPO / "pipelines/wholebrain/worker/data",
}
OUT = REPO / "results/reframe/registration_accuracy"
CACHE = OUT / "_cache.npz"

PX_UM = 0.2125  # Xenium pixel size; aligned coords are in pixels
BIN_UM = 75.0  # fixed-grid bin size (~ the bbox-relative 1/40 bin)
BIN_FRACS = (1 / 40.0, 1 / 80.0)
# STATISTICAL_REGIONS = the original six hand-drawn regions + antennal lobe
# (nine_region_common). Ellipsoid body / fan-shaped body are deliberately
# excluded -- see nine_region_common.DESCRIPTIVE_ONLY_REGIONS.
FLAGS = [f"in_{r}" for r in STATISTICAL_REGIONS]
COMMON_N = 15000  # slices subsampled to a common n for fixed-grid work
SEED = 0
MIN_FLAG_CELLS = 50  # per-slice minimum for a flag to give a usable centroid


# -------------------------- cache --------------------------


def _cat(grp):
    return (np.array([s.decode() for s in grp["categories"][:]]), grp["codes"][:])


def build_cache():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[cache] reading {COMBINED.name} via h5py ...", flush=True)
    f = h5py.File(COMBINED, "r")
    o = f["obs"]
    brain_cats, brain_codes = _cat(o["brain"])
    slice_cats, slice_codes = _cat(o["slice_name"])
    cid_cats, cid_codes = _cat(o["cell_id"])
    d = {
        "aligned_um": f["obsm"]["spatial_aligned"][:].astype(np.float32) * PX_UM,
        "brain_cats": brain_cats,
        "brain_codes": brain_codes,
        "slice_cats": slice_cats,
        "slice_codes": slice_codes,
        "z_level": o["z_level"][:].astype(np.float32),
    }
    cell_id = cid_cats[cid_codes]
    slice_name = slice_cats[slice_codes]
    # The six original flags baked into this h5ad are not read here; all seven
    # are joined fresh below.
    f.close()

    # Join all seven fitted flags from the crop sidecars. The six baked in by
    # labeling.py are frozen at whatever it last wrote, but polygon repair
    # moves cells (e.g. four polygons on worker 0068401 R1 moved 119 cells
    # from the lobula to the antennal lobe); mixing stale and fresh flags would
    # make those cells both in_lobula and in_al and corrupt every per-flag Dice
    # and centroid below.
    #
    # in_al is a nullable boolean: pd.NA means "this section has no antennal-lobe
    # tracing yet" (worker slide 0069041 in full, plus section 4 of 0068401), not
    # "not antennal lobe". The value and the traced bit are cached separately
    # (in_al / al_traced); filling untraced cells with False would create a caste
    # difference out of missing data, so every downstream use of the al flag in a
    # Dice/aggregate/centroid computation gates on al_traced.
    print("[cache] joining all seven region flags from crop sidecars ...", flush=True)
    ns = SimpleNamespace(
        obs=pd.DataFrame({"slice_name": slice_name, "cell_id": cell_id})
    )
    attach_region_flags(ns, regions=STATISTICAL_REGIONS)
    for r in FLAGS:
        d[r] = ns.obs[r].to_numpy(dtype=bool, na_value=False)
    al = ns.obs["in_al"]
    d["al_traced"] = al.notna().to_numpy()

    # native (pre-alignment) coordinates, matched by cell_id within slice
    print("[cache] pulling native coords from per-slice crops ...", flush=True)
    raw = np.full_like(d["aligned_um"], np.nan)
    n_missing = 0
    for si, sname in enumerate(slice_cats):
        bi = int(d["brain_codes"][d["slice_codes"] == si][0])
        cdir = CROP_DIRS[brain_cats[bi]]
        p = cdir / f"{sname}.h5ad"
        if not p.exists():
            print(f"  MISSING crop for {sname}", flush=True)
            continue
        cf = h5py.File(p, "r")
        cid = np.array([s.decode() for s in cf["obs"]["cell_id"][:]])
        cxy = cf["obsm"]["spatial"][:].astype(np.float32)  # microns, crop frame
        cf.close()
        lut = {c: k for k, c in enumerate(cid)}
        m = np.flatnonzero(d["slice_codes"] == si)
        pos = np.array([lut.get(c, -1) for c in cell_id[m]])
        hit = pos >= 0
        n_missing += int((~hit).sum())
        raw[m[hit]] = cxy[pos[hit]]
    print(f"[cache] unmatched cells: {n_missing}", flush=True)
    d["raw_um"] = raw
    np.savez_compressed(CACHE, **d)
    print(f"[cache] wrote {CACHE}", flush=True)


def _newest_input_mtime():
    """Newest mtime across the region flags this cache's labels come from."""
    paths = sorted((REPO / "results/nine_region/crops").glob("*/*.parquet"))
    # The cache also depends on the code that fills it: changing how flags are
    # joined changes the values while every data mtime stays put, so this script
    # and the vocabulary module are included.
    paths += [
        Path(__file__).resolve(),
        REPO / "src" / "bimp_st" / "nine_region_common.py",
    ]
    return max((p.stat().st_mtime for p in paths), default=0.0)


def load_cache():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        # A cache is reused only if it has the in_al/al_traced keys (older caches
        # carry six flags only) and is newer than its inputs; a relabel leaves the
        # schema intact but the values stale, so freshness is checked too.
        fresh = CACHE.stat().st_mtime >= _newest_input_mtime()
        if fresh and all(
            k in z for k in ("raw_um", "aligned_um", "in_al", "al_traced")
        ):
            return z
        if not fresh:
            print(
                f"[cache] {CACHE.name} is older than the region flags -- rebuilding",
                flush=True,
            )
    build_cache()
    return np.load(CACHE, allow_pickle=True)


# -------------------------- metrics --------------------------


def occupancy_dice_bboxrel(pts_a, pts_b, bin_frac=1.0 / 40):
    """Verbatim reimplementation of si_mosaicfield.py::_occupancy_dice.

    Bin size is a fraction of each pair's own bounding-box extent.
    """
    both = np.vstack([pts_a, pts_b])
    lo = np.percentile(both, 0.5, axis=0)
    hi = np.percentile(both, 99.5, axis=0)
    span = hi - lo
    span[span <= 0] = 1.0
    nbins = max(8, int(round(1.0 / bin_frac)))
    edges = [np.linspace(lo[d], hi[d], nbins + 1) for d in range(2)]
    ha, _, _ = np.histogram2d(pts_a[:, 0], pts_a[:, 1], bins=edges)
    hb, _, _ = np.histogram2d(pts_b[:, 0], pts_b[:, 1], bins=edges)
    ma, mb = ha > 0, hb > 0
    denom = ma.sum() + mb.sum()
    return 2 * (ma & mb).sum() / denom if denom > 0 else np.nan


def _fixed_grid(pts_a, pts_b, bin_um):
    both = np.vstack([pts_a, pts_b])
    lo = np.percentile(both, 0.5, axis=0)
    hi = np.percentile(both, 99.5, axis=0)
    nb = [max(2, int(np.ceil((hi[d] - lo[d]) / bin_um))) for d in range(2)]
    edges = [lo[d] + bin_um * np.arange(nb[d] + 1) for d in range(2)]
    return lo, np.array(nb), edges


def occupancy_dice_fixed(pts_a, pts_b, bin_um=BIN_UM):
    """Dice on a grid whose bin size is a fixed physical length."""
    _, _, edges = _fixed_grid(pts_a, pts_b, bin_um)
    ha, _, _ = np.histogram2d(pts_a[:, 0], pts_a[:, 1], bins=edges)
    hb, _, _ = np.histogram2d(pts_b[:, 0], pts_b[:, 1], bins=edges)
    ma, mb = ha > 0, hb > 0
    denom = ma.sum() + mb.sum()
    return 2 * (ma & mb).sum() / denom if denom > 0 else np.nan


def _bin_ids(pts, lo, nb, bin_um):
    ij = np.floor((pts - lo) / bin_um).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < nb[0]) & (ij[:, 1] < nb[1])
    return ij[ok, 0] * nb[1] + ij[ok, 1], ok


def label_agreement(
    pts_a, flags_a, pts_b, flags_b, bin_um=BIN_UM, traced_a=None, traced_b=None
):
    """Anatomical-label agreement between two sections on a fixed grid.

    Returns per-flag Dice of the bin sets each flag occupies, plus an
    aggregate dominant-label agreement whose DENOMINATOR is bins occupied by
    cells in both sections AND carrying at least one flag in at least one of
    them (unflagged<->unflagged bins are excluded as trivially matching).

    `traced_a`/`traced_b` mark, per cell per flag, whether that region was
    actually traced on that cell's section (see build_cache -- the antennal
    lobe is untraced, not absent, on part of the worker). A flag untraced on
    EITHER side of the pair is dropped from `usable` entirely: excluded from
    its own Dice (reported NaN, not a spurious 0) AND from the dominant-label
    aggregate, which would otherwise read the untraced side as "never this
    region" and score every bin where the traced side's dominant label is
    that region as a disagreement -- not a lack of data. Default is
    "everything traced", matching the original six flags that are always
    fully drawn.
    """
    lo, nb, _ = _fixed_grid(pts_a, pts_b, bin_um)
    ia, oka = _bin_ids(pts_a, lo, nb, bin_um)
    ib, okb = _bin_ids(pts_b, lo, nb, bin_um)
    fa, fb = flags_a[oka], flags_b[okb]
    nbin = int(nb[0] * nb[1])

    if traced_a is None:
        traced_a = np.ones_like(flags_a, dtype=bool)
    if traced_b is None:
        traced_b = np.ones_like(flags_b, dtype=bool)
    ta, tb = traced_a[oka], traced_b[okb]
    # Tracing is section-level (all-or-nothing per region), so any() and
    # all() agree here; any() is used since it is well-defined even if a
    # section's grid-clipped subset happens to contain zero cells.
    usable = [k for k in range(len(FLAGS)) if ta[:, k].any() and tb[:, k].any()]

    occ_a = np.zeros(nbin, bool)
    occ_a[ia] = True
    occ_b = np.zeros(nbin, bool)
    occ_b[ib] = True

    cnt_a = np.zeros((nbin, len(usable)), np.int32)
    cnt_b = np.zeros((nbin, len(usable)), np.int32)
    for j, k in enumerate(usable):
        np.add.at(cnt_a[:, j], ia[fa[:, k]], 1)
        np.add.at(cnt_b[:, j], ib[fb[:, k]], 1)

    per_flag = {}
    for j, k in enumerate(usable):
        name = FLAGS[k]
        sa, sb = cnt_a[:, j] > 0, cnt_b[:, j] > 0
        na, nb_ = int(sa.sum()), int(sb.sum())
        den = na + nb_
        # Dice <= 2*min(|A|,|B|)/(|A|+|B|).  If the two sections annotate very
        # different areas for a region, Dice is capped BELOW 1 no matter how
        # perfect the registration is, so the raw Dice alone cannot be read as
        # a registration score.
        bound = (2 * min(na, nb_) / den) if den else float("nan")
        obs = (2 * (sa & sb).sum() / den) if den else float("nan")
        per_flag[name] = {
            "dice": float(obs),
            "dice_coverage_bound": float(bound),
            "dice_frac_of_bound": float(obs / bound)
            if den and bound > 0
            else float("nan"),
            "bins_a": na,
            "bins_b": nb_,
        }
    for name in FLAGS:
        if name not in per_flag:
            per_flag[name] = {
                "dice": float("nan"),
                "dice_coverage_bound": float("nan"),
                "dice_frac_of_bound": float("nan"),
                "bins_a": 0,
                "bins_b": 0,
            }

    if usable:
        any_a, any_b = cnt_a.sum(1) > 0, cnt_b.sum(1) > 0
        usable_arr = np.array(usable, dtype=np.int64)
        dom_a = np.where(any_a, usable_arr[cnt_a.argmax(1)], -1)
        dom_b = np.where(any_b, usable_arr[cnt_b.argmax(1)], -1)
    else:
        any_a = any_b = np.zeros(nbin, bool)
        dom_a = dom_b = np.full(nbin, -1, dtype=np.int64)
    consider = occ_a & occ_b & (any_a | any_b)
    n_con = int(consider.sum())
    match = int(((dom_a == dom_b) & (dom_a >= 0) & consider).sum())
    # strict subset: both sections flagged -- does the flag identity agree?
    both_f = consider & any_a & any_b
    n_both = int(both_f.sum())
    id_match = int(((dom_a == dom_b) & both_f).sum())
    return {
        "per_flag": per_flag,
        "agreement": match / n_con if n_con else float("nan"),
        "n_considered_bins": n_con,
        "identity_agreement_both_flagged": (id_match / n_both)
        if n_both
        else float("nan"),
        "n_both_flagged_bins": n_both,
        "flagged_bins_co_occupied_frac": (n_both / n_con) if n_con else float("nan"),
    }


def _rot(pts, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return pts @ np.array([[c, -s], [s, c]], dtype=np.float64).T


def best_rotation_dice(pts_a, pts_b, bin_um=BIN_UM):
    """Ceiling for a rigid (rotation+translation) registration of this pair."""
    a = pts_a - pts_a.mean(0)
    b = pts_b - pts_b.mean(0)
    coarse = np.arange(-180, 180, 5.0)
    sc = [occupancy_dice_fixed(a, _rot(b, g), bin_um) for g in coarse]
    g0 = coarse[int(np.argmax(sc))]
    fine = np.arange(g0 - 5, g0 + 5.01, 1.0)
    sf = [occupancy_dice_fixed(a, _rot(b, g), bin_um) for g in fine]
    return float(np.max(sf)), float(fine[int(np.argmax(sf))])


def residual_rigid(pts_a, pts_b, bin_um=50.0, max_deg=12.0, max_shift_um=100.0):
    """Residual rigid misalignment of an ALREADY-ALIGNED adjacent pair.

    Rotates/shifts section B about the pair's common centroid to maximise
    occupancy Dice against section A.  Under a perfect pairwise registration
    the optimum sits at (0 deg, 0 um).  Because PASTE3 composes pairwise
    transforms sequentially, a residual with a CONSISTENT SIGN accumulates
    linearly along the stack, whereas sign-free residual accumulates as
    sqrt(n) -- which is what distinguishes systematic drift from noise.
    """
    c = np.vstack([pts_a, pts_b]).mean(0)
    a = pts_a - c
    b = pts_b - c
    grid = np.arange(-max_deg, max_deg + 1e-9, 0.5)
    sc = np.array([occupancy_dice_fixed(a, _rot(b, g), bin_um) for g in grid])
    deg = float(grid[int(np.argmax(sc))])
    br = _rot(b, deg)
    steps = np.arange(-max_shift_um, max_shift_um + 1e-9, bin_um / 2.0)
    best, bs = -1.0, (0.0, 0.0)
    for dx in steps:
        for dy in steps:
            v = occupancy_dice_fixed(a, br + np.array([dx, dy]), bin_um)
            if v > best:
                best, bs = v, (float(dx), float(dy))
    base = occupancy_dice_fixed(a, b, bin_um)
    return {
        "deg": deg,
        "dx_um": bs[0],
        "dy_um": bs[1],
        "dice_as_is": float(base),
        "dice_best": float(best),
    }


def kabsch_angle(P, Q):
    """Rotation (deg) taking configuration P onto Q, translation removed."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    rmsd = float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), rmsd


# -------------------------- driver --------------------------


def slices_for(d, brain):
    bi = int(np.flatnonzero(d["brain_cats"] == brain)[0])
    m = d["brain_codes"] == bi
    z = d["z_level"][m]
    order = sorted(np.unique(z))
    idx_all = np.flatnonzero(m)
    return [idx_all[z == t] for t in order], order


def subsample(idx, rng, n=COMMON_N):
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def main():
    d = load_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    aligned = d["aligned_um"]
    raw = d["raw_um"]
    flagmat = np.column_stack([d[r] for r in FLAGS])
    # Per-cell, per-flag "was this region traced on this cell's section".
    # Always True for the original six (baked into the h5ad, never null);
    # for the antennal lobe it is al_traced, which is False wherever in_al
    # is pd.NA. label_agreement() uses this to exclude untraced sides from
    # the al Dice/aggregate rather than reading them as "not antennal lobe".
    tracedmat = np.ones_like(flagmat, dtype=bool)
    tracedmat[:, FLAGS.index("in_al")] = d["al_traced"]
    res = {
        "units": "microns",
        "px_um": PX_UM,
        "bin_um": BIN_UM,
        "common_n": COMMON_N,
        "seed": SEED,
    }
    rows = []

    for brain in ["queen", "worker"]:
        rng = np.random.default_rng(SEED)
        sl, zs = slices_for(d, brain)
        nsl = len(sl)
        sub = [subsample(s, rng) for s in sl]
        print(
            f"\n=== {brain}: {nsl} sections, "
            f"{min(len(s) for s in sl)}-{max(len(s) for s in sl)} cells "
            f"(subsampled to {min(COMMON_N, min(len(s) for s in sl))})",
            flush=True,
        )

        B = {"n_sections": nsl, "cells_per_section": [int(len(s)) for s in sl]}

        # ---- A. occupancy Dice, bbox-relative (comparable to SI 0.832) ----
        for bf in BIN_FRACS:
            v = np.array(
                [
                    occupancy_dice_bboxrel(aligned[sl[i]], aligned[sl[i + 1]], bf)
                    for i in range(nsl - 1)
                ]
            )
            key = f"occupancy_dice_bboxrel_1over{int(round(1 / bf))}"
            B[key] = {
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)),
                "min": float(v.min()),
                "max": float(v.max()),
                "n_pairs": int(len(v)),
                "per_pair": [round(float(x), 4) for x in v],
            }
            print(
                f"  {key}: {v.mean():.4f} +/- {v.std(ddof=1):.4f} "
                f"(min {v.min():.3f}, n={len(v)})",
                flush=True,
            )

        # ---- A/B. fixed-micron grid, baseline ladder ----
        ladder = {}
        for cond in ["native_cropframe", "native_centred", "paste3", "rigid_ceiling"]:
            vals, angs = [], []
            for i in range(nsl - 1):
                a_i, b_i = sub[i], sub[i + 1]
                if cond == "paste3":
                    pa, pb = aligned[a_i], aligned[b_i]
                else:
                    pa, pb = raw[a_i], raw[b_i]
                    pa = pa[~np.isnan(pa).any(1)]
                    pb = pb[~np.isnan(pb).any(1)]
                    if cond != "native_cropframe":
                        pa = pa - pa.mean(0)
                        pb = pb - pb.mean(0)
                if cond == "rigid_ceiling":
                    v, g = best_rotation_dice(pa, pb)
                    angs.append(g)
                else:
                    v = occupancy_dice_fixed(pa, pb)
                vals.append(v)
            vals = np.array(vals)
            ladder[cond] = {
                "mean": float(np.nanmean(vals)),
                "sd": float(np.nanstd(vals, ddof=1)),
                "min": float(np.nanmin(vals)),
                "per_pair": [round(float(x), 4) for x in vals],
            }
            if angs:
                ladder[cond]["best_angle_deg"] = [round(a, 1) for a in angs]
            print(
                f"  fixed-{BIN_UM:.0f}um Dice [{cond}]: "
                f"{np.nanmean(vals):.4f} +/- {np.nanstd(vals, ddof=1):.4f} "
                f"(min {np.nanmin(vals):.3f})",
                flush=True,
            )
        B["occupancy_dice_fixed_grid"] = ladder

        # ---- 90-degree misregistration null ----
        null = []
        for i in range(nsl - 1):
            pa = aligned[sub[i]]
            pb = aligned[sub[i + 1]]
            null.append(
                occupancy_dice_fixed(pa - pa.mean(0), _rot(pb - pb.mean(0), 90.0))
            )
        B["null_90deg_rotation_dice"] = {
            "mean": float(np.mean(null)),
            "sd": float(np.std(null, ddof=1)),
        }
        print(f"  90-deg misregistration null: {np.mean(null):.4f}", flush=True)

        # ---- C/D. label agreement, delta-z ladder ----
        lab = {}
        for step in [1, 2, 3, 4]:
            pf = {k: [] for k in FLAGS}
            pb_ = {k: [] for k in FLAGS}
            agg, ident, nb_con, occ, covfrac = [], [], [], [], []
            for i in range(nsl - step):
                a_i, b_i = sub[i], sub[i + step]
                r = label_agreement(
                    aligned[a_i],
                    flagmat[a_i],
                    aligned[b_i],
                    flagmat[b_i],
                    traced_a=tracedmat[a_i],
                    traced_b=tracedmat[b_i],
                )
                for k in FLAGS:
                    pf[k].append(r["per_flag"][k]["dice"])
                    pb_[k].append(r["per_flag"][k]["dice_coverage_bound"])
                agg.append(r["agreement"])
                ident.append(r["identity_agreement_both_flagged"])
                nb_con.append(r["n_considered_bins"])
                covfrac.append(r["flagged_bins_co_occupied_frac"])
                occ.append(occupancy_dice_fixed(aligned[a_i], aligned[b_i]))
            dz = step * 10
            lab[dz] = {
                "n_pairs": nsl - step,
                "per_flag_dice": {
                    k: {
                        "mean": float(np.nanmean(pf[k])),
                        "sd": float(np.nanstd(pf[k], ddof=1)),
                        "coverage_bound_mean": float(np.nanmean(pb_[k])),
                        "frac_of_bound": float(np.nanmean(pf[k]) / np.nanmean(pb_[k])),
                        "n_finite": int(np.isfinite(pf[k]).sum()),
                    }
                    for k in FLAGS
                },
                "aggregate_agreement": {
                    "mean": float(np.nanmean(agg)),
                    "sd": float(np.nanstd(agg, ddof=1)),
                    "coverage_bound_mean": float(np.nanmean(covfrac)),
                    "frac_of_bound": float(np.nanmean(agg) / np.nanmean(covfrac)),
                },
                "identity_agreement_both_flagged": {
                    "mean": float(np.nanmean(ident)),
                    "sd": float(np.nanstd(ident, ddof=1)),
                },
                "median_considered_bins": float(np.median(nb_con)),
                "occupancy_dice_fixed": {
                    "mean": float(np.nanmean(occ)),
                    "sd": float(np.nanstd(occ, ddof=1)),
                },
            }
            if step == 1:
                print("  label agreement (dz=10um):", flush=True)
                for k in FLAGS:
                    print(
                        f"      {k:<11s} Dice {np.nanmean(pf[k]):.3f} "
                        f"+/- {np.nanstd(pf[k], ddof=1):.3f}   "
                        f"coverage bound {np.nanmean(pb_[k]):.3f} "
                        f"({np.nanmean(pf[k]) / np.nanmean(pb_[k]):.2f} of "
                        f"bound)",
                        flush=True,
                    )
                print(
                    f"      aggregate dominant-label agreement "
                    f"{np.nanmean(agg):.3f} (coverage bound "
                    f"{np.nanmean(covfrac):.3f}, "
                    f"{np.nanmean(agg) / np.nanmean(covfrac):.2f} of bound); "
                    f"both-flagged identity {np.nanmean(ident):.3f}",
                    flush=True,
                )

        # ---- C2. does registration beat naive centring for ANATOMY? ----
        # and how far apart, in microns, are the same region's centroids in
        # adjacent sections?  (the directly interpretable "same anatomy in the
        # same place" number)
        cmp_cond = {}
        for cond in ["native_centred", "paste3"]:
            pf = {k: [] for k in FLAGS}
            bnd = {k: [] for k in FLAGS}
            off = {k: [] for k in FLAGS}
            rad = {k: [] for k in FLAGS}
            agg = []
            for i in range(nsl - 1):
                a_i, b_i = sub[i], sub[i + 1]
                if cond == "paste3":
                    pa, pb = aligned[a_i], aligned[b_i]
                else:
                    pa, pb = raw[a_i].copy(), raw[b_i].copy()
                    pa = pa - np.nanmean(pa, 0)
                    pb = pb - np.nanmean(pb, 0)
                fa, fb = flagmat[a_i], flagmat[b_i]
                ta, tb = tracedmat[a_i], tracedmat[b_i]
                keep_a = ~np.isnan(pa).any(1)
                keep_b = ~np.isnan(pb).any(1)
                r = label_agreement(
                    pa[keep_a],
                    fa[keep_a],
                    pb[keep_b],
                    fb[keep_b],
                    traced_a=ta[keep_a],
                    traced_b=tb[keep_b],
                )
                agg.append(r["agreement"])
                for k, name in enumerate(FLAGS):
                    pf[name].append(r["per_flag"][name]["dice"])
                    bnd[name].append(r["per_flag"][name]["dice_coverage_bound"])
                    # Untraced al cells are False in fa/fb (see build_cache),
                    # so an untraced section contributes 0 cells here and the
                    # MIN_FLAG_CELLS gate below drops that pair's offset/
                    # radius for al without needing traced_a/traced_b -- it
                    # only guards label_agreement's own Dice/aggregate above.
                    ca = pa[keep_a][fa[keep_a][:, k]]
                    cb = pb[keep_b][fb[keep_b][:, k]]
                    if len(ca) >= MIN_FLAG_CELLS and len(cb) >= MIN_FLAG_CELLS:
                        off[name].append(float(np.linalg.norm(ca.mean(0) - cb.mean(0))))
                        rad[name].append(
                            float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
                        )
            cmp_cond[cond] = {
                "aggregate_agreement_mean": float(np.nanmean(agg)),
                "per_flag_dice_mean": {k: float(np.nanmean(pf[k])) for k in FLAGS},
                "per_flag_dice_coverage_bound_mean": {
                    k: float(np.nanmean(bnd[k])) for k in FLAGS
                },
                "per_flag_centroid_offset_um": {
                    k: {
                        "median": float(np.median(off[k])) if off[k] else np.nan,
                        "mean": float(np.mean(off[k])) if off[k] else np.nan,
                        "p90": float(np.percentile(off[k], 90)) if off[k] else np.nan,
                        "n_pairs": len(off[k]),
                    }
                    for k in FLAGS
                },
                "per_flag_rms_radius_um": {
                    k: float(np.mean(rad[k])) if rad[k] else np.nan for k in FLAGS
                },
            }
        B["labels_registration_vs_centring"] = cmp_cond
        print(
            "  region-centroid offset between adjacent sections (um, "
            "median | region RMS radius):",
            flush=True,
        )
        for k in FLAGS:
            o1 = cmp_cond["native_centred"]["per_flag_centroid_offset_um"][k]
            o2 = cmp_cond["paste3"]["per_flag_centroid_offset_um"][k]
            rr_ = cmp_cond["paste3"]["per_flag_rms_radius_um"][k]
            print(
                f"      {k:<11s} centred {o1['median']:7.1f}  PASTE3 "
                f"{o2['median']:7.1f}   (radius {rr_:.0f})",
                flush=True,
            )
        print(
            f"      per-flag label Dice, centred -> PASTE3: "
            f"{np.mean(list(cmp_cond['native_centred']['per_flag_dice_mean'].values())):.3f}"
            f" -> "
            f"{np.mean(list(cmp_cond['paste3']['per_flag_dice_mean'].values())):.3f}",
            flush=True,
        )

        # ---- C2b. label-agreement null: deliberately misregistered pair ----
        # 0.74 "both-flagged identity" is only interpretable against the value
        # a KNOWN-BAD registration produces on the same sections.
        nag, nid, npf = [], [], {k: [] for k in FLAGS}
        for i in range(nsl - 1):
            a_i, b_i = sub[i], sub[i + 1]
            pa = aligned[a_i]
            pb = aligned[b_i]
            pb90 = _rot(pb - pb.mean(0), 90.0) + pa.mean(0)
            r = label_agreement(
                pa,
                flagmat[a_i],
                pb90,
                flagmat[b_i],
                traced_a=tracedmat[a_i],
                traced_b=tracedmat[b_i],
            )
            nag.append(r["agreement"])
            nid.append(r["identity_agreement_both_flagged"])
            for k in FLAGS:
                npf[k].append(r["per_flag"][k]["dice"])
        B["label_null_90deg"] = {
            "aggregate_agreement": float(np.nanmean(nag)),
            "identity_agreement_both_flagged": float(np.nanmean(nid)),
            "per_flag_dice": {k: float(np.nanmean(npf[k])) for k in FLAGS},
        }
        print(
            f"  90-deg-misregistered label null: aggregate "
            f"{np.nanmean(nag):.3f}, both-flagged identity "
            f"{np.nanmean(nid):.3f}, mean per-flag Dice "
            f"{np.nanmean([np.nanmean(npf[k]) for k in FLAGS]):.3f}",
            flush=True,
        )

        # ---- C3. within- vs across-scan-region pairs ----
        # each Xenium scan region carries two consecutive sections annotated in
        # one sitting; pairs that cross a scan-region boundary additionally
        # carry annotator redraw and cross-run variation.
        def _reg(i):
            nm = str(d["slice_cats"][d["slice_codes"][sl[i][0]]])
            return nm.rsplit("_", 1)[0]

        within, across = [], []
        wl, al = [], []
        for i in range(nsl - 1):
            a_i, b_i = sub[i], sub[i + 1]
            occ = occupancy_dice_fixed(aligned[a_i], aligned[b_i])
            lb = label_agreement(
                aligned[a_i],
                flagmat[a_i],
                aligned[b_i],
                flagmat[b_i],
                traced_a=tracedmat[a_i],
                traced_b=tracedmat[b_i],
            )
            tgt_o, tgt_l = (within, wl) if _reg(i) == _reg(i + 1) else (across, al)
            tgt_o.append(occ)
            tgt_l.append(lb["agreement"])
        B["within_vs_across_scan_region"] = {
            "n_within": len(within),
            "n_across": len(across),
            "occupancy_dice": {
                "within": float(np.mean(within)),
                "across": float(np.mean(across)),
                "mwu_p": float(stats.mannwhitneyu(within, across).pvalue),
            },
            "label_agreement": {
                "within": float(np.nanmean(wl)),
                "across": float(np.nanmean(al)),
                "mwu_p": float(stats.mannwhitneyu(wl, al).pvalue),
            },
        }
        print(
            f"  within-scan-region pairs (n={len(within)}) vs across "
            f"(n={len(across)}): occupancy Dice "
            f"{np.mean(within):.4f} vs {np.mean(across):.4f} "
            f"(p={stats.mannwhitneyu(within, across).pvalue:.3g}); "
            f"label agreement {np.nanmean(wl):.3f} vs {np.nanmean(al):.3f} "
            f"(p={stats.mannwhitneyu(wl, al).pvalue:.3g})",
            flush=True,
        )

        # extrapolate to dz = 0
        dzs = np.array([10, 20, 30, 40], float)
        extrap = {}
        # B023: these close over `lab` from the enclosing loop, but every
        # getter is consumed by the `for name, getter in getters` loop directly
        # below, inside the same iteration, so late binding cannot bite.
        getters = [
            ("occupancy_dice_fixed", lambda s: lab[s]["occupancy_dice_fixed"]["mean"]),  # noqa: B023
            ("aggregate_agreement", lambda s: lab[s]["aggregate_agreement"]["mean"]),  # noqa: B023
        ]
        for k in FLAGS:
            getters.append(
                (
                    f"flag::{k}",
                    (
                        lambda kk: (
                            lambda s: (  # noqa: B023  (see note above)
                                lab[s]["per_flag_dice"][kk]["mean"]
                            )
                        )
                    )(k),
                )
            )
        for name, getter in getters:
            y = np.array([getter(int(z)) for z in dzs])
            lr = stats.linregress(dzs, y)
            extrap[name] = {
                "intercept_dz0": float(lr.intercept),
                "slope_per_um": float(lr.slope),
                "r2": float(lr.rvalue**2),
                "values": [round(float(val), 4) for val in y],
            }
        B["label_agreement_dz_ladder"] = lab
        B["dz0_extrapolation"] = extrap
        print(
            f"  dz->0 extrapolation: occupancy Dice "
            f"{extrap['occupancy_dice_fixed']['intercept_dz0']:.3f}, "
            f"aggregate label agreement "
            f"{extrap['aggregate_agreement']['intercept_dz0']:.3f}",
            flush=True,
        )

        # ---- E1. accumulated drift: residual rigid misalignment ----
        rr = [
            residual_rigid(aligned[sl[i]], aligned[sl[i + 1]]) for i in range(nsl - 1)
        ]
        dg = np.array([r["deg"] for r in rr])
        dxy = np.array([[r["dx_um"], r["dy_um"]] for r in rr])
        gain = np.array([r["dice_best"] - r["dice_as_is"] for r in rr])
        t_stat, t_p = stats.ttest_1samp(dg, 0.0)
        n_pos = int((dg > 0).sum())
        n_nz = int((dg != 0).sum())
        sign_p = float(stats.binomtest(n_pos, n_nz, 0.5).pvalue) if n_nz else np.nan
        cum_rot = float(dg.sum())
        rw_rot = float(np.sqrt((dg**2).sum()))  # sign-free (random-walk)
        cum_xy = dxy.sum(0)
        rw_xy = float(np.sqrt((dxy**2).sum()))
        # sensitivity check: inject a known +2 deg error into every pair
        inj = np.array(
            [
                residual_rigid(
                    aligned[sl[i]],
                    _rot(aligned[sl[i + 1]] - aligned[sl[i + 1]].mean(0), 2.0)
                    + aligned[sl[i + 1]].mean(0),
                )["deg"]
                for i in range(min(5, nsl - 1))
            ]
        )
        B["residual_rigid_drift"] = {
            "bin_um": 50.0,
            "residual_rotation_deg": [round(float(x), 2) for x in dg],
            "mean_deg": float(dg.mean()),
            "sd_deg": float(dg.std(ddof=1)),
            "median_abs_deg": float(np.median(np.abs(dg))),
            "ttest_vs_zero_p": float(t_p),
            "t_stat": float(t_stat),
            "n_positive": n_pos,
            "n_nonzero": n_nz,
            "sign_test_p": sign_p,
            "cumulative_rotation_deg": cum_rot,
            "randomwalk_rotation_deg": rw_rot,
            "residual_shift_um": [
                [round(float(a), 1), round(float(b_), 1)] for a, b_ in dxy
            ],
            "median_abs_shift_um": float(np.median(np.abs(dxy))),
            "cumulative_shift_um": [float(cum_xy[0]), float(cum_xy[1])],
            "cumulative_shift_norm_um": float(np.linalg.norm(cum_xy)),
            "randomwalk_shift_um": rw_xy,
            "dice_gain_from_residual_correction": {
                "mean": float(gain.mean()),
                "max": float(gain.max()),
            },
            "injected_2deg_recovered_deg": [round(float(x), 2) for x in inj],
        }
        print(
            f"  residual rotation per pair: mean {dg.mean():+.2f} deg "
            f"(sd {dg.std(ddof=1):.2f}, median|.| {np.median(np.abs(dg)):.2f}), "
            f"t-test vs 0 p={t_p:.3g}, sign test p={sign_p:.3g}",
            flush=True,
        )
        print(
            f"  cumulative rotation if summed: {cum_rot:+.1f} deg; "
            f"random-walk bound {rw_rot:.1f} deg",
            flush=True,
        )
        print(
            f"  residual shift: median|.| {np.median(np.abs(dxy)):.1f} um; "
            f"cumulative |sum| {np.linalg.norm(cum_xy):.1f} um",
            flush=True,
        )
        print(
            f"  sanity: injected +2.0 deg recovered as {inj} "
            f"(expect ~-2.0 offset from baseline)",
            flush=True,
        )
        print(
            f"  Dice gain from correcting residual: "
            f"{gain.mean():+.4f} mean, {gain.max():+.4f} max",
            flush=True,
        )

        # ---- E2. region-centroid Procrustes (secondary; anatomy-confounded) ----
        # This gate already excludes an untraced flag correctly without any
        # extra al_traced plumbing: an untraced section's in_al is False
        # everywhere in flagmat (see build_cache), so its per-section count is
        # 0 < MIN_FLAG_CELLS and `all(...)` fails for the whole brain. That
        # drops al from worker's drift fit entirely (0069041 + two 0068401
        # sections are untraced) while leaving it in queen's (fully traced).
        usable = [
            k
            for k in range(len(FLAGS))
            if all(flagmat[s][:, k].sum() >= MIN_FLAG_CELLS for s in sl)
        ]
        drift = {
            "flags_used": [FLAGS[k] for k in usable],
            "min_flag_cells": MIN_FLAG_CELLS,
        }
        if len(usable) >= 3:
            cents = np.array(
                [[aligned[s][flagmat[s][:, k]].mean(0) for k in usable] for s in sl]
            )
            ref = cents[0]
            theta, rmsd = [], []
            for i in range(nsl):
                t, r = kabsch_angle(cents[i], ref)
                theta.append(t)
                rmsd.append(r)
            theta = np.array(theta)
            rmsd = np.array(rmsd)
            x = np.arange(nsl, dtype=float)
            lr = stats.linregress(x, theta)
            rho, pv = stats.spearmanr(x, theta)
            gc = np.array([aligned[s].mean(0) for s in sl])
            disp = np.linalg.norm(gc - gc[0], axis=1)
            lrd = stats.linregress(x, disp)
            rho_d, pv_d = stats.spearmanr(x, disp)
            drift.update(
                {
                    "theta_deg": [round(float(t), 3) for t in theta],
                    "theta_slope_deg_per_section": float(lr.slope),
                    "theta_slope_se": float(lr.stderr),
                    "theta_slope_p": float(lr.pvalue),
                    "theta_slope_ci95": [
                        float(lr.slope - 1.96 * lr.stderr),
                        float(lr.slope + 1.96 * lr.stderr),
                    ],
                    "theta_spearman_rho": float(rho),
                    "theta_spearman_p": float(pv),
                    "theta_max_abs_deg": float(np.abs(theta).max()),
                    "theta_total_ramp_deg": float(lr.slope * (nsl - 1)),
                    "region_config_rmsd_um": {
                        "mean": float(rmsd[1:].mean()),
                        "max": float(rmsd.max()),
                    },
                    "centroid_disp_um": [round(float(v), 2) for v in disp],
                    "centroid_disp_slope_um_per_section": float(lrd.slope),
                    "centroid_disp_slope_p": float(lrd.pvalue),
                    "centroid_disp_spearman_rho": float(rho_d),
                    "centroid_disp_spearman_p": float(pv_d),
                    "centroid_disp_max_um": float(disp.max()),
                }
            )
            print(
                f"  drift: theta slope {lr.slope:+.3f} deg/section "
                f"(p={lr.pvalue:.3g}, total ramp {lr.slope * (nsl - 1):+.2f} "
                f"deg), max|theta| {np.abs(theta).max():.2f} deg",
                flush=True,
            )
            print(
                f"  drift: centroid displacement slope {lrd.slope:+.2f} "
                f"um/section (p={lrd.pvalue:.3g}), max {disp.max():.1f} um",
                flush=True,
            )
        else:
            drift["note"] = "fewer than 3 flags present in every section"
        B["accumulated_drift"] = drift

        # ---- per-pair Dice vs index (local quality trend) ----
        v = np.array(B["occupancy_dice_bboxrel_1over40"]["per_pair"])
        rho, pv = stats.spearmanr(np.arange(len(v)), v)
        B["pairwise_dice_vs_index"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(pv),
        }
        print(
            f"  per-pair Dice vs section index: rho={rho:+.3f}, p={pv:.3g}", flush=True
        )

        # ---- is the depth trend registration, or tissue? ----
        # If per-pair Dice falls toward the posterior but the per-pair RIGID
        # CEILING falls with it and the gap to that ceiling is flat, the
        # decline is a property of the sections, not of the registration.
        vp = np.array(ladder["paste3"]["per_pair"])
        vc = np.array(ladder["rigid_ceiling"]["per_pair"])
        gapc = vc - vp
        xi = np.arange(len(vp), dtype=float)
        k = 15 if brain == "queen" else 17  # sections on the first slide
        trend = {}
        for nm, arr in [
            ("paste3", vp),
            ("rigid_ceiling", vc),
            ("native_centred", np.array(ladder["native_centred"]["per_pair"])),
            ("gap_to_ceiling", gapc),
        ]:
            rho_, p_ = stats.spearmanr(xi, arr)
            trend[nm] = {
                "spearman_rho": float(rho_),
                "spearman_p": float(p_),
                "mean": float(arr.mean()),
            }
        trend["gap_to_ceiling_stats"] = {
            "mean": float(gapc.mean()),
            "sd": float(gapc.std(ddof=1)),
            "max": float(gapc.max()),
            "frac_of_ceiling_achieved": float((vp / vc).mean()),
        }
        trend["slide1_vs_slide2"] = {
            "paste3": {
                "slide1": float(vp[: k - 1].mean()),
                "slide2": float(vp[k:].mean()),
                "mwu_p": float(stats.mannwhitneyu(vp[: k - 1], vp[k:]).pvalue),
            },
            "rigid_ceiling": {
                "slide1": float(vc[: k - 1].mean()),
                "slide2": float(vc[k:].mean()),
                "mwu_p": float(stats.mannwhitneyu(vc[: k - 1], vc[k:]).pvalue),
            },
        }
        B["depth_trend_decomposition"] = trend
        print(
            f"  depth trend: PASTE3 rho={trend['paste3']['spearman_rho']:+.3f} "
            f"(p={trend['paste3']['spearman_p']:.3g}); rigid ceiling "
            f"rho={trend['rigid_ceiling']['spearman_rho']:+.3f} "
            f"(p={trend['rigid_ceiling']['spearman_p']:.3g}); GAP to ceiling "
            f"rho={trend['gap_to_ceiling']['spearman_rho']:+.3f} "
            f"(p={trend['gap_to_ceiling']['spearman_p']:.3g})",
            flush=True,
        )
        _frac = trend["gap_to_ceiling_stats"]["frac_of_ceiling_achieved"] * 100
        print(
            f"  PASTE3 reaches {_frac:.1f}% "
            f"{gapc.mean():.4f} +/- {gapc.std(ddof=1):.4f}, max {gapc.max():.4f})",
            flush=True,
        )

        # ---- worst pairs ----
        order = np.argsort(v)
        B["worst_pairs"] = [
            {
                "pair_index": int(j),
                "sections": [
                    str(d["slice_cats"][d["slice_codes"][sl[j][0]]]),
                    str(d["slice_cats"][d["slice_codes"][sl[j + 1][0]]]),
                ],
                "dice": float(v[j]),
                "sd_below_mean": float((v.mean() - v[j]) / v.std(ddof=1)),
            }
            for j in order[:5]
        ]
        print("  worst pairs:", flush=True)
        for w in B["worst_pairs"]:
            print(
                f"      {w['sections'][0][-6:]}->{w['sections'][1][-6:]}  "
                f"Dice {w['dice']:.3f}  ({w['sd_below_mean']:.2f} sd below "
                f"mean)",
                flush=True,
            )

        for j in range(nsl - 1):
            rows.append((brain, j, v[j]))
        res[brain] = B

    (OUT / "registration_accuracy.json").write_text(json.dumps(res, indent=2))
    with open(OUT / "per_pair_dice.csv", "w") as fh:
        fh.write("brain,pair_index,occupancy_dice_bboxrel_1over40\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.6f}\n")
    print(f"\nwrote {OUT / 'registration_accuracy.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
