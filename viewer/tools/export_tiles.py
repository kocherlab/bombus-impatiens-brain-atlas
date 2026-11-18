#!/usr/bin/env python
"""High-resolution tiled pyramid of every histology section and stain.

The overview images (``histo/``, ``histo_stains/``) are ~1024 px across a
3-3.6 mm section, ~3.4 um/px, so nuclei are 1-2 px. This renders each section
again at ``--upscale`` (default 4: ~0.85 um/px) and cuts it into 512 px WebP
tiles, at three levels: 0 (full), 1 (1/2) and 2 (1/4 = the overview's own
w x h). Every stain, DAPI included, is sampled from the per-crop h5ad
``uns/image`` through the crop's own cells, exactly as ``export_stains.py``
does (see there for the frame and the cell-id join), so all stains share one
source and one frame. Each level covers the section's existing ``bounds``.

Brightness: background subtraction, then one 0.5-99.7 percentile stretch per
brain, stain and LEVEL (levels differ in how much averaging flattens the peaks),
from ``--stretch-sample`` evenly spaced sections, then gamma. Alpha is the
overview DAPI image's tissue alpha, upsampled. Only tiles that hold tissue are
written. Gray is premultiplied by alpha (transparent and black outside).

Writes ONLY into ``--out`` (default ``<histo>/../histo_tiles``):
``{caste}/{ii}/{key}/{l}/{tx}_{ty}.webp``, ``tiles.json`` (the manifest
fragment) and ``tiles_check.json``. Manifest shape (agreed with the viewer,
final)::

    "tiles": {"path": "../histo_tiles/{caste}/{ii}/{key}/{l}/{tx}_{ty}.webp",
              "tile": 512, "overlap": 1}
    section["tiles"] = {"stains": [keys], "levels": [
        {"l": 0, "um_per_px": .., "w": W0, "h": H0, "nx": .., "ny": ..,
         "have": "0110..",
         "stains": [keys]   # only where a level lacks a stain (--drop-l0)
        }, ...]}

Tile (tx, ty) at level l covers pixels [tx*512, min((tx+1)*512, w_l)) x
[ty*512, ...) of a w_l x h_l image spanning the section's bounds (column 0 at
x0, row 0 at y0, outer edges); edge tiles are smaller, not padded. With
``"overlap": 1`` each tile IMAGE also carries 1 extra pixel on every side that
has a neighbour (not on the level's outer edges), so linear filtering has no
seam: the image of tile (tx, ty) holds pixels
[max(0, tx*512-1), min(w_l, (tx+1)*512+1)), the grid geometry is unchanged. ``have`` is
row-major '0'/'1', nx*ny long, the same for every stain (tissue is shared).
The last level has the overview's own w x h, from the same stretch as the
finer levels: show it in place of the overview to avoid a brightness step.

``--estimate`` renders only ``--sections`` (per brain), measures the bytes and
extrapolates to every section.

    PYTHONPATH=src .venv/bin/python viewer/tools/export_tiles.py \\
        --histo viewer/pages/data/histo --atlas results/tangram/<atlas>.h5ad
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_stains import (
    BRAINS,
    PIXEL_UM,
    STAINS,
    join,
    log,
    obs_col,
    obs_ids,
    read_binned,
    rigid_fit,
)  # noqa: E402

TILE = 512
OVERLAP = (
    1  # px each tile image extends into its neighbours (not past the level's edges)
)


def render(sec, nm, repo, bdir, atlas, frame, caste, histo, up, a):
    """Section at `up` x the overview: background-subtracted planes (C,H,W) + alpha,
    um/px."""
    axy, brain, sname, aid = atlas
    m = (brain == BRAINS[caste][0]) & (sname == nm)
    raw_px, al_px, n_ok = join(repo / bdir / "data" / f"{nm}.h5ad", aid[m], axy[m])
    rot, t, p95 = rigid_fit(raw_px, al_px)
    x0, y0, x1, y1 = sec["bounds"]
    ov = Image.open(histo / sec["file"]).convert("RGBA")
    w, h = ov.size
    W, H = w * up, h * up
    alpha = (
        np.asarray(ov.getchannel("A").resize((W, H), Image.BILINEAR)).astype(np.float32)
        / 255
    )
    scale, (cx, cy) = frame["scale_um"], frame["center_um"][caste]
    um = (x1 - x0) * scale / W
    k = max(1, int(um / PIXEL_UM))
    with h5py.File(repo / bdir / "data" / f"{nm}.h5ad", "r") as f:
        ch = read_binned(f["uns/image"], [c for _, _, c in STAINS], k)
    planes = np.empty((len(STAINS), H, W), np.float32)
    for r0 in range(0, H, 512):  # in row bands, to bound the coordinate arrays
        r1 = min(H, r0 + 512)
        col, row = np.meshgrid(np.arange(W) + 0.5, np.arange(r0, r1) + 0.5)
        al = (
            np.stack(
                [
                    (x0 + col / W * (x1 - x0)) * scale + cx,
                    (y0 + row / H * (y1 - y0)) * scale + cy,
                ],
                -1,
            ).reshape(-1, 2)
            / PIXEL_UM
        )
        crop = (al - t) @ rot.T
        cr, cc = (crop[:, 1] - (k - 1) / 2) / k, (crop[:, 0] - (k - 1) / 2) / k
        for j in range(len(STAINS)):
            planes[j, r0:r1] = ndi.map_coordinates(
                ch[j], [cr, cc], order=1, mode="constant", cval=0
            ).reshape(r1 - r0, W)
    del ch
    if a.bg_strength > 0:
        sig = a.bg_sigma_um / um
        for j in range(len(STAINS)):
            planes[j] = np.clip(
                planes[j] - a.bg_strength * ndi.gaussian_filter(planes[j], sig), 0, None
            )
    return planes, alpha, um, dict(cells_joined=n_ok, rigid_p95_px=round(p95, 3))


def merge_write(out, name, check_name, frag, check, caste):
    """Write this caste into the index, keeping every other caste already there.

    Re-reads the files under an exclusive lock at write time, so one SLURM array
    task per brain can run at once without either dropping the other's entries.
    """
    import fcntl

    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".index.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.loads((out / name).read_text()) if (out / name).exists() else {}
        cur.setdefault("tiles", frag["tiles"]).update(frag["tiles"])
        cur.setdefault("sections", {})[caste] = frag["sections"][caste]
        chk = (
            json.loads((out / check_name).read_text())
            if (out / check_name).exists()
            else {}
        )
        chk[caste] = check[caste]
        for fn, obj, ind in ((name, cur, None), (check_name, chk, 1)):
            tmp = out / f".{fn}.tmp"
            tmp.write_text(json.dumps(obj, indent=ind))
            tmp.replace(out / fn)
        fcntl.flock(lk, fcntl.LOCK_UN)


def halve(x):
    """2x2 mean over the last two axes (dims are multiples of 2 by construction)."""
    h, w = x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2
    x = x[..., :h, :w]
    return x.reshape(*x.shape[:-2], h // 2, 2, w // 2, 2).mean(axis=(-3, -1))


def pyramid(planes, alpha, n):
    out = [(planes, alpha)]
    for _ in range(n - 1):
        planes, alpha = halve(planes), halve(alpha)
        out.append((planes, alpha))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--histo", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--castes", nargs="+", default=["queen", "worker"])
    ap.add_argument("--sections", nargs="+", type=int, default=None)
    ap.add_argument(
        "--estimate",
        action="store_true",
        help="render only --sections; extrapolate the size",
    )
    ap.add_argument("--upscale", type=int, default=4, choices=[2, 4, 8])
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument(
        "--stains",
        nargs="+",
        default=None,
        help="write only these stain keys (e.g. dapi)",
    )
    ap.add_argument(
        "--url-dir",
        default="histo_tiles",
        help="folder name the manifest's tile path points at",
    )
    ap.add_argument(
        "--drop-l0", nargs="*", default=[], help="stain keys to leave out of level 0"
    )
    ap.add_argument(
        "--stretch-sample",
        type=int,
        default=6,
        help="sections per brain for the stretch",
    )
    ap.add_argument("--pct", nargs=2, type=float, default=[0.5, 99.7])
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--bg-strength", type=float, default=0.6)
    ap.add_argument("--bg-sigma-um", type=float, default=60.0)
    a = ap.parse_args()
    repo = a.repo.resolve()
    out = a.out or a.histo.parent / "histo_tiles"
    man = json.loads((a.histo / "manifest.json").read_text())
    reg = json.loads((a.histo / "registration.json").read_text())
    frame = man["frame"]
    with h5py.File(a.atlas, "r") as f:
        atlas = (
            f["obsm/spatial_aligned"][:].astype(np.float64),
            obs_col(f, "brain"),
            obs_col(f, "slice_name"),
            obs_ids(f),
        )
    zall = None
    log(
        f"atlas {a.atlas.name}: {len(atlas[0]):,} cells; out {out}; "
        f"upscale {a.upscale}, {a.levels} levels"
    )
    frag = {
        "tiles": {
            "path": "../" + a.url_dir + "/{caste}/{ii}/{key}/{l}/{tx}_{ty}.webp",
            "tile": TILE,
            "overlap": OVERLAP,
        },
        "sections": {},
    }
    check, est = {}, {}
    name = "tiles_estimate.json" if a.estimate else "tiles.json"
    keys = [k for k, _, _ in STAINS]
    sel = [k for k in keys if a.stains is None or k in a.stains]
    assert sel, f"--stains {a.stains} matches none of {keys}"
    for caste in a.castes:
        code, bdir = BRAINS[caste]
        secs_all = man["castes"][caste]["sections"]
        regsec = {
            s["i"]: s.get("slice_name") for s in reg.get(caste, {}).get("sections", [])
        }
        if any(regsec.get(s["i"]) is None for s in secs_all):
            if zall is None:
                with h5py.File(a.atlas, "r") as f:
                    zall = f["obs/z_level"][:]
            mb = atlas[1] == code
            names = sorted(
                set(atlas[2][mb]),
                key=lambda s: float(np.median(zall[mb & (atlas[2] == s)])),
            )
            regsec = {s["i"]: regsec.get(s["i"]) or names[s["i"]] for s in secs_all}
        secs = [s for s in secs_all if a.sections is None or s["i"] in a.sections]
        log(f"===== {caste}: {len(secs)} of {len(secs_all)} sections")

        # ---- per-brain, per-level, per-stain stretch from a sample of sections
        samp = secs_all[:: max(1, len(secs_all) // a.stretch_sample)][
            : a.stretch_sample
        ]
        vals = [[[] for _ in STAINS] for _ in range(a.levels)]
        for s in samp:
            t0 = time.time()
            pl, al, _, _ = render(
                s,
                regsec[s["i"]],
                repo,
                bdir,
                atlas,
                frame,
                caste,
                a.histo,
                a.upscale,
                a,
            )
            for lv, (p, al_) in enumerate(pyramid(pl, al, a.levels)):
                tis = al_ > 0.5
                for j in range(len(STAINS)):
                    v = p[j][tis]
                    vals[lv][j].append(v[:: max(1, v.size // 200_000)])
            log(f"  stretch sample {s['i']:2d} ({time.time() - t0:.0f}s)")
        lohi = [
            [
                tuple(np.percentile(np.concatenate(vals[lv][j]), a.pct))
                for j in range(len(STAINS))
            ]
            for lv in range(a.levels)
        ]
        del vals
        log(
            "  stretch "
            + "; ".join(
                f"L{lv} "
                + ", ".join(
                    f"{keys[j]} {lo:.0f}..{hi:.0f}"
                    for j, (lo, hi) in enumerate(lohi[lv])
                )
                for lv in range(a.levels)
            )
        )

        frag["sections"][caste], check[caste], est[caste] = {}, [], []
        for s in secs:
            i, nm, t0 = s["i"], regsec[s["i"]], time.time()
            pl, al, um, info = render(
                s, nm, repo, bdir, atlas, frame, caste, a.histo, a.upscale, a
            )
            levels, nbytes = [], {k: [0] * a.levels for k in sel}
            for lv, (p, al_) in enumerate(pyramid(pl, al, a.levels)):
                hh, ww = al_.shape
                nx, ny = -(-ww // TILE), -(-hh // TILE)
                lv_keys = [k for k in sel if not (lv == 0 and k in a.drop_l0)]
                have = []
                a8 = (np.clip(al_, 0, 1) * 255).round().astype(np.uint8)
                g8 = {}
                for j, k in enumerate(keys):
                    if k not in lv_keys:
                        continue
                    lo, hi = lohi[lv][j]
                    g = np.clip(
                        (p[j] - lo) / max(hi - lo, 1e-6), 0, 1
                    ) ** a.gamma * np.clip(al_, 0, 1)
                    g8[k] = (g * 255).round().astype(np.uint8)
                for ty in range(ny):
                    for tx in range(nx):
                        sl = (
                            slice(ty * TILE, min(hh, (ty + 1) * TILE)),
                            slice(tx * TILE, min(ww, (tx + 1) * TILE)),
                        )
                        if a8[sl].max() < 3:
                            have.append("0")
                            continue
                        have.append("1")
                        so = (
                            slice(
                                max(0, ty * TILE - OVERLAP),
                                min(hh, (ty + 1) * TILE + OVERLAP),
                            ),
                            slice(
                                max(0, tx * TILE - OVERLAP),
                                min(ww, (tx + 1) * TILE + OVERLAP),
                            ),
                        )
                        A = Image.fromarray(a8[so], "L")
                        for k in lv_keys:
                            G = Image.fromarray(g8[k][so], "L")
                            path = (
                                out
                                / caste
                                / f"{i:02d}"
                                / k
                                / str(lv)
                                / f"{tx}_{ty}.webp"
                            )
                            path.parent.mkdir(parents=True, exist_ok=True)
                            Image.merge("RGBA", [G, G, G, A]).save(
                                path, format="WEBP", quality=a.quality, method=4
                            )
                            nbytes[k][lv] += path.stat().st_size
                entry = dict(
                    l=lv,
                    um_per_px=round(um * 2**lv, 4),
                    w=ww,
                    h=hh,
                    nx=nx,
                    ny=ny,
                    have="".join(have),
                )
                if lv_keys != sel:
                    entry["stains"] = lv_keys
                levels.append(entry)
            del pl, al
            frag["sections"][caste][str(i)] = dict(stains=sel, levels=levels)
            tot = sum(sum(v) for v in nbytes.values())
            est[caste].append(nbytes)
            check[caste].append(dict(i=i, slice_name=nm, **info, bytes=nbytes))
            log(
                f"  {i:2d} {nm[-20:]:>20} {levels[0]['w']}x{levels[0]['h']} "
                f"@ {um:.3f} um/px  tiles "
                + "/".join(str(lv_["have"].count("1")) for lv_ in levels)
                + f"  {tot / 1e6:5.1f} MB  rigid "
                f"p95 {info['rigid_p95_px']:.2f}px  {time.time() - t0:.0f}s"
            )
        if a.estimate and est[caste]:
            n_all = len(secs_all)
            per = {
                k: [
                    float(np.mean([e[k][lv] for e in est[caste]]))
                    for lv in range(a.levels)
                ]
                for k in sel
            }
            log(f"  ESTIMATE {caste}: x{n_all} sections")
            for k in sel:
                log(
                    f"    {k:9s} "
                    + "  ".join(
                        f"L{lv} {per[k][lv] * n_all / 1e6:6.0f} MB"
                        for lv in range(a.levels)
                    )
                    + f"  = {sum(per[k]) * n_all / 1e6:6.0f} MB"
                )
            log(f"    total {sum(sum(v) for v in per.values()) * n_all / 1e6:.0f} MB")
        merge_write(out, name, "tiles_check.json", frag, check, caste)
    sz = sum(p.stat().st_size for p in out.rglob("*.webp"))
    n = sum(1 for _ in out.rglob("*.webp"))
    log(f"done: {n:,} tiles, {sz / 1e6:.0f} MB in {out}")


if __name__ == "__main__":
    main()
