#!/usr/bin/env python
"""Export the Xenium morphology stains (besides DAPI) onto the histology layer's
sections.

The DAPI layer (``histo/``) comes from the PASTE3-aligned DAPI stacks. The other
three Xenium multimodal stains were never written to an aligned stack, but every
per-crop h5ad (``pipelines/<brain>/data/<slice_name>.h5ad``) holds the crop's
full morphology image in ``uns/image`` -- (C, H, W) uint16 at 0.2125 um/px, in
RAW CROP pixels (cells sit at ``obsm/spatial / 0.2125 - uns/crop_coords/min``).
The crops are the inputs of the alignment the atlas uses (queen 2026-04-15,
worker 2026-04-06 -> aligned 2026-04-08), so each crop maps into the atlas frame
through its own cells -- a rigid fit from raw crop px to ``spatial_aligned`` px,
joined by ``(slice_name, cell_id)`` -- without any image registration and
without the worker's February DAPI stack.

For every section the existing DAPI manifest ships, each stain is resampled
onto EXACTLY that section's ``bounds`` and pixel size (output pixel centre ->
viewer xy -> um -> aligned px -> inverse rigid -> crop px), so a stain differs
from the DAPI image only in its pixels. The DAPI WebP's alpha is reused as the
tissue mask. Intensities are stretched per BRAIN and channel (pooled tissue
percentiles), so the dim stains stay comparable along the stack.

Channels (OME metadata of the raw morphology_focus images; the queen's uns/image
has a 5th, all-zero channel): 0 DAPI, 1 ATP1A1/CD45/E-cadherin (boundary),
2 18S rRNA, 3 alphaSMA/vimentin. These are mammalian antibody cocktails; their
specificity in bumblebee tissue is not validated, so labels name what was applied.

Check: channel 0 (DAPI) is rendered the same way, never written out, and compared
with each section's deployed DAPI WebP (Pearson r over the tissue, and the best
residual shift by FFT, px and um) -- an independent test of the DAPI layer's
own registration, which for the worker is a per-section fit.

Writes ONLY into ``--out`` (default ``viewer/pages/data/histo_stains``):
``{caste}/{ii}-{key}.webp``, ``{caste}/{ii}-{key}-s.webp``, ``stains.json``
(the manifest fragment: ``{"stains": [...], "sections": {caste: {i: [keys]}}}``)
and ``stains_check.json`` (per-section numbers). Merge ``stains`` into
``histo/manifest.json`` (and the per-section key lists as ``"stains"``) after.

    PYTHONPATH=src .venv/bin/python viewer/tools/export_stains.py \\
        --histo viewer/pages/data/histo --atlas results/tangram/<atlas>.h5ad

Reads each crop's 4 channels in row strips (~110 GB in all, ~1 h);
4 cpu / 24 GB is plenty.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import fft as sfft
from scipy import ndimage as ndi

PIXEL_UM = 0.2125
BRAINS = {
    "queen": ("queen", "pipelines/wholebrain/queen"),
    "worker": ("worker", "pipelines/wholebrain/worker"),
}
STAINS = [  # key, label, uns/image channel. DAPI (ch 0) is the existing layer.
    ("dapi", "DAPI (nuclei)", 0),
    ("boundary", "Boundary (ATP1A1/CD45/E-cad)", 1),
    ("rna18s", "18S rRNA", 2),
    ("smavim", "Interior (αSMA/vimentin)", 3),
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _decode(a):
    return np.array(
        [x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object
    )


def obs_col(f, name):
    node = f["obs"][name]
    if hasattr(node, "keys"):
        cats, codes = _decode(node["categories"][:]), node["codes"][:]
        return cats[codes]
    v = node[:]
    return _decode(v) if v.dtype.kind in "SO" else v


def obs_ids(f):
    if "cell_id" in f["obs"]:
        return obs_col(f, "cell_id")
    key = f["obs"].attrs.get("_index", "_index")
    return _decode(f["obs"][key.decode() if isinstance(key, bytes) else key][:])


def rigid_fit(src, dst):
    """dst ~ src @ rot + t (row vectors); returns rot, t, p95 residual."""
    ms, md = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    rot = u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt
    t = md - ms @ rot
    return rot, t, float(np.percentile(np.linalg.norm(src @ rot + t - dst, axis=1), 95))


def join(raw_path, aid, axy):
    """Raw crop px and aligned px of the atlas cells, joined by cell id."""
    with h5py.File(raw_path, "r") as f:
        sp = f["obsm/spatial"][:, :2].astype(np.float64)
        cmin = (
            np.asarray(f["uns/crop_coords/min"][()], np.float64)
            if "uns/crop_coords/min" in f
            else np.zeros(2)
        )
        rid = obs_ids(f)
    pos = {k: j for j, k in enumerate(rid)}
    best = None
    for ids in (
        aid,
        np.array(
            [
                k.rsplit("-", 1)[0] if k.rsplit("-", 1)[-1].isdigit() else k
                for k in map(str, aid)
            ],
            dtype=object,
        ),
    ):
        jr = np.array([pos.get(k, -1) for k in ids])
        if best is None or (jr >= 0).sum() > (best >= 0).sum():
            best = jr
    ok = best >= 0
    return sp[best[ok]] / PIXEL_UM - cmin, axy[ok], int(ok.sum())


def read_binned(ds, channels, k):
    """uns/image channels, k x k block-mean, read in row strips.

    -> (C, H//k, W//k) float32."""
    _, h, w = ds.shape
    hh, ww = h // k, w // k
    out = np.empty((len(channels), hh, ww), np.float32)
    rows = max(1, 4096 // k)
    for r0 in range(0, hh, rows):
        r1 = min(hh, r0 + rows)
        for j, c in enumerate(channels):
            blk = ds[c, r0 * k : r1 * k, : ww * k].astype(np.float32)
            out[j, r0:r1] = blk.reshape(r1 - r0, k, ww, k).mean(axis=(1, 3))
    return out


def xshift(a, b, win):
    """Shift (row, col) of b that best matches a, |shift| <= win, by FFT."""
    h, w = a.shape
    s = (sfft.next_fast_len(2 * h), sfft.next_fast_len(2 * w))
    a, b = a - a.mean(), b - b.mean()
    c = sfft.irfft2(sfft.rfft2(a, s) * np.conj(sfft.rfft2(b, s)), s)
    rr = np.minimum(np.arange(s[0]), s[0] - np.arange(s[0]))
    cc = np.minimum(np.arange(s[1]), s[1] - np.arange(s[1]))
    c[(rr[:, None] > win) | (cc[None, :] > win)] = -np.inf
    k = np.unravel_index(int(np.argmax(c)), c.shape)
    return [k[i] if k[i] < s[i] / 2 else k[i] - s[i] for i in (0, 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument(
        "--histo",
        type=Path,
        required=True,
        help="the DAPI layer dir (manifest.json, registration.json)",
    )
    ap.add_argument(
        "--atlas",
        type=Path,
        required=True,
        help="the atlas h5ad the viewer was exported from",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="default: <histo>/../histo_stains"
    )
    ap.add_argument("--castes", nargs="+", default=["queen", "worker"])
    ap.add_argument("--sections", nargs="+", type=int, default=None)
    ap.add_argument("--quality", type=int, default=74)
    ap.add_argument("--pct", nargs=2, type=float, default=[0.5, 99.7])
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--bg-strength", type=float, default=0.6)
    ap.add_argument("--bg-sigma-um", type=float, default=60.0)
    ap.add_argument("--max-rigid-p95-px", type=float, default=2.0)
    a = ap.parse_args()
    repo = a.repo.resolve()
    out = a.out or a.histo.parent / "histo_stains"
    man = json.loads((a.histo / "manifest.json").read_text())
    reg = json.loads((a.histo / "registration.json").read_text())
    fr = man["frame"]
    scale, centers = fr["scale_um"], fr["center_um"]
    log(
        f"DAPI layer {a.histo}: "
        + ", ".join(f"{c} {len(v['sections'])}" for c, v in man["castes"].items())
        + f"; out {out}"
    )

    with h5py.File(a.atlas, "r") as f:
        axy_all = f["obsm/spatial_aligned"][:].astype(np.float64)
        brain, sname, z = (
            obs_col(f, "brain"),
            obs_col(f, "slice_name"),
            f["obs/z_level"][:],
        )
        aid_all = obs_ids(f)
    log(f"atlas {a.atlas.name}: {len(axy_all):,} cells")

    frag = {
        "stains": [
            dict(
                key=k,
                label=lab,
                path="{caste}/{ii}.webp"
                if k == "dapi"
                else f"{{caste}}/{{ii}}-{k}.webp",
                small="{caste}/{ii}-s.webp"
                if k == "dapi"
                else f"{{caste}}/{{ii}}-{k}-s.webp",
            )
            for k, lab, _ in STAINS
        ],
        "sections": {},
    }
    check = {}
    for caste in a.castes:
        code, bdir = BRAINS[caste]
        mb = brain == code
        names = sorted(
            set(sname[mb]), key=lambda s: float(np.median(z[mb & (sname == s)]))
        )
        regsec = {
            s["i"]: s.get("slice_name") for s in reg.get(caste, {}).get("sections", [])
        }
        secs = [
            s
            for s in man["castes"][caste]["sections"]
            if a.sections is None or s["i"] in a.sections
        ]
        cx, cy = centers[caste]
        log(f"===== {caste}: {len(secs)} sections")
        rendered, check[caste], frag["sections"][caste] = [], [], {}
        for s in secs:
            i = s["i"]
            nm = regsec.get(i) or names[i]
            if regsec.get(i) and regsec[i] != names[i]:
                log(
                    f"  WARNING section {i}: registration.json says {regsec[i]}, "
                    f"z order says {names[i]}; using the former"
                )
            t0 = time.time()
            m = mb & (sname == nm)
            raw_px, al_px, n_ok = join(
                repo / bdir / "data" / f"{nm}.h5ad", aid_all[m], axy_all[m]
            )
            rot, t, p95 = rigid_fit(raw_px, al_px)
            x0, y0, x1, y1 = s["bounds"]
            dapi = (
                np.asarray(Image.open(a.histo / s["file"]).convert("RGBA")).astype(
                    np.float32
                )
                / 255
            )
            H, W = dapi.shape[:2]
            if (W, H) != (s["w"], s["h"]):
                log(
                    f"  NOTE section {i}: WebP is {W}x{H}, manifest "
                    f"says {s['w']}x{s['h']}; using the WebP"
                )
            alpha, dgray = dapi[..., 3], dapi[..., 0]
            # output pixel centres -> crop px
            col, row = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
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
            crop = (al - t) @ rot.T  # inverse of the row-form rigid map
            out_um = (x1 - x0) * scale / W
            k = max(1, int(out_um / PIXEL_UM))
            with h5py.File(repo / bdir / "data" / f"{nm}.h5ad", "r") as f:
                ch = read_binned(f["uns/image"], [c for _, _, c in STAINS], k)
            cr = (crop[:, 1] - (k - 1) / 2) / k
            cc = (crop[:, 0] - (k - 1) / 2) / k
            planes = np.stack(
                [
                    ndi.map_coordinates(
                        ch[j], [cr, cc], order=1, mode="constant", cval=0
                    ).reshape(H, W)
                    for j in range(len(STAINS))
                ]
            )
            if a.bg_strength > 0:
                sig = a.bg_sigma_um / out_um
                planes = np.clip(
                    planes
                    - a.bg_strength
                    * np.stack([ndi.gaussian_filter(p, sig) for p in planes]),
                    0,
                    None,
                )
            # check: crop DAPI vs the deployed DAPI, over the tissue
            tis = alpha > 0.5
            mine = np.log1p(planes[0])
            r = (
                float(np.corrcoef(mine[tis], dgray[tis])[0, 1])
                if tis.sum() > 100
                else float("nan")
            )
            dsh = xshift(dgray * alpha, mine / max(mine.max(), 1e-9) * alpha, win=40)
            sh_um = float(np.hypot(*dsh) * out_um)
            ok = p95 <= a.max_rigid_p95_px
            check[caste].append(
                dict(
                    i=i,
                    slice_name=nm,
                    cells_joined=n_ok,
                    rigid_p95_px=round(p95, 3),
                    rigid_rot_deg=round(
                        float(np.degrees(np.arctan2(rot[0, 1], rot[0, 0]))), 3
                    ),
                    dapi_r=round(r, 3),
                    dapi_shift_px=[int(v) for v in dsh],
                    dapi_shift_um=round(sh_um, 1),
                    out_um_per_px=round(out_um, 3),
                    shipped=bool(ok),
                )
            )
            log(
                f"  {i:2d} {nm[-24:]:>24} cells {n_ok:6,d} rigid "
                f"p95 {p95:5.2f}px  DAPI vs layer r={r:.3f} "
                f"shift ({dsh[1]:+d},{dsh[0]:+d})px = {sh_um:4.1f}um "
                f" {time.time() - t0:.0f}s"
                + ("" if ok else "  RIGID FIT POOR -- not shipped")
            )
            if ok:
                rendered.append((s, planes[1:], alpha))
                frag["sections"][caste][str(i)] = [k_ for k_, _, _ in STAINS]
        # per-brain stretch, then encode
        for j, (key, _, _) in enumerate(STAINS[1:]):
            vals = (
                np.concatenate([p[j][al_ > 0.5] for _, p, al_ in rendered])
                if rendered
                else np.zeros(1)
            )
            lo, hi = np.percentile(vals, a.pct)
            log(f"  {key}: per-brain stretch {lo:.0f}..{hi:.0f}")
            for s, p, al_ in rendered:
                g = np.clip((p[j] - lo) / max(hi - lo, 1e-6), 0, 1) ** a.gamma * al_
                g8, a8 = (
                    (g * 255).round().astype(np.uint8),
                    (al_ * 255).round().astype(np.uint8),
                )
                img = Image.merge(
                    "RGBA", [Image.fromarray(g8, "L")] * 3 + [Image.fromarray(a8, "L")]
                )
                small_size = Image.open(a.histo / s["small"]).size
                for path, im in (
                    (f"{caste}/{s['i']:02d}-{key}.webp", img),
                    (
                        f"{caste}/{s['i']:02d}-{key}-s.webp",
                        img.resize(small_size, Image.LANCZOS),
                    ),
                ):
                    (out / path).parent.mkdir(parents=True, exist_ok=True)
                    im.save(out / path, format="WEBP", quality=a.quality, method=4)
        (out / "stains.json").write_text(json.dumps(frag, indent=1, ensure_ascii=False))
        (out / "stains_check.json").write_text(json.dumps(check, indent=1))
    sz = sum(p.stat().st_size for p in out.rglob("*.webp"))
    log(f"done: {sz / 1e6:.1f} MB of WebP in {out}")


if __name__ == "__main__":
    main()
