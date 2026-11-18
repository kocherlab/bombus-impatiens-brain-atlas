"""Synthetic stand-in for the Xenium DAPI histology layer, for testing the viewer only.

Rasterises the cells of pages/data/base-lite.bin (or the full base.bin) into one
blurred, noisy grayscale image per section, so a correctly-placed image sits exactly
under its cells. Writes
pages/data/histo/{manifest.json, queen/NN.webp, queen/NN-s.webp, ...} in the
contract the real export (tools/export_histology.py) produces. pages/ is build
output.

    uv run --no-project --with pillow --with numpy --with scipy \
        python viewer/tools/fake_histology.py [--full]
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "pages" / "data"
OUT = DATA / "histo"
LARGE, SMALL = 1024, 256


def load(full):
    m = json.loads((DATA / "manifest.json").read_text())
    tier = (
        {"file": m["base"], "n": m["n"], "offsets": m["offsets"]} if full else m["lite"]
    )
    buf = (DATA / tier["file"]).read_bytes()
    if buf[:2] == b"\x1f\x8b":
        buf = gzip.decompress(buf)

    def col(k):
        o, ln, t = tier["offsets"][k]
        return np.frombuffer(
            buf[o : o + ln], dtype=np.int16 if t == "int16" else np.uint8
        )

    return (col("x") / 32767, col("y") / 32767, col("z") / 32767, col("caste"))


def main():
    x, y, z, caste = load("--full" in sys.argv)
    rng = np.random.default_rng(0)
    man = {"version": 1, "synthetic": True, "castes": {}}
    for ci, name in enumerate(("queen", "worker")):
        m = caste == ci
        zs = np.unique(np.round(z[m], 4))
        secs = []
        (OUT / name).mkdir(parents=True, exist_ok=True)
        for i, zv in enumerate(zs):
            k = m & (np.abs(z - zv) < 1e-4)
            if k.sum() < 50:
                continue
            xs, ys = x[k], y[k]
            pad = 0.03
            x0, x1 = xs.min() - pad, xs.max() + pad
            y0, y1 = ys.min() - pad, ys.max() + pad
            ar = (y1 - y0) / (x1 - x0)
            w, h = (
                (LARGE, max(8, round(LARGE * ar)))
                if ar <= 1
                else (max(8, round(LARGE / ar)), LARGE)
            )
            # image column ↔ x, image ROW 0 ↔ y0 (rows run along +y; no flip)
            cx = np.clip(((xs - x0) / (x1 - x0) * w).astype(int), 0, w - 1)
            ry = np.clip(((ys - y0) / (y1 - y0) * h).astype(int), 0, h - 1)
            img = np.zeros((h, w), np.float32)
            np.add.at(img, (ry, cx), 1.0)
            img = gaussian_filter(img, 1.6) * 0.7 + gaussian_filter(img, 9) * 3.0
            img *= 1 + 0.35 * rng.standard_normal(img.shape).astype(np.float32)
            img = np.clip(img / np.percentile(img[img > 0], 99.5), 0, 1) ** 0.8
            im = Image.fromarray((img * 255).astype(np.uint8), "L")
            f, fs = f"{name}/{i:02d}.webp", f"{name}/{i:02d}-s.webp"
            im.save(OUT / f, "WEBP", quality=70, method=6)
            sw, sh = max(4, round(w * SMALL / LARGE)), max(4, round(h * SMALL / LARGE))
            im.resize((sw, sh), Image.LANCZOS).save(
                OUT / fs, "WEBP", quality=70, method=6
            )
            secs.append(
                {
                    "i": i,
                    "z": float(zv),
                    "bounds": [float(x0), float(y0), float(x1), float(y1)],
                    "file": f,
                    "small": fs,
                    "w": w,
                    "h": h,
                    "sw": sw,
                    "sh": sh,
                }
            )
        man["castes"][name] = {"sections": secs}
        print(name, len(secs), "sections")
    (OUT / "manifest.json").write_text(json.dumps(man))
    tot = sum(p.stat().st_size for p in OUT.rglob("*.webp"))
    print(f"wrote {OUT}  ({tot / 1e6:.1f} MB webp)")


if __name__ == "__main__":
    main()
