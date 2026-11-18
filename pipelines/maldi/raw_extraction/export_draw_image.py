# export_draw_image.py
"""Build a native-resolution multi-channel TIFF (TIC + peptide ion images) for
hand-drawing regions in napari, plus a JSON sidecar.

Usage: export_draw_image.py <features.parquet> <sample> <out.tif> <out.json>
"""

import json
import sys

import numpy as np
import pandas as pd
import tifffile

CHANNELS = ["TIC", "sNPF_v1", "Tk2"]  # order matters; ch0 is the drawing background


def main(features_pq, sample, out_tif, out_json):
    df = pd.read_parquet(features_pq)
    px = df["px"].to_numpy(np.int32)
    py = df["py"].to_numpy(np.int32)
    H, W = int(py.max()), int(px.max())
    stack = []
    for ch in CHANNELS:
        img = np.zeros((H, W), dtype=np.float32)
        vals = df[ch].to_numpy(dtype=np.float32)
        img[py - 1, px - 1] = vals
        if ch == "TIC":
            img = np.log1p(img)  # compress dynamic range for visibility
        nz = img[img > 0]
        if nz.size:
            img = np.clip(img, 0, np.percentile(nz, 99.5))
        stack.append(img)
    arr = np.stack(stack, axis=0)  # (C, H, W)
    tifffile.imwrite(
        out_tif,
        arr,
        photometric="minisblack",
        metadata={"axes": "CYX", "channels": CHANNELS},
    )
    json.dump(
        {
            "sample": sample,
            "shape": [H, W],
            "channels": CHANNELS,
            "coord_convention": "img[py-1, px-1], 1-based pixel coords",
        },
        open(out_json, "w"),
        indent=2,
    )
    print(f"wrote {out_tif} {arr.shape} and {out_json}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
