# import_regions.py
"""Map a hand-drawn napari label TIFF onto MSI pixels and emit:
  1) pixel_regions_<sample>.parquet  (px, py, region_id, region_name)
  2) region_stats_<sample>.parquet   (per region: n_pixels + per-peptide stats)
  3) region_crops/<sample>__<region>__<peptide>.png  (masked, bbox-cropped)

Usage: import_regions.py <features.parquet> <labels.tif> <regions.json> <sample>
           <outdir>
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import regionlib
import tifffile

PEPTIDES = ["sNPF_v1", "Tk2"]


def _crop_to_uint8(crop):
    """Normalize a float crop to uint8, treating NaN as 0."""
    out = np.where(np.isnan(crop), 0.0, crop.astype(np.float64))
    vmin, vmax = out.min(), out.max()
    if vmax > vmin:
        out = (out - vmin) / (vmax - vmin)
    else:
        out = np.zeros_like(out)
    return (out * 255).astype(np.uint8)


def main(features_pq, labels_tif, regions_json, sample, outdir):
    os.makedirs(outdir, exist_ok=True)
    crops_dir = os.path.join(outdir, "region_crops")
    os.makedirs(crops_dir, exist_ok=True)
    df = pd.read_parquet(features_pq)
    px = df["px"].to_numpy(np.int32)
    py = df["py"].to_numpy(np.int32)
    H, W = int(py.max()), int(px.max())
    label_img = tifffile.imread(labels_tif).astype(np.int32)
    names = {int(k): v for k, v in json.load(open(regions_json)).items()}

    ids = regionlib.map_labels_to_pixels(px, py, label_img, expected_shape=(H, W))

    def name_for(i):
        return names.get(int(i), (f"region_{i}" if i != 0 else "unassigned"))

    pixel_regions = pd.DataFrame(
        {
            "px": px,
            "py": py,
            "region_id": ids,
            "region_name": [name_for(i) for i in ids],
        }
    )
    pr_path = os.path.join(outdir, f"pixel_regions_{sample}.parquet")
    pixel_regions.to_parquet(pr_path, index=False)

    stats = regionlib.compute_region_stats(pixel_regions, df, PEPTIDES)
    st_path = os.path.join(outdir, f"region_stats_{sample}.parquet")
    stats.to_parquet(st_path, index=False)

    for rid, rname in sorted(names.items()):
        if rid == 0:
            continue
        mask = label_img == rid
        for pep in PEPTIDES:
            img = np.zeros((H, W), dtype=np.float32)
            img[py - 1, px - 1] = df[pep].to_numpy(dtype=np.float32)
            crop = regionlib.crop_region_image(img, mask)
            if crop.size == 0:
                continue
            safe = "".join(c if c.isalnum() else "_" for c in rname)
            png_path = os.path.join(crops_dir, f"{sample}__{safe}__{pep}.png")
            tifffile.imwrite(png_path, _crop_to_uint8(crop))

    print(f"wrote:\n  {pr_path}\n  {st_path}\n  {crops_dir}/")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
