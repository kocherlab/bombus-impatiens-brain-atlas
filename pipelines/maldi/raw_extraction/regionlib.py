"""Pure functions for mapping hand-drawn napari labels onto MSI pixels,
computing per-region stats, and cropping region images. No file I/O."""

import numpy as np
import pandas as pd


def map_labels_to_pixels(px, py, label_img, expected_shape=None):
    """Return the integer region id for each pixel.

    px, py: 1-based pixel coords (arrays, length N).
    label_img: 2D int array, indexed [py-1, px-1] (shape H x W).
    expected_shape: optional (H, W) to validate against.
    """
    px = np.asarray(px)
    py = np.asarray(py)
    if expected_shape is not None and tuple(label_img.shape) != tuple(expected_shape):
        raise ValueError(
            f"label image shape {label_img.shape} != expected {expected_shape}"
        )
    return label_img[py - 1, px - 1].astype(np.int64)


def compute_region_stats(pixel_regions, features, peptides):
    """One row per named region with n_pixels and mean/median/sum/std per peptide.
    Excludes region_id == 0 (unassigned)."""
    pr = pixel_regions[pixel_regions["region_id"] != 0]
    merged = pr.merge(features[["px", "py", *peptides]], on=["px", "py"], how="inner")
    rows = []
    for name, grp in merged.groupby("region_name", sort=True):
        row = {"region_name": name, "n_pixels": int(len(grp))}
        for pep in peptides:
            v = grp[pep].to_numpy(dtype=float)
            row[f"{pep}_mean"] = float(np.mean(v)) if len(v) else 0.0
            row[f"{pep}_median"] = float(np.median(v)) if len(v) else 0.0
            row[f"{pep}_sum"] = float(np.sum(v))
            row[f"{pep}_std"] = float(np.std(v)) if len(v) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def crop_region_image(img, mask):
    """Mask `img` to `mask` and crop to the mask's bounding box.
    Pixels outside the mask (but inside bbox) become NaN."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.empty((0, 0), dtype=float)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub_img = img[y0:y1, x0:x1].astype(float)
    sub_mask = mask[y0:y1, x0:x1]
    out = np.full(sub_img.shape, np.nan, dtype=float)
    out[sub_mask] = sub_img[sub_mask]
    return out
