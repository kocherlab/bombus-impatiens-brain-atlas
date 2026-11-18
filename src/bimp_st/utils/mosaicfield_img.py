"""Shared MOSAICField image helpers (normalize, enhance, load, crop, warp).

Used by ``scripts/atlas/si_mosaicfield.py``.

Everything is pure numpy except ``warp_img``, which imports ``torch`` lazily
so importing this module does not pay for torch initialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def norm(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to ``[0, 1]`` (float32)."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def enhance(img: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Percentile-based contrast enhancement.

    Normalizes first, then rescales so the *pct* percentile of positive
    pixels maps to 1.0, clipping above.
    """
    img = norm(img)
    hi = np.percentile(img[img > 0], pct) if (img > 0).any() else 1.0
    return np.clip(img / max(hi, 1e-8), 0, 1)


def load_img(sid: str, img_dir: Path | str) -> np.ndarray:
    """Load ``{sid}_img.npy`` from *img_dir*, returning a 2D array.

    If the file is HxWxC, the first channel is returned.
    """
    arr = np.load(Path(img_dir) / f"{sid}_img.npy")
    return arr[:, :, 0] if arr.ndim == 3 else arr


def tissue_bbox(
    imgs: Mapping[str, np.ndarray],
    pad_frac: float = 0.03,
    percentile: float = 10.0,
) -> tuple[int, int, int, int]:
    """Tight ``(y0, y1, x0, x1)`` bounding box around non-zero tissue.

    Computes a union mask across all images by thresholding at
    ``percentile`` and expanding the bbox by ``pad_frac`` of its extent on
    each side. Defaults match ``si_mosaicfield.py``.
    """
    first = next(iter(imgs.values()))
    mask = np.zeros_like(first, dtype=bool)
    for img in imgs.values():
        mask |= img > np.percentile(img, percentile)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    pad_y = int((ys.max() - ys.min()) * pad_frac)
    pad_x = int((xs.max() - xs.min()) * pad_frac)
    return (
        max(0, ys.min() - pad_y),
        min(mask.shape[0], ys.max() + pad_y),
        max(0, xs.min() - pad_x),
        min(mask.shape[1], xs.max() + pad_x),
    )


def crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Slice an image with a ``(y0, y1, x0, x1)`` bbox."""
    y0, y1, x0, x1 = bbox
    return img[y0:y1, x0:x1]


def warp_img(img: np.ndarray, flow_tensor) -> np.ndarray:
    """Warp a 2D image through a displacement field.

    ``flow_tensor`` must be a torch tensor with shape ``(1, 2, H, W)``,
    displacement in pixels with ``flow[:, 0] = dy``, ``flow[:, 1] = dx``
    (the same convention as ``MOSAICField.Utils.SpatialTransformer``).

    Reimplements that transformer's ``forward()`` with
    ``torch.nn.functional.grid_sample`` because importing ``MOSAICField``
    pulls in the optional ``ot`` (POT) dependency, which warping does not need.
    """
    import torch  # noqa: PLC0415 — deliberate lazy import
    import torch.nn.functional as Fnn  # noqa: PLC0415

    H, W = img.shape
    ys = torch.arange(0, H)
    xs = torch.arange(0, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_y, grid_x), dim=0).unsqueeze(0).float()

    locs_yx_pix = grid + flow_tensor
    y = 2 * (locs_yx_pix[:, 0] / (H - 1) - 0.5)
    x = 2 * (locs_yx_pix[:, 1] / (W - 1) - 0.5)
    new_locs = torch.stack((x, y), dim=1).permute(0, 2, 3, 1)

    t = torch.tensor(img[None, None], dtype=torch.float32)
    out = Fnn.grid_sample(
        t, new_locs, align_corners=True, mode="bilinear", padding_mode="border"
    )
    return out[0, 0].numpy()


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert ``'#rrggbb'`` (or ``'rrggbb'``) to a ``(r, g, b)`` byte tuple."""
    hex_str = hex_str.lstrip("#")
    r, g, b = (int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    return r, g, b


__all__ = [
    "norm",
    "enhance",
    "load_img",
    "tissue_bbox",
    "crop",
    "warp_img",
    "hex_to_rgb",
]
