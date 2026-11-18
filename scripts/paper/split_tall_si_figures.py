#!/usr/bin/env python3
"""Split the over-tall Xenium QC SI figure into page-sized halves.

si_xenium_run_qc.py emits one tall multi-row stack, too tall for one page.
The SI ships only the lower half (panels A-C) as Supplementary figure
figS_xenium_qc; the unlettered upper half is the Supplementary Data item
xenium_qc_metrics.csv (scripts/deposit/supplementary_data_csvs.py), so it is not
written.

Run this after that script, then sync as usual:

    PYTHONPATH=src:scripts .venv/bin/python scripts/atlas/si_xenium_run_qc.py
    PYTHONPATH=src:scripts .venv/bin/python scripts/paper/split_tall_si_figures.py
    PYTHONPATH=src:scripts .venv/bin/python scripts/paper/sync_manuscript_figures.py \
        --apply

The half is written to results/figures/si/figS_xenium_qc.png. PIL's PNG encoder
is not byte-stable across versions, so a re-split matches the shipped file pixel
for pixel (checked 2026-11-17), not necessarily byte for byte.

The cut row is found automatically: the widest band of pure background between
panels, searched near a target fraction of the height. That keeps the split
stable if a figure is regenerated at a different size, and guarantees the cut
lands in a gutter rather than through a panel -- which the script verifies
before writing anything.

Reshaping these at the source would be better than cutting pixels, and is worth
doing if either script is revisited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures" / "si"

# source stem -> (target cut as a fraction of height, panels above, panels
#                 below, {half: output stem} for the halves that ship)
SPLITS = {
    "fig_s_xenium_qc": (0.57, "unlettered", "A-C", {"_b": "figS_xenium_qc"}),
}

BACKGROUND = 245  # >= this grey counts as background
MIN_GUTTER = 20  # px; ignore incidental one-line gaps
SEARCH = 0.12  # look this far either side of the target fraction


def _gutters(img: np.ndarray) -> list[tuple[int, int]]:
    """Row ranges that are pure background across the full width."""
    is_gap = img.min(axis=1) > BACKGROUND
    runs, start = [], None
    for y, gap in enumerate(is_gap):
        if gap and start is None:
            start = y
        elif not gap and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_gap) - 1))
    return [r for r in runs if r[1] - r[0] + 1 >= MIN_GUTTER]


def _cut_row(img: np.ndarray, target: float) -> int:
    """Midpoint of the widest gutter near `target` of the image height."""
    h = img.shape[0]
    lo, hi = (target - SEARCH) * h, (target + SEARCH) * h
    candidates = [r for r in _gutters(img) if lo <= (r[0] + r[1]) / 2 <= hi]
    if not candidates:
        raise SystemExit(
            f"no background gutter within {SEARCH:.0%} of {target:.0%} height; "
            "the figure's panel layout has changed -- re-check the split point"
        )
    widest = max(candidates, key=lambda r: r[1] - r[0])
    return (widest[0] + widest[1]) // 2


def main() -> int:
    # Optional first argument: a directory to split in place of
    # results/figures/si (to preview the halves of a test render).
    fig_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else FIG_DIR
    for name, (target, top_panels, bottom_panels, ship) in SPLITS.items():
        src = fig_dir / f"{name}.png"
        if not src.exists():
            print(f"skip  {name}: not in {fig_dir}")
            continue
        im = Image.open(src)
        grey = np.array(im.convert("L"))
        y = _cut_row(grey, target)
        w, h = im.size

        top, bottom = im.crop((0, 0, w, y)), im.crop((0, y, w, h))
        for suffix, half in (("_a", top), ("_b", bottom)):
            if suffix in ship:
                half.save(fig_dir / f"{ship[suffix]}.png")
        print(
            f"{name}: {w}x{h} (aspect {h / w:.2f}) cut at row {y} "
            f"({y / h:.3f})\n"
            f"   _a {top.size[0]}x{top.size[1]} aspect "
            f"{top.size[1] / top.size[0]:.2f}  panels {top_panels}\n"
            f"   _b {bottom.size[0]}x{bottom.size[1]} aspect "
            f"{bottom.size[1] / bottom.size[0]:.2f}  panels {bottom_panels}"
        )
    return 0


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        sys.exit(main())
