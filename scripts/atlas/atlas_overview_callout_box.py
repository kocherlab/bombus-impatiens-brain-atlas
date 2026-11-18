"""Find the queen equivalent of fig:atlas_overview's KC-IC selection cube.

The worker version of the fig:atlas_overview panel C callout cube is
hardcoded to worker
coordinates:

    _KC_X0, _KC_X1 = -1854, -1454
    _KC_Y0, _KC_Y1 =  2523,  2923
    _KC_SLICE_IDXS = [19, 20, 21]

chosen for its cell count and Kenyon-cell purity. The queen box with the same
properties has to be searched for, since the two brains are not in a common
coordinate frame.

Sweeps three-consecutive-section windows and a grid of 400 µm boxes inside the
queen KC-IC annotation, scoring each by Kenyon-cell purity (canonical labels)
subject to a cell-count floor, and prints the winner as a drop-in code block.

Outputs to the results/reframe/ directory of the queen box:
    candidates.csv      every box scored
    queen_box.png       the chosen box drawn on its three sections
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import (  # noqa: E402
    anatomy_guided_vote,
    get_pooled_color,
)

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
QUEEN = REPO / "pipelines" / "wholebrain" / "queen" / "data" / "aligned_clustered.h5ad"
COMBINED = (
    REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
)
OUT = REPO / "results" / "reframe" / "fig1_queen_box"
OUT.mkdir(parents=True, exist_ok=True)

BOX = 400.0  # µm, matches the worker box edge
STRIDE = 100.0
MIN_CELLS = 500
TARGET_CELLS = 800
N_WIN = 3  # consecutive sections, as in the worker cube
LABEL = "cluster_vote"  # the pooled anatomy-guided vote, as fig_atlas_overview uses


def main() -> None:
    # Coordinates and slice order come from the per-brain object, the frame
    # fig_atlas_overview.py's _kc_ic_region_mask() works in. The per-brain h5ad has no
    # cell-type labels (Tangram was run on the combined object), so labels are
    # joined on (slice_name, cell_id), which is unique within a brain.
    print(f"loading {QUEEN.name} ...", flush=True)
    a = ad.read_h5ad(QUEEN)
    print(f"  {a.n_obs:,} cells (per-brain frame)", flush=True)

    print(f"loading labels from {COMBINED.name} ...", flush=True)
    c = ad.read_h5ad(COMBINED)

    # Score purity on the same quantity fig_atlas_overview.py does:
    # prepare_kc_selection() uses obs["cluster_vote"], the pooled anatomy-guided
    # vote, and the worker box's purity is measured on that rather than on raw
    # per-cell Tangram argmax (raw labels give a lower purity).
    # This purity is partly circular, since the anatomy-guided vote assigns
    # calyx-dominated clusters to "Kenyon cells"; the aim here is to match the
    # figure, not to test Kenyon-cell identity.
    anatomy_guided_vote(
        c,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        pool=True,
        dst="cluster_vote",
    )
    cq = c.obs[c.obs["brain"].astype(str) == "queen"]
    print(f"  {len(cq):,} queen cells in the combined object", flush=True)

    key_c = (cq["slice_name"].astype(str) + "|" + cq["cell_id"].astype(str)).values
    lab_c = cq["cluster_vote"].astype(str).values
    lut = dict(zip(key_c, lab_c, strict=False))
    if len(lut) != len(key_c):
        print(
            f"  WARNING: {len(key_c) - len(lut):,} duplicate (slice, cell_id) keys",
            flush=True,
        )

    key_a = (
        a.obs["slice_name"].astype(str) + "|" + a.obs["cell_id"].astype(str)
    ).values
    labels = np.array([lut.get(k, "unmapped") for k in key_a])
    n_un = int((labels == "unmapped").sum())
    print(
        f"  joined labels onto {a.n_obs - n_un:,} / {a.n_obs:,} cells "
        f"({n_un:,} unmapped)",
        flush=True,
    )
    if n_un > 0.5 * a.n_obs:
        raise SystemExit("more than half the cells failed to join; the key is wrong")
    a.obs[LABEL] = labels
    del c, cq

    obs = a.obs
    xy = np.asarray(a.obsm["spatial"], float)
    slices = sorted(obs["slice_name"].astype(str).unique())
    sl_arr = obs["slice_name"].astype(str).values
    lab = obs[LABEL].astype(str).values
    # cluster_vote is already pooled, so Kenyon cells is a single category
    is_kc = lab == "Kenyon cells"

    in_ic = (
        obs["in_kc_ic"].astype(bool).values
        if "in_kc_ic" in obs.columns
        else np.ones(a.n_obs, bool)
    )
    print(
        f"  KC-IC annotated cells: {in_ic.sum():,}; Kenyon-labelled: {is_kc.sum():,}",
        flush=True,
    )

    rows = []
    for w0 in range(len(slices) - N_WIN + 1):
        win = slices[w0 : w0 + N_WIN]
        wm = np.isin(sl_arr, win) & in_ic
        if wm.sum() < MIN_CELLS:
            continue
        wx, wy = xy[wm, 0], xy[wm, 1]
        kc_w = is_kc[wm]
        for x0 in np.arange(wx.min(), max(wx.max() - BOX, wx.min()) + 1, STRIDE):
            for y0 in np.arange(wy.min(), max(wy.max() - BOX, wy.min()) + 1, STRIDE):
                b = (wx >= x0) & (wx <= x0 + BOX) & (wy >= y0) & (wy <= y0 + BOX)
                n = int(b.sum())
                if n < MIN_CELLS:
                    continue
                rows.append(
                    dict(
                        slice_start=w0,
                        slices="|".join(win),
                        x0=float(x0),
                        x1=float(x0 + BOX),
                        y0=float(y0),
                        y1=float(y0 + BOX),
                        n_cells=n,
                        kc_purity=float(kc_w[b].mean()),
                    )
                )

    if not rows:
        raise SystemExit("no box met the cell-count floor; lower MIN_CELLS")

    cand = pd.DataFrame(rows)
    # prefer purity, then closeness to the worker box's ~800 cells
    cand["size_penalty"] = (cand.n_cells - TARGET_CELLS).abs() / TARGET_CELLS
    cand["score"] = cand.kc_purity - 0.05 * cand.size_penalty
    cand = cand.sort_values("score", ascending=False)
    cand.to_csv(OUT / "candidates.csv", index=False)

    best = cand.iloc[0]
    print(f"\n  {len(cand):,} candidate boxes scored", flush=True)
    print(
        f"  best: purity {best.kc_purity:.3f}, {int(best.n_cells)} cells, "
        f"sections {best.slices}",
        flush=True,
    )
    print(
        "\n  drop-in replacement for scripts/atlas/fig_atlas_overview.py:\n",
        flush=True,
    )
    print(f"    _KC_X0, _KC_X1 = {best.x0:.0f}, {best.x1:.0f}", flush=True)
    print(f"    _KC_Y0, _KC_Y1 = {best.y0:.0f}, {best.y1:.0f}", flush=True)
    print(
        "    _KC_SLICE_IDXS = "
        f"{list(range(int(best.slice_start), int(best.slice_start) + N_WIN))}",
        flush=True,
    )

    win = best.slices.split("|")
    fig, axes = plt.subplots(1, len(win), figsize=(5.0 * len(win), 5.4))
    for ax, sl in zip(np.atleast_1d(axes), win, strict=False):
        k = sl_arr == sl
        cols = [get_pooled_color(v) for v in lab[k]]
        ax.scatter(xy[k, 0], xy[k, 1], c=cols, s=1.4, linewidths=0, rasterized=True)
        ax.add_patch(
            plt.Rectangle(
                (best.x0, best.y0), BOX, BOX, fill=False, ec="#241D14", lw=1.8, ls="--"
            )
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(sl, fontsize=10)
    fig.suptitle(
        f"Queen Small-type compartment selection box — {int(best.n_cells)} cells, "
        f"{100 * best.kc_purity:.1f}% Kenyon "
        f"(worker box was ~800 at ~96%)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "queen_box.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
