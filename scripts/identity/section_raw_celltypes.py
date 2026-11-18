"""A single Xenium section coloured by the raw Tangram cell-type call.

Not the anatomy-guided cluster vote: this shows what the label transfer
produces on its own, collapsed to the canonical vocabulary (16 annotated
central-brain types + a single "Optic lobe"), so it can be used to judge how
much work the anatomical prior does.

Renders the queen mid-brain section with the most cells, plus a small strip of
neighbouring sections for context.

Outputs to results/reframe/slice_celltypes/:
    slice_celltypes.png / .pdf       the main section, large
    slice_strip.png                  five sections through the volume
    slice_composition.csv            cell-type counts on the main section
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
    CANONICAL_ORDER,
    add_canonical,
    get_canonical_color,
)

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
H5 = REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
OUT = REPO / "results" / "reframe" / "slice_celltypes"
OUT.mkdir(parents=True, exist_ok=True)

BRAIN = "queen"  # the atlas-overview section panels are all queen
LABEL = "cell_type_canonical"
DOT = 2.4
N_STRIP = 5

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "savefig.facecolor": "white",
    }
)


def scalebar(ax, xy, span_um=500, label="500 µm"):
    """Bar with its label below it. The y axis is inverted, so 'below' on screen
    means a larger data y."""
    (x0, y0) = xy
    ax.plot(
        [x0, x0 + span_um],
        [y0, y0],
        lw=3,
        color="#241D14",
        solid_capstyle="butt",
        zorder=6,
    )
    ax.text(
        x0 + span_um / 2,
        y0 + span_um * 0.22,
        label,
        ha="center",
        va="top",
        fontsize=9,
        color="#241D14",
        zorder=6,
    )


def draw_slice(ax, xy, labels, cats, title, dot=DOT, legend=False):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(xy))
    cols = np.array([get_canonical_color(v) for v in labels])
    ax.scatter(
        xy[order, 0], xy[order, 1], c=cols[order], s=dot, linewidths=0, rasterized=True
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11)
    if legend:
        present = [c for c in cats if (labels == c).sum() > 0]
        h = [
            plt.Line2D([], [], marker="o", ls="", ms=7, color=get_canonical_color(c))
            for c in present
        ]
        ax.legend(
            h,
            present,
            frameon=False,
            fontsize=8.5,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )


def main() -> None:
    print(f"loading {H5.name} ...", flush=True)
    a = ad.read_h5ad(H5)
    # raw Tangram call only -- no anatomy-guided vote
    src = add_canonical(a, src="tangram_cell_type")
    print(f"  canonical labels built from obs['{src}']", flush=True)

    m = (a.obs["brain"].astype(str) == BRAIN).values
    sub = a[m]
    xy_all = np.asarray(sub.obsm["spatial"], float)
    slices = sub.obs["slice_name"].astype(str).values
    labels_all = sub.obs[LABEL].astype(str).values

    counts = pd.Series(slices).value_counts()
    # Order sections by anatomical depth, not by name: sorting slice_name
    # alphabetically puts Region_1 first, which says nothing about position along
    # the anterior-posterior axis.
    z = (
        pd.Series(np.asarray(sub.obs["z_level"], float))
        .groupby(pd.Series(slices))
        .median()
    )
    uniq = list(z.sort_values().index)
    mid = uniq[len(uniq) // 3 : 2 * len(uniq) // 3]
    main_slice = counts[counts.index.isin(mid)].idxmax()
    print(
        f"  sections ordered by median z_level ({z.min():.0f} to {z.max():.0f})",
        flush=True,
    )
    print(
        f"  {BRAIN}: {len(uniq)} sections; main = {main_slice} "
        f"({counts[main_slice]:,} cells)",
        flush=True,
    )

    cats = [c for c in CANONICAL_ORDER]

    # ── main section ────────────────────────────────────────────────────
    sm = slices == main_slice
    xy, lab = xy_all[sm], labels_all[sm]

    comp = pd.Series(lab).value_counts().rename_axis("cell_type").reset_index(name="n")
    comp["pct"] = 100 * comp.n / comp.n.sum()
    comp.to_csv(OUT / "slice_composition.csv", index=False)
    print("\n  composition of the main section:", flush=True)
    print(comp.head(10).to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(10.5, 9))
    draw_slice(
        ax,
        xy,
        lab,
        cats,
        f"{BRAIN.capitalize()} brain, section {main_slice} — "
        f"{sm.sum():,} cells\nraw Tangram cell type, canonical 17",
        dot=DOT,
        legend=True,
    )
    scalebar(ax, (xy[:, 0].min() + 60, xy[:, 1].max() + 40))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slice_celltypes.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)

    # ── strip through the volume ────────────────────────────────────────
    step = max(1, len(uniq) // (N_STRIP + 1))
    picks = uniq[step::step][:N_STRIP]  # uniq is z-ordered, so this is a
    # genuine sweep through the volume
    fig, axes = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 4.4))
    for ax, sl in zip(np.atleast_1d(axes), picks, strict=False):
        k = slices == sl
        draw_slice(
            ax, xy_all[k], labels_all[k], cats, f"{sl}\n{k.sum():,} cells", dot=1.1
        )
    fig.suptitle(
        f"{BRAIN.capitalize()} brain through the volume — "
        "raw Tangram cell type, canonical 17",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "slice_strip.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
