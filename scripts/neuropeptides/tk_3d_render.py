"""A genuine 3D rendering of the Tk optic-lobe spillover, not a projection.

tk_3d_spillover.py / si_tk_3d_spillover.py established the
quantitative case (per-section fold consistency, a true 3D distance
gradient). This script makes the same finding visible directly: render the
registered volume in 3D, with Kenyon cell compartments as a landmark cloud
and Tk-expressing non-KC cells picked out against a faint whole-brain
silhouette, queen and worker side by side at matched axes and viewing angle.

Coordinates: identical construction to tk_3d_spillover.py --
obsm['spatial_aligned'] for (x, y), z = z_level * Z_SCALE (imported from
region_prediction_3d, the one validated 3D coordinate system in
this codebase). Gene resolution and expression extraction reuse
fig_neuropeptide_spatial.py's
own resolve_gene/_get_expr rather than re-deriving them.

Region masks (Kenyon compartments; optic lobe = lamina + medulla + lobula)
are the nine-region flags, via tk_3d_spillover.nine_region_flags.

Run directly, writes fig_tk_3d_render.png to bimp_st.paths.TK_3D_SPILLOVER
(not registered; si_tk_3d_spillover.py imports load() and panel() from here).
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuropeptides.fig_neuropeptide_spatial import resolve_gene  # noqa: E402
from regions.region_prediction_3d import Z_SCALE  # noqa: E402

from bimp_st.si_common import SPATIAL_H5AD  # noqa: E402
from bimp_st.si_common import get_expr as _get_expr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuropeptides.tk_3d_spillover import nine_region_flags, nine_region_kc_flags  # noqa: E402

from bimp_st.paths import REPO, TK_3D_SPILLOVER  # noqa: E402

# Standalone render only (main); si_tk_3d_spillover.py imports load/panel.
OUT = TK_3D_SPILLOVER

SEED = 0
N_CONTEXT = 35_000  # faint whole-brain silhouette, per caste
N_KC = 12_000  # Kenyon-cell landmark cloud, per caste
N_OPTIC = 12_000  # optic-lobe landmark cloud, per caste
N_TK_STORY_MAX = 9_000  # cap on Tk+-in-optic-lobe points actually drawn

KC_COLOR = "#B8AF9E"
OPTIC_COLOR = "#9EB6C4"
TK_QUEEN_COLOR = "#C1440E"
TK_WORKER_COLOR = "#1B5FA6"
ELEV, AZIM = 14, -55


def load():
    print(f"loading {SPATIAL_H5AD} ...", flush=True)
    adata = sc.read_h5ad(REPO / SPATIAL_H5AD)
    queen_mask = adata.obs["brain"].values == "queen"
    z_level = adata.obs["z_level"].to_numpy(dtype=float)
    sp = adata.obsm["spatial_aligned"]
    z = z_level * Z_SCALE
    xyz = np.column_stack([sp, z]).astype(np.float32)

    var = resolve_gene("Tk", adata.var_names)
    expr_tk = _get_expr(adata, var)

    # Nine-region flags, joined on (slice_name, cell_id); the in_* columns
    # stored in combined_clustered.h5ad predate them and are not read.
    kc_flags, _ = nine_region_kc_flags(adata.obs)  # inside only, for drawing
    optic_flags = nine_region_flags(adata.obs, ("medulla", "lamina", "lobula"))

    return dict(
        xyz=xyz,
        expr_tk=expr_tk,
        kc_flags=kc_flags,
        optic_flags=optic_flags,
        queen_mask=queen_mask,
    )


def panel(
    ax, d, brain_mask, tk_color, title, pad=0.18, legend=True, zoom=1.0, title_size=13
):
    rng = np.random.default_rng(SEED)
    xyz, expr = d["xyz"], d["expr_tk"]
    kc, optic = d["kc_flags"], d["optic_flags"]

    all_idx = np.flatnonzero(brain_mask)
    ctx_idx = rng.choice(all_idx, size=min(N_CONTEXT, len(all_idx)), replace=False)
    ax.scatter(
        xyz[ctx_idx, 0],
        xyz[ctx_idx, 1],
        xyz[ctx_idx, 2],
        s=1.4,
        c="#CFCFCF",
        alpha=0.10,
        linewidths=0,
        zorder=1,
        label="Whole brain (context)",
    )

    kc_idx_all = np.flatnonzero(brain_mask & kc)
    kc_idx = rng.choice(kc_idx_all, size=min(N_KC, len(kc_idx_all)), replace=False)
    ax.scatter(
        xyz[kc_idx, 0],
        xyz[kc_idx, 1],
        xyz[kc_idx, 2],
        s=2.0,
        c=KC_COLOR,
        alpha=0.35,
        linewidths=0,
        zorder=2,
        label="Kenyon cell compartments",
    )

    optic_idx_all = np.flatnonzero(brain_mask & optic)
    optic_idx = rng.choice(
        optic_idx_all, size=min(N_OPTIC, len(optic_idx_all)), replace=False
    )
    ax.scatter(
        xyz[optic_idx, 0],
        xyz[optic_idx, 1],
        xyz[optic_idx, 2],
        s=2.0,
        c=OPTIC_COLOR,
        alpha=0.30,
        linewidths=0,
        zorder=2,
        label="Optic lobe (lamina/medulla/lobula)",
    )

    story_idx_all = np.flatnonzero(brain_mask & optic & (expr > 0))
    n_story = len(story_idx_all)
    story_idx = rng.choice(
        story_idx_all, size=min(N_TK_STORY_MAX, n_story), replace=False
    )
    ax.scatter(
        xyz[story_idx, 0],
        xyz[story_idx, 1],
        xyz[story_idx, 2],
        s=5.0,
        c=tk_color,
        alpha=0.85,
        linewidths=0,
        zorder=4,
        label=f"Tk+ in the optic lobe\n(n={n_story:,} cells)",
    )

    # Padded (not tight) limits, symmetric on every axis: at this oblique view
    # tight limits push the point cloud into the top-left legend corner. A
    # flat fraction of headroom shrinks the apparent brain without changing
    # the view angle or aspect.
    def _padded(lo, hi, frac=None):
        frac = pad if frac is None else frac
        span = hi - lo
        return lo - span * frac, hi + span * frac

    ax.set_xlim(*_padded(xyz[all_idx, 0].min(), xyz[all_idx, 0].max()))
    ax.set_ylim(*_padded(xyz[all_idx, 1].min(), xyz[all_idx, 1].max()))
    ax.set_zlim(*_padded(xyz[all_idx, 2].min(), xyz[all_idx, 2].max()))
    ax.set_box_aspect((1, 1, 0.6), zoom=zoom)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=-10)
    if legend:
        leg = ax.legend(
            loc="upper left",
            fontsize=7.5,
            frameon=False,
            bbox_to_anchor=(0.0, 0.98),
            markerscale=5,
            labelspacing=0.9,
        )
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)


def main():
    d = load()
    fig = plt.figure(figsize=(13.0, 6.5))
    ax_q = fig.add_subplot(1, 2, 1, projection="3d")
    ax_w = fig.add_subplot(1, 2, 2, projection="3d")
    panel(
        ax_q,
        d,
        d["queen_mask"],
        TK_QUEEN_COLOR,
        "Queen -- Tk reaches beyond the Kenyon cells",
    )
    panel(
        ax_w,
        d,
        ~d["queen_mask"],
        TK_WORKER_COLOR,
        "Worker -- Tk stays close to the Kenyon cell boundary",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, wspace=0.02)
    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / "fig_tk_3d_render.png"
    fig.savefig(out_png, dpi=220, facecolor="white")
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
