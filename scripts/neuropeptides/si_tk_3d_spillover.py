"""SI figure for the Tk 3D spillover analysis (tk_3d_spillover.py).

Four panels, quantitative evidence on top and the same finding rendered
directly in 3D underneath:
  A. Per-section KC-vs-rest Tk fold enrichment across the full A-P (z) axis
     of the reconstruction, one point per section -- whether the pooled
     per-caste folds hold across the whole volume rather than a few
     sections.
  B. True 3D distance from each non-KC cell to its nearest Kenyon-cell
     neighbor, Tk-expressing vs not, per caste -- the "spillover" language
     as a measured spatial gradient.
  C, D. The registered volume itself, queen and worker, with the optic
     lobe and Kenyon-cell compartments as landmark clouds and Tk-expressing
     optic-lobe cells picked out -- the same enrichment contrast as
     something you can see rather than only read off an axis.

Reads bimp_st.paths.TK_3D_SPILLOVER/{per_section_fold.csv,
per_cell_distance_to_kc.parquet}, and (for C/D) loads combined_clustered.h5ad
directly via tk_3d_render's load()/panel() -- reused, not reimplemented,
so the two views of the same finding cannot silently drift apart.
Writes results/figures/si/figS_tk_3d_spillover.png
"""

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from bimp_st.print_scale import PRINTED_WIDTH_FRAC  # noqa: E402
from bimp_st.si_common import (
    LABEL_SIZE,
    LEGEND_SIZE,
    QUEEN_COLOR,
    TICK_SIZE,
    WORKER_COLOR,
    add_panel_headers,
    set_style,
)
from bimp_st.si_common import (
    save as _save,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuropeptides.tk_3d_render import TK_WORKER_COLOR  # noqa: E402
from neuropeptides.tk_3d_render import load as _load_3d  # noqa: E402
from neuropeptides.tk_3d_render import panel as _panel_3d  # noqa: E402

from bimp_st.paths import TK_3D_SPILLOVER  # noqa: E402

DATA_DIR = TK_3D_SPILLOVER
# BIMP_TK3D_OUT writes the figure and its render counts to a scratch dir
# (render-identity checks) instead of the shared results/.
SCRATCH = os.environ.get("BIMP_TK3D_OUT")
SCRATCH_DIR = Path(SCRATCH) if SCRATCH else DATA_DIR
# Darker gold (dark goldenrod), the QUEEN_COLOR hue with enough contrast
# against the grey/tan landmark clouds of panels C/D.
QUEEN_TK_3D_COLOR = "#B8860B"


def panel_a(ax, per_section):
    for brain, color in [("queen", QUEEN_COLOR), ("worker", WORKER_COLOR)]:
        sub = per_section[per_section.brain == brain].sort_values("z_level")
        z_norm = (sub.z_level - sub.z_level.min()) / (
            sub.z_level.max() - sub.z_level.min()
        )
        ax.scatter(
            z_norm,
            sub.fold,
            color=color,
            s=22,
            alpha=0.85,
            label=f"{brain.capitalize()} ($n$={len(sub)} sections)",
            zorder=3,
        )
        med = sub.fold.median()
        ax.axhline(med, color=color, ls="--", lw=1.0, alpha=0.6, zorder=2)

    ax.set_xlabel("Normalized anterior$\\to$posterior position", fontsize=LABEL_SIZE)
    ax.set_ylabel(
        "Tk fold enrichment,\nKC vs. non-KC (per section)", fontsize=LABEL_SIZE
    )
    # No title: caption (A) already reads "The fold within every section
    # along the anterior-posterior axis..."
    ax.tick_params(labelsize=TICK_SIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center right", fontsize=LEGEND_SIZE, frameon=False)


def panel_b(ax, per_cell):
    for i, (brain, color) in enumerate(
        [("queen", QUEEN_COLOR), ("worker", WORKER_COLOR)]
    ):
        sub = per_cell[(per_cell.brain == brain) & (~per_cell.is_kc)]
        pos = sub[sub.expr_tk > 0].dist_to_kc_um.dropna()
        neg = sub[sub.expr_tk == 0].dist_to_kc_um.dropna()
        offset = i * 2.4
        for j, (vals, _label, alpha) in enumerate(
            [(pos, "Tk+", 0.9), (neg, "Tk$-$", 0.35)]
        ):
            parts = ax.violinplot(
                [vals],
                positions=[offset + j],
                widths=0.9,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(alpha)
                pc.set_edgecolor("none")
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.3)

    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(
        ["Queen\nTk+", "Queen\nTk$-$", "Worker\nTk+", "Worker\nTk$-$"],
        fontsize=TICK_SIZE,
    )
    ax.set_ylabel(
        "3D distance to nearest\nKenyon cell neighbor (µm)", fontsize=LABEL_SIZE
    )
    # No title: caption (B) already reads "3D distance from each non-Kenyon
    # cell to its nearest Kenyon cell..."
    ax.tick_params(labelsize=TICK_SIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def compose():
    per_section = pd.read_csv(DATA_DIR / "per_section_fold.csv")
    per_cell = pd.read_parquet(DATA_DIR / "per_cell_distance_to_kc.parquet")
    d3 = _load_3d()

    fig = plt.figure(figsize=(12.5, 8.9))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.05],
        hspace=0.26,
        left=0.06,
        right=0.98,
        top=0.96,
        bottom=0.075,
    )

    top = outer[0].subgridspec(1, 2, wspace=0.32)
    ax_a = fig.add_subplot(top[0])
    ax_b = fig.add_subplot(top[1])
    panel_a(ax_a, per_section)
    panel_b(ax_b, per_cell)

    bottom = outer[1].subgridspec(1, 2, wspace=0.02)
    ax_c = fig.add_subplot(bottom[0], projection="3d")
    ax_d = fig.add_subplot(bottom[1], projection="3d")
    # Caste is named in the panel tag, so the in-axes title is left empty.
    # Queen Tk+ cells are drawn in a darker gold (same hue as QUEEN_COLOR,
    # which washes out against the grey context and tan Kenyon-cell clouds).
    # The legend is drawn once under both panels, not inside each axes.
    for ax, mask, color in (
        (ax_c, d3["queen_mask"], QUEEN_TK_3D_COLOR),
        (ax_d, ~d3["queen_mask"], TK_WORKER_COLOR),
    ):
        _panel_3d(
            ax, d3, mask, color, "", pad=0.02, legend=False, zoom=1.75, title_size=8
        )
    hc, lc = ax_c.get_legend_handles_labels()
    hd, ld = ax_d.get_legend_handles_labels()
    # Two rows: the three landmark clouds, then the two Tk+ sets (one per
    # caste), so neither row is left with a stranded entry.
    rows = [
        (hc[:3], lc[:3]),
        (
            [hc[3], hd[3]],
            [
                lc[3].replace("\n", " ").replace("Tk+ in", "Queen Tk+ in"),
                ld[3].replace("\n", " ").replace("Tk+ in", "Worker Tk+ in"),
            ],
        ),
    ]
    for (h, labels), y in zip(rows, (0.032, 0.0), strict=True):
        leg = fig.legend(
            h,
            labels,
            loc="lower center",
            ncol=len(h),
            frameon=False,
            fontsize=LEGEND_SIZE,
            markerscale=5,
            columnspacing=2.0,
            bbox_to_anchor=(0.5, y),
        )
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)

    add_panel_headers(
        fig,
        [
            (ax_a, "A", "Enrichment by section"),
            (ax_b, "B", "Distance to Kenyon cells"),
            (ax_c, "C", "Queen, 3D"),
            (ax_d, "D", "Worker, 3D"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["figS_tk_3d_spillover"],
    )

    n_story_q = int((d3["queen_mask"] & d3["optic_flags"] & (d3["expr_tk"] > 0)).sum())
    n_story_w = int((~d3["queen_mask"] & d3["optic_flags"] & (d3["expr_tk"] > 0)).sum())
    n_optic_q = int((d3["queen_mask"] & d3["optic_flags"]).sum())
    n_optic_w = int((~d3["queen_mask"] & d3["optic_flags"]).sum())
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            dict(
                brain="queen",
                n_tk_pos_optic=n_story_q,
                n_optic=n_optic_q,
                frac=n_story_q / n_optic_q,
            ),
            dict(
                brain="worker",
                n_tk_pos_optic=n_story_w,
                n_optic=n_optic_w,
                frac=n_story_w / n_optic_w,
            ),
        ]
    ).to_csv(SCRATCH_DIR / "render_optic_tk_counts.csv", index=False)

    fig_dir = SCRATCH_DIR if SCRATCH else Path("results/figures/si")
    fig_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, fig_dir, "figS_tk_3d_spillover")
    print(f"Saved {fig_dir / 'figS_tk_3d_spillover.png'}", flush=True)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        compose()
