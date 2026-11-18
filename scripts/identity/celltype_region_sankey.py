"""Sankey diagrams: cell type -> anatomical region.

One labeling rule for every cell: the full Tangram posterior. Each cell's
29-way probability vector (``adata.obsm["tangram_ct_pred"]``) is pooled to the
canonical-17 vocabulary as the hard label is (``cluster_labels.get_canonical``:
identity for the 16 named types, OL_1..OL_13 summed into "Optic lobe"), and the
whole vector contributes to the cell's region flow. No cluster-level override,
compartment restriction or excluded residual is applied.

  - Right (region) totals are exact hand-annotated cell counts (each
    posterior sums to 1).
  - Left (cell type) totals are expected counts under the posterior (can be
    fractional, labelled "~"); they sum to the same N.

Region assignment is categorical: a cell sits in exactly one hand-annotated
region or is dropped (~35% of the atlas; see nine_region_common.py). Region
labels are joined onto the canonical Tangram object by (slice_name, cell_id),
never positionally.

Combined (both castes) plus queen-only and worker-only. Pure matplotlib (no
plotly/kaleido dependency).

Outputs to results/reframe/celltype_region_sankey/:
    sankey_combined.png / sankey_queen.png / sankey_worker.png
    flow_counts_combined.csv   (cell_type x region, wide, probability mass)
    flow_counts_long.csv       (per-cell join table: id cols + 17 prob cols + region)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import (  # noqa: E402
    CANONICAL_ORDER,
    get_canonical,
    get_canonical_color,
)
from bimp_st.paths import load_tangram  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.nine_region_common import (  # noqa: E402
    DESCRIPTIVE_REGIONS,
    REGION_COLORS,
)
from bimp_st.nine_region_common import (
    display as region_display,
)

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
NINE_REGION_ATLAS = REPO / "results" / "nine_region" / "atlas"

# Rendered links below this fraction of a diagram's own total are omitted
# from the ribbon drawing only: with a dense posterior every type has some mass
# in every region, and hairline links obscure the structure. Node totals and the
# CSV stay complete, and the dropped mass is printed.
MIN_LINK_FRAC = 0.0008


def load_region_flags(brain: str) -> pd.DataFrame:
    """(slice_name, cell_id) -> one region label, from the 9-region atlas.

    Nullable-boolean in_* columns: pd.NA means "not traced", never treated
    as False. A cell in >1 region (99/1.3M atlas-wide) or in none of the
    DESCRIPTIVE_REGIONS is dropped -- this is a flow diagram of cells with an
    unambiguous single region, not a claim about the other 65% of the atlas.
    """
    import anndata as ad

    p = NINE_REGION_ATLAS / f"{brain}_aligned_clustered_9region.h5ad"
    a = ad.read_h5ad(p, backed="r")
    cols = {r: f"in_{r}" for r in DESCRIPTIVE_REGIONS}
    obs = a.obs[["slice_name", "cell_id"] + list(cols.values())].copy()
    a.file.close()

    flags = np.stack(
        [obs[c].to_numpy(dtype=bool, na_value=False) for c in cols.values()],
        axis=1,
    )
    n_hit = flags.sum(axis=1)
    single = n_hit == 1
    region_idx = flags.argmax(axis=1)
    region = np.array(DESCRIPTIVE_REGIONS, dtype=object)[region_idx]
    region = np.where(single, region, None)

    out = pd.DataFrame(
        {
            "slice_name": obs["slice_name"].astype(str).to_numpy(),
            "cell_id": obs["cell_id"].astype(str).to_numpy(),
            "region": region,
        }
    )
    return out[out["region"].notna()]


def build_flow_table() -> pd.DataFrame:
    """One row per cell: id columns + 17 canonical-type probability columns
    (each cell's full Tangram posterior, pooled OL_1..13 -> "Optic lobe") +
    its single hand-annotated region.
    """
    adata = load_tangram()
    probs = adata.obsm["tangram_ct_pred"]
    assert probs.shape[0] == adata.n_obs, "tangram_ct_pred not cell-aligned"

    canon_of = {c: get_canonical(c) for c in probs.columns}
    missing = [c for c in CANONICAL_ORDER if c not in canon_of.values()]
    if missing:
        raise ValueError(f"canonical types with no posterior column at all: {missing}")

    pooled = pd.DataFrame(index=np.arange(adata.n_obs))
    for canon in CANONICAL_ORDER:
        cols = [c for c, k in canon_of.items() if k == canon]
        pooled[canon] = probs[cols].to_numpy().sum(axis=1)

    row_sums = pooled.to_numpy().sum(axis=1)
    off = np.abs(row_sums - 1.0)
    print(
        f"posterior row-sum check: max |1 - sum| = {off.max():.4f}  "
        f"(mean {off.mean():.5f}); renormalizing per-cell to exactly 1"
    )
    pooled = pooled.div(row_sums, axis=0)

    obs = pd.DataFrame(
        {
            "slice_name": adata.obs["slice_name"].astype(str).to_numpy(),
            "cell_id": adata.obs["cell_id"].astype(str).to_numpy(),
            "brain": adata.obs["brain"].astype(str).to_numpy(),
        }
    )
    obs = pd.concat([obs, pooled], axis=1)

    regions = pd.concat(
        [load_region_flags(b) for b in ("queen", "worker")], ignore_index=True
    )
    merged = obs.merge(regions, on=["slice_name", "cell_id"], how="inner")
    return merged


def _ribbon(x0, y0a, y0b, x1, y1a, y1b):
    """Smooth cubic-bezier sankey ribbon between a left and a right segment."""
    xm = (x0 + x1) / 2
    verts = [
        (x0, y0a),
        (xm, y0a),
        (xm, y1a),
        (x1, y1a),
        (x1, y1b),
        (xm, y1b),
        (xm, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def _minimize_crossings(tab_full: pd.DataFrame, n_iter: int = 12):
    """Order left/right nodes to minimize ribbon crossings (barycenter
    heuristic with alternating passes, the standard 2-layer crossing-reduction
    method).

    Starts from a size-descending order on both sides, then repeatedly
    re-ranks each side by the flow-weighted average rank of what it connects
    to on the other side, alternating sides, until the order stops changing
    or ``n_iter`` passes are used. Node stacking is bottom-up (see
    node_positions), so "rank 0" = bottom; we want the biggest, most
    connected nodes near the top, which the barycenter pass naturally
    clusters together regardless of raw size.
    """
    left = list(tab_full.sum(axis=1).sort_values(ascending=True).index)
    right = list(tab_full.sum(axis=0).sort_values(ascending=True).index)

    def barycenter(order_self, order_other, axis):
        rank_other = {name: i for i, name in enumerate(order_other)}
        scores = {}
        for name in order_self:
            row = tab_full.loc[name] if axis == 0 else tab_full[name]
            w = row[row > 0]
            if w.sum() == 0:
                scores[name] = -1  # no connections: park at the bottom
                continue
            scores[name] = sum(rank_other[k] * v for k, v in w.items()) / w.sum()
        return sorted(order_self, key=lambda n: scores[n])

    for _ in range(n_iter):
        new_left = barycenter(left, right, axis=0)
        new_right = barycenter(right, new_left, axis=1)
        if new_left == left and new_right == right:
            left, right = new_left, new_right
            break
        left, right = new_left, new_right
    return left, right


def sankey(
    flow: pd.DataFrame,
    title: str,
    out_path: Path,
    type_cols: list[str] = CANONICAL_ORDER,
) -> None:
    """Two-column flow diagram: cell type (left, probability-weighted) ->
    hand-annotated region (right, hard)."""
    n_cells = len(flow)  # exact int; summed per-cell floats round to n-1 sometimes
    tab_full = flow.groupby("region")[type_cols].sum().T
    tab_full = tab_full.reindex(index=type_cols).fillna(0.0)
    ct_order, reg_order = _minimize_crossings(tab_full)
    tab = tab_full.reindex(index=ct_order, columns=reg_order, fill_value=0.0)

    left_totals = tab.sum(axis=1)
    right_totals = tab.sum(axis=0)
    total = left_totals.sum()

    gap = total * 0.032
    fig, ax = plt.subplots(figsize=(12, 14))

    def node_positions(totals, order):
        pos, y = {}, 0.0
        for name in order:
            h = totals[name]
            pos[name] = (y, y + h)
            y += h + gap
        return pos, y - gap

    left_pos, left_h = node_positions(left_totals, ct_order)
    right_pos, right_h = node_positions(right_totals, reg_order)
    y_shift = (left_h - right_h) / 2
    right_pos = {k: (v0 + y_shift, v1 + y_shift) for k, (v0, v1) in right_pos.items()}

    x_left, x_right = 0.0, 1.0
    node_w = 0.018

    left_cursor = {k: v0 for k, (v0, v1) in left_pos.items()}
    right_cursor = {k: v0 for k, (v0, v1) in right_pos.items()}

    all_links = [
        (ct, reg, tab.loc[ct, reg])
        for ct in ct_order
        for reg in reg_order
        if tab.loc[ct, reg] > 0
    ]
    min_render = total * MIN_LINK_FRAC
    links = [link for link in all_links if link[2] >= min_render]
    dropped_mass = sum(n for _, _, n in all_links if n < min_render)
    links.sort(key=lambda t: t[2], reverse=True)

    for ct, reg, n in links:
        color = get_canonical_color(ct)
        ly0 = left_cursor[ct]
        ly1 = ly0 + n
        left_cursor[ct] = ly1
        ry0 = right_cursor[reg]
        ry1 = ry0 + n
        right_cursor[reg] = ry1
        path = _ribbon(x_left + node_w, ly0, ly1, x_right - node_w, ry0, ry1)
        ax.add_patch(
            PathPatch(
                path, facecolor=color, edgecolor="none", alpha=0.62, lw=0, zorder=2
            )
        )

    # Cells whose mass fell below the render threshold for a given link still
    # count toward the node's total height (left_pos/right_pos above), so the
    # cursor must still advance past that mass or the last rendered ribbon on
    # a node would fall short of the node's top edge.
    for ct, reg, n in all_links:
        if n < min_render:
            left_cursor[ct] += n
            right_cursor[reg] += n

    for ct in ct_order:
        y0, y1 = left_pos[ct]
        ax.add_patch(
            Rectangle(
                (x_left, y0),
                node_w,
                y1 - y0,
                facecolor=get_canonical_color(ct),
                edgecolor="white",
                lw=0.6,
                zorder=3,
            )
        )
        ax.text(
            x_left - 0.014,
            (y0 + y1) / 2,
            f"{ct}\n(~{left_totals[ct]:,.0f})",
            ha="right",
            va="center",
            fontsize=9.5,
            linespacing=1.15,
        )

    for reg in reg_order:
        y0, y1 = right_pos[reg]
        ax.add_patch(
            Rectangle(
                (x_right - node_w, y0),
                node_w,
                y1 - y0,
                facecolor=REGION_COLORS.get(reg, "#999999"),
                edgecolor="white",
                lw=0.6,
                zorder=3,
            )
        )
        ax.text(
            x_right + 0.014,
            (y0 + y1) / 2,
            f"{region_display(reg)}\n({int(right_totals[reg]):,})",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            linespacing=1.15,
        )

    ax.set_xlim(-0.42, 1.30)
    ax.set_ylim(-total * 0.015, max(left_h, right_h) + total * 0.015)
    ax.axis("off")
    ax.set_title(
        f"{title} — cell type → hand-annotated region\n"
        f"(probability-mass weighted: every cell's full Tangram posterior, "
        f"no overrides; n={n_cells:,} cells in exactly one region)",
        fontsize=11,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(
        f"wrote {out_path}  (n={n_cells:,}, "
        f"{len(links)}/{len(all_links)} links rendered, "
        f"{100 * dropped_mass / total:.2f}% of mass in unrendered hairline links)"
    )


if __name__ == "__main__":
    flow = build_flow_table()
    OUT = REPO / "results" / "reframe" / "celltype_region_sankey"
    OUT.mkdir(parents=True, exist_ok=True)
    flow.to_csv(OUT / "flow_counts_long.csv", index=False)

    sankey(flow, "Both castes", OUT / "sankey_combined.png")
    sankey(flow[flow["brain"] == "queen"], "Queen", OUT / "sankey_queen.png")
    sankey(flow[flow["brain"] == "worker"], "Worker", OUT / "sankey_worker.png")

    ct_reg = flow.groupby("region")[CANONICAL_ORDER].sum().T
    ct_reg = ct_reg.loc[ct_reg.sum(axis=1).sort_values(ascending=False).index]
    ct_reg.to_csv(OUT / "flow_counts_combined.csv")
    print(
        f"\n{len(flow):,} cells with exactly one hand-annotated region "
        f"used ({100 * len(flow) / 1_300_192:.1f}% of the atlas)"
    )
