"""SI figure: Spatial slices colored by probe-only Leiden clusters.

Same slices as the cell-type section panels, but colored by the spatial Leiden
clustering (from 101 Xenium panel genes) instead of Tangram-transferred
snRNA cell types. Shows what the probe panel alone can resolve
without the snRNA reference.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from bimp_st.paths import CANONICAL_TANGRAM
from bimp_st.si_common import (
    OUTPUT_DIR,
    PIXEL_SCALE,
    SCALEBAR_KW,
    TICK_SIZE,
    save,
    set_style,
    strip_spines,
)

set_style()


def _scalebar_below(ax, span, cx, cy, bar_um=1000, label="1 mm"):
    """Draw a scale bar just below an inverted-y spatial panel.

    These panels use an inverted y-axis (image convention), so the visual
    bottom is the *larger* data-y value; the shared ``add_scalebar`` helper
    (written for non-inverted axes) would place the bar inside the plot.
    Coordinates are in Xenium pixels (PIXEL_SCALE µm/px).
    """
    bar_len_px = bar_um / PIXEL_SCALE
    x0 = cx - bar_len_px / 2
    y0 = cy + span / 2 + span * 0.03
    ax.plot([x0, x0 + bar_len_px], [y0, y0], **SCALEBAR_KW)
    ax.text(
        cx,
        y0 + span * 0.02,
        label,
        ha="center",
        va="top",
        fontsize=TICK_SIZE,
        clip_on=False,
    )


def main():
    import scanpy as sc

    from bimp_st.cluster_labels import (
        DISPLAY_NAME_OVERRIDES,
        display_color,
        display_legend_sort,
        get_pooled_display,
        hierarchical_vote,
    )

    def _color(ct):
        # display_color separates the three glial entries (pooled "Glia" and
        # Perineurial glia share one hex in the pooled palette).
        return display_color(ct)

    def _display(ct):
        d = get_pooled_display(ct)
        return DISPLAY_NAME_OVERRIDES.get(d, DISPLAY_NAME_OVERRIDES.get(ct, d))

    print("SI: Probe-only Leiden clusters on spatial slices", flush=True)

    # Load spatial data
    SPATIAL_H5AD = "pipelines/combined_atlas/data/combined_clustered.h5ad"
    print(f"  Loading {SPATIAL_H5AD}...", flush=True)
    sp = sc.read_h5ad(SPATIAL_H5AD)
    print(f"  {sp.n_obs:,} cells", flush=True)

    # Canonical Tangram cell-type labels (same cells/order as the spatial
    # h5ad; the canonical Tangram output is written by tangram_mapping.py
    # train). Raw per-cell ``tangram_cell_type`` is panel-unresolvable argmax
    # noise — the
    # canonical label is the combined run plus the expression-only per-cluster
    # vote, which also pools the fine OL_*/glia/KC subtypes into the coarse
    # display groups. Same helper fig_atlas_overview.py uses.
    # Must be the canonical object (cell-type columns keyed by name): the
    # top row is the voted label, with 12 clusters voting Optic lobe and
    # none voting Other neurons.
    TANGRAM_H5AD = str(CANONICAL_TANGRAM)
    print(f"  Loading Tangram labels from {TANGRAM_H5AD}...", flush=True)
    tg = sc.read_h5ad(TANGRAM_H5AD, backed="r")
    for _req in ("tangram_cell_type", "leiden"):
        if _req not in tg.obs.columns:
            raise KeyError(
                f"{_req} column missing from tangram h5ad — rerun "
                "tangram_mapping.py train"
            )
    # hierarchical_vote names the decision column "label"; cluster_vote
    # called it "voted".
    cluster_to_ct = hierarchical_vote(
        tg,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        dst="cluster_vote",
    )["label"].to_dict()
    print(
        f"  Expression-only cluster vote: {len(cluster_to_ct)} Leiden clusters -> "
        f"{len(set(cluster_to_ct.values()))} cell-type groups",
        flush=True,
    )

    slices_arr = sp.obs["slice_name"].values.astype(str)
    brains_arr = sp.obs["brain"].values.astype(str)
    sp_coords = sp.obsm["spatial"]

    # ── Pick one mid-brain slice per caste (1 queen mid, 1 worker mid) ──
    unique_slices = np.unique(slices_arr)
    slice_brain = {}
    slice_n = {}
    aligned = sp.obsm.get("spatial_aligned", sp_coords)
    slice_z = {}
    for sl in unique_slices:
        mask = slices_arr == sl
        pos = np.where(mask)[0]
        slice_z[sl] = (
            aligned[pos, 2].mean()
            if aligned.shape[1] > 2
            else float(np.where(unique_slices == sl)[0][0])
        )
        slice_brain[sl] = brains_arr[pos[0]]
        slice_n[sl] = len(pos)

    sorted_slices = sorted(unique_slices, key=lambda s: slice_z[s])
    queen_slices = [s for s in sorted_slices if slice_brain[s] == "queen"]
    worker_slices = [s for s in sorted_slices if slice_brain[s] == "worker"]

    pick = []
    for brain_sl in [queen_slices, worker_slices]:
        if brain_sl:
            mid_range = brain_sl[len(brain_sl) // 3 : 2 * len(brain_sl) // 3]
            if not mid_range:
                mid_range = brain_sl
            pick.append(max(mid_range, key=lambda s: slice_n[s]))

    # ── Figure: 2 rows x 2 spatial cols + UMAP col + legend col ──────────────
    # The probe UMAP is a third column spanning both rows. It shares one colour
    # mapping (probe_colors, clusters 0-31) and one legend with the bottom
    # (probe-Leiden) row.
    rng = np.random.default_rng(42)

    # Spatial Leiden mapped to canonical (cluster vote) cell types so
    # the top row can be colored by the pooled cell-type palette.
    ct_spatial = sp.obs["leiden"].values.astype(str)
    ct_sp_unique = sorted(
        np.unique(ct_spatial),
        key=lambda s: int(s),
        reverse=True,
    )
    dominant_ct = {cl: cluster_to_ct.get(cl, "Mixed") for cl in ct_sp_unique}

    # Probe-only Leiden
    ct_probe = sp.obs["leiden"].values.astype(str)
    ct_pr_unique = sorted(np.unique(ct_probe), key=int, reverse=True)

    # Generate a large palette for 31 probe clusters
    import colorsys

    n_probe = len(ct_pr_unique)
    probe_colors = {}
    for i, cl in enumerate(sorted(ct_pr_unique, key=int)):
        hue = i / n_probe
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.7)
        probe_colors[cl] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    # Shared axis limits, computed before the figure exists so the panel
    # boxes can be cut to the tissue's aspect. A single square span left a
    # wide empty margin either side of each (tall, narrow) section.
    all_x, all_y = [], []
    slice_data = []
    _sect_n = {}  # cells in each displayed section, quoted in the caption
    for sl in pick[:2]:
        pos = np.where(slices_arr == sl)[0]
        sl_sp = sp_coords[pos]
        slice_data.append((sl, pos, sl_sp))
        _sect_n[slice_brain[sl]] = int(len(pos))
        all_x.extend([sl_sp[:, 0].min(), sl_sp[:, 0].max()])
        all_y.extend([sl_sp[:, 1].min(), sl_sp[:, 1].max()])
    x_range = (
        max(all_x[1] - all_x[0], all_x[3] - all_x[2])
        if len(all_x) >= 4
        else all_x[1] - all_x[0]
    )
    y_range = (
        max(all_y[1] - all_y[0], all_y[3] - all_y[2])
        if len(all_y) >= 4
        else all_y[1] - all_y[0]
    )
    # Separate x/y spans (still shared across the two slices, so the scale is
    # identical everywhere); ``set_aspect("equal")`` keeps µm isotropic.
    x_span = x_range * 1.08
    y_span = y_range * 1.10

    # ── UMAP data, prepared here (not in a separate call) so its aspect can
    # size the layout before any axes are created. ──
    umap_coords = sp.obsm["X_umap"]
    n_umap_sub = min(200_000, sp.n_obs)
    umap_sub_idx = rng.choice(sp.n_obs, n_umap_sub, replace=False)
    umap_sub_coords = umap_coords[umap_sub_idx]
    umap_sub_ct = ct_probe[umap_sub_idx]
    umap_xr = float(np.ptp(umap_sub_coords[:, 0]))
    umap_yr = float(np.ptp(umap_sub_coords[:, 1]))

    # Layout in inches on a ~7.2 in canvas (prints at 6.7 in), so authored
    # sizes are close to printed sizes. Left block: two rows (A, B) of two
    # equal-aspect sections, each axes cut exactly to the shared data window
    # so nothing shrinks inside its box and the two columns sit close.
    # Right: the UMAP (C), its top level with row A's so A and C share a
    # header baseline. A's cell-type legend sits under row A; B has no legend
    # because its cluster numbers are printed on the UMAP it shares colours
    # with.
    fig_w = 7.2
    panel_h = 1.85
    panel_w = panel_h * x_span / y_span
    col_gap = 0.10
    left_in = 0.04
    hdr_in = 0.44  # panel letter + "Queen"/"Worker" title above each row
    sb_in = 0.30  # scale bar + its label below each section
    leg_in = 0.50  # A's legend strip
    fig_h = hdr_in + panel_h + sb_in + leg_in + hdr_in + panel_h + sb_in + 0.02
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _axes_in(x, y_top, w, h):
        return fig.add_axes(
            [x / fig_w, (fig_h - y_top - h) / fig_h, w / fig_w, h / fig_h]
        )

    row_top = [hdr_in, hdr_in + panel_h + sb_in + leg_in + hdr_in]
    axes = np.empty((2, 2), dtype=object)
    for _r in range(2):
        for _c in range(2):
            axes[_r, _c] = _axes_in(
                left_in + _c * (panel_w + col_gap), row_top[_r], panel_w, panel_h
            )
    umap_x0 = left_in + 2 * panel_w + col_gap + 0.42
    umap_w = fig_w - umap_x0 - 0.04
    umap_h = umap_w * umap_yr / umap_xr
    if umap_h > fig_h - hdr_in - 0.30:
        umap_h = fig_h - hdr_in - 0.30
        umap_w = umap_h * umap_xr / umap_yr
    ax_umap = _axes_in(umap_x0, hdr_in, umap_w, umap_h)

    for si, (sl, pos, sl_sp) in enumerate(slice_data):
        brain_label = slice_brain[sl].capitalize()

        # ── Top row: Spatial Leiden colored by dominant Tangram cell type ──
        ax = axes[0, si]
        sl_ct = ct_spatial[pos]
        for cl in ct_sp_unique:
            ct_mask = sl_ct == cl
            if ct_mask.sum() == 0:
                continue
            idx = rng.permutation(ct_mask.sum())
            ax.scatter(
                sl_sp[ct_mask][idx, 0],
                sl_sp[ct_mask][idx, 1],
                c=_color(dominant_ct[cl]),
                s=0.5,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
            )
        ax.set_title(brain_label, fontsize=TICK_SIZE + 1, fontweight="bold", pad=2)
        print(f"  {brain_label}: {slice_n[sl]:,} cells (caption)", flush=True)
        # Centre on the bounding box, not the mean: for a C-shaped section
        # the two differ, and a mean-centred window clipped one side.
        cx = 0.5 * (sl_sp[:, 0].min() + sl_sp[:, 0].max())
        cy = 0.5 * (sl_sp[:, 1].min() + sl_sp[:, 1].max())
        ax.set_xlim(cx - x_span / 2, cx + x_span / 2)
        ax.set_ylim(cy + y_span / 2, cy - y_span / 2)
        ax.set_aspect("equal")
        strip_spines(ax)
        _scalebar_below(ax, y_span, cx, cy)

        # ── Bottom row: Probe-only Leiden ──
        ax = axes[1, si]
        sl_ct = ct_probe[pos]
        for cl in ct_pr_unique:
            ct_mask = sl_ct == cl
            if ct_mask.sum() == 0:
                continue
            idx = rng.permutation(ct_mask.sum())
            ax.scatter(
                sl_sp[ct_mask][idx, 0],
                sl_sp[ct_mask][idx, 1],
                c=probe_colors[cl],
                s=0.5,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
            )
        ax.set_title(brain_label, fontsize=TICK_SIZE + 1, fontweight="bold", pad=2)
        ax.set_xlim(cx - x_span / 2, cx + x_span / 2)
        ax.set_ylim(cy + y_span / 2, cy - y_span / 2)
        ax.set_aspect("equal")
        strip_spines(ax)
        _scalebar_below(ax, y_span, cx, cy)

    # ── UMAP column: probe-only Leiden clusters, same colours as bottom row ──
    for cl in ct_pr_unique:
        mask = umap_sub_ct == cl
        if mask.sum() == 0:
            continue
        idx = rng.permutation(mask.sum())
        ax_umap.scatter(
            umap_sub_coords[mask][idx, 0],
            umap_sub_coords[mask][idx, 1],
            c=probe_colors[cl],
            s=0.3,
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
        )
    strip_spines(ax_umap)
    ax_umap.set_aspect("equal")
    ax_umap.set_xlabel("UMAP1")
    ax_umap.set_ylabel("UMAP2")
    # No title: single panel; the caption states the subsample size.
    # 32 rainbow swatches cannot be decoded from a legend, so each cluster's
    # number is printed on the UMAP at the densest bin of its (subsampled)
    # points -- its core, not a median that can land between lobes -- then
    # nudged apart where two labels would overlap. B's sections use the same
    # colours, so this is B's key too. Deterministic: no randomness.
    import matplotlib.patheffects as pe

    lo = umap_sub_coords.min(axis=0)
    hi = umap_sub_coords.max(axis=0)
    nb = 70
    anchors, names = [], []
    for cl in sorted(ct_pr_unique, key=int):
        m = umap_sub_ct == cl
        if not m.any():
            continue
        h, ex, ey = np.histogram2d(
            umap_sub_coords[m, 0],
            umap_sub_coords[m, 1],
            bins=nb,
            range=[[lo[0], hi[0]], [lo[1], hi[1]]],
        )
        i, j = np.unravel_index(np.argmax(h), h.shape)
        anchors.append(((ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2))
        names.append(cl)
    # Repel in units of the label box (about two by one character heights at
    # the printed size), converted to data units through the axes' scale.
    pos = np.array(anchors, dtype=float)
    ax_w_in = ax_umap.get_position().width * fig_w
    data_per_in = (hi[0] - lo[0]) * 1.1 / ax_w_in
    char_in = TICK_SIZE * 0.75 / 72.0  # printed text is 0.75 of authored size
    min_dx, min_dy = 2.0 * char_in * data_per_in, 1.25 * char_in * data_per_in
    for _ in range(200):
        moved = False
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                d = pos[b] - pos[a]
                ox, oy = min_dx - abs(d[0]), min_dy - abs(d[1])
                if ox > 0 and oy > 0:
                    moved = True
                    # push along the axis that needs the smaller move
                    if ox / min_dx < oy / min_dy:
                        step = np.array([np.sign(d[0]) or 1.0, 0.0]) * ox / 2
                    else:
                        step = np.array([0.0, np.sign(d[1]) or 1.0]) * oy / 2
                    pos[a] -= step
                    pos[b] += step
        if not moved:
            break
    for (mx, my), cl in zip(pos, names, strict=True):
        ax_umap.text(
            mx,
            my,
            cl,
            ha="center",
            va="center",
            fontsize=TICK_SIZE,
            fontweight="bold",
            color="black",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            zorder=5,
        )
    print(
        f"  UMAP column: {len(ct_pr_unique)} clusters; "
        f"{n_umap_sub:,} of {sp.n_obs:,} cells shown",
        flush=True,
    )

    # ── Row-A legend: every cell type present across the two panels ──
    # Colors match the dots (cluster-voted cell type per Leiden cluster).
    present_clusters = set()
    for _sl, _pos, _sp in slice_data:
        present_clusters.update(np.unique(ct_spatial[_pos]))
    present_cts = {dominant_ct[cl] for cl in present_clusters}
    legend_cts = display_legend_sort(present_cts)
    print(f"  Top-row cell-type legend: {', '.join(legend_cts)}", flush=True)
    handles_ct = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_color(ct),
            markersize=5,
            label=_display(ct),
        )
        for ct in legend_cts
    ]

    # Panel letters (A, B, C); the caption owns the figure title. A and C
    # axes share a top edge, so one lift puts all three on matched baselines.
    from bimp_st.si_common import add_panel_headers

    add_panel_headers(
        fig,
        [
            (axes[0, 0], "A", "Cell type clusters"),
            (axes[1, 0], "B", "Probe clusters"),
            (ax_umap, "C", "Probe UMAP"),
        ],
        width_frac=1.0,
        dy_pts=13,
    )  # clear of Queen/Worker headers

    # Under row A, below its scale bars, spanning both sections.
    leg_top = (fig_h - hdr_in - panel_h - sb_in) / fig_h
    fig.legend(
        handles=handles_ct,
        fontsize=TICK_SIZE,
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(left_in / fig_w, leg_top),
        handletextpad=0.2,
        labelspacing=0.25,
        columnspacing=0.8,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    import json

    (OUTPUT_DIR / "fig_s_probe_clusters_stats.json").write_text(
        json.dumps(
            {
                "section_cells": _sect_n,
                "umap_subsample": int(n_umap_sub),
                "total_cells": int(sp.n_obs),
            },
            indent=2,
        )
    )
    save(fig, OUTPUT_DIR, "figS_probe_clusters")


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
